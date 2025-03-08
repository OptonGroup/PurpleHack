import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tqdm import tqdm

COLORS = ['beige', 'white', 'turquoise', 'burgundy', 'blue', 'yellow', 'green', 
          'gold', 'brown', 'red', 'orange', 'variegated', 'pink', 'silver', 
          'gray', 'blue', 'purple', 'black']
CATEGORIES = ['girls_clothes', 'tables', 'chairs', 'bags']

class ProductDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, is_test=False):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test
        self.category_to_idx = {cat: idx for idx, cat in enumerate(CATEGORIES)}
        self.color_to_idx = {color: idx for idx, color in enumerate(COLORS)}
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_id = self.df.iloc[idx]['id']
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        category = self.category_to_idx[self.df.iloc[idx]['category']]
        
        if not self.is_test:
            color = self.color_to_idx[self.df.iloc[idx]['target']]
            return image, category, color
        
        return image, category, img_id

class ColorClassifier(nn.Module):
    def __init__(self, num_colors, num_categories):
        super().__init__()
        self.backbone = models.efficientnet_b0(pretrained=True)
        for param in list(self.backbone.parameters())[:-30]:
            param.requires_grad = False
            
        num_features = self.backbone.classifier[1].in_features
        self.category_embedding = nn.Embedding(num_categories, 32)
        
        self.classifier = nn.Sequential(
            nn.Linear(num_features + 32, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_colors)
        )
        
    def forward(self, x, category):
        features = self.backbone.features(x)
        features = self.backbone.avgpool(features)
        features = torch.flatten(features, 1)
        
        category_emb = self.category_embedding(category)
        combined = torch.cat([features, category_emb], dim=1)
        
        return self.classifier(combined)

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    best_f1 = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        
        for images, categories, colors in tqdm(train_loader):
            images = images.to(device)
            categories = categories.to(device)
            colors = colors.to(device)
            
            optimizer.zero_grad()
            outputs = model(images, categories)
            loss = criterion(outputs, colors)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        model.eval()
        val_preds = []
        val_true = []
        
        with torch.no_grad():
            for images, categories, colors in val_loader:
                images = images.to(device)
                categories = categories.to(device)
                
                outputs = model(images, categories)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_true.extend(colors.numpy())
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            val_true, val_preds, average='macro'
        )
        
        if f1 > best_f1:
            best_f1 = f1
            best_model_state = model.state_dict().copy()
            
        scheduler.step()
        
    return best_model_state

def predict(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    predictions = []
    ids = []
    
    with torch.no_grad():
        for images, categories, img_ids in test_loader:
            images = images.to(device)
            categories = categories.to(device)
            
            outputs = model(images, categories)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            
            for img_id, prob in zip(img_ids, probs):
                pred_dict = {COLORS[i]: float(p) for i, p in enumerate(prob)}
                pred_color = COLORS[np.argmax(prob)]
                
                predictions.append({
                    'id': img_id,
                    'predict_proba': json.dumps(pred_dict),
                    'predict_color': pred_color
                })
                
    return pd.DataFrame(predictions)

def main():
    train_df = pd.read_csv('train_data.csv')
    test_df = pd.read_csv('test_data.csv')
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df['target'])
    train_dataset = ProductDataset(train_df, 'train_data', transform=train_transform)
    val_dataset = ProductDataset(val_df, 'train_data', transform=test_transform)
    test_dataset = ProductDataset(test_df, 'test_data', transform=test_transform, is_test=True)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    

    model = ColorClassifier(len(COLORS), len(CATEGORIES))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    
    best_model_state = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler
    )
    model.load_state_dict(best_model_state)
    predictions_df = predict(model, test_loader)
    predictions_df.to_csv('submission.csv', index=False)

if __name__ == '__main__':
    main()