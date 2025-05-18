#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Скрипт для инициализации модели и загрузки весов.
"""

import os
import torch
import torch.nn as nn
import timm
import requests
from tqdm import tqdm

# Словари соответствия
TRANSLIT_TO_RU = {
    'bezhevyi': 'бежевый',
    'belyi': 'белый',
    'biryuzovyi': 'бирюзовый',
    'bordovyi': 'бордовый',
    'goluboi': 'голубой',
    'zheltyi': 'желтый',
    'zelenyi': 'зеленый',
    'zolotoi': 'золотой',
    'korichnevyi': 'коричневый',
    'krasnyi': 'красный',
    'oranzhevyi': 'оранжевый',
    'raznocvetnyi': 'разноцветный',
    'rozovyi': 'розовый',
    'serebristyi': 'серебряный',
    'seryi': 'серый',
    'sinii': 'синий',
    'fioletovyi': 'фиолетовый',
    'chernyi': 'черный'
}

# Создаем обратное соответствие с русского на транслитерацию
RU_TO_TRANSLIT = {v: k for k, v in TRANSLIT_TO_RU.items()}

# Словарь цветов
COLORS = {
    'бежевый': 'beige',
    'белый': 'white',
    'бирюзовый': 'turquoise',
    'бордовый': 'burgundy',
    'голубой': 'blue',
    'желтый': 'yellow',
    'зеленый': 'green',
    'золотой': 'gold',
    'коричневый': 'brown',
    'красный': 'red',
    'оранжевый': 'orange',
    'разноцветный': 'variegated',
    'розовый': 'pink',
    'серебряный': 'silver',
    'серый': 'gray',
    'синий': 'blue',
    'фиолетовый': 'purple',
    'черный': 'black'
}

# Категории
CATEGORIES = ['одежда для девочек', 'столы', 'стулья', 'сумки']

# Глобальная переменная для хранения загруженной модели
MODEL = None
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ColorClassifier(nn.Module):
    def __init__(self, num_colors, num_categories):
        super().__init__()
        # Используем более легкий и быстрый вариант ViT
        self.backbone = timm.create_model(
            'beitv2_large_patch16_224', 
            pretrained=True, 
            num_classes=0,  # Без верхнего слоя классификации
        )
        
        # Фиксируем большую часть весов для ускорения обучения
        for param in list(self.backbone.parameters())[:-30]:
            param.requires_grad = False
            
        # Расширение для быстрой инференции с кэшированием
        self.backbone.reset_classifier(0)
        
        # Размерность признаков модели
        self.feature_dim = self.backbone.embed_dim
        
        # Эмбеддинг категории
        self.category_embedding = nn.Embedding(num_categories, 32)
        
        # Классификационная голова
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim + 32, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_colors)
        )
        
        # Для оптимизации torch.jit
        self.example_input = torch.zeros(1, 3, 224, 224)
        self.example_category = torch.LongTensor([0])
        
    def forward(self, x, category):
        features = self.backbone(x)
        
        category_emb = self.category_embedding(category)
        combined = torch.cat([features, category_emb], dim=1)
        
        return self.classifier(combined)

def download_weights(url, destination, chunk_size=8192):
    """
    Загружает файл с весами модели по указанному URL.
    
    Args:
        url (str): URL для загрузки весов
        destination (str): Путь для сохранения файла
        chunk_size (int): Размер чанка для загрузки
    """
    # Проверяем, существует ли уже файл
    if os.path.exists(destination):
        print(f"Файл с весами уже существует: {destination}")
        return True
    
    # Создаем директорию, если не существует
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Проверка на ошибки HTTP
        
        total_size = int(response.headers.get('content-length', 0))
        with open(destination, 'wb') as f, tqdm(
            desc=os.path.basename(destination),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                size = f.write(chunk)
                bar.update(size)
        
        print(f"Файл успешно загружен и сохранен в {destination}")
        return True
    except Exception as e:
        print(f"Ошибка при загрузке файла: {str(e)}")
        return False

def load_model(model_path):
    """
    Загружает ранее обученную модель из указанного пути.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Пробуем загрузить как TorchScript модель
        model = torch.jit.load(model_path, map_location=device)
        print("Загружена оптимизированная TorchScript модель")
        return model
    except:
        # Загружаем как обычную модель
        print("Загрузка модели из стандартных весов...")
        model = ColorClassifier(len(COLORS), len(CATEGORIES))
        
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        # Установка модели в режим оценки
        model.eval()
        model = model.to(device)
        
        return model

def initialize_model(model_path="/app/weights/model_weights.pth", weights_url=None):
    """
    Инициализирует модель один раз и сохраняет её в глобальной переменной.
    Если указан URL весов и файл с весами не существует, загружает веса.
    
    Args:
        model_path (str): Путь к весам модели
        weights_url (str): URL для загрузки весов модели, если файл не существует
        
    Returns:
        Загруженная модель
    """
    global MODEL
    
    # Если указан URL весов и файл не существует, загружаем веса
    if weights_url and not os.path.exists(model_path):
        print(f"Файл с весами не найден: {model_path}")
        print(f"Загрузка весов с URL: {weights_url}")
        if not download_weights(weights_url, model_path):
            raise RuntimeError("Не удалось загрузить веса модели")
    
    if MODEL is None:
        print("Загрузка модели в первый раз...")
        MODEL = load_model(model_path)
    else:
        print("Модель уже загружена, повторное использование...")
    
    return MODEL

# Пример использования
if __name__ == "__main__":
    # URL для загрузки весов (актуальный URL)
    WEIGHTS_URL = "https://drive.usercontent.google.com/download?id=17p0eXDdtpcpHF4kmApanexzqGWFFXdir&export=download&authuser=0&confirm=t&uuid=d335c718-4016-44be-8212-f8846e1bc333&at=AEz70l75v8L-U5sRzAy--ChfoK6D%3A1741921834963"
    # Путь для сохранения весов
    WEIGHTS_PATH = "weights/model_weights.pth"
    
    # Инициализация модели с автоматической загрузкой весов при необходимости
    model = initialize_model(WEIGHTS_PATH, WEIGHTS_URL)
    print("Модель успешно инициализирована") 