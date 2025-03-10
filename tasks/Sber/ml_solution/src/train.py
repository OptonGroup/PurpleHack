"""Main training script for project scheduler model."""

import logging
import os
import sys
from pathlib import Path

# Добавляем корневую директорию проекта в PYTHONPATH
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

import hydra
import torch
import torch.backends.cudnn as cudnn
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.data_generation.synthetic_data import SyntheticDataGenerator
from src.models.scheduler_model import SchedulerModel
from src.preprocessing.data_processor import DataProcessor
from src.training.trainer import SchedulerDataset, Trainer


def setup_environment(device: str) -> None:
    """Setup training environment.
    
    Args:
        device: Device to use for training ('cuda' or 'cpu')
    """
    if device == "cuda":
        # Оптимизация CUDA
        cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Создание директорий
    dirs = ['checkpoints', 'logs']
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)


def setup_logging() -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/training.log")
        ]
    )
    return logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="model_config")
def main(config: DictConfig) -> None:
    """Main training function."""
    # Определение устройства
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Базовая настройка
    setup_environment(device)
    logger = setup_logging()
    logger.info("Starting training pipeline")
    logger.info(f"Using device: {device.upper()}")
    
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    try:
        # Инициализация модели
        model = SchedulerModel(config).to(device)
        logger.info("Model initialized")
        
        # Генерация данных
        generator = SyntheticDataGenerator(config)
        data_processor = DataProcessor(config)
        
        logger.info("Generating training data...")
        dataset = generator.generate_dataset(
            num_samples=config.data.num_samples,
            min_tasks=config.data.synthetic.min_tasks,
            max_tasks=config.data.synthetic.max_tasks,
            min_resources=config.data.synthetic.min_resources,
            max_resources=config.data.synthetic.max_resources
        )
        
        # Разделение на train/val
        train_size = int(len(dataset) * config.data.train_val_split)
        val_size = len(dataset) - train_size
        
        train_dataset = SchedulerDataset(dataset[:train_size])
        val_dataset = SchedulerDataset(dataset[train_size:])
        
        logger.info(f"Dataset split: {len(train_dataset)} train, {len(val_dataset)} val")
        
        # Создание DataLoader'ов
        # Для CPU уменьшаем количество workers
        num_workers = 4 if device == "cuda" else 2
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(device == "cuda")
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device == "cuda")
        )
        
        # Инициализация тренера и запуск обучения
        trainer = Trainer(model, config, device)
        trainer.train(
            train_loader,
            val_loader,
            config.training.num_epochs,
            Path(config.training.checkpoint_dir)
        )
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main() 