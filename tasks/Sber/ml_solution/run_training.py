"""
Скрипт для быстрого запуска обучения модели.
Поддерживает запуск как на GPU (RTX 3070), так и на CPU.
"""

import os
import subprocess
import sys
from pathlib import Path


def check_device():
    """Проверка доступного устройства для обучения."""
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            device_name = torch.cuda.get_device_name(0)
            print(f"Найден GPU: {device_name}")
            if "RTX 3070" in device_name:
                print("Обнаружена RTX 3070 - будут использованы оптимизированные настройки")
            else:
                print("Предупреждение: конфигурация оптимизирована для RTX 3070, возможно потребуется корректировка параметров")
        else:
            device = "cpu"
            print("GPU не найден, будет использован CPU (обучение может занять значительно больше времени)")
        return device
    except ImportError:
        print("PyTorch не установлен. Установите зависимости.")
        sys.exit(1)


def setup_environment():
    """Подготовка окружения."""
    # Создание необходимых директорий
    dirs = ['checkpoints', 'logs', 'logs/tensorboard']
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)


def install_requirements():
    """Установка зависимостей."""
    try:
        # Определяем версию PyTorch в зависимости от наличия CUDA
        import torch
        if torch.cuda.is_available():
            print("Установка PyTorch с поддержкой CUDA...")
            torch_command = "--extra-index-url https://download.pytorch.org/whl/cu118 torch==2.2.0+cu118"
        else:
            print("Установка PyTorch для CPU...")
            torch_command = "torch==2.2.0"
        
        # Сначала устанавливаем PyTorch
        subprocess.run(
            [sys.executable, "-m", "pip", "install", torch_command],
            check=True
        )
        
        # Затем остальные зависимости
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "tasks\Sber\ml_solution\requirements.txt"],
            check=True
        )
        print("Зависимости успешно установлены")
    except subprocess.CalledProcessError as e:
        print(f"Ошибка при установке зависимостей: {str(e)}")
        sys.exit(1)


def adjust_config_for_cpu():
    """Корректировка конфигурации для CPU."""
    try:
        import yaml
        config_path = "configs/model_config.yaml"
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Уменьшаем размеры батчей и модели для CPU
        config['training']['batch_size'] = 16
        config['model']['architecture']['task_embedding_dim'] = 128
        config['model']['architecture']['resource_embedding_dim'] = 64
        config['model']['architecture']['hidden_dim'] = 256
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True)
            
        print("Конфигурация скорректирована для CPU")
    except Exception as e:
        print(f"Ошибка при корректировке конфигурации: {str(e)}")


def main():
    """Основная функция запуска."""
    print("Подготовка к запуску обучения...")
    
    # Проверка наличия requirements.txt
    if not Path("requirements.txt").exists():
        print("Файл requirements.txt не найден")
        sys.exit(1)
    
    # Установка зависимостей
    # print("Установка зависимостей...")
    # install_requirements()
    
    # Проверка устройства
    print("Проверка доступного устройства...")
    device = check_device()
    
    # Подготовка окружения
    print("Подготовка окружения...")
    setup_environment()
    
    # Корректировка конфигурации для CPU при необходимости
    if device == "cpu":
        print("Корректировка конфигурации для CPU...")
        adjust_config_for_cpu()
    
    # Запуск обучения
    print(f"Запуск обучения на {device.upper()}...")
    try:
        subprocess.run(
            [sys.executable, "src/train.py"],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Ошибка при запуске обучения: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 