#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import requests
from tqdm import tqdm


def download_file(url, output_path):
    """
    Скачивает файл с указанного URL в указанный путь с отображением прогресса.
    
    Args:
        url (str): URL для скачивания файла
        output_path (str): Путь для сохранения файла
    
    Returns:
        bool: True в случае успеха, False в случае ошибки
    """
    try:
        # Создаем директории, если они не существуют
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Инициализируем запрос и получаем размер файла
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 KB
        
        # Открываем файл для записи и запускаем прогресс-бар
        with open(output_path, 'wb') as file, tqdm(
            desc=f"Загрузка {os.path.basename(output_path)}",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                file.write(data)
                bar.update(len(data))
        
        print(f"Файл успешно скачан в {output_path}")
        return True
    except Exception as e:
        print(f"Ошибка при скачивании файла: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Скачать веса модели для классификации цветов')
    parser.add_argument('--url', type=str, required=True, help='URL для скачивания весов модели')
    parser.add_argument('--output', type=str, default='weights/model_weights.pth', 
                       help='Путь для сохранения весов (по умолчанию: weights/model_weights.pth)')
    
    args = parser.parse_args()
    
    print(f"Начинаем загрузку весов модели с {args.url}")
    success = download_file(args.url, args.output)
    
    if success:
        print("Загрузка весов модели завершена успешно")
    else:
        print("Не удалось загрузить веса модели")
        exit(1)


if __name__ == "__main__":
    main() 