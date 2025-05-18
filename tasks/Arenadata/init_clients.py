#!/usr/bin/env python3
"""
Скрипт для инициализации данных клиентов.
Загружает данные из telecom100k/psx и обрабатывает их для создания файлов клиентов.
"""

import os
from utils.data_extraction import load_client_dataframes
from utils.main import update_data
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """
    Основная функция инициализации данных клиентов.
    """
    try:
        # Проверка наличия файлов в директории clients
        clients_dir = 'clients'
        files_count = len(os.listdir(clients_dir)) if os.path.exists(clients_dir) else 0
        
        if files_count > 0:
            logger.info(f"Директория {clients_dir} уже содержит {files_count} файлов. Пропуск инициализации.")
            return True
        
        logger.info("Начало загрузки и обработки данных клиентов...")
        
        # Загрузка файлов
        files = load_client_dataframes()
        logger.info(f"Загружено {len(files)} файлов данных клиентов.")
        
        # Обновление данных клиентов
        stats = update_data(files, clinets_path=clients_dir)
        logger.info(f"Обновление данных клиентов завершено: {stats}")
        
        # Проверка результатов
        new_files_count = len(os.listdir(clients_dir)) if os.path.exists(clients_dir) else 0
        logger.info(f"После инициализации директория {clients_dir} содержит {new_files_count} файлов.")
        
        return True
    except Exception as e:
        logger.error(f"Ошибка при инициализации данных клиентов: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        logger.info("Инициализация данных клиентов успешно завершена.")
    else:
        logger.error("Инициализация данных клиентов завершилась с ошибкой.") 