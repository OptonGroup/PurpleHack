#!/usr/bin/env python3
"""
Скрипт-обертка для запуска кластеризации с предварительной инициализацией данных клиентов.
Оптимизирован для эффективного использования памяти.
"""

import os
import sys
import logging
import gc
from init_clients import main as init_clients

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """
    Основная функция запуска кластеризации с инициализацией.
    Оптимизирована для эффективного использования памяти.
    """
    try:
        # Сначала инициализируем данные клиентов
        logger.info("Запуск инициализации данных клиентов...")
        init_success = init_clients()
        
        if not init_success:
            logger.error("Инициализация данных клиентов завершилась с ошибкой. Прерывание процесса.")
            return False
        
        # Проверка наличия файлов в директории clients
        clients_dir = 'clients'
        files_list = os.listdir(clients_dir) if os.path.exists(clients_dir) else []
        files_count = len(files_list)
        
        if files_count == 0:
            logger.error(f"Директория {clients_dir} пуста после инициализации. Прерывание процесса.")
            return False
        
        logger.info(f"Директория {clients_dir} содержит {files_count} файлов. Запуск кластеризации...")
        
        # Вместо загрузки всех файлов сразу, загрузим только необходимые данные
        # и вызовем нужную функцию напрямую
        from utils.main import detection
        
        # Принудительная сборка мусора перед началом кластеризации
        gc.collect()
        
        # Запуск кластеризации с оптимизированными параметрами
        logger.info("Начало процесса кластеризации...")
        
        # Задаем меньшие значения eps и min_samples для снижения нагрузки на память
        # при сохранении качества кластеризации
        result = detection(
            clients_path='clients', 
            merge_threshold=0.1, 
            eps=0.25,  # Можно уменьшить для экономии памяти
            min_samples=10, 
            min_distance=0.6
        )
        
        # Сохраняем результаты и очищаем память
        result.to_csv('RESULT.csv', index=False)
        logger.info(f"Кластеризация успешно завершена. Результаты сохранены в RESULT.csv ({len(result)} записей).")
        
        # Освобождаем память
        del result
        gc.collect()
        
        return True
    except MemoryError as me:
        logger.error(f"Ошибка нехватки памяти при выполнении кластеризации: {str(me)}")
        logger.error("Попробуйте увеличить объем доступной памяти для Docker-контейнера или уменьшить параметры кластеризации.")
        return False
    except Exception as e:
        logger.error(f"Ошибка при выполнении кластеризации: {str(e)}")
        return False

if __name__ == "__main__":
    try:
        success = main()
        if success:
            logger.info("Процесс успешно завершен.")
            sys.exit(0)
        else:
            logger.error("Процесс завершен с ошибкой.")
            sys.exit(1)
    except MemoryError:
        logger.critical("Критическая ошибка нехватки памяти!")
        sys.exit(2) 