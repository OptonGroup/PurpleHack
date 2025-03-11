#!/usr/bin/env python
"""
Скрипт для обучения и сохранения модели оптимизации календарного плана.

Этот скрипт загружает данные из файла JSON, обучает RL-агента и
сохраняет модель в указанный файл для последующего использования.
"""
import os
import sys
import logging
import argparse
from rich.logging import RichHandler
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn

# Настраиваем пути
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.optimizer import optimize_schedule

def train_model():
    """
    Обучает модель и сохраняет ее.
    """
    parser = argparse.ArgumentParser(description="Обучение модели оптимизации календарного плана")
    parser.add_argument("-i", "--input", default="dataset.json", help="Путь к входному файлу JSON")
    parser.add_argument("-m", "--model", default="trained_model.pt", help="Путь для сохранения модели")
    parser.add_argument("-d", "--duration-weight", type=float, default=7.0, help="Вес длительности проекта")
    parser.add_argument("-r", "--resource-weight", type=float, default=3.0, help="Вес ресурсов")
    parser.add_argument("-c", "--cost-weight", type=float, default=1.0, help="Вес стоимости проекта")
    parser.add_argument("-e", "--episodes", type=int, default=500, help="Количество эпизодов обучения")
    parser.add_argument("-v", "--verbose", action="store_true", help="Подробный вывод")
    args = parser.parse_args()

    # Настраиваем логирование
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)]
    )
    
    logger = logging.getLogger("train_model")
    console = Console()
    
    # Проверяем наличие входного файла
    if not os.path.exists(args.input):
        logger.error(f"Входной файл {args.input} не найден")
        return 1
    
    try:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Обучение модели...", total=100)
            
            # Запускаем обучение с прогресс-баром
            def update_progress(percentage):
                progress.update(task, completed=percentage)
            
            # Запускаем оптимизацию, но не сохраняем результаты в файл, только модель
            optimize_schedule(
                input_file=args.input,
                output_file=None,  # Не сохраняем результаты
                duration_weight=args.duration_weight,
                resource_weight=args.resource_weight,
                cost_weight=args.cost_weight,
                num_episodes=args.episodes,
                model_path=None,  # Не загружаем модель
                save_model=True,
                model_save_path=args.model,
                log_level=log_level,
                progress_callback=update_progress
            )
            
            progress.update(task, completed=100)
        
        console.print(f"[bold green]Модель успешно обучена и сохранена в файл {args.model}[/bold green]")
        return 0
    except Exception as e:
        logger.exception(f"Произошла ошибка при обучении модели: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(train_model()) 