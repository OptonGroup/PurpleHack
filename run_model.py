#!/usr/bin/env python
"""
Скрипт для запуска оптимизации с использованием предобученной модели.

Этот скрипт загружает предобученную модель и применяет ее для
оптимизации календарного плана из указанного файла JSON.
"""
import os
import sys
import json
import logging
import argparse
from datetime import datetime
from rich.logging import RichHandler
from rich.console import Console
from rich.table import Table
from rich import print as rprint

# Настраиваем пути
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.optimizer import optimize_schedule

def run_model():
    """
    Загружает модель и применяет ее для оптимизации.
    """
    parser = argparse.ArgumentParser(description="Запуск оптимизации с использованием предобученной модели")
    parser.add_argument("-i", "--input", default="dataset.json", help="Путь к входному файлу JSON")
    parser.add_argument("-o", "--output", default="optimized_schedule.json", help="Путь для сохранения оптимизированного плана")
    parser.add_argument("-m", "--model", default="trained_model.pt", help="Путь к предобученной модели")
    parser.add_argument("-d", "--duration-weight", type=float, default=7.0, help="Вес длительности проекта")
    parser.add_argument("-r", "--resource-weight", type=float, default=3.0, help="Вес ресурсов")
    parser.add_argument("-c", "--cost-weight", type=float, default=1.0, help="Вес стоимости проекта")
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
    
    logger = logging.getLogger("run_model")
    console = Console()
    
    # Проверяем наличие входного файла и модели
    if not os.path.exists(args.input):
        logger.error(f"Входной файл {args.input} не найден")
        return 1
    
    if not os.path.exists(args.model):
        logger.error(f"Файл модели {args.model} не найден")
        return 1
    
    try:
        console.print(f"[bold]Запуск оптимизации с использованием модели {args.model}[/bold]")
        
        # Запускаем оптимизацию с предобученной моделью
        optimized_schedule = optimize_schedule(
            input_file=args.input,
            output_file=args.output,
            duration_weight=args.duration_weight,
            resource_weight=args.resource_weight,
            cost_weight=args.cost_weight,
            num_episodes=0,  # Не обучаем заново, используем только предобученную модель
            model_path=args.model,
            save_model=False,  # Не перезаписываем модель
            log_level=log_level
        )
        
        # Выводим результаты на экран
        console.print("[bold]Результаты оптимизации:[/bold]")
        
        table = Table(title="Оптимизированное расписание")
        table.add_column("ID задачи", style="cyan")
        table.add_column("Начало", style="green")
        table.add_column("Окончание", style="red")
        table.add_column("Ресурс", style="magenta")
        
        for task in optimized_schedule.tasks:
            start_date = task.startDate if isinstance(task.startDate, str) else task.startDate.strftime("%Y-%m-%d")
            end_date = task.endDate if isinstance(task.endDate, str) else task.endDate.strftime("%Y-%m-%d")
            
            table.add_row(
                task.id,
                start_date,
                end_date,
                task.assignedResourceId
            )
        
        console.print(table)
        
        # Выводим общие результаты
        rprint(f"[bold]Общая длительность проекта:[/bold] {optimized_schedule.totalDuration:.2f} дней")
        rprint(f"[bold]Использование ресурсов:[/bold] {optimized_schedule.resourceUtilization:.2f}%")
        rprint(f"[bold]Общая стоимость проекта:[/bold] {optimized_schedule.totalCost:.2f}")
        
        console.print(f"[bold green]Результаты сохранены в файл {args.output}[/bold green]")
        return 0
    except Exception as e:
        logger.exception(f"Произошла ошибка при запуске оптимизации: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(run_model()) 