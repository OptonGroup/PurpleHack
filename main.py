"""
Основной файл для запуска оптимизации календарного плана проекта.
"""
import argparse
import logging
import sys
import os
from rich.console import Console
from rich.logging import RichHandler

from src.optimizer import optimize_schedule, OptimizationAlgorithm


def main():
    """
    Основная функция для запуска оптимизации календарного плана.
    """
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)]
    )
    logger = logging.getLogger("main")
    console = Console()
    
    # Создание парсера аргументов командной строки
    parser = argparse.ArgumentParser(description="Оптимизация календарного плана проекта с использованием машинного обучения")
    parser.add_argument(
        "-i", "--input", 
        type=str, 
        default="dataset.json", 
        help="Путь к входному файлу (по умолчанию: dataset.json)"
    )
    parser.add_argument(
        "-o", "--output", 
        type=str, 
        default="optimized_schedule.json", 
        help="Путь для сохранения оптимизированного плана (по умолчанию: optimized_schedule.json)"
    )
    parser.add_argument(
        "-d", "--duration-weight", 
        type=float, 
        default=1.0, 
        help="Вес длительности проекта при расчете награды (по умолчанию: 1.0)"
    )
    parser.add_argument(
        "-r", "--resource-weight", 
        type=float, 
        default=1.0, 
        help="Вес количества ресурсов при расчете награды (по умолчанию: 1.0)"
    )
    parser.add_argument(
        "-c", "--cost-weight", 
        type=float, 
        default=1.0, 
        help="Вес стоимости проекта при расчете награды (по умолчанию: 1.0)"
    )
    parser.add_argument(
        "-e", "--episodes", 
        type=int, 
        default=100, 
        help="Количество эпизодов для обучения (по умолчанию: 100)"
    )
    parser.add_argument(
        "-m", "--model", 
        type=str, 
        help="Путь к сохраненной модели (по умолчанию: None)"
    )
    parser.add_argument(
        "--save-model", 
        action="store_true", 
        help="Сохранять ли модель после обучения (по умолчанию: True)"
    )
    parser.add_argument(
        "--model-save-path", 
        type=str, 
        default="model.pt", 
        help="Путь для сохранения модели (по умолчанию: model.pt)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Выводить детальную информацию о процессе оптимизации"
    )
    # Добавляем опцию выбора алгоритма
    parser.add_argument(
        "-a", "--algorithm",
        type=str,
        choices=["reinforcement_learning", "simulated_annealing"],
        default="reinforcement_learning",
        help="Алгоритм оптимизации (по умолчанию: reinforcement_learning)"
    )
    # Добавляем параметры для алгоритма имитации отжига
    parser.add_argument(
        "--initial-temperature",
        type=float,
        default=100.0,
        help="Начальная температура для алгоритма имитации отжига (по умолчанию: 100.0)"
    )
    parser.add_argument(
        "--cooling-rate",
        type=float,
        default=0.95,
        help="Скорость охлаждения для алгоритма имитации отжига (по умолчанию: 0.95)"
    )
    parser.add_argument(
        "--min-temperature",
        type=float,
        default=0.1,
        help="Минимальная температура для алгоритма имитации отжига (по умолчанию: 0.1)"
    )
    parser.add_argument(
        "--iterations-per-temp",
        type=int,
        default=100,
        help="Количество итераций на каждой температуре для алгоритма имитации отжига (по умолчанию: 100)"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10000,
        help="Максимальное количество итераций для алгоритма имитации отжига (по умолчанию: 10000)"
    )
    
    # Парсинг аргументов
    args = parser.parse_args()
    
    # Проверка существования входного файла
    if not os.path.exists(args.input):
        logger.error(f"Входной файл {args.input} не найден!")
        return 1
    
    # Установка уровня логирования
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger.setLevel(log_level)
    
    # Также установим уровень логирования для корневого логгера и для логгера src
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    optimizer_logger = logging.getLogger("src")
    optimizer_logger.setLevel(log_level)
    
    try:
        # Запуск оптимизации
        logger.info("Запуск оптимизации календарного плана...")
        logger.info(f"Веса оптимизации: длительность={args.duration_weight}, ресурсы={args.resource_weight}, стоимость={args.cost_weight}")
        logger.info(f"Выбранный алгоритм: {args.algorithm}")
        
        if args.algorithm == "simulated_annealing":
            logger.info(f"Параметры имитации отжига: начальная температура={args.initial_temperature}, "
                         f"скорость охлаждения={args.cooling_rate}, минимальная температура={args.min_temperature}, "
                         f"итераций на температуру={args.iterations_per_temp}, максимум итераций={args.max_iterations}")
        
        optimized_schedule = optimize_schedule(
            input_file=args.input,
            output_file=args.output,
            duration_weight=args.duration_weight,
            resource_weight=args.resource_weight,
            cost_weight=args.cost_weight,
            num_episodes=args.episodes,
            model_path=args.model,
            save_model=args.save_model,
            model_save_path=args.model_save_path,
            log_level=log_level,
            algorithm=args.algorithm,
            initial_temperature=args.initial_temperature,
            cooling_rate=args.cooling_rate,
            min_temperature=args.min_temperature,
            iterations_per_temp=args.iterations_per_temp,
            max_iterations=args.max_iterations
        )
        
        logger.info("Оптимизация успешно завершена!")
        logger.info(f"Общая длительность проекта: {optimized_schedule.totalDuration:.2f} дней")
        logger.info(f"Использование ресурсов: {optimized_schedule.resourceUtilization:.2f}%")
        logger.info(f"Общая стоимость проекта: {optimized_schedule.totalCost:.2f}")
        logger.info(f"Оптимизированный план сохранен в {args.output}")
        
        return 0
    
    except Exception as e:
        logger.exception(f"Произошла ошибка при оптимизации: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 