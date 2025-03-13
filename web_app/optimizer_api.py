#!/usr/bin/env python
"""
API для работы с оптимизатором календарного плана.

Предоставляет функции для валидации входных данных, запуска оптимизации,
отслеживания прогресса и получения результатов.
"""
import os
import sys
import json
import uuid
import time
import logging
import threading
from typing import Dict, Any, List, Tuple, Optional, Union, Literal
from datetime import datetime, timedelta
from fastapi import BackgroundTasks

# Добавляем корневую директорию проекта в путь для импорта модулей
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Импортируем модули оптимизатора (замените на фактические импорты)
try:
    from src.optimizer import optimize_schedule, OptimizationAlgorithm
    from src.data.models import Dataset, convert_dataset_to_project_input, ProjectOutput
    OPTIMIZER_AVAILABLE = True
except ImportError:
    OPTIMIZER_AVAILABLE = False
    logging.warning("Модули оптимизатора не найдены, будет использован эмулятор")

# Настраиваем логирование
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Директория для временных файлов
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

# Словарь для хранения статусов задач оптимизации
optimization_tasks = {}

def validate_input_json(file_path: str) -> Tuple[bool, List[str], List[str]]:
    """
    Валидирует JSON-файл на соответствие формату входных данных.
    
    Args:
        file_path: Путь к JSON-файлу
        
    Returns:
        Кортеж (is_valid, errors, warnings)
    """
    errors = []
    warnings = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Проверка наличия основных разделов
        if not data.get("project"):
            errors.append("Отсутствует раздел 'project'")
        
        if not data.get("tasks") or not data.get("tasks").get("rows"):
            errors.append("Отсутствует раздел 'tasks' или 'tasks.rows'")
        elif len(data.get("tasks", {}).get("rows", [])) == 0:
            warnings.append("Раздел 'tasks.rows' пуст")
        
        if not data.get("resources") or not data.get("resources").get("rows"):
            errors.append("Отсутствует раздел 'resources' или 'resources.rows'")
        elif len(data.get("resources", {}).get("rows", [])) == 0:
            warnings.append("Раздел 'resources.rows' пуст")
        
        if not data.get("dependencies") or not data.get("dependencies").get("rows"):
            warnings.append("Отсутствует раздел 'dependencies' или 'dependencies.rows'")
        
        # Проверка стартовой даты проекта
        if "project" in data and not data["project"].get("startDate"):
            errors.append("Отсутствует поле 'startDate' в разделе 'project'")
        
        # Если используем OPTIMIZER_AVAILABLE, можем выполнить более глубокую валидацию
        if OPTIMIZER_AVAILABLE:
            try:
                # Создаем объект Dataset для валидации
                dataset = Dataset(
                    requestId=data.get("requestId", str(uuid.uuid4())),
                    project=data.get("project", {}),
                    success=data.get("success", True),
                    tasks=data.get("tasks", {"rows": []}),
                    resources=data.get("resources", {"rows": []}),
                    dependencies=data.get("dependencies", {"rows": []}),
                    projectCalendar=data.get("projectCalendar")
                )
                
                # Пытаемся преобразовать Dataset в ProjectInput
                project_input = convert_dataset_to_project_input(dataset)
                
                # Проверяем наличие задач и ресурсов
                if not project_input.tasks:
                    errors.append("Не удалось загрузить задачи из JSON")
                
                if not project_input.resources:
                    errors.append("Не удалось загрузить ресурсы из JSON")
                
            except Exception as e:
                errors.append(f"Ошибка валидации через оптимизатор: {str(e)}")
        
        is_valid = len(errors) == 0
        return is_valid, errors, warnings
    
    except json.JSONDecodeError as e:
        return False, [f"Некорректный JSON: {str(e)}"], []
    except Exception as e:
        return False, [f"Ошибка при валидации: {str(e)}"], []

def _run_optimization_task(
    job_id: str,
    input_file: str,
    output_file: str,
    duration_weight: float,
    resource_weight: float,
    cost_weight: float,
    num_episodes: int,
    model_path: Optional[str],
    algorithm: OptimizationAlgorithm,
    # Параметры для алгоритма обучения с подкреплением
    learning_rate: float,
    gamma: float,
    # Параметры для алгоритма имитации отжига
    initial_temperature: float,
    cooling_rate: float,
    min_temperature: float,
    iterations_per_temp: int,
    max_iterations: int
) -> None:
    """
    Выполняет задачу оптимизации в отдельном потоке.
    
    Args:
        job_id: Идентификатор задачи
        input_file: Путь к входному файлу JSON
        output_file: Путь для сохранения оптимизированного плана
        duration_weight: Вес длительности проекта
        resource_weight: Вес ресурсов
        cost_weight: Вес стоимости проекта
        num_episodes: Количество эпизодов обучения
        model_path: Путь к предобученной модели
        algorithm: Алгоритм оптимизации ('reinforcement_learning' или 'simulated_annealing')
        learning_rate: Темп обучения для алгоритма обучения с подкреплением
        gamma: Коэффициент дисконтирования для алгоритма обучения с подкреплением
        initial_temperature: Начальная температура для алгоритма имитации отжига
        cooling_rate: Скорость охлаждения для алгоритма имитации отжига
        min_temperature: Минимальная температура для алгоритма имитации отжига
        iterations_per_temp: Количество итераций на каждой температуре для алгоритма имитации отжига
        max_iterations: Максимальное количество итераций для алгоритма имитации отжига
    """
    try:
        # Обновляем статус
        optimization_tasks[job_id]["status"] = "running"
        optimization_tasks[job_id]["progress"] = 0
        optimization_tasks[job_id]["start_time"] = time.time()
        
        # Функция для отслеживания прогресса
        def update_progress(progress):
            optimization_tasks[job_id]["progress"] = progress
        
        if OPTIMIZER_AVAILABLE:
            # Выполняем оптимизацию с использованием настоящего оптимизатора
            result = optimize_schedule(
                input_file=input_file,
                output_file=output_file,
                duration_weight=duration_weight,
                resource_weight=resource_weight,
                cost_weight=cost_weight,
                num_episodes=num_episodes,
                model_path=model_path,
                progress_callback=update_progress,
                algorithm=algorithm,
                # Параметры для алгоритма обучения с подкреплением
                learning_rate=learning_rate,
                gamma=gamma,
                # Параметры для алгоритма имитации отжига
                initial_temperature=initial_temperature,
                cooling_rate=cooling_rate,
                min_temperature=min_temperature,
                iterations_per_temp=iterations_per_temp,
                max_iterations=max_iterations
            )
            
            # Сохраняем результаты
            optimization_tasks[job_id]["result"] = {
                "tasks": [task.dict() for task in result.tasks],
                "totalDuration": result.totalDuration,
                "resourceUtilization": result.resourceUtilization,
                "totalCost": result.totalCost
            }
            
            optimization_tasks[job_id]["status"] = "completed"
            optimization_tasks[job_id]["progress"] = 100
            optimization_tasks[job_id]["end_time"] = time.time()
        else:
            # Эмулируем выполнение оптимизации
            for i in range(10):
                time.sleep(1)  # Имитация работы
                update_progress(i * 10)
            
            # Генерируем случайный план
            random_schedule = generate_random_plan()
            optimization_tasks[job_id]["result"] = random_schedule
            optimization_tasks[job_id]["status"] = "completed"
            optimization_tasks[job_id]["progress"] = 100
            optimization_tasks[job_id]["end_time"] = time.time()
            
            # Сохраняем результаты в файл
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(random_schedule, f, ensure_ascii=False, indent=2)
        
    except Exception as e:
        # В случае ошибки
        logging.error(f"Ошибка при выполнении оптимизации {job_id}: {str(e)}")
        optimization_tasks[job_id]["status"] = "failed"
        optimization_tasks[job_id]["error"] = str(e)
        optimization_tasks[job_id]["end_time"] = time.time()

def optimize_plan(
    data: Dict[str, Any],
    duration_weight: float = 7.0,
    resource_weight: float = 3.0,
    cost_weight: float = 1.0,
    num_episodes: int = 500,
    use_pretrained_model: bool = False,
    model_path: Optional[str] = None,
    background_tasks: Optional[BackgroundTasks] = None,
    algorithm: OptimizationAlgorithm = "reinforcement_learning",
    # Параметры для алгоритма обучения с подкреплением
    learning_rate: float = 0.001,
    gamma: float = 0.99,
    # Параметры для алгоритма имитации отжига
    initial_temperature: float = 100.0,
    cooling_rate: float = 0.95,
    min_temperature: float = 0.1,
    iterations_per_temp: int = 100,
    max_iterations: int = 10000
) -> str:
    """
    Запускает оптимизацию плана проекта.
    
    Args:
        data: Данные проекта в формате JSON
        duration_weight: Вес длительности проекта
        resource_weight: Вес ресурсов
        cost_weight: Вес стоимости проекта
        num_episodes: Количество эпизодов обучения
        use_pretrained_model: Использовать ли предобученную модель
        model_path: Путь к предобученной модели
        background_tasks: Объект BackgroundTasks для фоновых задач
        algorithm: Алгоритм оптимизации ('reinforcement_learning' или 'simulated_annealing')
        learning_rate: Темп обучения для алгоритма обучения с подкреплением
        gamma: Коэффициент дисконтирования для алгоритма обучения с подкреплением
        initial_temperature: Начальная температура для алгоритма имитации отжига
        cooling_rate: Скорость охлаждения для алгоритма имитации отжига
        min_temperature: Минимальная температура для алгоритма имитации отжига
        iterations_per_temp: Количество итераций на каждой температуре для алгоритма имитации отжига
        max_iterations: Максимальное количество итераций для алгоритма имитации отжига
        
    Returns:
        Идентификатор задачи оптимизации
    """
    # Директория для временных файлов
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # Создаем идентификатор задачи
    job_id = str(uuid.uuid4())
    
    # Пути к файлам
    input_file = os.path.join(TEMP_DIR, f"{job_id}_input.json")
    output_file = os.path.join(TEMP_DIR, f"{job_id}_output.json")
    model_file = os.path.join(TEMP_DIR, f"{job_id}_model.pt")
    
    # Сохраняем входные данные
    with open(input_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    # Определяем, какую модель использовать
    model_to_use = None
    if use_pretrained_model and model_path:
        if os.path.exists(model_path):
            # Используем указанную модель
            model_to_use = model_path
        else:
            # Модель не найдена, используем стандартную
            model_to_use = "trained_model.pt"
    
    # Сохраняем информацию о задаче
    optimization_tasks[job_id] = {
        "id": job_id,
        "status": "pending",
        "progress": 0,
        "start_time": None,
        "end_time": None,
        "input_file": input_file,
        "output_file": output_file,
        "duration_weight": duration_weight,
        "resource_weight": resource_weight,
        "cost_weight": cost_weight,
        "num_episodes": num_episodes,
        "use_pretrained_model": use_pretrained_model,
        "model_path": model_to_use,
        "algorithm": algorithm,
        # Параметры для алгоритма обучения с подкреплением
        "learning_rate": learning_rate,
        "gamma": gamma,
        # Параметры для алгоритма имитации отжига
        "initial_temperature": initial_temperature,
        "cooling_rate": cooling_rate,
        "min_temperature": min_temperature,
        "iterations_per_temp": iterations_per_temp,
        "max_iterations": max_iterations,
        "result": None,
        "error": None
    }
    
    # Запускаем задачу оптимизации в фоновом режиме
    if background_tasks:
        background_tasks.add_task(
            _run_optimization_task,
            job_id,
            input_file,
            output_file,
            duration_weight,
            resource_weight,
            cost_weight,
            num_episodes if not use_pretrained_model else 0,
            model_to_use,
            algorithm,
            # Параметры для алгоритма обучения с подкреплением
            learning_rate,
            gamma,
            # Параметры для алгоритма имитации отжига
            initial_temperature,
            cooling_rate,
            min_temperature,
            iterations_per_temp,
            max_iterations
        )
    else:
        thread = threading.Thread(
            target=_run_optimization_task,
            args=(
                job_id,
                input_file,
                output_file,
                duration_weight,
                resource_weight,
                cost_weight,
                num_episodes if not use_pretrained_model else 0,
                model_to_use,
                algorithm,
                # Параметры для алгоритма обучения с подкреплением
                learning_rate,
                gamma,
                # Параметры для алгоритма имитации отжига
                initial_temperature,
                cooling_rate,
                min_temperature,
                iterations_per_temp,
                max_iterations
            )
        )
        thread.daemon = True
        thread.start()
    
    return job_id

def get_optimization_status(job_id: str) -> Dict[str, Any]:
    """
    Возвращает статус задачи оптимизации.
    
    Args:
        job_id: Идентификатор задачи
        
    Returns:
        Словарь с информацией о статусе задачи
    """
    if job_id not in optimization_tasks:
        return {"id": job_id, "status": "not_found"}
    
    # Получаем задачу
    task = optimization_tasks[job_id]
    
    # Группируем параметры в подсловарь params
    result = {
        "id": task["id"],
        "status": task["status"],
        "progress": task["progress"],
        "start_time": task["start_time"],
        "end_time": task["end_time"],
        "input_file": task["input_file"],
        "output_file": task["output_file"],
        "result": task["result"],
        "error": task["error"],
        "params": {
            "duration_weight": task["duration_weight"],
            "resource_weight": task["resource_weight"],
            "cost_weight": task["cost_weight"],
            "num_episodes": task["num_episodes"],
            "use_pretrained_model": task["use_pretrained_model"],
            "model_path": task["model_path"],
            "algorithm": task["algorithm"],
            "learning_rate": task["learning_rate"],
            "gamma": task["gamma"],
            "initial_temperature": task["initial_temperature"],
            "cooling_rate": task["cooling_rate"],
            "min_temperature": task["min_temperature"],
            "iterations_per_temp": task["iterations_per_temp"],
            "max_iterations": task["max_iterations"]
        }
    }
    
    # Вычисляем дополнительные поля
    if task["start_time"]:
        result["created_at"] = datetime.fromtimestamp(task["start_time"]).strftime("%Y-%m-%d %H:%M:%S")
    
    if task["end_time"]:
        result["end_time"] = datetime.fromtimestamp(task["end_time"]).strftime("%Y-%m-%d %H:%M:%S")
    
    return result

def get_all_optimization_results() -> List[Dict[str, Any]]:
    """
    Возвращает список всех результатов оптимизации.
    
    Returns:
        Список словарей с информацией о задачах оптимизации
    """
    results = []
    
    for job_id in optimization_tasks:
        # Используем уже существующую функцию для получения результата в нужном формате
        results.append(get_optimization_status(job_id))
    
    # Сортируем результаты по времени создания (по убыванию)
    return sorted(
        results,
        key=lambda x: x.get("created_at", ""),
        reverse=True
    )

def generate_random_plan() -> Dict[str, Any]:
    """
    Генерирует случайный план проекта для тестирования.
    
    Returns:
        Словарь с тестовым планом проекта
    """
    # Текущая дата
    current_date = datetime.now()
    
    # Генерируем случайные задачи
    tasks = []
    for i in range(5):
        start_date = current_date + timedelta(days=i*3)
        end_date = start_date + timedelta(days=2)
        
        tasks.append({
            "id": f"task_{i+1}",
            "startDate": start_date.isoformat(),
            "endDate": end_date.isoformat(),
            "assignedResourceId": f"resource_{(i % 3) + 1}"
        })
    
    # Создаем тестовый результат
    return {
        "tasks": tasks,
        "totalDuration": 14,
        "resourceUtilization": 85.7,
        "totalCost": 500.0
    } 