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
from typing import Dict, Any, List, Tuple, Optional, Union
from datetime import datetime, timedelta
from fastapi import BackgroundTasks

# Добавляем корневую директорию проекта в путь для импорта модулей
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Импортируем модули оптимизатора (замените на фактические импорты)
try:
    from src.optimizer import optimize_schedule
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
    model_path: Optional[str]
):
    """
    Запускает задачу оптимизации в фоновом режиме.
    
    Args:
        job_id: Идентификатор задачи
        input_file: Путь к входному файлу
        output_file: Путь к выходному файлу
        duration_weight: Вес длительности проекта
        resource_weight: Вес ресурсов
        cost_weight: Вес стоимости
        num_episodes: Количество эпизодов обучения
        model_path: Путь к предобученной модели
    """
    try:
        # Обновляем статус задачи
        optimization_tasks[job_id]["status"] = "running"
        optimization_tasks[job_id]["start_time"] = datetime.now().isoformat()
        
        if OPTIMIZER_AVAILABLE:
            # Запускаем оптимизацию с помощью реального оптимизатора
            result = optimize_schedule(
                input_file=input_file,
                output_file=output_file,
                duration_weight=duration_weight,
                resource_weight=resource_weight,
                cost_weight=cost_weight,
                num_episodes=num_episodes,
                model_path=model_path,
                save_model=True,
                model_save_path=f"{TEMP_DIR}/{job_id}_model.pt"
            )
            
            # Сохраняем результат в формате JSON
            with open(output_file, 'w', encoding='utf-8') as f:
                # Если result - объект ProjectOutput, преобразуем его в словарь
                if isinstance(result, ProjectOutput):
                    result_dict = {
                        "tasks": [
                            {
                                "id": task.id,
                                "startDate": task.startDate.isoformat(),
                                "endDate": task.endDate.isoformat(),
                                "assignedResourceId": task.assignedResourceId
                            }
                            for task in result.tasks
                        ],
                        "totalDuration": result.totalDuration,
                        "resourceUtilization": result.resourceUtilization,
                        "totalCost": result.totalCost
                    }
                    json.dump(result_dict, f, indent=2)
                else:
                    # Если result уже словарь, сохраняем как есть
                    json.dump(result, f, indent=2)
        else:
            # Эмулируем оптимизацию, если оптимизатор недоступен
            logger.info("Эмуляция оптимизации...")
            time.sleep(5)  # Эмулируем длительный процесс
            
            # Генерируем тестовый результат
            test_result = generate_random_plan()
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(test_result, f, indent=2)
        
        # Обновляем статус задачи
        optimization_tasks[job_id]["status"] = "completed"
        optimization_tasks[job_id]["end_time"] = datetime.now().isoformat()
        
    except Exception as e:
        logger.exception(f"Ошибка при оптимизации задачи {job_id}")
        optimization_tasks[job_id]["status"] = "failed"
        optimization_tasks[job_id]["error"] = str(e)
        optimization_tasks[job_id]["end_time"] = datetime.now().isoformat()

def optimize_plan(
    data: Dict[str, Any],
    duration_weight: float = 7.0,
    resource_weight: float = 3.0,
    cost_weight: float = 1.0,
    num_episodes: int = 500,
    use_pretrained_model: bool = False,
    model_path: Optional[str] = None,
    background_tasks: Optional[BackgroundTasks] = None
) -> str:
    """
    Запускает оптимизацию календарного плана.
    
    Args:
        data: Данные проекта в формате JSON
        duration_weight: Вес длительности проекта
        resource_weight: Вес ресурсов
        cost_weight: Вес стоимости
        num_episodes: Количество эпизодов обучения
        use_pretrained_model: Использовать предобученную модель
        model_path: Путь к предобученной модели
        background_tasks: Объект BackgroundTasks для запуска фоновых задач
        
    Returns:
        Идентификатор задачи оптимизации
    """
    # Генерируем уникальный идентификатор задачи
    job_id = str(uuid.uuid4())
    
    # Сохраняем входные данные во временный файл
    input_file = f"{TEMP_DIR}/{job_id}_input.json"
    with open(input_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    # Путь для сохранения результатов
    output_file = f"{TEMP_DIR}/{job_id}_result.json"
    
    # Создаем запись о задаче
    optimization_tasks[job_id] = {
        "id": job_id,
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "input_file": input_file,
        "output_file": output_file,
        "params": {
            "duration_weight": duration_weight,
            "resource_weight": resource_weight,
            "cost_weight": cost_weight,
            "num_episodes": num_episodes,
            "use_pretrained_model": use_pretrained_model,
            "model_path": model_path
        }
    }
    
    # Определяем путь к модели
    actual_model_path = model_path if use_pretrained_model and model_path else None
    
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
            num_episodes,
            actual_model_path
        )
    else:
        # Если background_tasks не предоставлен, используем threading
        thread = threading.Thread(
            target=_run_optimization_task,
            args=(
                job_id,
                input_file,
                output_file,
                duration_weight,
                resource_weight,
                cost_weight,
                num_episodes,
                actual_model_path
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
    
    return optimization_tasks[job_id]

def get_all_optimization_results() -> List[Dict[str, Any]]:
    """
    Возвращает список всех результатов оптимизации.
    
    Returns:
        Список словарей с информацией о задачах оптимизации
    """
    return sorted(
        [task for task in optimization_tasks.values()],
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