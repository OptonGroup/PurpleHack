"""
Основной модуль для оптимизации календарного плана проекта.

Этот модуль содержит класс Optimizer, который объединяет все компоненты
системы для оптимизации календарного плана.
"""
from datetime import datetime, date
from typing import Dict, Any, List, Tuple, Optional, Callable
import json
import logging
import os
import numpy as np
from rich.logging import RichHandler
from rich.progress import Progress, TaskID
import traceback
import uuid
from dateutil import parser
from pydantic import ValidationError

from src.data.models import (
    ProjectInput, ProjectOutput, OptimizedTask, Dataset, Calendar,
    convert_dataset_to_project_input, Project, Task, Resource, Dependency,
    DatasetContainer
)
from src.models.rl_environment import ProjectSchedulingEnvironment
from src.models.rl_agent import RLAgent


class Optimizer:
    """
    Класс для оптимизации календарного плана проекта.
    
    Этот класс объединяет все компоненты системы для оптимизации календарного плана.
    """
    
    def __init__(
        self, 
        project_input: Optional[ProjectInput] = None,
        duration_weight: float = 1.0,
        resource_weight: float = 1.0,
        cost_weight: float = 1.0,
        model_path: Optional[str] = None,
        log_level: int = logging.INFO
    ):
        """
        Инициализирует оптимизатор.
        
        Args:
            project_input: Входные данные проекта
            duration_weight: Вес длительности проекта при расчете награды
            resource_weight: Вес количества ресурсов при расчете награды
            cost_weight: Вес стоимости проекта при расчете награды
            model_path: Путь к сохраненной модели
            log_level: Уровень логирования
        """
        # Настройка логирования
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        
        # Если обработчики еще не настроены
        if not self.logger.handlers:
            handler = RichHandler(rich_tracebacks=True)
            handler.setLevel(log_level)
            formatter = logging.Formatter("%(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        self.project_input = project_input
        self.duration_weight = duration_weight
        self.resource_weight = resource_weight
        self.cost_weight = cost_weight
        self.model_path = model_path
        
        # Если входные данные предоставлены, инициализируем среду и агента
        if project_input:
            self._initialize_environment_and_agent()
    
    def load_data_from_file(self, file_path: str) -> Dataset:
        """
        Загружает данные из файла JSON.
        
        Args:
            file_path: Путь к файлу с данными
            
        Returns:
            Объект Dataset с загруженными данными
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Логируем основную структуру данных
        self.logger.info(f"Загружен файл JSON с данными проекта")
        self.logger.info(f"Структура данных:")
        self.logger.info(f"- requestId: {data.get('requestId', 'Отсутствует')}")
        self.logger.info(f"- project: {data.get('project', {}).get('name', 'Отсутствует')}")
        self.logger.info(f"- tasks: {len(data.get('tasks', {}).get('rows', []))} задач")
        self.logger.info(f"- resources: {len(data.get('resources', {}).get('rows', []))} ресурсов")
        self.logger.info(f"- dependencies: {len(data.get('dependencies', {}).get('rows', []))} зависимостей")
        
        # Проверка наличия календаря проекта
        if "projectCalendar" in data:
            self.logger.info(f"- projectCalendar: {data['projectCalendar'].get('workingDays', [])} рабочих дней")
        else:
            self.logger.warning(f"- projectCalendar отсутствует, будет создан календарь по умолчанию")
        
        # Извлекаем информацию о проекте
        project_info = data.get("project", {})
        
        # Создаем модель проекта
        project = Project(
            id=project_info.get("id", ""),
            name=project_info.get("name", "Unnamed Project"),
            startDate=datetime.fromisoformat(project_info.get("startDate", "2024-01-01").replace('Z', '+00:00'))
        )
        
        # Получаем задачи
        tasks_data = data.get("tasks", {}).get("rows", [])
        tasks = {}
        
        def process_tasks(task_list, parent_id=None):
            """
            Рекурсивно обрабатываем задачи, включая вложенные.
            
            Args:
                task_list: Список задач для обработки
                parent_id: ID родительской задачи
            """
            for task_data in task_list:
                # Получаем ID задачи или генерируем новый, если отсутствует
                task_id = task_data.get("id", str(uuid.uuid4()))
                
                # Проверяем наличие необходимых полей
                if "startDate" in task_data and "endDate" in task_data:
                    try:
                        # Преобразуем строки дат в объекты datetime
                        if isinstance(task_data["startDate"], str):
                            start_date = parser.parse(task_data["startDate"])
                        else:
                            start_date = task_data["startDate"]
                            
                        if isinstance(task_data["endDate"], str):
                            end_date = parser.parse(task_data["endDate"])
                        else:
                            end_date = task_data["endDate"]
                        
                        # Копируем данные и устанавливаем родительский ID
                        task_dict = task_data.copy()
                        task_dict["parentId"] = parent_id
                        
                        # Создаем задачу с использованием **task_dict для всех полей
                        try:
                            task = Task(
                                id=task_id,
                                name=task_data.get("name", f"Task {task_id}"),
                                startDate=start_date,
                                endDate=end_date,
                                duration=task_data.get("duration", 1.0),
                                durationUnit=task_data.get("durationUnit", "d"),
                                effort=task_data.get("effort", 1.0),
                                effortUnit=task_data.get("effortUnit", "d"),
                                percentDone=task_data.get("percentDone", 0.0),
                                schedulingMode=task_data.get("schedulingMode", "FixedUnits"),
                                constraintType=task_data.get("constraintType"),
                                constraintDate=task_data.get("constraintDate"),
                                projectRole=task_data.get("projectRole"),
                                priority=task_data.get("priority", 1),
                                parentId=parent_id
                            )
                            
                            tasks[task_id] = task
                            logging.debug(f"Создана задача: {task.name}, ID: {task.id}")
                        except Exception as e:
                            logging.warning(f"Ошибка при создании задачи {task_id}: {e}")
                            continue
                    except (ValueError, TypeError) as e:
                        logging.warning(f"Ошибка при обработке дат задачи {task_id}: {e}")
                        continue
                
                # Рекурсивно обрабатываем дочерние задачи
                if "children" in task_data and isinstance(task_data["children"], list):
                    process_tasks(task_data["children"], task_id)
        
        # Начинаем обработку с верхнего уровня
        process_tasks(tasks_data)
        
        # Получаем ресурсы
        resources_data = data.get("resources", {}).get("rows", [])
        resources = {}
        
        for resource_data in resources_data:
            resource_id = resource_data.get("id", str(uuid.uuid4()))
            
            try:
                resource = Resource(
                    id=resource_id,
                    name=resource_data.get("name", f"Resource {resource_id}"),
                    projectRole=resource_data.get("projectRole", ""),
                    reservationType=resource_data.get("reservationType", ""),
                    reservationPercent=resource_data.get("reservationPercent", 100.0),
                    reservationStatus=resource_data.get("reservationStatus", ""),
                    projectRoleId=resource_data.get("projectRoleId", ""),
                    reservePartially=resource_data.get("reservePartially", False),
                    performedTaskKinds=resource_data.get("performedTaskKinds", [])
                )
                
                resources[resource_id] = resource
                logging.debug(f"Создан ресурс: {resource.name}, ID: {resource.id}")
            except Exception as e:
                logging.warning(f"Ошибка при создании ресурса {resource_id}: {e}")
                continue
        
        # Получаем зависимости
        dependencies_data = data.get("dependencies", {}).get("rows", [])
        dependencies = []
        
        for dep_data in dependencies_data:
            try:
                dependency = Dependency(**dep_data)
                dependencies.append(dependency)
                logging.debug(f"Создана зависимость: {dependency.id}")
            except ValidationError as e:
                logging.warning(f"Ошибка при обработке зависимости: {e}, данные: {dep_data}")
        
        # Создаем календарь проекта (если отсутствует)
        project_calendar = data.get("projectCalendar", {})
        if not project_calendar:
            # Создаем календарь по умолчанию (8-часовой рабочий день, 5-дневная рабочая неделя)
            project_calendar = {
                "id": "default",
                "name": "Default Calendar",
                "workingDays": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],  # Пн-Пт
                "holidays": []  # Нет праздников по умолчанию
            }
        
        # Создаем Dataset
        try:
            dataset = Dataset(
                requestId=data.get("requestId", str(uuid.uuid4())),
                project=project_info,
                success=data.get("success", True),
                tasks=DatasetContainer(rows=tasks_data),
                resources=DatasetContainer(rows=resources_data),
                dependencies=DatasetContainer(rows=dependencies_data),
                projectCalendar=project_calendar
            )
            logging.debug(f"Dataset успешно создан: {len(tasks)} задач, {len(resources)} ресурсов, {len(dependencies)} зависимостей")
            return dataset
        except Exception as e:
            logging.error(f"Ошибка при создании Dataset: {e}")
            # Создаем минимальный датасет, если что-то пошло не так
            empty_rows = []
            dataset = Dataset(
                requestId=data.get("requestId", str(uuid.uuid4())),
                project=project_info,
                success=False,
                tasks=DatasetContainer(rows=empty_rows),
                resources=DatasetContainer(rows=empty_rows),
                dependencies=DatasetContainer(rows=empty_rows),
                projectCalendar=project_calendar
            )
            return dataset
    
    def _initialize_environment_and_agent(self):
        """
        Инициализирует среду и агента обучения с подкреплением.
        """
        self.logger.info("Инициализация среды и агента обучения с подкреплением...")
        
        # Проверка, что project_input существует
        if not self.project_input:
            raise ValueError("Не заданы входные данные проекта (project_input)")
        
        # Проверка, что в project_input есть задачи и ресурсы
        if not self.project_input.tasks:
            raise ValueError("В project_input нет задач")
        if not self.project_input.resources:
            raise ValueError("В project_input нет ресурсов")
        
        # Создаем среду
        self.env = ProjectSchedulingEnvironment(
            project_input=self.project_input,
            weights={
                "duration": self.duration_weight,
                "resources": self.resource_weight,
                "cost": self.cost_weight
            }
        )
        
        # Создаем агента
        self.agent = RLAgent(
            env=self.env,
            gamma=0.99,  # Коэффициент дисконтирования
            epsilon_start=1.0,  # Начальное значение для epsilon-greedy
            epsilon_end=0.01,  # Конечное значение для epsilon-greedy
            epsilon_decay=0.995,  # Скорость убывания epsilon
            learning_rate=0.001,  # Скорость обучения
            batch_size=64,  # Размер батча для обучения
            memory_capacity=10000,  # Размер памяти для опыта
            target_update_frequency=10  # Частота обновления целевой сети
        )
        
        # Загружаем модель, если указан путь
        if self.model_path and os.path.exists(self.model_path):
            self.logger.info(f"Загрузка модели из {self.model_path}...")
            self.agent.load_model(self.model_path)
            self.logger.info("Модель успешно загружена")
        
        self.logger.info("Среда и агент успешно инициализированы")
    
    def train(
        self, 
        num_episodes: int = 100, 
        max_steps_per_episode: int = 1000,
        save_model: bool = True,
        model_save_path: Optional[str] = None,
        episode_callback: Optional[Callable[[float], None]] = None
    ) -> List[float]:
        """
        Обучает RL-агента на заданном количестве эпизодов.
        
        Args:
            num_episodes: Количество эпизодов для обучения
            max_steps_per_episode: Максимальное количество шагов в эпизоде
            save_model: Сохранять ли модель после обучения
            model_save_path: Путь для сохранения модели
            episode_callback: Функция обратного вызова для отображения прогресса (0-100%)
            
        Returns:
            Список наград за каждый эпизод
        """
        self.logger.info(f"Начало обучения агента на {num_episodes} эпизодах...")
        
        # Убедимся, что среда и агент инициализированы
        if not hasattr(self, 'env') or not hasattr(self, 'agent'):
            self.logger.info("Инициализация среды и агента...")
            self._initialize_environment_and_agent()
        
        # Обучаем агента
        rewards = self.agent.train(num_episodes, max_steps_per_episode, verbose=True)
        
        self.logger.info(f"Обучение завершено. Средняя награда: {np.mean(rewards):.4f}")
        
        # Сохраняем модель, если требуется
        if save_model:
            model_path = model_save_path or "model.pt"
            self.agent.save_model(model_path)
            self.logger.info(f"Модель сохранена в {model_path}")
        
        # Если указан callback для прогресса, обновляем его во время обучения
        if episode_callback:
            # Создаем класс-обертку для отслеживания эпизодов
            class EpisodeTracker:
                def __init__(self, total_episodes, callback):
                    self.total_episodes = total_episodes
                    self.callback = callback
                    self.current_episode = 0
                
                def update(self, episode_reward):
                    self.current_episode += 1
                    progress = (self.current_episode / self.total_episodes) * 100
                    self.callback(progress)
            
            # Создаем трекер эпизодов
            tracker = EpisodeTracker(num_episodes, episode_callback)
            
            # Запускаем обучение с отслеживанием прогресса
            for reward in rewards:
                tracker.update(reward)
        
        return rewards
    
    def optimize(self) -> ProjectOutput:
        """
        Оптимизирует календарный план проекта.
        
        Returns:
            Оптимизированный план проекта
        """
        self.logger.info("Запуск оптимизации календарного плана...")
        
        # Убедимся, что среда и агент инициализированы
        if not hasattr(self, 'env') or not hasattr(self, 'agent'):
            self.logger.info("Инициализация среды и агента...")
            self._initialize_environment_and_agent()
        
        # Получаем оптимизированное расписание
        schedule = self.agent.optimize_schedule()
        
        # Создаем выходной объект
        optimized_tasks = []
        
        for task_id, (start_date, end_date, resource_id) in schedule.items():
            optimized_task = OptimizedTask(
                id=task_id,
                startDate=start_date,
                endDate=end_date,
                assignedResourceId=resource_id
            )
            optimized_tasks.append(optimized_task)
        
        # Рассчитываем общую длительность проекта
        if optimized_tasks:
            project_end = max(task.endDate for task in optimized_tasks)
            project_start = min(task.startDate for task in optimized_tasks)
            total_duration = (project_end - project_start).days
        else:
            total_duration = 0
        
        # Рассчитываем утилизацию ресурсов
        total_resource_days = len(self.project_input.resources) * total_duration
        total_task_days = sum((task.endDate - task.startDate).days for task in optimized_tasks)
        resource_utilization = (total_task_days / total_resource_days * 100) if total_resource_days > 0 else 0
        
        # Рассчитываем общую стоимость
        total_cost = 0
        for task in optimized_tasks:
            resource = next((r for r in self.project_input.resources if r.id == task.assignedResourceId), None)
            if resource:
                # Логируем информацию о ресурсе для отладки
                self.logger.debug(f"Ресурс для задачи {task.id}: {resource.id}, имеет атрибут cost: {hasattr(resource, 'cost')}")
                if hasattr(resource, 'cost'):
                    self.logger.debug(f"Стоимость ресурса {resource.id}: {resource.cost}")
                    task_duration = (task.endDate - task.startDate).days
                    task_cost = task_duration * resource.cost
                    total_cost += task_cost
                    self.logger.debug(f"Стоимость задачи {task.id}: {task_cost} (длительность: {task_duration} дней)")
        
        # Логируем итоговую стоимость
        self.logger.info(f"Общая стоимость проекта: {total_cost}")
        
        return ProjectOutput(
            tasks=optimized_tasks,
            totalDuration=total_duration,
            resourceUtilization=resource_utilization,
            totalCost=total_cost
        )
    
    def save_optimized_schedule(self, schedule: ProjectOutput, file_path: str):
        """
        Сохраняет оптимизированный план в файл.
        
        Args:
            schedule: Оптимизированный план
            file_path: Путь для сохранения
        """
        self.logger.info(f"Сохранение оптимизированного плана в {file_path}...")
        
        # Преобразуем даты в строки
        schedule_dict = schedule.dict()
        for task in schedule_dict["tasks"]:
            task["startDate"] = task["startDate"].isoformat()
            task["endDate"] = task["endDate"].isoformat()
        
        # Сохраняем в файл
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(schedule_dict, f, ensure_ascii=False, indent=2)
        
        self.logger.info("Оптимизированный план успешно сохранен")


def optimize_schedule(
    input_file: str,
    output_file: Optional[str] = None,
    duration_weight: float = 1.0,
    resource_weight: float = 1.0,
    cost_weight: float = 1.0,
    num_episodes: int = 100,
    model_path: Optional[str] = None,
    save_model: bool = True,
    model_save_path: Optional[str] = None,
    log_level: int = logging.INFO,
    progress_callback: Optional[Callable[[float], None]] = None
) -> ProjectOutput:
    """
    Оптимизирует календарный план проекта.
    
    Args:
        input_file: Путь к входному файлу JSON
        output_file: Путь для сохранения оптимизированного плана
        duration_weight: Вес длительности проекта
        resource_weight: Вес ресурсов
        cost_weight: Вес стоимости
        num_episodes: Количество эпизодов обучения
        model_path: Путь к сохраненной модели
        save_model: Сохранять ли модель после обучения
        model_save_path: Путь для сохранения модели
        log_level: Уровень логирования
        progress_callback: Функция обратного вызова для отображения прогресса (0-100%)
        
    Returns:
        Оптимизированный план проекта
    """
    # Настраиваем логирование
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("optimizer")
    
    # Создаем оптимизатор без project_input (загрузим его из файла)
    optimizer = Optimizer(
        project_input=None,
        duration_weight=duration_weight,
        resource_weight=resource_weight,
        cost_weight=cost_weight,
        model_path=model_path,
        log_level=log_level
    )
    
    # Загружаем данные из файла и конвертируем в ProjectInput
    logger.info(f"Загрузка данных из {input_file}...")
    dataset = optimizer.load_data_from_file(input_file)
    
    # Конвертируем Dataset в ProjectInput
    project_input = convert_dataset_to_project_input(dataset)
    
    # Устанавливаем project_input в оптимизаторе
    optimizer.project_input = project_input
    
    # Проверяем, что данные загружены корректно
    if not project_input or not project_input.tasks:
        logger.error("Ошибка загрузки данных: проект не содержит задач")
        return ProjectOutput(
            tasks=[],
            totalDuration=0.0,
            resourceUtilization=0.0,
            totalCost=0.0
        )
    
    logger.info(f"Данные загружены успешно: {len(project_input.tasks)} задач, {len(project_input.resources)} ресурсов")
    
    # Обучаем агента, если необходимо
    if num_episodes > 0:
        logger.info(f"Запуск обучения на {num_episodes} эпизодах...")
        
        # Если указан callback для прогресса, обновляем его во время обучения
        if progress_callback:
            # Создаем класс-обертку для отслеживания эпизодов
            class EpisodeTracker:
                def __init__(self, total_episodes, callback):
                    self.total_episodes = total_episodes
                    self.callback = callback
                    self.current_episode = 0
                
                def update(self, episode_reward):
                    self.current_episode += 1
                    progress = (self.current_episode / self.total_episodes) * 100
                    self.callback(progress)
            
            # Создаем трекер эпизодов
            tracker = EpisodeTracker(num_episodes, progress_callback)
            
            # Запускаем обучение с отслеживанием прогресса
            optimizer.train(
                num_episodes=num_episodes,
                save_model=save_model,
                model_save_path=model_save_path,
                episode_callback=tracker.update
            )
        else:
            # Обычное обучение без отслеживания прогресса
            optimizer.train(
                num_episodes=num_episodes,
                save_model=save_model,
                model_save_path=model_save_path
            )
    else:
        logger.info("Обучение пропущено (num_episodes=0), используется предобученная модель")
    
    # Оптимизируем план
    logger.info("Запуск оптимизации...")
    optimized_schedule = optimizer.optimize()
    
    # Сохраняем результаты
    if output_file:
        logger.info(f"Сохранение оптимизированного плана в {output_file}...")
        optimizer.save_optimized_schedule(optimized_schedule, output_file)
    else:
        logger.info("Сохранение результатов пропущено (output_file=None)")
    
    return optimized_schedule 