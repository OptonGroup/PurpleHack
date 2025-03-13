"""
Основной модуль для оптимизации календарного плана проекта.

Этот модуль содержит класс Optimizer, который объединяет все компоненты
системы для оптимизации календарного плана.
"""
from datetime import datetime, date
from typing import Dict, Any, List, Tuple, Optional, Callable, Literal
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
from src.models.simulated_annealing import optimize_schedule_with_sa

# Типы поддерживаемых алгоритмов оптимизации
OptimizationAlgorithm = Literal["reinforcement_learning", "simulated_annealing"]

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
        log_level: int = logging.INFO,
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
            algorithm: Алгоритм оптимизации ('reinforcement_learning' или 'simulated_annealing')
            learning_rate: Темп обучения для алгоритма обучения с подкреплением
            gamma: Коэффициент дисконтирования для алгоритма обучения с подкреплением
            initial_temperature: Начальная температура для алгоритма имитации отжига
            cooling_rate: Скорость охлаждения для алгоритма имитации отжига
            min_temperature: Минимальная температура для алгоритма имитации отжига
            iterations_per_temp: Количество итераций на каждой температуре для алгоритма имитации отжига
            max_iterations: Максимальное количество итераций для алгоритма имитации отжига
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
        self.algorithm = algorithm
        
        # Параметры для алгоритма обучения с подкреплением
        self.learning_rate = learning_rate
        self.gamma = gamma
        
        # Параметры для алгоритма имитации отжига
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
        self.iterations_per_temp = iterations_per_temp
        self.max_iterations = max_iterations
        
        # Если входные данные предоставлены и выбран алгоритм RL, инициализируем среду и агента
        if project_input and algorithm == "reinforcement_learning":
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
            gamma=self.gamma,  # Коэффициент дисконтирования
            epsilon_start=1.0,  # Начальное значение для epsilon-greedy
            epsilon_end=0.01,  # Конечное значение для epsilon-greedy
            epsilon_decay=0.995,  # Скорость убывания epsilon
            learning_rate=self.learning_rate,  # Скорость обучения
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
            Оптимизированный календарный план в формате ProjectOutput
        """
        if not self.project_input:
            raise ValueError("Входные данные проекта не предоставлены")
        
        if self.algorithm == "reinforcement_learning":
            # Проверяем, что среда и агент инициализированы
            if not hasattr(self, 'env') or not hasattr(self, 'agent'):
                self.logger.info("Инициализация среды и агента...")
                self._initialize_environment_and_agent()
            
            self.logger.info("Оптимизация с использованием Reinforcement Learning...")
            
            # Если модель не загружена, ничего не делаем
            # Обучение модели выполняется отдельным методом train()
            
            # Запрашиваем у агента оптимальное расписание
            schedule = self.agent.optimize_schedule()
            
            # Преобразуем расписание в формат ProjectOutput
            optimized_tasks = []
            
            for task_id, (start_date, end_date, resource_id) in schedule.items():
                optimized_task = OptimizedTask(
                    id=task_id,
                    startDate=start_date,
                    endDate=end_date,
                    assignedResourceId=resource_id
                )
                optimized_tasks.append(optimized_task)
            
            # Вычисляем общую длительность проекта и использование ресурсов
            if optimized_tasks:
                end_dates = [task.endDate for task in optimized_tasks]
                project_end_date = max(end_dates)
                
                total_duration = (project_end_date - self.project_input.startDate).days
                
                # Вычисляем загрузку ресурсов
                resources = {resource.id: resource for resource in self.project_input.resources}
                resource_costs = {}
                resource_busy_days = {}
                
                for resource_id in resources:
                    resource_busy_days[resource_id] = 0
                    if hasattr(resources[resource_id], 'cost'):
                        resource_costs[resource_id] = resources[resource_id].cost
                    else:
                        resource_costs[resource_id] = 0
                
                total_cost = 0
                
                for task in optimized_tasks:
                    resource_id = task.assignedResourceId
                    task_duration = (task.endDate - task.startDate).days
                    resource_busy_days[resource_id] += task_duration
                    total_cost += resource_costs[resource_id] * task_duration
                
                # Вычисляем процент использования ресурсов
                resource_utilization = 0
                if total_duration > 0:
                    for resource_id, busy_days in resource_busy_days.items():
                        resource_utilization += busy_days / total_duration
                    
                    resource_utilization = resource_utilization / len(resources) * 100
            else:
                total_duration = 0
                resource_utilization = 0
                total_cost = 0
            
            return ProjectOutput(
                tasks=optimized_tasks,
                totalDuration=total_duration,
                resourceUtilization=resource_utilization,
                totalCost=total_cost
            )
        
        elif self.algorithm == "simulated_annealing":
            self.logger.info("Оптимизация с использованием алгоритма имитации отжига...")
            
            # Используем функцию оптимизации на основе алгоритма имитации отжига
            return optimize_schedule_with_sa(
                project_input=self.project_input,
                duration_weight=self.duration_weight,
                resource_weight=self.resource_weight,
                cost_weight=self.cost_weight,
                initial_temperature=self.initial_temperature,
                cooling_rate=self.cooling_rate,
                min_temperature=self.min_temperature,
                iterations_per_temp=self.iterations_per_temp,
                max_iterations=self.max_iterations,
                log_level=self.logger.level
            )
        
        else:
            raise ValueError(f"Неизвестный алгоритм оптимизации: {self.algorithm}")
    
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
    progress_callback: Optional[Callable[[float], None]] = None,
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
) -> ProjectOutput:
    """
    Оптимизирует календарный план проекта.
    
    Args:
        input_file: Путь к файлу с данными
        output_file: Путь для сохранения оптимизированного плана
        duration_weight: Вес длительности проекта при расчете награды
        resource_weight: Вес количества ресурсов при расчете награды
        cost_weight: Вес стоимости проекта при расчете награды
        num_episodes: Количество эпизодов для обучения (только для RL)
        model_path: Путь к предобученной модели (только для RL)
        save_model: Сохранять ли обученную модель (только для RL)
        model_save_path: Путь для сохранения модели (только для RL)
        log_level: Уровень логирования
        progress_callback: Функция обратного вызова для отслеживания прогресса
        algorithm: Алгоритм оптимизации ('reinforcement_learning' или 'simulated_annealing')
        learning_rate: Темп обучения для алгоритма обучения с подкреплением
        gamma: Коэффициент дисконтирования для алгоритма обучения с подкреплением
        initial_temperature: Начальная температура для алгоритма имитации отжига
        cooling_rate: Скорость охлаждения для алгоритма имитации отжига
        min_temperature: Минимальная температура для алгоритма имитации отжига
        iterations_per_temp: Количество итераций на каждой температуре для алгоритма имитации отжига
        max_iterations: Максимальное количество итераций для алгоритма имитации отжига
        
    Returns:
        Оптимизированный план проекта
    """
    # Получаем логгер
    logger = logging.getLogger("src.optimizer")
    logger.setLevel(log_level)
    
    # Создаем обработчик для логирования
    if not logger.handlers:
        handler = RichHandler(rich_tracebacks=True)
        handler.setLevel(log_level)
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    try:
        # Создаем оптимизатор
        optimizer = Optimizer(
            project_input=None,  # Пока не загружаем данные
            duration_weight=duration_weight,
            resource_weight=resource_weight,
            cost_weight=cost_weight,
            model_path=model_path,
            log_level=log_level,
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
        
        # Загружаем данные
        logger.info(f"Загрузка данных из файла {input_file}...")
        dataset = optimizer.load_data_from_file(input_file)
        
        # Преобразуем данные в формат ProjectInput
        logger.info("Преобразование данных в формат ProjectInput...")
        project_input = convert_dataset_to_project_input(dataset)
        
        # Устанавливаем проект в оптимизатор
        optimizer.project_input = project_input
        
        # Если выбран алгоритм RL и нужно обучение
        if algorithm == "reinforcement_learning" and num_episodes > 0:
            # Инициализируем среду и агента
            optimizer._initialize_environment_and_agent()
            
            # Запускаем обучение
            logger.info(f"Запуск обучения на {num_episodes} эпизодах...")
            
            # Класс для отслеживания прогресса обучения
            class EpisodeTracker:
                def __init__(self, total_episodes, callback):
                    self.total_episodes = total_episodes
                    self.callback = callback
                    self.current_episode = 0
                
                def update(self, episode_reward):
                    self.current_episode += 1
                    if self.callback:
                        progress = min(100.0, (self.current_episode / self.total_episodes) * 100)
                        self.callback(progress)
            
            # Создаем трекер для отслеживания прогресса
            tracker = EpisodeTracker(num_episodes, progress_callback) if progress_callback else None
            episode_callback = tracker.update if tracker else None
            
            # Запускаем обучение
            rewards = optimizer.train(
                num_episodes=num_episodes,
                save_model=save_model,
                model_save_path=model_save_path or "model.pt",
                episode_callback=episode_callback
            )
            
            logger.info(f"Обучение завершено. Средняя награда: {sum(rewards) / len(rewards)}")
        
        # Запускаем оптимизацию
        logger.info("Запуск оптимизации...")
        optimized_schedule = optimizer.optimize()
        
        # Сохраняем результаты, если указан выходной файл
        if output_file:
            logger.info(f"Сохранение результатов в файл {output_file}...")
            optimizer.save_optimized_schedule(optimized_schedule, output_file)
        
        return optimized_schedule
    
    except Exception as e:
        logger.exception(f"Ошибка при оптимизации: {str(e)}")
        raise 