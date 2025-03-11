"""
Среда обучения с подкреплением для оптимизации календарного плана.

Этот модуль содержит реализацию среды для обучения с подкреплением,
которая используется для оптимизации календарного плана проекта.
"""
from datetime import datetime, date
from typing import Dict, List, Tuple, Optional, Any, Set
import numpy as np
import pandas as pd
from collections import defaultdict
import logging

from src.data.models import ProjectInput, Task, Resource, Dependency
from src.utils.calendar_utils import (
    is_working_day, 
    add_working_days, 
    count_working_days_between,
    get_earliest_start_date
)


class ProjectSchedulingEnvironment:
    """
    Среда для обучения с подкреплением для оптимизации календарного плана проекта.
    
    Эта среда моделирует процесс построения календарного плана, где агент
    принимает решения о порядке выполнения задач и назначении ресурсов.
    """
    
    def __init__(
        self, 
        project_input: ProjectInput,
        weights: Dict[str, float] = None
    ):
        """
        Инициализирует среду для обучения с подкреплением.
        
        Args:
            project_input: Входные данные проекта
            weights: Веса для различных компонентов награды
        """
        # Устанавливаем веса для расчета награды
        self.weights = weights or {"duration": 1.0, "resources": 1.0, "cost": 1.0}
        
        # Сохраняем входные данные
        self.project_input = project_input
        
        # Извлекаем информацию о задачах
        self.tasks = {task.id: task for task in project_input.tasks}
        
        # Извлекаем информацию о ресурсах
        self.resources = {resource.id: resource for resource in project_input.resources}
        
        # Строим словарь зависимостей
        self.dependencies = self._build_dependencies_dict()
        
        # Строим обратный словарь зависимостей
        self.reverse_dependencies = self._build_reverse_dependencies_dict()
        
        # Календарь проекта
        if hasattr(project_input, 'calendar'):
            self.working_days = project_input.calendar.workingDays
            self.holidays = project_input.calendar.holidays
        else:
            # Значения по умолчанию
            self.working_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
            self.holidays = []
        
        # Группируем ресурсы по ролям
        self.resources_by_role = self._group_resources_by_role()
        
        # Устанавливаем дату начала проекта
        self.start_time = project_input.startDate
        
        # Инициализируем текущее время
        self.current_time = self.start_time
        
        # Инициализируем выполненные задачи
        self.completed_tasks = []
        
        # Инициализируем назначенные задачи
        self.scheduled_tasks = {}
        
        # Инициализируем назначения ресурсов
        self.resource_assignments = {}
        
        # Инициализируем времена окончания работы ресурсов
        self.resource_end_times = {resource_id: self.start_time for resource_id in self.resources}
        
        # Инициализируем расписание
        self.schedule = {}
        
        # Инициализируем информацию о доступности ресурсов
        self.resources_info = {
            resource_id: {"available_at": self.start_time}
            for resource_id in self.resources
        }
        
        # Вычисляем среднюю длительность задач
        self.average_task_duration = np.mean([task.duration for task in self.tasks.values()])
        
        # Вычисляем среднюю стоимость ресурсов
        resource_costs = [resource.cost for resource in self.resources.values() if hasattr(resource, 'cost')]
        self.average_resource_cost = np.mean(resource_costs) if resource_costs else 1.0
        
        # Вычисляем ожидаемую длительность проекта (сумма длительностей всех задач)
        self.expected_project_duration = sum(task.duration for task in self.tasks.values())
        
        # Вычисляем среднюю стоимость задачи
        self.average_task_cost = 0.0
        for task in self.tasks.values():
            task_cost = task.duration * self.average_resource_cost
            self.average_task_cost += task_cost
        self.average_task_cost /= len(self.tasks) if self.tasks else 1.0
        
        logging.debug(f"Среда инициализирована: {len(self.tasks)} задач, {len(self.resources)} ресурсов")
        logging.debug(f"Веса для расчета награды: {self.weights}")
        logging.debug(f"Рабочие дни: {self.working_days}")
        logging.debug(f"Праздничные дни: {self.holidays}")
        logging.debug(f"Средняя длительность задачи: {self.average_task_duration}")
        logging.debug(f"Средняя стоимость ресурса: {self.average_resource_cost}")
        logging.debug(f"Ожидаемая длительность проекта: {self.expected_project_duration}")
        logging.debug(f"Средняя стоимость задачи: {self.average_task_cost}")
    
    def reset(self) -> Dict[str, Any]:
        """
        Сбрасывает среду в начальное состояние.
        
        Returns:
            Наблюдение начального состояния среды
        """
        # Начальное время - самая ранняя дата начала проекта
        self.current_time = min(task.startDate for task in self.project_input.tasks)
        
        # Списки выполненных и запланированных задач
        self.completed_tasks: Set[str] = set()
        self.scheduled_tasks: Dict[str, Tuple[datetime, datetime, str]] = {}  # task_id -> (start_date, end_date, resource_id)
        
        # Назначение ресурсов и времена освобождения ресурсов
        self.resource_assignments: Dict[str, List[str]] = defaultdict(list)  # resource_id -> [task_id1, task_id2, ...]
        self.resource_end_times: Dict[str, datetime] = {resource_id: self.current_time for resource_id in self.resources}
        
        # Возвращаем начальное наблюдение
        return self._get_observation()
    
    def step(self, action: Tuple[str, str]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Выполняет шаг в среде на основе выбранного действия.
        
        Args:
            action: Кортеж (task_id, resource_id), где task_id - идентификатор задачи,
                   а resource_id - идентификатор ресурса для ее выполнения
                   
        Returns:
            Кортеж (observation, reward, done, info), где:
            - observation: наблюдение нового состояния среды
            - reward: награда за выполненное действие
            - done: флаг завершения эпизода
            - info: дополнительная информация
        """
        task_id, resource_id = action
        
        # Получаем задачу и ресурс
        task = self.tasks[task_id]
        resource = self.resources[resource_id]
        
        # Проверяем, что задача не выполнена и все ее зависимости выполнены
        if task_id in self.completed_tasks:
            return self._get_observation(), -10, False, {"error": "Task already completed"}
        
        # Проверяем, что все зависимости выполнены
        dependencies = self.dependencies.get(task_id, [])
        for dep_id in dependencies:
            if dep_id not in self.completed_tasks:
                return self._get_observation(), -10, False, {"error": f"Dependency {dep_id} not completed"}
        
        # Проверяем, что ресурс может выполнить задачу (соответствие ролей)
        if task.projectRole and task.projectRole != resource.projectRole:
            return self._get_observation(), -10, False, {"error": "Resource cannot perform task"}
        
        # Определяем время начала задачи
        # Учитываем время освобождения ресурса
        resource_available_time = self.resource_end_times[resource_id]
        
        # Находим время завершения всех зависимостей
        dependencies_end_time = self.current_time
        if dependencies:
            dependencies_end_time = max(
                self.scheduled_tasks[dep_id][1] for dep_id in dependencies if dep_id in self.scheduled_tasks
            )
        
        # Определяем самую раннюю возможную дату начала задачи
        earliest_start = get_earliest_start_date(
            task.constraintType,
            task.constraintDate,
            max(resource_available_time, dependencies_end_time),
            self.working_days,
            self.holidays,
            resource.unavailableDates
        )
        
        # Рассчитываем дату окончания задачи
        task_end_date = add_working_days(
            earliest_start,
            task.duration,
            self.working_days,
            self.holidays,
            resource.unavailableDates
        )
        
        # Обновляем время освобождения ресурса
        self.resource_end_times[resource_id] = task_end_date
        
        # Добавляем задачу в список запланированных
        self.scheduled_tasks[task_id] = (earliest_start, task_end_date, resource_id)
        
        # Добавляем задачу в список выполненных
        self.completed_tasks.add(task_id)
        
        # Добавляем задачу в список назначений ресурса
        self.resource_assignments[resource_id].append(task_id)
        
        # Обновляем текущее время
        self.current_time = max(self.current_time, task_end_date)
        
        # Проверяем, завершены ли все задачи
        done = len(self.completed_tasks) == len(self.tasks)
        
        # Рассчитываем награду
        reward = self._calculate_reward(task, earliest_start, task_end_date, resource)
        
        return self._get_observation(), reward, done, {}
    
    def get_available_actions(self) -> List[Tuple[str, str]]:
        """
        Возвращает список доступных действий (пары task_id, resource_id).
        
        Returns:
            Список кортежей (task_id, resource_id)
        """
        available_actions = []
        
        # Определяем задачи, которые можно запланировать (не выполнены и все зависимости выполнены)
        available_tasks = []
        for task_id, task in self.tasks.items():
            if task_id in self.completed_tasks:
                continue
            
            # Проверяем, что все зависимости выполнены
            dependencies = self.dependencies.get(task_id, [])
            all_dependencies_completed = all(dep_id in self.completed_tasks for dep_id in dependencies)
            
            if all_dependencies_completed:
                available_tasks.append(task_id)
        
        # Для каждой доступной задачи определяем подходящие ресурсы
        for task_id in available_tasks:
            task = self.tasks[task_id]
            role = task.projectRole
            
            # Если роль не указана, любой ресурс может выполнить задачу
            if not role:
                suitable_resources = list(self.resources.keys())
            else:
                # Иначе выбираем ресурсы с подходящей ролью
                suitable_resources = [
                    res_id for res_id, res in self.resources.items() 
                    if res.projectRole == role
                ]
            
            # Добавляем все возможные пары (задача, ресурс)
            for resource_id in suitable_resources:
                available_actions.append((task_id, resource_id))
        
        return available_actions
    
    def get_schedule(self) -> Dict[str, Tuple[datetime, datetime, str]]:
        """
        Возвращает текущее расписание задач.
        
        Returns:
            Словарь {task_id: (start_date, end_date, resource_id)}
        """
        return self.scheduled_tasks
    
    def _build_dependencies_dict(self) -> Dict[str, List[str]]:
        """
        Строит словарь зависимостей между задачами.
        
        Returns:
            Словарь, где ключ - ID задачи, значение - список ID задач, от которых она зависит
        """
        dependencies_dict = defaultdict(list)
        
        for dependency in self.project_input.dependencies:
            # Используем правильные имена атрибутов
            to_task_id = dependency.to_task if dependency.to_task else dependency.toEvent
            from_task_id = dependency.from_task if dependency.from_task else dependency.fromEvent
            
            # Проверка существования задач
            if to_task_id in self.tasks and from_task_id in self.tasks:
                dependencies_dict[to_task_id].append(from_task_id)
                logging.debug(f"Добавлена зависимость: {from_task_id} -> {to_task_id}")
            else:
                logging.warning(f"Не удалось добавить зависимость: {from_task_id} -> {to_task_id}. Одна из задач не найдена.")
        
        return dependencies_dict
    
    def _build_reverse_dependencies_dict(self) -> Dict[str, List[str]]:
        """
        Строит обратный словарь зависимостей (какие задачи зависят от текущей).
        
        Returns:
            Словарь {task_id: [dependent_task_1, dependent_task_2, ...]}
        """
        reverse_dependencies = defaultdict(list)
        
        for dependency in self.project_input.dependencies:
            # Используем правильные атрибуты из модели Dependency
            from_task_id = dependency.from_task if dependency.from_task else dependency.fromEvent
            to_task_id = dependency.to_task if dependency.to_task else dependency.toEvent
            
            # Проверка существования задач
            if from_task_id in self.tasks and to_task_id in self.tasks:
                reverse_dependencies[from_task_id].append(to_task_id)
                logging.debug(f"Добавлена обратная зависимость: {from_task_id} -> {to_task_id}")
            else:
                logging.warning(f"Не удалось добавить обратную зависимость: {from_task_id} -> {to_task_id}. Одна из задач не найдена.")
        
        return reverse_dependencies
    
    def _group_resources_by_role(self) -> Dict[str, List[str]]:
        """
        Группирует ресурсы по ролям.
        
        Returns:
            Словарь {role: [resource_id1, resource_id2, ...]}
        """
        resources_by_role = defaultdict(list)
        
        for resource_id, resource in self.resources.items():
            resources_by_role[resource.projectRole].append(resource_id)
        
        return resources_by_role
    
    def _get_observation(self) -> Dict[str, Any]:
        """
        Формирует наблюдение текущего состояния среды.
        
        Returns:
            Словарь с информацией о текущем состоянии
        """
        # Создаем список доступных задач (не выполненных и с выполненными зависимостями)
        available_tasks = []
        for task_id, task in self.tasks.items():
            if task_id in self.completed_tasks:
                continue
            
            # Проверяем, что все зависимости выполнены
            dependencies = self.dependencies.get(task_id, [])
            all_dependencies_completed = all(dep_id in self.completed_tasks for dep_id in dependencies)
            
            if all_dependencies_completed:
                available_tasks.append(task_id)
        
        # Информация о задачах
        tasks_info = []
        for task_id in available_tasks:
            task = self.tasks[task_id]
            
            # Определяем, когда задача может быть начата
            dependencies = self.dependencies.get(task_id, [])
            dependencies_end_time = self.current_time
            if dependencies:
                dependencies_end_time = max(
                    self.scheduled_tasks[dep_id][1] for dep_id in dependencies if dep_id in self.scheduled_tasks
                )
            
            # Проверяем ограничения
            earliest_possible_start = get_earliest_start_date(
                task.constraintType,
                task.constraintDate,
                dependencies_end_time,
                self.working_days,
                self.holidays,
                []
            )
            
            tasks_info.append({
                "id": task_id,
                "duration": task.duration,
                "role": task.projectRole,
                "earliest_start": earliest_possible_start,
                "priority": task.priority
            })
        
        # Информация о ресурсах
        resources_info = []
        for resource_id, resource in self.resources.items():
            resources_info.append({
                "id": resource_id,
                "role": resource.projectRole,
                "available_at": self.resource_end_times[resource_id]
            })
        
        return {
            "current_time": self.current_time,
            "available_tasks": tasks_info,
            "resources": resources_info,
            "completed_tasks": list(self.completed_tasks),
            "scheduled_tasks": self.scheduled_tasks
        }
    
    def _calculate_reward(
        self, 
        task: Task, 
        start_date: datetime, 
        end_date: datetime, 
        resource: Resource
    ) -> float:
        """
        Рассчитывает награду за выполнение задачи.
        
        Args:
            task: Задача
            start_date: Дата начала задачи
            end_date: Дата окончания задачи
            resource: Назначенный ресурс
            
        Returns:
            Награда за действие
        """
        # Базовая награда за выполнение задачи
        reward = 1.0
        
        # Учитываем близость даты начала к самой ранней возможной дате
        if task.constraintDate and task.constraintType == "startnoearlierthan":
            delay = count_working_days_between(
                task.constraintDate, 
                start_date, 
                self.working_days, 
                self.holidays
            )
            # Штраф за задержку
            reward -= self.weights["duration"] * min(delay, 5) * 0.1
        
        # Учитываем количество задействованных ресурсов
        # Чем меньше ресурсов используется, тем лучше
        used_resources = len(set(res_id for _, _, res_id in self.scheduled_tasks.values()))
        resource_utilization = used_resources / len(self.resources)
        
        # Штраф за использование ресурсов
        # Чем больше ресурсов задействовано, тем выше штраф
        reward -= self.weights["resources"] * resource_utilization * 0.5
        
        # Учитываем приоритет задачи
        # Задачи с высоким приоритетом должны выполняться раньше
        reward += task.priority * 0.1
        
        return reward

    def calculate_reward(self, action: Tuple[str, str]) -> float:
        """
        Рассчитывает награду за выполненное действие.
        
        Args:
            action: Кортеж (task_id, resource_id)
            
        Returns:
            Награда (штраф) за действие
        """
        task_id, resource_id = action
        task = self.tasks[task_id]
        resource = self.resources[resource_id]
        
        # Базовая награда за выполнение задачи
        reward = 1.0
        
        # Оцениваем длительность выполнения задачи
        task_duration = task.duration
        if task_duration > self.average_task_duration * 1.5:
            reward -= self.weights["duration"] * 0.5  # Штраф за длительные задачи
        
        # Оцениваем стоимость ресурса
        if hasattr(resource, 'cost') and resource.cost > 0:
            if resource.cost > self.average_resource_cost:
                reward -= self.weights["resources"] * (resource.cost / self.average_resource_cost - 1)
        
        # Проверяем, соответствует ли роль ресурса роли задачи
        if task.projectRole is not None and task.projectRole and hasattr(resource, 'projectRole'):
            if resource.projectRole != task.projectRole:
                reward -= self.weights["resources"] * 2  # Большой штраф за несоответствие ролей
            else:
                reward += self.weights["resources"] * 0.5  # Бонус за соответствие ролей
        
        # Оцениваем общую стоимость выполнения
        task_cost = task_duration * (resource.cost if hasattr(resource, 'cost') else 0)
        if task_cost > self.average_task_cost * 1.2:
            reward -= self.weights["cost"] * 0.5
        
        # Поощряем завершение проекта раньше срока
        if len(self.completed_tasks) == len(self.tasks):
            project_duration = (self.current_time - self.start_time).days
            if project_duration < self.expected_project_duration:
                reward += self.weights["duration"] * 2.0
                logging.info(f"Проект завершен за {project_duration} дней (ожидалось {self.expected_project_duration})")
        
        logging.debug(f"Награда за действие {action}: {reward}")
        return reward

    def _is_working_day(self, date_to_check: datetime) -> bool:
        """
        Проверяет, является ли указанная дата рабочим днем.
        
        Args:
            date_to_check: Дата для проверки
            
        Returns:
            True, если дата является рабочим днем, False иначе
        """
        # Приводим к date, если передан datetime
        if isinstance(date_to_check, datetime):
            date_to_check = date_to_check.date()
        
        # Проверяем, не является ли дата праздничным днем
        if hasattr(self, 'holidays') and date_to_check in self.holidays:
            return False
        
        # Проверяем, является ли день недели рабочим
        weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        weekday_name = weekday_names[date_to_check.weekday()]
        
        # По умолчанию рабочие дни - с понедельника по пятницу
        default_working_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        working_days = getattr(self, 'working_days', default_working_days)
        
        return weekday_name in working_days 