"""
Модуль реализации алгоритма имитации отжига для оптимизации календарного плана.

Алгоритм имитации отжига - метаэвристический алгоритм оптимизации, 
который хорошо подходит для задач с большим пространством состояний,
когда нужно найти глобальный оптимум.
"""
from typing import Dict, List, Tuple, Any, Optional, Callable, Set
import numpy as np
import random
import math
import logging
from datetime import datetime, timedelta
from copy import deepcopy

from src.data.models import (
    ProjectInput, ProjectOutput, OptimizedTask, Task, Resource, Dependency,
    Project, Calendar
)
from src.utils.calendar_utils import (
    is_working_day, 
    add_working_days, 
    count_working_days_between,
    get_earliest_start_date
)


class ScheduleSolution:
    """
    Класс, представляющий решение для календарного плана.
    
    Хранит текущее расписание задач и методы для его изменения и оценки.
    """
    
    def __init__(
        self, 
        project_input: ProjectInput,
        weights: Dict[str, float] = None
    ):
        """
        Инициализирует решение.
        
        Args:
            project_input: Входные данные проекта
            weights: Веса для различных компонентов функции оценки
        """
        # Устанавливаем веса для расчета оценки
        self.weights = weights or {"duration": 1.0, "resources": 1.0, "cost": 1.0}
        
        # Сохраняем входные данные
        self.project_input = project_input
        
        # Извлекаем информацию о задачах
        self.tasks = {task.id: task for task in project_input.tasks}
        
        # Извлекаем информацию о ресурсах
        self.resources = {resource.id: resource for resource in project_input.resources}
        
        # Строим словарь зависимостей
        self.dependencies_dict = self._build_dependencies_dict()
        
        # Строим список задач в топологическом порядке
        self.sorted_tasks = self._topological_sort()
        
        # Календарь проекта
        if hasattr(project_input, 'calendar'):
            self.working_days = project_input.calendar.workingDays
            self.holidays = project_input.calendar.holidays
        else:
            # Значения по умолчанию
            self.working_days = [0, 1, 2, 3, 4]  # Понедельник - Пятница (0-4)
            self.holidays = []
        
        # Группируем ресурсы по ролям
        self.resources_by_role = self._group_resources_by_role()
        
        # Устанавливаем дату начала проекта
        self.start_time = project_input.startDate
        
        # Инициализируем расписание случайным образом
        self.schedule = self._generate_initial_schedule()
        
        # Вычисляем оценку текущего расписания
        self.score = self._calculate_schedule_score()
    
    def _build_dependencies_dict(self) -> Dict[str, List[str]]:
        """
        Строит словарь зависимостей задач.
        
        Returns:
            Словарь, где ключ - id задачи, значение - список id задач, от которых она зависит
        """
        dependencies_dict = {task_id: [] for task_id in self.tasks}
        
        if hasattr(self.project_input, 'dependencies'):
            for dependency in self.project_input.dependencies:
                successor_id = dependency.toEvent
                predecessor_id = dependency.fromEvent
                
                if successor_id in dependencies_dict:
                    dependencies_dict[successor_id].append(predecessor_id)
        
        return dependencies_dict
    
    def _topological_sort(self) -> List[str]:
        """
        Выполняет топологическую сортировку задач с учетом зависимостей.
        
        Returns:
            Список ID задач в топологическом порядке
        """
        # Создаем копию словаря зависимостей
        temp_dependencies = deepcopy(self.dependencies_dict)
        
        # Список задач, у которых нет зависимостей
        no_dependencies = [
            task_id for task_id, dependencies in temp_dependencies.items() 
            if not dependencies
        ]
        
        # Результат топологической сортировки
        sorted_tasks = []
        
        # Основной алгоритм
        while no_dependencies:
            # Выбираем одну из задач без зависимостей
            task_id = no_dependencies.pop(0)
            sorted_tasks.append(task_id)
            
            # Удаляем эту задачу из списка зависимостей других задач
            for other_task_id, dependencies in list(temp_dependencies.items()):
                if task_id in dependencies:
                    temp_dependencies[other_task_id].remove(task_id)
                    # Если у задачи больше нет зависимостей, добавляем ее в список
                    if not temp_dependencies[other_task_id]:
                        no_dependencies.append(other_task_id)
        
        # Проверяем циклические зависимости
        if len(sorted_tasks) != len(self.tasks):
            logging.warning("Обнаружены циклические зависимости в задачах")
            # Добавляем оставшиеся задачи (если есть циклы)
            for task_id in self.tasks:
                if task_id not in sorted_tasks:
                    sorted_tasks.append(task_id)
        
        return sorted_tasks
    
    def _group_resources_by_role(self) -> Dict[str, List[str]]:
        """
        Группирует ресурсы по ролям.
        
        Returns:
            Словарь, где ключ - тип ресурса, значение - список ID ресурсов
        """
        resources_by_role = {}
        
        for resource_id, resource in self.resources.items():
            resource_type = resource.projectRole
            
            if resource_type not in resources_by_role:
                resources_by_role[resource_type] = []
            
            resources_by_role[resource_type].append(resource_id)
        
        return resources_by_role
    
    def _generate_initial_schedule(self) -> Dict[str, Dict[str, Any]]:
        """
        Генерирует начальное расписание задач.
        
        Returns:
            Словарь с расписанием задач
        """
        schedule = {}
        resource_end_times = {resource_id: self.start_time for resource_id in self.resources}
        
        # Проверка и исправление даты начала проекта
        max_year = 4000  # Установим максимальный год для предотвращения переполнения
        try:
            # Если дата начала проекта некорректна, используем текущую дату
            if self.start_time.year > max_year:
                logging.warning(f"Дата начала проекта {self.start_time} слишком далеко в будущем. Используем текущую дату.")
                self.start_time = datetime.now()
        except (ValueError, OverflowError) as e:
            logging.warning(f"Некорректная дата начала проекта: {e}. Используем текущую дату.")
            self.start_time = datetime.now()
        
        # Сброс времени в часовом поясе для всех ресурсов
        resource_end_times = {resource_id: self.start_time for resource_id in self.resources}
        
        # Для каждой задачи в топологическом порядке
        for task_id in self.sorted_tasks:
            task = self.tasks[task_id]
            
            # Определяем, какие ресурсы подходят для этой задачи
            suitable_resources = []
            if hasattr(task, 'projectRole') and task.projectRole:
                # Находим ресурсы с соответствующей ролью
                for resource_id, resource in self.resources.items():
                    if resource.projectRole == task.projectRole:
                        suitable_resources.append(resource_id)
            
            # Если нет подходящих ресурсов, берем любой
            if not suitable_resources:
                suitable_resources = list(self.resources.keys())
            
            # Выбираем ресурс случайным образом
            assigned_resource_id = random.choice(suitable_resources)
            
            # Определяем самое раннее время начала задачи с учетом зависимостей
            earliest_start_date = self.start_time
            
            if task_id in self.dependencies_dict:
                for predecessor_id in self.dependencies_dict[task_id]:
                    if predecessor_id in schedule:
                        predecessor_end_date = schedule[predecessor_id]["endDate"]
                        if predecessor_end_date > earliest_start_date:
                            earliest_start_date = predecessor_end_date
            
            # Учитываем доступность ресурса
            if resource_end_times[assigned_resource_id] > earliest_start_date:
                earliest_start_date = resource_end_times[assigned_resource_id]
            
            # Находим первый рабочий день, начиная с earliest_start_date
            start_date = earliest_start_date
            max_iterations = 366  # Максимальное количество итераций, чтобы избежать бесконечного цикла
            iterations = 0
            
            try:
                while not self._is_working_day(start_date) and iterations < max_iterations:
                    start_date += timedelta(days=1)
                    iterations += 1
                
                if iterations >= max_iterations:
                    logging.warning(f"Достигнуто максимальное количество итераций при поиске рабочего дня для задачи {task_id}")
                    # Используем текущую дату как запасной вариант
                    start_date = earliest_start_date
            except (ValueError, OverflowError) as e:
                logging.warning(f"Ошибка при расчете даты начала для задачи {task_id}: {e}. Используем начальную дату проекта.")
                start_date = self.start_time
                        
            # Вычисляем дату окончания задачи с учетом календаря
            end_date = start_date
            remaining_duration = max(1, min(task.duration, 365))  # Ограничиваем длительность разумным диапазоном
            iterations = 0
            
            try:
                while remaining_duration > 0 and iterations < max_iterations:
                    end_date += timedelta(days=1)
                    if self._is_working_day(end_date):
                        remaining_duration -= 1
                    iterations += 1
                
                if iterations >= max_iterations:
                    logging.warning(f"Достигнуто максимальное количество итераций при расчете даты окончания для задачи {task_id}")
                    # Используем дату начала + фиксированное количество дней как запасной вариант
                    end_date = start_date + timedelta(days=task.duration)
            except (ValueError, OverflowError) as e:
                logging.warning(f"Ошибка при расчете даты окончания для задачи {task_id}: {e}. Используем дату начала + длительность.")
                end_date = start_date + timedelta(days=task.duration)
            
            # Обновляем время окончания работы ресурса
            resource_end_times[assigned_resource_id] = end_date
            
            # Сохраняем расписание задачи
            schedule[task_id] = {
                "startDate": start_date,
                "endDate": end_date,
                "assignedResourceId": assigned_resource_id
            }
        
        return schedule
    
    def _is_working_day(self, date_to_check: datetime) -> bool:
        """
        Проверяет, является ли день рабочим.
        
        Args:
            date_to_check: Дата для проверки
            
        Returns:
            True, если день рабочий, иначе False
        """
        try:
            # Проверка на адекватность даты
            max_year = 4000
            if date_to_check.year > max_year:
                logging.warning(f"Дата {date_to_check} вне допустимого диапазона")
                return False
            
            # Преобразуем день недели из формата Python (0-6, где 0 - понедельник) в наш формат
            weekday = date_to_check.weekday()
            
            # Проверяем, входит ли день в список праздничных дней
            date_str = date_to_check.strftime("%Y-%m-%d")
            is_holiday = date_str in self.holidays
            
            # День рабочий, если это не праздник и день входит в список рабочих дней
            return not is_holiday and weekday in self.working_days
        except (ValueError, OverflowError, AttributeError) as e:
            logging.warning(f"Ошибка при проверке рабочего дня: {e}")
            return False  # В случае ошибки считаем день нерабочим
    
    def _calculate_schedule_score(self) -> float:
        """
        Вычисляет оценку текущего расписания.
        
        Returns:
            Оценка расписания (чем меньше, тем лучше)
        """
        if not self.schedule:
            return float('inf')
        
        try:
            # Вычисляем длительность проекта
            project_end_date = max(info["endDate"] for info in self.schedule.values())
            project_duration = max(0, min((project_end_date - self.start_time).days, 10000))
            
            # Вычисляем загрузку ресурсов
            resource_utilization = {}
            total_resource_cost = 0
            
            for resource_id in self.resources:
                # Подсчитываем количество дней, в которые ресурс занят
                resource_busy_days = 0
                
                for task_id, info in self.schedule.items():
                    if info["assignedResourceId"] == resource_id:
                        task_start_date = info["startDate"]
                        task_end_date = info["endDate"]
                        try:
                            task_duration = count_working_days_between(
                                task_start_date, 
                                task_end_date,
                                self.working_days,
                                self.holidays
                            )
                            resource_busy_days += task_duration
                            
                            # Суммируем стоимость ресурса
                            resource = self.resources[resource_id]
                            if hasattr(resource, 'cost'):
                                total_resource_cost += resource.cost * task_duration
                        except (ValueError, OverflowError) as e:
                            logging.warning(f"Ошибка при расчете длительности задачи {task_id}: {e}")
                
                # Вычисляем процент загрузки ресурса
                try:
                    total_project_days = count_working_days_between(
                        self.start_time,
                        project_end_date,
                        self.working_days,
                        self.holidays
                    )
                    
                    if total_project_days > 0:
                        resource_utilization[resource_id] = resource_busy_days / total_project_days
                    else:
                        resource_utilization[resource_id] = 0
                except (ValueError, OverflowError) as e:
                    logging.warning(f"Ошибка при расчете рабочих дней проекта: {e}")
                    resource_utilization[resource_id] = 0
            
            # Вычисляем среднюю загрузку ресурсов
            avg_utilization = sum(resource_utilization.values()) / len(resource_utilization) if resource_utilization else 0
            
            # Неравномерность загрузки ресурсов (стандартное отклонение)
            utilization_values = list(resource_utilization.values())
            utilization_std = np.std(utilization_values) if utilization_values else 0
            
            # Вычисляем итоговую оценку с учетом весов
            score = (
                self.weights["duration"] * project_duration +
                self.weights["resources"] * (1 - avg_utilization + utilization_std) * 100 +
                self.weights["cost"] * total_resource_cost / 1000
            )
            
            return score
        except Exception as e:
            logging.error(f"Ошибка при расчете оценки расписания: {e}")
            return float('inf')
    
    def get_neighbor_solution(self) -> 'ScheduleSolution':
        """
        Генерирует соседнее решение путем случайного изменения текущего.
        
        Returns:
            Новое решение
        """
        # Создаем глубокую копию текущего решения
        neighbor = deepcopy(self)
        
        # Выбираем случайную операцию для генерации соседа
        operation = random.choice(["swap_resources", "reschedule_task", "swap_tasks"])
        
        if operation == "swap_resources":
            # Операция 1: Поменять ресурсы местами для двух задач
            if len(self.schedule) < 2:
                return neighbor
            
            task_ids = list(self.schedule.keys())
            task1_id, task2_id = random.sample(task_ids, 2)
            
            # Меняем ресурсы местами
            resource1_id = neighbor.schedule[task1_id]["assignedResourceId"]
            resource2_id = neighbor.schedule[task2_id]["assignedResourceId"]
            
            # Проверяем соответствие ролей
            task1 = neighbor.tasks[task1_id]
            task2 = neighbor.tasks[task2_id]
            resource1 = neighbor.resources[resource1_id]
            resource2 = neighbor.resources[resource2_id]
            
            if not hasattr(task1, 'projectRole') or not hasattr(task2, 'projectRole'):
                # Если требования к ресурсам не заданы, просто меняем их местами
                neighbor.schedule[task1_id]["assignedResourceId"] = resource2_id
                neighbor.schedule[task2_id]["assignedResourceId"] = resource1_id
            else:
                # Проверяем соответствие ролей
                if (not task1.projectRole or task1.projectRole == resource2.projectRole) and \
                   (not task2.projectRole or task2.projectRole == resource1.projectRole):
                    neighbor.schedule[task1_id]["assignedResourceId"] = resource2_id
                    neighbor.schedule[task2_id]["assignedResourceId"] = resource1_id
        
        elif operation == "reschedule_task":
            # Операция 2: Изменить время начала задачи
            if not self.schedule:
                return neighbor
            
            task_id = random.choice(list(self.schedule.keys()))
            
            # Определяем ограничения на время начала задачи
            earliest_start_date = self.start_time
            
            # Учитываем зависимости
            if task_id in self.dependencies_dict:
                for predecessor_id in self.dependencies_dict[task_id]:
                    if predecessor_id in self.schedule:
                        predecessor_end_date = self.schedule[predecessor_id]["endDate"]
                        if predecessor_end_date > earliest_start_date:
                            earliest_start_date = predecessor_end_date
            
            # Случайно сдвигаем время начала в пределах допустимого диапазона
            max_shift = 10  # Максимальный сдвиг в днях
            shift_days = random.randint(0, max_shift)
            
            try:
                new_start_date = earliest_start_date + timedelta(days=shift_days)
                
                # Находим первый рабочий день, начиная с new_start_date
                max_iterations = 366  # Максимальное количество итераций
                iterations = 0
                while not self._is_working_day(new_start_date) and iterations < max_iterations:
                    new_start_date += timedelta(days=1)
                    iterations += 1
                
                # Вычисляем новую дату окончания
                task = neighbor.tasks[task_id]
                new_end_date = new_start_date
                remaining_duration = max(1, min(task.duration, 365))  # Ограничиваем длительность
                iterations = 0
                
                while remaining_duration > 0 and iterations < max_iterations:
                    new_end_date += timedelta(days=1)
                    if self._is_working_day(new_end_date):
                        remaining_duration -= 1
                    iterations += 1
                
                # Обновляем расписание
                neighbor.schedule[task_id]["startDate"] = new_start_date
                neighbor.schedule[task_id]["endDate"] = new_end_date
            except (ValueError, OverflowError) as e:
                logging.warning(f"Ошибка при пересчете дат в reschedule_task: {e}")
                # В случае ошибки не меняем расписание
        
        elif operation == "swap_tasks":
            # Операция 3: Поменять порядок выполнения двух независимых задач
            if len(self.schedule) < 2:
                return neighbor
            
            # Выбираем две случайные задачи, которые не зависят друг от друга
            task_ids = list(self.schedule.keys())
            random.shuffle(task_ids)
            
            # Ищем пару независимых задач
            for i in range(len(task_ids)):
                for j in range(i + 1, len(task_ids)):
                    task1_id = task_ids[i]
                    task2_id = task_ids[j]
                    
                    # Проверяем, зависит ли task1 от task2 или наоборот
                    task1_depends_on_task2 = task2_id in self.dependencies_dict.get(task1_id, [])
                    task2_depends_on_task1 = task1_id in self.dependencies_dict.get(task2_id, [])
                    
                    if not task1_depends_on_task2 and not task2_depends_on_task1:
                        # Нашли независимые задачи, меняем их порядок
                        task1_start = neighbor.schedule[task1_id]["startDate"]
                        task1_end = neighbor.schedule[task1_id]["endDate"]
                        task2_start = neighbor.schedule[task2_id]["startDate"]
                        task2_end = neighbor.schedule[task2_id]["endDate"]
                        
                        try:
                            # Учитываем длительность при обмене
                            task1_duration = (task1_end - task1_start).days
                            task2_duration = (task2_end - task2_start).days
                            
                            # Меняем местами времена начала
                            neighbor.schedule[task1_id]["startDate"] = task2_start
                            neighbor.schedule[task2_id]["startDate"] = task1_start
                            
                            # Обновляем времена окончания с учетом длительности
                            task1 = neighbor.tasks[task1_id]
                            remaining_duration1 = min(task1.duration, 365)
                            new_end_date1 = task2_start
                            max_iterations = 366
                            iterations = 0
                            
                            while remaining_duration1 > 0 and iterations < max_iterations:
                                new_end_date1 += timedelta(days=1)
                                if self._is_working_day(new_end_date1):
                                    remaining_duration1 -= 1
                                iterations += 1
                            
                            task2 = neighbor.tasks[task2_id]
                            remaining_duration2 = min(task2.duration, 365)
                            new_end_date2 = task1_start
                            iterations = 0
                            
                            while remaining_duration2 > 0 and iterations < max_iterations:
                                new_end_date2 += timedelta(days=1)
                                if self._is_working_day(new_end_date2):
                                    remaining_duration2 -= 1
                                iterations += 1
                            
                            neighbor.schedule[task1_id]["endDate"] = new_end_date1
                            neighbor.schedule[task2_id]["endDate"] = new_end_date2
                        except (ValueError, OverflowError, AttributeError) as e:
                            logging.warning(f"Ошибка при обмене задач {task1_id} и {task2_id}: {e}")
                            # В случае ошибки оставляем расписание без изменений
                        
                        break
                else:
                    continue
                break
        
        # Пересчитываем оценку соседнего решения
        neighbor.score = neighbor._calculate_schedule_score()
        
        return neighbor
    
    def to_project_output(self) -> ProjectOutput:
        """
        Преобразует решение в выходной формат ProjectOutput.
        
        Returns:
            Объект ProjectOutput с оптимизированным расписанием
        """
        try:
            # Создаем список оптимизированных задач
            optimized_tasks = []
            
            for task_id, info in self.schedule.items():
                optimized_task = OptimizedTask(
                    id=task_id,
                    startDate=info["startDate"],
                    endDate=info["endDate"],
                    assignedResourceId=info["assignedResourceId"]
                )
                optimized_tasks.append(optimized_task)
            
            # Вычисляем общую длительность проекта
            total_duration = 0
            if self.schedule:
                try:
                    project_end_date = max(info["endDate"] for info in self.schedule.values())
                    total_duration = count_working_days_between(
                        self.start_time,
                        project_end_date,
                        self.working_days,
                        self.holidays
                    )
                except (ValueError, OverflowError) as e:
                    logging.warning(f"Ошибка при расчете общей длительности проекта: {e}")
                    # Резервный вариант - вычисляем примерную длительность
                    project_end_date = max(info["endDate"] for info in self.schedule.values())
                    total_duration = max(0, min((project_end_date - self.start_time).days, 10000))
            
            # Вычисляем загрузку ресурсов
            resource_utilization = {}
            total_resource_cost = 0
            
            for resource_id in self.resources:
                # Подсчитываем количество дней, в которые ресурс занят
                resource_busy_days = 0
                
                for task_id, info in self.schedule.items():
                    if info["assignedResourceId"] == resource_id:
                        try:
                            task_start_date = info["startDate"]
                            task_end_date = info["endDate"]
                            task_duration = count_working_days_between(
                                task_start_date, 
                                task_end_date,
                                self.working_days,
                                self.holidays
                            )
                            resource_busy_days += task_duration
                            
                            # Суммируем стоимость ресурса
                            resource = self.resources[resource_id]
                            if hasattr(resource, 'cost'):
                                total_resource_cost += resource.cost * task_duration
                        except (ValueError, OverflowError) as e:
                            logging.warning(f"Ошибка при расчете длительности задачи {task_id}: {e}")
                
                # Вычисляем процент загрузки ресурса
                if total_duration > 0:
                    resource_utilization[resource_id] = resource_busy_days / total_duration
                else:
                    resource_utilization[resource_id] = 0
            
            # Вычисляем среднюю загрузку ресурсов в процентах
            avg_utilization = sum(resource_utilization.values()) / len(resource_utilization) if resource_utilization else 0
            avg_utilization_percent = avg_utilization * 100
            
            # Создаем объект ProjectOutput
            project_output = ProjectOutput(
                tasks=optimized_tasks,
                totalDuration=total_duration,
                resourceUtilization=avg_utilization_percent,
                totalCost=total_resource_cost
            )
            
            return project_output
        except Exception as e:
            logging.error(f"Ошибка при создании ProjectOutput: {e}")
            # Возвращаем пустой результат в случае ошибки
            return ProjectOutput(
                tasks=[],
                totalDuration=0,
                resourceUtilization=0,
                totalCost=0
            )


class SimulatedAnnealing:
    """
    Класс реализации алгоритма имитации отжига для оптимизации календарного плана.
    """
    
    def __init__(
        self, 
        project_input: ProjectInput,
        duration_weight: float = 1.0,
        resource_weight: float = 1.0,
        cost_weight: float = 1.0,
        initial_temperature: float = 100.0,
        cooling_rate: float = 0.95,
        min_temperature: float = 0.1,
        iterations_per_temp: int = 100,
        max_iterations: int = 10000,
        log_level: int = logging.INFO
    ):
        """
        Инициализирует оптимизатор на основе алгоритма имитации отжига.
        
        Args:
            project_input: Входные данные проекта
            duration_weight: Вес длительности проекта при расчете оценки
            resource_weight: Вес использования ресурсов при расчете оценки
            cost_weight: Вес стоимости при расчете оценки
            initial_temperature: Начальная температура для алгоритма
            cooling_rate: Скорость охлаждения (0 < cooling_rate < 1)
            min_temperature: Минимальная температура для остановки алгоритма
            iterations_per_temp: Количество итераций на каждой температуре
            max_iterations: Максимальное количество итераций
            log_level: Уровень логирования
        """
        # Настройка логирования
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        
        # Параметры алгоритма имитации отжига
        self.initial_temperature = initial_temperature
        self.temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
        self.iterations_per_temp = iterations_per_temp
        self.max_iterations = max_iterations
        
        # Веса для функции оценки
        self.weights = {
            "duration": duration_weight,
            "resources": resource_weight,
            "cost": cost_weight
        }
        
        # Инициализируем начальное решение
        self.current_solution = ScheduleSolution(
            project_input=project_input,
            weights=self.weights
        )
        
        # Лучшее найденное решение и его оценка
        self.best_solution = deepcopy(self.current_solution)
        self.best_score = self.current_solution.score
        
        # Статистика оптимизации
        self.iteration_scores = []
        self.temperatures = []
        self.acceptance_probabilities = []
    
    def run(
        self, 
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> ProjectOutput:
        """
        Запускает алгоритм имитации отжига.
        
        Args:
            progress_callback: Функция обратного вызова для отслеживания прогресса
            
        Returns:
            Оптимизированное расписание в формате ProjectOutput
        """
        self.logger.info("Запуск алгоритма имитации отжига...")
        self.logger.info(f"Начальная температура: {self.initial_temperature}")
        self.logger.info(f"Скорость охлаждения: {self.cooling_rate}")
        self.logger.info(f"Веса: длительность={self.weights['duration']}, "
                        f"ресурсы={self.weights['resources']}, "
                        f"стоимость={self.weights['cost']}")
        
        temperature = self.initial_temperature
        iteration = 0
        total_iterations = 0
        
        while temperature > self.min_temperature and total_iterations < self.max_iterations:
            self.temperatures.append(temperature)
            
            # На каждой температуре выполняем несколько итераций
            for i in range(self.iterations_per_temp):
                # Получаем соседнее решение
                neighbor = self.current_solution.get_neighbor_solution()
                
                # Вычисляем разницу в оценке (энергии)
                delta_score = neighbor.score - self.current_solution.score
                
                # Критерий принятия решения Метрополиса
                if delta_score <= 0:  # Если новое решение лучше или такое же
                    acceptance_probability = 1.0
                    self.current_solution = neighbor
                else:  # Если новое решение хуже
                    acceptance_probability = math.exp(-delta_score / temperature)
                    if random.random() < acceptance_probability:
                        self.current_solution = neighbor
                
                self.acceptance_probabilities.append(acceptance_probability)
                
                # Сохраняем лучшее решение
                if self.current_solution.score < self.best_score:
                    self.best_solution = deepcopy(self.current_solution)
                    self.best_score = self.current_solution.score
                    self.logger.info(f"Найдено лучшее решение с оценкой: {self.best_score}")
                
                # Сохраняем текущую оценку
                self.iteration_scores.append(self.current_solution.score)
                
                total_iterations += 1
                if total_iterations >= self.max_iterations:
                    break
            
            # Снижаем температуру
            temperature *= self.cooling_rate
            iteration += 1
            
            # Вызываем функцию обратного вызова для отслеживания прогресса
            if progress_callback:
                progress = min(100.0, (total_iterations / self.max_iterations) * 100)
                progress_callback(progress)
            
            self.logger.info(f"Итерация {iteration}, температура: {temperature:.2f}, "
                            f"текущая оценка: {self.current_solution.score:.2f}, "
                            f"лучшая оценка: {self.best_score:.2f}")
        
        self.logger.info(f"Оптимизация завершена. Лучшая оценка: {self.best_score}")
        
        # Преобразуем лучшее решение в формат ProjectOutput
        return self.best_solution.to_project_output()
    
    def get_statistics(self) -> Dict[str, List[float]]:
        """
        Возвращает статистику оптимизации.
        
        Returns:
            Словарь со статистикой процесса оптимизации
        """
        return {
            "scores": self.iteration_scores,
            "temperatures": self.temperatures,
            "acceptance_probabilities": self.acceptance_probabilities
        }


def optimize_schedule_with_sa(
    project_input: ProjectInput,
    duration_weight: float = 1.0,
    resource_weight: float = 1.0,
    cost_weight: float = 1.0,
    initial_temperature: float = 100.0,
    cooling_rate: float = 0.95,
    min_temperature: float = 0.1,
    iterations_per_temp: int = 100,
    max_iterations: int = 10000,
    log_level: int = logging.INFO,
    progress_callback: Optional[Callable[[float], None]] = None
) -> ProjectOutput:
    """
    Оптимизирует календарный план с использованием алгоритма имитации отжига.
    
    Args:
        project_input: Входные данные проекта
        duration_weight: Вес длительности проекта при расчете оценки
        resource_weight: Вес использования ресурсов при расчете оценки
        cost_weight: Вес стоимости при расчете оценки
        initial_temperature: Начальная температура для алгоритма
        cooling_rate: Скорость охлаждения (0 < cooling_rate < 1)
        min_temperature: Минимальная температура для остановки алгоритма
        iterations_per_temp: Количество итераций на каждой температуре
        max_iterations: Максимальное количество итераций
        log_level: Уровень логирования
        progress_callback: Функция обратного вызова для отслеживания прогресса
        
    Returns:
        Оптимизированное расписание в формате ProjectOutput
    """
    # Создаем оптимизатор на основе алгоритма имитации отжига
    optimizer = SimulatedAnnealing(
        project_input=project_input,
        duration_weight=duration_weight,
        resource_weight=resource_weight,
        cost_weight=cost_weight,
        initial_temperature=initial_temperature,
        cooling_rate=cooling_rate,
        min_temperature=min_temperature,
        iterations_per_temp=iterations_per_temp,
        max_iterations=max_iterations,
        log_level=log_level
    )
    
    # Запускаем алгоритм
    return optimizer.run(progress_callback=progress_callback) 