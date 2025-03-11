"""
Утилиты для работы с календарем проекта.

Этот модуль содержит функции для работы с календарем проекта,
расчета рабочих дней, смещения дат и т.д.
"""
from datetime import datetime, date, timedelta
from typing import List, Optional


def is_working_day(
    day: date, 
    working_days: List[str], 
    holidays: List[date],
    resource_unavailable_dates: Optional[List[date]] = None
) -> bool:
    """
    Проверяет, является ли день рабочим с учетом выходных, праздников и недоступности ресурса.
    
    Args:
        day: Проверяемая дата
        working_days: Список рабочих дней недели (например, ["Monday", "Tuesday", ...])
        holidays: Список праздничных дней
        resource_unavailable_dates: Список дней, когда ресурс недоступен
        
    Returns:
        True, если день является рабочим, False в противном случае
    """
    # Преобразуем дату к типу date, если передан datetime
    if isinstance(day, datetime):
        day = day.date()
    
    # Проверяем, входит ли день в рабочие дни недели
    day_of_week = day.strftime("%A")
    if day_of_week not in working_days:
        return False
    
    # Проверяем, не является ли день праздником
    if day in holidays:
        return False
    
    # Проверяем, не является ли день недоступным для ресурса
    if resource_unavailable_dates and day in resource_unavailable_dates:
        return False
    
    return True


def add_working_days(
    start_date: datetime, 
    days: float, 
    working_days: List[str], 
    holidays: List[date],
    resource_unavailable_dates: Optional[List[date]] = None
) -> datetime:
    """
    Добавляет указанное количество рабочих дней к дате.
    
    Args:
        start_date: Начальная дата
        days: Количество рабочих дней для добавления
        working_days: Список рабочих дней недели
        holidays: Список праздничных дней
        resource_unavailable_dates: Список дней, когда ресурс недоступен
        
    Returns:
        Новая дата после добавления указанного количества рабочих дней
    """
    # Целая часть дней
    whole_days = int(days)
    # Дробная часть дней (часы)
    fractional_days = days - whole_days
    
    current_date = start_date
    working_days_added = 0
    
    # Добавляем целое число рабочих дней
    while working_days_added < whole_days:
        current_date += timedelta(days=1)
        if is_working_day(current_date, working_days, holidays, resource_unavailable_dates):
            working_days_added += 1
    
    # Добавляем дробную часть (часы)
    if fractional_days > 0:
        hours_to_add = int(fractional_days * 8)  # Считаем, что рабочий день - 8 часов
        current_date += timedelta(hours=hours_to_add)
    
    return current_date


def count_working_days_between(
    start_date: datetime, 
    end_date: datetime, 
    working_days: List[str], 
    holidays: List[date],
    resource_unavailable_dates: Optional[List[date]] = None
) -> float:
    """
    Рассчитывает количество рабочих дней между двумя датами.
    
    Args:
        start_date: Начальная дата
        end_date: Конечная дата
        working_days: Список рабочих дней недели
        holidays: Список праздничных дней
        resource_unavailable_dates: Список дней, когда ресурс недоступен
        
    Returns:
        Количество рабочих дней между указанными датами
    """
    if start_date > end_date:
        return 0
    
    count = 0
    current_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
    
    while current_date.date() < end_date.date():
        if is_working_day(current_date, working_days, holidays, resource_unavailable_dates):
            count += 1
        current_date += timedelta(days=1)
    
    # Добавляем дробную часть для последнего дня, если это не полный день
    if current_date.date() == end_date.date():
        if is_working_day(current_date, working_days, holidays, resource_unavailable_dates):
            # Рассчитываем долю рабочего дня
            hours = end_date.hour + end_date.minute / 60
            count += min(hours / 8, 1.0)  # Предполагаем 8-часовой рабочий день
    
    return count


def get_earliest_start_date(
    constraint_type: Optional[str],
    constraint_date: Optional[datetime],
    base_date: datetime,
    working_days: List[str],
    holidays: List[date],
    resource_unavailable_dates: Optional[List[date]] = None
) -> datetime:
    """
    Определяет самую раннюю возможную дату начала задачи с учетом ограничений.
    
    Args:
        constraint_type: Тип ограничения
        constraint_date: Дата ограничения
        base_date: Базовая дата (например, дата окончания предыдущей задачи)
        working_days: Список рабочих дней недели
        holidays: Список праздничных дней
        resource_unavailable_dates: Список дней, когда ресурс недоступен
        
    Returns:
        Самая ранняя возможная дата начала задачи
    """
    if constraint_type is None or constraint_date is None:
        # Если нет ограничений, используем базовую дату
        earliest_date = base_date
    elif constraint_type == "startnoearlierthan":
        # Задача не может начаться раньше указанной даты
        earliest_date = max(base_date, constraint_date)
    elif constraint_type == "muststarton":
        # Задача должна начаться точно в указанную дату
        earliest_date = constraint_date
    elif constraint_type == "startnolaterthan":
        # Задача не может начаться позже указанной даты
        earliest_date = min(base_date, constraint_date)
    elif constraint_type == "aslataspossible":
        # Задача должна начаться как можно позже (этот случай требует дополнительной логики)
        # Для простоты используем базовую дату
        earliest_date = base_date
    else:
        # Для остальных типов ограничений используем базовую дату
        earliest_date = base_date
    
    # Если дата начала не является рабочим днем, сдвигаем ее на ближайший рабочий день
    current_date = earliest_date
    while not is_working_day(current_date, working_days, holidays, resource_unavailable_dates):
        current_date += timedelta(days=1)
    
    return current_date 