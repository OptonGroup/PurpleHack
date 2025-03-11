"""
Модели данных для проекта оптимизации календарного плана.

Этот модуль содержит Pydantic модели для валидации и преобразования входных и выходных данных.
"""
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Any, Union
from enum import Enum
from pydantic import BaseModel, Field, validator
import uuid
import logging
from dateutil import parser


class ConstraintType(str, Enum):
    """Типы ограничений для задач."""
    MUST_START_ON = "muststarton"
    START_NO_EARLIER_THAN = "startnoearlierthan"
    START_NO_LATER_THAN = "startnolaterthan"
    MUST_FINISH_ON = "mustfinishon"
    FINISH_NO_EARLIER_THAN = "finishnoearlierthan"
    FINISH_NO_LATER_THAN = "finishnolaterthan"
    AS_SOON_AS_POSSIBLE = "assoonaspossible"
    AS_LATE_AS_POSSIBLE = "aslataspossible"


class EntityRef(BaseModel):
    """Ссылка на сущность."""
    entityId: str
    rootEntityId: str


class Task(BaseModel):
    """Модель задачи."""
    id: str
    name: str
    startDate: datetime
    endDate: datetime
    effort: float
    effortUnit: str
    duration: float
    durationUnit: str
    percentDone: float = 0.0  # Устанавливаем значение по умолчанию
    schedulingMode: str = "FixedUnits"  # Устанавливаем значение по умолчанию
    constraintType: Optional[ConstraintType] = None
    constraintDate: Optional[datetime] = None
    parentId: Optional[str] = None
    projectRole: Optional[str] = None  # Роль, необходимая для выполнения
    
    # Дополнительные поля из dataset.json
    manuallyScheduled: bool = False
    effortDriven: bool = False
    expanded: bool = True
    rollup: bool = False
    inactive: bool = False
    rootTask: bool = False
    priority: int = 1
    assignmentsUnitsSum: float = 100.0
    guid: str = Field(default_factory=lambda: str(uuid.uuid4()))  # Генерируем GUID по умолчанию
    
    # Поля для вывода результатов
    assignedResourceId: Optional[str] = None
    optimizedStartDate: Optional[datetime] = None
    optimizedEndDate: Optional[datetime] = None
    
    class Config:
        extra = "ignore"  # Игнорировать лишние поля в JSON


class Dependency(BaseModel):
    """
    Модель для представления зависимости между задачами.
    """
    id: str
    fromEvent: str  # ID задачи, от которой зависит
    from_task: Optional[str] = Field(None, alias="from")  # Альтернативное поле
    toEvent: str    # ID задачи, которая зависит
    to_task: Optional[str] = Field(None, alias="to")  # Альтернативное поле
    lag: float = 0.0  # Задержка между задачами
    lagUnit: str = "d"  # Единица измерения задержки
    type: int = 2  # Тип зависимости
    guid: Optional[str] = None  # Уникальный идентификатор

    class Config:
        populate_by_name = True  # Позволяет использовать алиасы полей
        extra = "ignore"  # Игнорировать лишние поля


class Resource(BaseModel):
    """Модель ресурса (исполнителя)."""
    id: str
    name: str
    projectRole: str = ""
    reservationType: str = ""
    reservationPercent: float = 100.0
    reservationStatus: str = ""
    projectRoleId: str = ""
    reservePartially: bool = False
    performedTaskKinds: List[EntityRef] = []
    cost: float = 0.0  # Добавляем стоимость ресурса для расчетов
    calendarId: Optional[str] = None  # ID календаря ресурса
    
    # Календарь исполнителя (нерабочие дни)
    unavailableDates: List[date] = []
    
    class Config:
        extra = "ignore"  # Игнорировать лишние поля


class Calendar(BaseModel):
    """Модель календаря проекта."""
    workingDays: List[str]  # Рабочие дни недели
    holidays: List[date]  # Нерабочие дни (праздники)


class OptimizationWeights(BaseModel):
    """Веса для параметров оптимизации."""
    duration: float = 1.0
    resources: float = 1.0
    cost: float = 1.0


class ProjectInput(BaseModel):
    """Входные данные для оптимизации проекта."""
    projectId: str
    projectName: str
    startDate: datetime
    tasks: List[Task]
    resources: List[Resource]
    dependencies: List[Dependency]
    calendar: Calendar
    
    class Config:
        extra = "ignore"  # Игнорировать лишние поля


class OptimizedTask(BaseModel):
    """Оптимизированная задача для вывода."""
    id: str
    startDate: datetime
    endDate: datetime
    assignedResourceId: str


class ProjectOutput(BaseModel):
    """Выходные данные проекта после оптимизации."""
    tasks: List[OptimizedTask]
    totalDuration: float  # Общая длительность проекта в днях
    resourceUtilization: float  # Процент использования ресурсов
    totalCost: float  # Общая стоимость проекта


class DatasetTask(BaseModel):
    """Модель задачи из dataset.json."""
    parentId: Optional[str] = None
    name: str
    startDate: datetime
    endDate: datetime
    effort: float
    effortUnit: str
    duration: float
    durationUnit: str
    percentDone: float
    schedulingMode: str
    constraintType: Optional[ConstraintType] = None
    constraintDate: Optional[datetime] = None
    manuallyScheduled: bool
    effortDriven: bool
    parentIndex: Optional[int] = None
    expanded: bool
    rollup: bool
    inactive: bool
    rootTask: bool
    taskKind: Optional[EntityRef] = None
    isTransferred: bool
    priority: int
    assignmentsUnitsSum: float
    guid: str
    id: str
    children: Optional[List[Any]] = None


class DatasetDependency(BaseModel):
    """Модель зависимости из dataset.json."""
    id: str
    guid: str
    fromEvent: str
    toEvent: str
    lag: float
    lagUnit: str
    type: int
    # Использование разных имен для полей, которые могут конфликтовать с Python
    from_task: Optional[str] = Field(None, alias="from")
    to_task: Optional[str] = Field(None, alias="to")


class DatasetResource(BaseModel):
    """Модель ресурса из dataset.json."""
    name: str
    projectRole: str
    reservationType: str
    reservationPercent: float
    reservationStatus: str
    projectRoleId: str
    reservePartially: bool
    performedTaskKinds: Optional[List[EntityRef]] = None
    id: str


class DatasetCalendar(BaseModel):
    """Модель календаря из dataset.json."""
    workingDays: List[str]
    holidays: List[date]


class DatasetContainer(BaseModel):
    """Контейнер для элементов dataset.json."""
    rows: List[Dict[str, Any]] = []
    
    class Config:
        extra = "ignore"  # Игнорировать лишние поля


class Dataset(BaseModel):
    """Модель для dataset.json."""
    requestId: str
    project: Dict[str, Any]
    success: bool
    tasks: DatasetContainer
    resources: DatasetContainer
    dependencies: DatasetContainer
    # Сделаем projectCalendar необязательным полем
    projectCalendar: Optional[Dict[str, Any]] = None
    
    class Config:
        extra = "ignore"  # Игнорировать лишние поля


class Project(BaseModel):
    """
    Модель для представления проекта.
    """
    id: str
    name: str
    startDate: datetime
    endDate: Optional[datetime] = None
    description: Optional[str] = None
    
    class Config:
        extra = "ignore"  # Игнорировать лишние поля


def convert_dataset_to_project_input(dataset: Dataset) -> ProjectInput:
    """
    Конвертирует Dataset в ProjectInput для оптимизации.
    
    Args:
        dataset: Объект Dataset
        
    Returns:
        ProjectInput для передачи в оптимизатор
    """
    # Создаем объекты Task из данных в dataset
    tasks = []
    for task_raw in dataset.tasks.rows:
        # Рекурсивно обрабатываем задачи и их потомков
        tasks.extend(process_task_tree(task_raw))
    
    # Создаем объекты Resource из данных в dataset
    resources = []
    for i, res_row in enumerate(dataset.resources.rows):
        # Генерируем случайную стоимость, если не указана
        default_cost = 100.0 + (i * 10)  # Разные стоимости для разных ресурсов
        
        resource = Resource(
            id=res_row.get("id", str(uuid.uuid4())),
            name=res_row.get("name", f"Resource {len(resources)+1}"),
            projectRole=res_row.get("projectRole", ""),
            reservationType=res_row.get("reservationType", ""),
            reservationPercent=res_row.get("reservationPercent", 100.0),
            reservationStatus=res_row.get("reservationStatus", ""),
            projectRoleId=res_row.get("projectRoleId", ""),
            reservePartially=res_row.get("reservePartially", False),
            performedTaskKinds=res_row.get("performedTaskKinds", []),
            # Устанавливаем cost явно и добавляем логирование
            cost=float(res_row.get("cost", default_cost)),
            calendarId=res_row.get("calendarId"),
            unavailableDates=[]  # В dataset.json не указаны недоступные даты
        )
        # Логируем стоимость ресурса для отладки
        logging.debug(f"Создан ресурс: {resource.name}, ID: {resource.id}, Cost: {resource.cost}")
        resources.append(resource)
    
    # Создаем объекты Dependency из данных в dataset
    dependencies = []
    for dep_row in dataset.dependencies.rows:
        try:
            dependency = Dependency(
                id=dep_row.get("id", str(uuid.uuid4())),
                fromEvent=dep_row.get("fromEvent", ""),
                from_task=dep_row.get("from", ""),
                toEvent=dep_row.get("toEvent", ""),
                to_task=dep_row.get("to", ""),
                lag=dep_row.get("lag", 0.0),
                lagUnit=dep_row.get("lagUnit", "d"),
                type=dep_row.get("type", 2),
                guid=dep_row.get("guid", str(uuid.uuid4()))
            )
            dependencies.append(dependency)
        except Exception as e:
            logging.warning(f"Ошибка при обработке зависимости: {e}, данные: {dep_row}")
    
    # Создаем календарь проекта
    calendar = None
    if dataset.projectCalendar:
        calendar = Calendar(
            workingDays=dataset.projectCalendar.get("workingDays", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]),
            holidays=dataset.projectCalendar.get("holidays", [])
        )
    else:
        # Календарь по умолчанию
        calendar = Calendar(
            workingDays=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
            holidays=[]
        )
        logging.info("Создан календарь по умолчанию: рабочие дни Пн-Пт, без праздников")
    
    # Создаем ProjectInput
    project_data = dataset.project
    project_start_date = None
    if "startDate" in project_data:
        project_start_date = parse_date(project_data["startDate"])
    else:
        project_start_date = datetime.now()
    
    project_input = ProjectInput(
        projectId=project_data.get("id", str(uuid.uuid4())),
        projectName=project_data.get("name", "Unnamed Project"),
        startDate=project_start_date,
        tasks=tasks,
        resources=resources,
        dependencies=dependencies,
        calendar=calendar
    )
    
    return project_input

def process_task_tree(task_data, parent_id=None):
    """
    Рекурсивно обрабатывает дерево задач.
    
    Args:
        task_data: Данные о задаче
        parent_id: ID родительской задачи
        
    Returns:
        Список объектов Task
    """
    tasks = []
    
    # Получаем ID задачи или генерируем новый
    task_id = task_data.get("id", str(uuid.uuid4()))
    
    # Проверяем наличие необходимых полей
    if "name" in task_data and ("startDate" in task_data or "constraintDate" in task_data):
        try:
            # Определяем даты начала и окончания
            start_date = None
            if "startDate" in task_data:
                start_date = parse_datetime(task_data["startDate"])
            elif "constraintDate" in task_data:
                start_date = parse_datetime(task_data["constraintDate"])
            else:
                # Используем текущую дату, если нет указанной
                start_date = datetime.now()
            
            end_date = None
            if "endDate" in task_data:
                end_date = parse_datetime(task_data["endDate"])
            elif "duration" in task_data:
                # Вычисляем дату окончания на основе длительности
                duration = task_data.get("duration", 1.0)
                end_date = start_date + timedelta(days=duration)
            else:
                # Используем дату начала + 1 день, если нет указанной
                end_date = start_date + timedelta(days=1)
            
            # Создаем задачу
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
                constraintDate=start_date if "constraintDate" in task_data else None,
                projectRole=task_data.get("projectRole"),
                priority=task_data.get("priority", 1),
                parentId=parent_id
            )
            
            tasks.append(task)
            
        except Exception as e:
            logging.warning(f"Ошибка при создании задачи {task_id}: {e}")
    
    # Рекурсивно обрабатываем дочерние задачи
    if "children" in task_data and isinstance(task_data["children"], list):
        for child in task_data["children"]:
            tasks.extend(process_task_tree(child, task_id))
    
    return tasks

def parse_datetime(date_str):
    """
    Преобразует строку даты в объект datetime.
    
    Args:
        date_str: Строка с датой
        
    Returns:
        Объект datetime
    """
    if not date_str:
        return datetime.now()
    
    if isinstance(date_str, datetime):
        return date_str
    
    try:
        return parser.parse(date_str)
    except:
        # Если не удалось разобрать формат, используем текущую дату
        return datetime.now()

def parse_date(date_str):
    """
    Преобразует строку даты в объект datetime (только дата).
    
    Args:
        date_str: Строка с датой
        
    Returns:
        Объект datetime
    """
    dt = parse_datetime(date_str)
    return datetime(dt.year, dt.month, dt.day) 