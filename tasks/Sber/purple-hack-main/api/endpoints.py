"""
API эндпоинты для взаимодействия с ML моделью планирования проекта.
"""
from typing import Dict, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
from datetime import datetime

from ..ml.models.scheduler import SchedulingModel
from ..ml.data.preprocessor import ProjectDataPreprocessor

app = FastAPI(
    title="Project Scheduling API",
    description="API для оптимизации календарного плана проекта с использованием ML",
    version="1.0.0"
)

# Загружаем предобученную модель
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SchedulingModel(
        task_feature_dim=10,  # Будет определено при первом использовании
        resource_feature_dim=94,  # 4 роли + 90 дней календаря
        hidden_dim=64
    )
    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    model.eval()
    preprocessor = ProjectDataPreprocessor()
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    preprocessor = None

class Task(BaseModel):
    """Модель данных для задачи."""
    id: int
    duration: int
    role: str
    constraintType: str = None
    constraintDate: str = None
    dependencies: List[int] = []

class Resource(BaseModel):
    """Модель данных для ресурса."""
    id: int
    roles: List[str]
    calendar: List[str] = []

class ProjectCalendar(BaseModel):
    """Модель данных для календаря проекта."""
    workingDays: List[str]
    holidays: List[str] = []

class Project(BaseModel):
    """Модель данных для проекта."""
    tasks: List[Task]
    resources: List[Resource]
    projectCalendar: ProjectCalendar

class OptimizationWeights(BaseModel):
    """Модель данных для весов оптимизации."""
    duration: float = 0.4
    resources: float = 0.3
    constraints: float = 0.3

class OptimizationRequest(BaseModel):
    """Модель данных для запроса оптимизации."""
    project: Project
    weights: OptimizationWeights = OptimizationWeights()

class TaskAssignment(BaseModel):
    """Модель данных для назначения задачи."""
    task_id: int
    resource_id: int
    start_date: str
    end_date: str

class OptimizationResponse(BaseModel):
    """Модель данных для ответа на запрос оптимизации."""
    assignments: List[TaskAssignment]
    total_duration: int
    resource_utilization: Dict[int, float]
    optimization_score: float

@app.post("/optimize", response_model=OptimizationResponse)
async def optimize_project(request: OptimizationRequest) -> OptimizationResponse:
    """
    Оптимизация календарного плана проекта.
    
    Args:
        request: Данные проекта и параметры оптимизации
        
    Returns:
        OptimizationResponse: Оптимизированный план проекта
    """
    if model is None or preprocessor is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
        
    try:
        # Преобразуем входные данные в формат для модели
        project_dict = request.project.dict()
        task_features, dep_matrix, resource_features, calendar = preprocessor.preprocess_project(project_dict)
        
        # Конвертируем в тензоры
        task_features = torch.FloatTensor(task_features).unsqueeze(0).to(device)
        dep_matrix = torch.FloatTensor(dep_matrix).unsqueeze(0).to(device)
        resource_features = torch.FloatTensor(resource_features).unsqueeze(0).to(device)
        calendar = torch.FloatTensor(calendar).unsqueeze(0).to(device)
        
        # Получаем предсказания модели
        with torch.no_grad():
            sequence, assignments, times = model(task_features, dep_matrix, resource_features, calendar)
            
        # Преобразуем предсказания в формат ответа
        task_order = torch.argsort(sequence[0]).cpu().numpy()
        resource_assignments = torch.argmax(assignments[0], dim=1).cpu().numpy()
        start_times = torch.argmax(times[0], dim=1).cpu().numpy()
        
        # Формируем назначения
        assignments_list = []
        total_duration = 0
        resource_utilization = {r.id: 0.0 for r in request.project.resources}
        
        for task_idx in task_order:
            task = request.project.tasks[task_idx]
            resource_id = request.project.resources[resource_assignments[task_idx]].id
            start_date = datetime.fromtimestamp(start_times[task_idx] * 86400)  # Конвертируем дни в timestamp
            end_date = start_date + task.duration
            
            assignments_list.append(TaskAssignment(
                task_id=task.id,
                resource_id=resource_id,
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat()
            ))
            
            # Обновляем метрики
            total_duration = max(total_duration, (end_date - start_date).days)
            resource_utilization[resource_id] += task.duration
            
        # Нормализуем утилизацию ресурсов
        for resource_id in resource_utilization:
            resource_utilization[resource_id] /= total_duration
            
        # Вычисляем общий скор оптимизации
        optimization_score = (
            request.weights.duration * (1 - total_duration / 90) +  # Нормализуем на 90 дней
            request.weights.resources * np.mean(list(resource_utilization.values())) +
            request.weights.constraints * (1 - torch.mean(torch.relu(times[0])).item())
        )
        
        return OptimizationResponse(
            assignments=assignments_list,
            total_duration=total_duration,
            resource_utilization=resource_utilization,
            optimization_score=float(optimization_score)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Проверка работоспособности сервиса.
    
    Returns:
        Dict[str, str]: Статус сервиса
    """
    return {
        "status": "healthy",
        "model_loaded": "yes" if model is not None else "no",
        "preprocessor_loaded": "yes" if preprocessor is not None else "no"
    } 