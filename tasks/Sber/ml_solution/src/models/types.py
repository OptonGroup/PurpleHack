"""Type definitions for the project scheduler model."""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from pydantic import BaseModel


class Task(BaseModel):
    """Representation of a project task."""
    
    id: int
    duration: int
    role: str
    constraint_type: Optional[str] = None
    constraint_date: Optional[datetime] = None
    dependencies: List[int] = []


class Resource(BaseModel):
    """Representation of a project resource."""
    
    id: int
    role: str
    calendar: List[datetime] = []


class ProjectCalendar(BaseModel):
    """Representation of a project calendar."""
    
    working_days: List[str]
    holidays: List[datetime] = []


@dataclass
class ProjectData:
    """Container for project data."""
    
    tasks: List[Task]
    resources: List[Resource]
    project_calendar: ProjectCalendar


@dataclass
class ModelInput:
    """Container for model input data."""
    
    task_features: torch.Tensor
    resource_features: torch.Tensor
    dependency_matrix: torch.Tensor
    calendar_features: torch.Tensor
    constraint_features: Optional[torch.Tensor] = None


@dataclass
class ModelOutput:
    """Container for model output data."""
    
    task_sequence: torch.Tensor
    resource_assignments: torch.Tensor
    start_times: torch.Tensor
    end_times: torch.Tensor


@dataclass
class OptimizationMetrics:
    """Container for optimization metrics."""
    
    duration_reduction: float
    resource_utilization: float
    constraint_violation: float
    schedule_compression: float
    load_balancing: float
    dependency_satisfaction: float


class ScheduleSolution(BaseModel):
    """Final schedule solution."""
    
    task_assignments: Dict[int, Dict[str, Union[int, datetime, str]]]
    resource_schedules: Dict[int, List[Dict[str, Union[int, datetime]]]]
    metrics: Dict[str, float]
    optimization_status: str 