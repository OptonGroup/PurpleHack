"""Data preprocessing module for project scheduling."""

from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
from omegaconf import DictConfig

from ..models.types import ModelInput, ModelOutput, ProjectData


class DataProcessor:
    """Preprocessor for project scheduling data."""
    
    def __init__(self, config: DictConfig):
        """Initialize the data processor.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.role_to_idx: Dict[str, int] = {}
        self.max_sequence_length = config.data.preprocessing.max_sequence_length
        
    def _encode_roles(self, roles: List[str]) -> None:
        """Create mapping for role encoding.
        
        Args:
            roles: List of unique roles
        """
        self.role_to_idx = {role: idx for idx, role in enumerate(sorted(set(roles)))}
        
    def _encode_dates(self, dates: List[datetime]) -> torch.Tensor:
        """Encode datetime objects to normalized values.
        
        Args:
            dates: List of datetime objects
            
        Returns:
            Tensor of encoded dates
        """
        if not dates:
            return torch.zeros(0)
            
        # Convert to timestamps
        timestamps = np.array([d.timestamp() for d in dates])
        
        # Normalize
        min_time = timestamps.min()
        max_time = timestamps.max()
        if max_time > min_time:
            normalized = (timestamps - min_time) / (max_time - min_time)
        else:
            normalized = np.zeros_like(timestamps)
            
        return torch.FloatTensor(normalized)
    
    def _create_dependency_matrix(
        self,
        tasks: List[Dict],
        num_tasks: int
    ) -> torch.Tensor:
        """Create adjacency matrix for task dependencies.
        
        Args:
            tasks: List of tasks
            num_tasks: Total number of tasks
            
        Returns:
            Adjacency matrix as tensor
        """
        matrix = torch.zeros((num_tasks, num_tasks))
        for task in tasks:
            for dep in task.dependencies:
                matrix[dep][task.id] = 1
        return matrix
    
    def _encode_task_features(
        self,
        project_data: ProjectData
    ) -> torch.Tensor:
        """Encode task features.
        
        Args:
            project_data: Project data
            
        Returns:
            Tensor of encoded task features
        """
        num_tasks = len(project_data.tasks)
        num_features = (
            1 +  # Duration
            len(self.role_to_idx) +  # One-hot encoded role
            1 +  # Has constraint
            1    # Constraint time (if any)
        )
        
        features = torch.zeros((num_tasks, num_features))
        
        for task in project_data.tasks:
            idx = task.id
            feature_idx = 0
            
            # Encode duration
            if self.config.data.preprocessing.normalize_durations:
                features[idx, feature_idx] = task.duration / 10.0  # Normalize by max duration
            else:
                features[idx, feature_idx] = task.duration
            feature_idx += 1
            
            # Encode role
            role_idx = self.role_to_idx[task.role]
            features[idx, feature_idx + role_idx] = 1
            feature_idx += len(self.role_to_idx)
            
            # Encode constraint information
            features[idx, feature_idx] = 1 if task.constraint_type else 0
            feature_idx += 1
            
            if task.constraint_date:
                features[idx, feature_idx] = self._encode_dates([task.constraint_date])[0]
            
        return features
    
    def _encode_resource_features(
        self,
        project_data: ProjectData
    ) -> torch.Tensor:
        """Encode resource features.
        
        Args:
            project_data: Project data
            
        Returns:
            Tensor of encoded resource features
        """
        num_resources = len(project_data.resources)
        num_features = (
            len(self.role_to_idx) +  # One-hot encoded role
            1                        # Availability ratio
        )
        
        features = torch.zeros((num_resources, num_features))
        
        for resource in project_data.resources:
            idx = resource.id
            feature_idx = 0
            
            # Encode role
            role_idx = self.role_to_idx[resource.role]
            features[idx, feature_idx + role_idx] = 1
            feature_idx += len(self.role_to_idx)
            
            # Encode availability
            if resource.calendar:
                availability_ratio = 1 - (
                    len(resource.calendar) /
                    (project_data.project_calendar.holidays[-1] -
                     project_data.project_calendar.holidays[0]).days
                )
                features[idx, feature_idx] = availability_ratio
            else:
                features[idx, feature_idx] = 1.0
                
        return features
    
    def _encode_calendar_features(
        self,
        project_data: ProjectData
    ) -> torch.Tensor:
        """Encode calendar features.
        
        Args:
            project_data: Project data
            
        Returns:
            Tensor of encoded calendar features
        """
        # Convert working days to one-hot encoding
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        working_days = torch.zeros(7)
        for day in project_data.project_calendar.working_days:
            working_days[days.index(day)] = 1
            
        # Encode holidays
        holiday_features = self._encode_dates(project_data.project_calendar.holidays)
        
        return torch.cat([working_days, holiday_features])
    
    def preprocess(
        self,
        project_data: ProjectData
    ) -> ModelInput:
        """Preprocess project data for model input.
        
        Args:
            project_data: Project data
            
        Returns:
            Model input
        """
        # Create role mapping if not exists
        if not self.role_to_idx:
            roles = set()
            for task in project_data.tasks:
                roles.add(task.role)
            for resource in project_data.resources:
                roles.add(resource.role)
            self._encode_roles(list(roles))
            
        # Encode features
        task_features = self._encode_task_features(project_data)
        resource_features = self._encode_resource_features(project_data)
        dependency_matrix = self._create_dependency_matrix(
            project_data.tasks,
            len(project_data.tasks)
        )
        calendar_features = self._encode_calendar_features(project_data)
        
        return ModelInput(
            task_features=task_features,
            resource_features=resource_features,
            dependency_matrix=dependency_matrix,
            calendar_features=calendar_features
        )
    
    def create_target(
        self,
        task_sequence: List[int],
        resource_assignments: List[int],
        start_times: List[datetime],
        end_times: List[datetime],
        num_tasks: int,
        num_resources: int
    ) -> ModelOutput:
        """Create target output for training.
        
        Args:
            task_sequence: Sequence of task IDs
            resource_assignments: Resource assignments for each task
            start_times: Start times for each task
            end_times: End times for each task
            num_tasks: Total number of tasks
            num_resources: Total number of resources
            
        Returns:
            Model output for training
        """
        # Create task sequence tensor
        sequence_tensor = torch.tensor(task_sequence)
        
        # Create resource assignment matrix
        assignment_matrix = torch.zeros((num_tasks, num_resources))
        for task_id, resource_id in enumerate(resource_assignments):
            if resource_id >= 0:
                assignment_matrix[task_id, resource_id] = 1
                
        # Create timing tensors
        start_tensor = self._encode_dates(start_times)
        end_tensor = self._encode_dates(end_times)
        
        return ModelOutput(
            task_sequence=sequence_tensor,
            resource_assignments=assignment_matrix,
            start_times=start_tensor,
            end_times=end_tensor
        ) 