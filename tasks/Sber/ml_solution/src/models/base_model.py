"""Base model for project scheduling optimization."""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig

from .types import ModelInput, ModelOutput, OptimizationMetrics


class BaseSchedulerModel(nn.Module, ABC):
    """Base class for project scheduler models."""

    def __init__(self, config: DictConfig):
        """Initialize the base scheduler model.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        self.task_embedding = nn.Embedding(
            config.model.architecture.task_embedding_dim,
            config.model.architecture.hidden_dim
        )
        self.resource_embedding = nn.Embedding(
            config.model.architecture.resource_embedding_dim,
            config.model.architecture.hidden_dim
        )
        
    @abstractmethod
    def encode_tasks(self, task_features: torch.Tensor) -> torch.Tensor:
        """Encode task features.
        
        Args:
            task_features: Tensor of task features
            
        Returns:
            Encoded task features
        """
        pass
    
    @abstractmethod
    def encode_resources(self, resource_features: torch.Tensor) -> torch.Tensor:
        """Encode resource features.
        
        Args:
            resource_features: Tensor of resource features
            
        Returns:
            Encoded resource features
        """
        pass
    
    @abstractmethod
    def process_constraints(
        self,
        task_encodings: torch.Tensor,
        dependency_matrix: torch.Tensor,
        constraint_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Process constraints and dependencies.
        
        Args:
            task_encodings: Encoded task features
            dependency_matrix: Task dependency matrix
            constraint_features: Optional constraint features
            
        Returns:
            Processed constraint features
        """
        pass
    
    @abstractmethod
    def decode_schedule(
        self,
        task_encodings: torch.Tensor,
        resource_encodings: torch.Tensor,
        constraint_features: torch.Tensor
    ) -> ModelOutput:
        """Decode schedule from encoded features.
        
        Args:
            task_encodings: Encoded task features
            resource_encodings: Encoded resource features
            constraint_features: Processed constraint features
            
        Returns:
            Model output containing schedule information
        """
        pass
    
    def forward(self, model_input: ModelInput) -> ModelOutput:
        """Forward pass of the model.
        
        Args:
            model_input: Input data for the model
            
        Returns:
            Model output containing schedule information
        """
        # Encode tasks and resources
        task_encodings = self.encode_tasks(model_input.task_features)
        resource_encodings = self.encode_resources(model_input.resource_features)
        
        # Process constraints
        constraint_features = self.process_constraints(
            task_encodings,
            model_input.dependency_matrix,
            model_input.constraint_features
        )
        
        # Decode schedule
        return self.decode_schedule(
            task_encodings,
            resource_encodings,
            constraint_features
        )
    
    def calculate_loss(
        self,
        output: ModelOutput,
        target: ModelOutput,
        weights: Optional[Dict[str, float]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Calculate loss for the model output.
        
        Args:
            output: Model output
            target: Target output
            weights: Optional loss weights
            
        Returns:
            Total loss and individual loss components
        """
        weights = weights or self.config.training.loss_weights
        
        # Calculate individual losses
        sequence_loss = nn.CrossEntropyLoss()(
            output.task_sequence.view(-1, output.task_sequence.size(-1)),
            target.task_sequence.view(-1)
        )
        
        assignment_loss = nn.BCEWithLogitsLoss()(
            output.resource_assignments,
            target.resource_assignments
        )
        
        timing_loss = nn.MSELoss()(
            torch.cat([output.start_times, output.end_times], dim=-1),
            torch.cat([target.start_times, target.end_times], dim=-1)
        )
        
        # Combine losses
        total_loss = (
            weights["duration"] * sequence_loss +
            weights["resource_utilization"] * assignment_loss +
            weights["constraint_violation"] * timing_loss
        )
        
        return total_loss, {
            "sequence_loss": sequence_loss,
            "assignment_loss": assignment_loss,
            "timing_loss": timing_loss
        }
    
    def calculate_metrics(
        self,
        output: ModelOutput,
        target: ModelOutput
    ) -> OptimizationMetrics:
        """Calculate optimization metrics.
        
        Args:
            output: Model output
            target: Target output
            
        Returns:
            Optimization metrics
        """
        # Calculate duration reduction
        pred_duration = output.end_times.max() - output.start_times.min()
        target_duration = target.end_times.max() - target.start_times.min()
        duration_reduction = (target_duration - pred_duration) / target_duration
        
        # Calculate resource utilization
        resource_utilization = output.resource_assignments.mean()
        
        # Calculate constraint violation
        timing_violations = torch.relu(
            output.start_times[:-1] - output.end_times[1:]
        ).mean()
        
        # Calculate schedule compression
        schedule_compression = pred_duration / target_duration
        
        # Calculate load balancing
        load_std = output.resource_assignments.std(dim=0).mean()
        load_balancing = 1 - load_std
        
        # Calculate dependency satisfaction
        dependency_satisfaction = 1 - timing_violations
        
        return OptimizationMetrics(
            duration_reduction=duration_reduction.item(),
            resource_utilization=resource_utilization.item(),
            constraint_violation=timing_violations.item(),
            schedule_compression=schedule_compression.item(),
            load_balancing=load_balancing.item(),
            dependency_satisfaction=dependency_satisfaction.item()
        ) 