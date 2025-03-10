"""Training module for project scheduler model."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlflow
import torch
from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..models.scheduler_model import SchedulerModel
from ..models.types import ModelInput, ModelOutput, OptimizationMetrics


class SchedulerDataset(Dataset):
    """Dataset for project scheduling."""
    
    def __init__(
        self,
        inputs: List[ModelInput],
        targets: List[ModelOutput]
    ):
        """Initialize dataset.
        
        Args:
            inputs: List of model inputs
            targets: List of target outputs
        """
        self.inputs = inputs
        self.targets = targets
        
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.inputs)
        
    def __getitem__(self, idx: int) -> Tuple[ModelInput, ModelOutput]:
        """Get dataset item.
        
        Args:
            idx: Item index
            
        Returns:
            Tuple of (input, target)
        """
        return self.inputs[idx], self.targets[idx]


class Trainer:
    """Trainer for project scheduler model."""
    
    def __init__(
        self,
        model: SchedulerModel,
        config: DictConfig,
        device: torch.device
    ):
        """Initialize trainer.
        
        Args:
            model: Model to train
            config: Training configuration
            device: Device to train on
        """
        self.model = model
        self.config = config
        self.device = device
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self._setup_mlflow()
        
    def _create_optimizer(self) -> Optimizer:
        """Create optimizer.
        
        Returns:
            Optimizer instance
        """
        optimizer_config = self.config.training.optimizer
        if optimizer_config.name.lower() == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                betas=optimizer_config.betas,
                weight_decay=optimizer_config.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_config.name}")
            
    def _create_scheduler(self) -> Optional[_LRScheduler]:
        """Create learning rate scheduler.
        
        Returns:
            Scheduler instance if configured, None otherwise
        """
        scheduler_config = self.config.training.scheduler
        if scheduler_config.name.lower() == "cosine_annealing":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_config.T_max,
                eta_min=scheduler_config.eta_min
            )
        else:
            return None
            
    def _setup_mlflow(self) -> None:
        """Setup MLflow tracking."""
        mlflow.set_tracking_uri(self.config.logging.mlflow.tracking_uri)
        mlflow.set_experiment(self.config.logging.mlflow.experiment_name)
        
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Tuple[float, Dict[str, float]]:
        """Train one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Tuple of (average loss, metrics)
        """
        self.model.train()
        total_loss = 0.0
        total_metrics = OptimizationMetrics(
            duration_reduction=0.0,
            resource_utilization=0.0,
            constraint_violation=0.0,
            schedule_compression=0.0,
            load_balancing=0.0,
            dependency_satisfaction=0.0
        )
        
        progress_bar = tqdm(
            train_loader,
            desc=f"Training epoch {epoch}",
            leave=False
        )
        
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            # Move data to device
            inputs = self._to_device(inputs)
            targets = self._to_device(targets)
            
            # Forward pass
            outputs = self.model(inputs)
            loss, loss_components = self.model.calculate_loss(
                outputs,
                targets,
                self.config.training.loss_weights
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Calculate metrics
            metrics = self.model.calculate_metrics(outputs, targets)
            
            # Update totals
            total_loss += loss.item()
            total_metrics.duration_reduction += metrics.duration_reduction
            total_metrics.resource_utilization += metrics.resource_utilization
            total_metrics.constraint_violation += metrics.constraint_violation
            total_metrics.schedule_compression += metrics.schedule_compression
            total_metrics.load_balancing += metrics.load_balancing
            total_metrics.dependency_satisfaction += metrics.dependency_satisfaction
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "dur_red": f"{metrics.duration_reduction:.4f}"
            })
            
            # Log to MLflow
            if batch_idx % self.config.logging.log_interval == 0:
                step = epoch * len(train_loader) + batch_idx
                self._log_metrics(
                    step,
                    loss.item(),
                    loss_components,
                    metrics,
                    prefix="train"
                )
                
        # Calculate averages
        num_batches = len(train_loader)
        avg_loss = total_loss / num_batches
        avg_metrics = {
            "duration_reduction": total_metrics.duration_reduction / num_batches,
            "resource_utilization": total_metrics.resource_utilization / num_batches,
            "constraint_violation": total_metrics.constraint_violation / num_batches,
            "schedule_compression": total_metrics.schedule_compression / num_batches,
            "load_balancing": total_metrics.load_balancing / num_batches,
            "dependency_satisfaction": total_metrics.dependency_satisfaction / num_batches
        }
        
        return avg_loss, avg_metrics
        
    def validate(
        self,
        val_loader: DataLoader,
        epoch: int
    ) -> Tuple[float, Dict[str, float]]:
        """Validate model.
        
        Args:
            val_loader: Validation data loader
            epoch: Current epoch number
            
        Returns:
            Tuple of (average loss, metrics)
        """
        self.model.eval()
        total_loss = 0.0
        total_metrics = OptimizationMetrics(
            duration_reduction=0.0,
            resource_utilization=0.0,
            constraint_violation=0.0,
            schedule_compression=0.0,
            load_balancing=0.0,
            dependency_satisfaction=0.0
        )
        
        with torch.no_grad():
            progress_bar = tqdm(
                val_loader,
                desc=f"Validation epoch {epoch}",
                leave=False
            )
            
            for batch_idx, (inputs, targets) in enumerate(progress_bar):
                # Move data to device
                inputs = self._to_device(inputs)
                targets = self._to_device(targets)
                
                # Forward pass
                outputs = self.model(inputs)
                loss, loss_components = self.model.calculate_loss(
                    outputs,
                    targets,
                    self.config.training.loss_weights
                )
                
                # Calculate metrics
                metrics = self.model.calculate_metrics(outputs, targets)
                
                # Update totals
                total_loss += loss.item()
                total_metrics.duration_reduction += metrics.duration_reduction
                total_metrics.resource_utilization += metrics.resource_utilization
                total_metrics.constraint_violation += metrics.constraint_violation
                total_metrics.schedule_compression += metrics.schedule_compression
                total_metrics.load_balancing += metrics.load_balancing
                total_metrics.dependency_satisfaction += metrics.dependency_satisfaction
                
                # Update progress bar
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "dur_red": f"{metrics.duration_reduction:.4f}"
                })
                
                # Log to MLflow
                if batch_idx % self.config.logging.log_interval == 0:
                    step = epoch * len(val_loader) + batch_idx
                    self._log_metrics(
                        step,
                        loss.item(),
                        loss_components,
                        metrics,
                        prefix="val"
                    )
                    
        # Calculate averages
        num_batches = len(val_loader)
        avg_loss = total_loss / num_batches
        avg_metrics = {
            "duration_reduction": total_metrics.duration_reduction / num_batches,
            "resource_utilization": total_metrics.resource_utilization / num_batches,
            "constraint_violation": total_metrics.constraint_violation / num_batches,
            "schedule_compression": total_metrics.schedule_compression / num_batches,
            "load_balancing": total_metrics.load_balancing / num_batches,
            "dependency_satisfaction": total_metrics.dependency_satisfaction / num_batches
        }
        
        return avg_loss, avg_metrics
        
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        checkpoint_dir: Path
    ) -> None:
        """Train model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            checkpoint_dir: Directory to save checkpoints
        """
        best_val_loss = float("inf")
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Train epoch
            train_loss, train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, val_metrics = self.validate(val_loader, epoch)
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
                
            # Log epoch metrics
            self.logger.info(
                f"Epoch {epoch}: "
                f"train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, "
                f"train_dur_red={train_metrics['duration_reduction']:.4f}, "
                f"val_dur_red={val_metrics['duration_reduction']:.4f}"
            )
            
            # Save checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self._save_checkpoint(
                    checkpoint_dir / "best_model.pt",
                    epoch,
                    val_loss,
                    val_metrics
                )
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= self.config.training.early_stopping_patience:
                self.logger.info(
                    f"Early stopping triggered after {epoch + 1} epochs"
                )
                break
                
            # Save periodic checkpoint
            if (epoch + 1) % self.config.logging.save_interval == 0:
                self._save_checkpoint(
                    checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt",
                    epoch,
                    val_loss,
                    val_metrics
                )
                
    def _to_device(self, data: ModelInput) -> ModelInput:
        """Move data to device.
        
        Args:
            data: Input data
            
        Returns:
            Data on device
        """
        return ModelInput(
            task_features=data.task_features.to(self.device),
            resource_features=data.resource_features.to(self.device),
            dependency_matrix=data.dependency_matrix.to(self.device),
            calendar_features=data.calendar_features.to(self.device),
            constraint_features=data.constraint_features.to(self.device)
            if data.constraint_features is not None
            else None
        )
        
    def _log_metrics(
        self,
        step: int,
        loss: float,
        loss_components: Dict[str, torch.Tensor],
        metrics: OptimizationMetrics,
        prefix: str = ""
    ) -> None:
        """Log metrics to MLflow.
        
        Args:
            step: Current step
            loss: Total loss
            loss_components: Individual loss components
            metrics: Optimization metrics
            prefix: Metric name prefix
        """
        with mlflow.start_run(nested=True):
            # Log loss
            mlflow.log_metric(f"{prefix}_loss", loss, step=step)
            
            # Log loss components
            for name, value in loss_components.items():
                mlflow.log_metric(
                    f"{prefix}_{name}",
                    value.item(),
                    step=step
                )
                
            # Log optimization metrics
            mlflow.log_metric(
                f"{prefix}_duration_reduction",
                metrics.duration_reduction,
                step=step
            )
            mlflow.log_metric(
                f"{prefix}_resource_utilization",
                metrics.resource_utilization,
                step=step
            )
            mlflow.log_metric(
                f"{prefix}_constraint_violation",
                metrics.constraint_violation,
                step=step
            )
            mlflow.log_metric(
                f"{prefix}_schedule_compression",
                metrics.schedule_compression,
                step=step
            )
            mlflow.log_metric(
                f"{prefix}_load_balancing",
                metrics.load_balancing,
                step=step
            )
            mlflow.log_metric(
                f"{prefix}_dependency_satisfaction",
                metrics.dependency_satisfaction,
                step=step
            )
            
    def _save_checkpoint(
        self,
        path: Path,
        epoch: int,
        val_loss: float,
        val_metrics: Dict[str, float]
    ) -> None:
        """Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
            epoch: Current epoch
            val_loss: Validation loss
            val_metrics: Validation metrics
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "val_metrics": val_metrics
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
            
        torch.save(checkpoint, path)
        self.logger.info(f"Saved checkpoint to {path}") 