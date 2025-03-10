"""Concrete implementation of the scheduler model."""

from typing import Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch_geometric.nn import GCNConv

from .base_model import BaseSchedulerModel
from .types import ModelOutput


class TaskEncoder(nn.Module):
    """Transformer-based task encoder."""
    
    def __init__(self, config: DictConfig):
        """Initialize the task encoder.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.model.architecture.hidden_dim,
            nhead=config.model.encoder.task_encoder.nhead,
            dim_feedforward=config.model.encoder.task_encoder.dim_feedforward,
            dropout=config.model.architecture.dropout_rate,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.model.encoder.task_encoder.num_layers
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the task encoder.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]
            
        Returns:
            Encoded task features
        """
        return self.transformer(x)


class ResourceEncoder(nn.Module):
    """BiLSTM-based resource encoder."""
    
    def __init__(self, config: DictConfig):
        """Initialize the resource encoder.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        self.lstm = nn.LSTM(
            input_size=config.model.architecture.hidden_dim,
            hidden_size=config.model.encoder.resource_encoder.lstm_hidden_size,
            num_layers=config.model.encoder.resource_encoder.lstm_num_layers,
            bidirectional=config.model.encoder.resource_encoder.bidirectional,
            batch_first=True,
            dropout=config.model.architecture.dropout_rate
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the resource encoder.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]
            
        Returns:
            Encoded resource features
        """
        output, _ = self.lstm(x)
        return output


class ConstraintProcessor(nn.Module):
    """GNN-based constraint processor."""
    
    def __init__(self, config: DictConfig):
        """Initialize the constraint processor.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        self.gnn_layers = nn.ModuleList([
            GCNConv(
                config.model.architecture.hidden_dim,
                config.model.constraint_processor.gnn_hidden_dim
            )
            for _ in range(config.model.constraint_processor.gnn_layers)
        ])
        self.output_layer = nn.Linear(
            config.model.constraint_processor.gnn_hidden_dim,
            config.model.architecture.hidden_dim
        )
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass of the constraint processor.
        
        Args:
            x: Node features
            edge_index: Edge indices
            edge_attr: Optional edge attributes
            
        Returns:
            Processed constraint features
        """
        for layer in self.gnn_layers:
            x = layer(x, edge_index)
            x = torch.relu(x)
        return self.output_layer(x)


class ScheduleDecoder(nn.Module):
    """Pointer network based schedule decoder."""
    
    def __init__(self, config: DictConfig):
        """Initialize the schedule decoder.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        self.pointer_lstm = nn.LSTM(
            input_size=config.model.architecture.hidden_dim,
            hidden_size=config.model.decoder.pointer_network.hidden_size,
            num_layers=config.model.decoder.pointer_network.num_layers,
            batch_first=True,
            dropout=config.model.architecture.dropout_rate
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=config.model.decoder.pointer_network.hidden_size,
            num_heads=config.model.architecture.num_attention_heads,
            dropout=config.model.architecture.dropout_rate,
            batch_first=True
        )
        self.output_projection = nn.Linear(
            config.model.decoder.pointer_network.hidden_size,
            config.model.decoder.output_dim
        )
        
    def forward(
        self,
        task_encodings: torch.Tensor,
        resource_encodings: torch.Tensor,
        constraint_features: torch.Tensor
    ) -> ModelOutput:
        """Forward pass of the schedule decoder.
        
        Args:
            task_encodings: Encoded task features
            resource_encodings: Encoded resource features
            constraint_features: Processed constraint features
            
        Returns:
            Model output containing schedule information
        """
        # Generate task sequence
        lstm_out, _ = self.pointer_lstm(task_encodings)
        attn_output, _ = self.attention(
            lstm_out,
            task_encodings,
            task_encodings
        )
        task_sequence = self.output_projection(attn_output)
        
        # Generate resource assignments
        resource_attn, _ = self.attention(
            task_sequence,
            resource_encodings,
            resource_encodings
        )
        resource_assignments = torch.sigmoid(
            self.output_projection(resource_attn)
        )
        
        # Generate timing information
        timing_features = torch.cat([
            task_sequence,
            resource_assignments,
            constraint_features
        ], dim=-1)
        timing_projection = nn.Linear(
            timing_features.size(-1),
            2
        ).to(timing_features.device)
        timing = timing_projection(timing_features)
        
        return ModelOutput(
            task_sequence=task_sequence,
            resource_assignments=resource_assignments,
            start_times=timing[..., 0],
            end_times=timing[..., 1]
        )


class SchedulerModel(BaseSchedulerModel):
    """Concrete implementation of the scheduler model."""
    
    def __init__(self, config: DictConfig):
        """Initialize the scheduler model.
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        self.task_encoder = TaskEncoder(config)
        self.resource_encoder = ResourceEncoder(config)
        self.constraint_processor = ConstraintProcessor(config)
        self.schedule_decoder = ScheduleDecoder(config)
        
    def encode_tasks(self, task_features: torch.Tensor) -> torch.Tensor:
        """Encode task features.
        
        Args:
            task_features: Task feature tensor
            
        Returns:
            Encoded task features
        """
        embedded = self.task_embedding(task_features)
        return self.task_encoder(embedded)
    
    def encode_resources(self, resource_features: torch.Tensor) -> torch.Tensor:
        """Encode resource features.
        
        Args:
            resource_features: Resource feature tensor
            
        Returns:
            Encoded resource features
        """
        embedded = self.resource_embedding(resource_features)
        return self.resource_encoder(embedded)
    
    def process_constraints(
        self,
        task_encodings: torch.Tensor,
        dependency_matrix: torch.Tensor,
        constraint_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Process constraints and dependencies.
        
        Args:
            task_encodings: Encoded task features
            dependency_matrix: Dependency adjacency matrix
            constraint_features: Optional constraint features
            
        Returns:
            Processed constraint features
        """
        # Convert dependency matrix to edge index format
        edge_index = dependency_matrix.nonzero().t()
        
        # Process constraints using GNN
        constraint_features = constraint_features if constraint_features is not None \
            else task_encodings
        
        return self.constraint_processor(
            constraint_features,
            edge_index
        )
    
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
        return self.schedule_decoder(
            task_encodings,
            resource_encodings,
            constraint_features
        ) 