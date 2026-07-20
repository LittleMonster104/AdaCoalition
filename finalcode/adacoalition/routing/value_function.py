"""Value function for information routing"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List


class ValueFunction(nn.Module):
    """
    Learnable value function V(m, A_j) for routing decisions
    
    Estimates the mutual information I(m; Y | context_j)
    """
    
    def __init__(self, 
                 embedding_dim: int = 128,
                 hidden_dim: int = 256):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Message encoder
        self.message_net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Context encoder
        self.context_net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Joint value estimator
        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def forward(self, message_emb: torch.Tensor, context_emb: torch.Tensor) -> torch.Tensor:
        """
        Compute value V(m, context)
        
        Args:
            message_emb: [batch_size, embedding_dim]
            context_emb: [batch_size, embedding_dim]
            
        Returns:
            value: [batch_size, 1]
        """
        msg_features = self.message_net(message_emb)
        ctx_features = self.context_net(context_emb)
        
        # Concatenate and estimate value
        combined = torch.cat([msg_features, ctx_features], dim=-1)
        value = self.value_net(combined)
        
        return value
    
    def train_step(self, 
                   message_embs: torch.Tensor,
                   context_embs: torch.Tensor,
                   outcomes: torch.Tensor,
                   optimizer: torch.optim.Optimizer) -> float:
        """
        Training step using outcome feedback
        
        Args:
            message_embs: [batch_size, embedding_dim]
            context_embs: [batch_size, embedding_dim]
            outcomes: [batch_size, 1] - actual task performance
            
        Returns:
            loss: scalar loss value
        """
        optimizer.zero_grad()
        
        predicted_values = self.forward(message_embs, context_embs)
        
        # MSE loss between predicted value and actual outcome
        loss = nn.functional.mse_loss(predicted_values, outcomes)
        
        loss.backward()
        optimizer.step()
        
        return loss.item()
