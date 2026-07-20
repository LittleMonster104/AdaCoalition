"""Information-theoretic message routing"""
import numpy as np
from typing import Dict, Any, List, Set
import torch
import torch.nn as nn


class InformationRouter:
    """Adaptive message routing based on information value"""
    
    def __init__(self, beta: float = 0.1):
        self.beta = beta  # Cost-performance tradeoff parameter
        self.value_estimator = ValueEstimator()
    
    def route_message(self, 
                     message: Dict[str, Any],
                     sender_id: str,
                     potential_recipients: List[str],
                     contexts: Dict[str, List]) -> Set[str]:
        """
        Decide which agents should receive this message
        
        Returns:
            Set of recipient agent IDs
        """
        recipients = set()
        message_length = len(str(message.get('content', '')).split())
        
        for recipient_id in potential_recipients:
            if recipient_id == sender_id:
                continue
            
            # Compute V(m, A_j) = I(m; Y | context_j) - beta * |m|
            context = contexts.get(recipient_id, [])
            info_value = self.value_estimator.estimate(message, context)
            
            value = info_value - self.beta * message_length
            
            if value > 0:
                recipients.add(recipient_id)
        
        return recipients


class ValueEstimator(nn.Module):
    """Neural network for estimating I(m; Y | context)"""
    
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.message_encoder = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.value_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, message_emb, context_emb):
        """Estimate information value"""
        msg_enc = self.message_encoder(message_emb)
        ctx_enc = self.context_encoder(context_emb)
        combined = torch.cat([msg_enc, ctx_enc], dim=-1)
        return self.value_head(combined)
    
    def estimate(self, message: Dict[str, Any], context: List) -> float:
        """Estimate I(m; Y | context) - simplified version"""
        # Simplified: random embedding for now
        # In real implementation, use proper text encoding
        message_emb = torch.randn(1, 128)
        context_emb = torch.randn(1, 128)
        
        with torch.no_grad():
            value = self.forward(message_emb, context_emb)
        
        return float(value.item())
