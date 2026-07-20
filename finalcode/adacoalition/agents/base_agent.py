"""Base agent class with capability profile"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List
import numpy as np


class BaseAgent(ABC):
    """Base class for all agents in AdaCoalition"""
    
    def __init__(self, agent_id: str, capability_profile: np.ndarray):
        self.agent_id = agent_id
        self.capability_profile = capability_profile  # C_i in paper
        self.context = []  # Message history
        
    @abstractmethod
    def forward(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input and return output"""
        pass
    
    def update_context(self, message: Dict[str, Any]):
        """Update agent's context with new message"""
        self.context.append(message)
    
    def get_capability_match(self, requirement: np.ndarray) -> float:
        """Compute match score with task requirement (cosine similarity)"""
        return np.dot(self.capability_profile, requirement) / (
            np.linalg.norm(self.capability_profile) * np.linalg.norm(requirement) + 1e-8
        )
