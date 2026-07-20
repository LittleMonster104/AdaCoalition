"""Capability profile definitions and utilities"""
import numpy as np
from typing import List, Dict


class CapabilityProfile:
    """Manages capability profiles for agents"""
    
    # Standard capability dimensions
    DIMENSIONS = {
        'reasoning': 0,
        'knowledge_retrieval': 1,
        'data_analysis': 2,
        'text_generation': 3,
        'domain_expertise': 4
    }
    
    @staticmethod
    def create_profile(capabilities: Dict[str, float]) -> np.ndarray:
        """
        Create a capability profile vector
        
        Args:
            capabilities: Dict mapping capability name to strength (0-1)
            
        Returns:
            Normalized capability vector
        """
        profile = np.zeros(len(CapabilityProfile.DIMENSIONS))
        
        for cap_name, strength in capabilities.items():
            if cap_name in CapabilityProfile.DIMENSIONS:
                idx = CapabilityProfile.DIMENSIONS[cap_name]
                profile[idx] = strength
        
        # Normalize
        norm = np.linalg.norm(profile)
        if norm > 0:
            profile = profile / norm
        
        return profile
    
    @staticmethod
    def random_profile(seed: int = None) -> np.ndarray:
        """Generate a random capability profile"""
        if seed is not None:
            np.random.seed(seed)
        
        profile = np.random.rand(len(CapabilityProfile.DIMENSIONS))
        profile = profile / np.linalg.norm(profile)
        
        return profile
    
    @staticmethod
    def specialist_profile(dimension: str, strength: float = 0.9) -> np.ndarray:
        """Create a specialist profile focused on one dimension"""
        profile = np.zeros(len(CapabilityProfile.DIMENSIONS))
        
        if dimension in CapabilityProfile.DIMENSIONS:
            idx = CapabilityProfile.DIMENSIONS[dimension]
            profile[idx] = strength
            
            # Add small random noise to other dimensions
            for i in range(len(profile)):
                if i != idx:
                    profile[i] = np.random.rand() * 0.1
        
        profile = profile / np.linalg.norm(profile)
        return profile
