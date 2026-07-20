"""Coalition performance metrics"""
import numpy as np
from typing import List, Set, Dict
from ..agents import BaseAgent


class CoalitionMetrics:
    """Compute various coalition performance metrics"""
    
    @staticmethod
    def communication_efficiency(messages: List[Dict], 
                                num_agents: int) -> float:
        """
        Compute communication efficiency
        
        Efficiency = useful_messages / total_possible_messages
        """
        total_messages = len(messages)
        max_messages = num_agents * (num_agents - 1)  # Fully connected
        
        if max_messages == 0:
            return 1.0
        
        return 1.0 - (total_messages / max_messages)
    
    @staticmethod
    def task_completion_rate(completed: int, total: int) -> float:
        """Compute task completion rate"""
        return completed / total if total > 0 else 0.0
    
    @staticmethod
    def coalition_coherence(coalition: Set[str], 
                          agents: List[BaseAgent]) -> float:
        """
        Measure how coherent a coalition is
        
        Coherence = average pairwise similarity of capabilities
        """
        if len(coalition) <= 1:
            return 1.0
        
        agent_dict = {a.agent_id: a for a in agents}
        
        similarities = []
        coalition_list = list(coalition)
        for i in range(len(coalition_list)):
            for j in range(i + 1, len(coalition_list)):
                a1 = agent_dict[coalition_list[i]]
                a2 = agent_dict[coalition_list[j]]
                
                sim = np.dot(a1.capability_profile, a2.capability_profile) / (
                    np.linalg.norm(a1.capability_profile) * 
                    np.linalg.norm(a2.capability_profile) + 1e-8
                )
                similarities.append(sim)
        
        return np.mean(similarities)
    
    @staticmethod
    def coverage_score(coalitions: Dict[str, Set[str]], 
                      agents: List[BaseAgent],
                      task_requirements: Dict[str, np.ndarray]) -> float:
        """
        Measure how well coalitions cover task requirements
        
        Coverage = average match score across all tasks
        """
        agent_dict = {a.agent_id: a for a in agents}
        
        scores = []
        for subtask_id, requirement in task_requirements.items():
            if subtask_id in coalitions:
                coalition = coalitions[subtask_id]
                
                # Max capability match in coalition
                matches = [agent_dict[aid].get_capability_match(requirement)
                          for aid in coalition]
                max_match = max(matches) if matches else 0.0
                scores.append(max_match)
        
        return np.mean(scores) if scores else 0.0
    
    @staticmethod
    def diversity_score(coalition: Set[str], 
                       agents: List[BaseAgent]) -> float:
        """
        Measure diversity within a coalition
        
        Higher diversity = more varied capabilities
        """
        if len(coalition) <= 1:
            return 0.0
        
        agent_dict = {a.agent_id: a for a in agents}
        
        capabilities = np.array([agent_dict[aid].capability_profile 
                                for aid in coalition])
        
        # Use standard deviation across dimensions as diversity measure
        return np.mean(np.std(capabilities, axis=0))
