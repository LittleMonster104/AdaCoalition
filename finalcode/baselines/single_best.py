"""Single Best Agent baseline"""
import numpy as np
from typing import List, Dict, Any
from ada_coalition.agents import BaseAgent


class SingleBestAgent:
    """
    Baseline: Select single best agent for each task
    
    No collaboration, just route to the agent with highest capability match.
    """
    
    def __init__(self, agents: List[BaseAgent]):
        self.agents = agents
        self.agent_dict = {a.agent_id: a for a in agents}
    
    def select_agent(self, task_requirement: np.ndarray) -> str:
        """
        Select the single best agent for a task
        
        Args:
            task_requirement: Requirement vector for the task
            
        Returns:
            agent_id of the best agent
        """
        best_agent = None
        best_score = -float('inf')
        
        for agent in self.agents:
            score = agent.get_capability_match(task_requirement)
            if score > best_score:
                best_score = score
                best_agent = agent
        
        return best_agent.agent_id if best_agent else self.agents[0].agent_id
    
    def process_task(self, 
                    task: Dict[str, Any],
                    task_requirement: np.ndarray) -> Dict[str, Any]:
        """
        Process a task using single best agent
        
        Args:
            task: Task data
            task_requirement: Requirement vector
            
        Returns:
            Task result
        """
        agent_id = self.select_agent(task_requirement)
        agent = self.agent_dict[agent_id]
        
        result = agent.forward(task)
        result['method'] = 'SingleBest'
        result['agents_used'] = [agent_id]
        
        return result
