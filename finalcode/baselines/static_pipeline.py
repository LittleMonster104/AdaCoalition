"""Static Pipeline baseline"""
import numpy as np
from typing import List, Dict, Any
from ada_coalition.agents import BaseAgent


class StaticPipeline:
    """
    Baseline: Fixed sequential pipeline of agents
    
    All agents process in a predefined order, regardless of task.
    """
    
    def __init__(self, agents: List[BaseAgent]):
        self.agents = agents
        self.agent_dict = {a.agent_id: a for a in agents}
    
    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process task through static pipeline
        
        Args:
            task: Task data
            
        Returns:
            Final task result after all agents
        """
        current_data = task
        agents_used = []
        
        # Pass through each agent in order
        for agent in self.agents:
            result = agent.forward(current_data)
            
            # Update data with agent's output
            if 'response' in result:
                current_data['context'] = current_data.get('context', '') + ' ' + result['response']
            
            agents_used.append(agent.agent_id)
            
            # Update agent context
            agent.update_context({
                'content': result.get('response', ''),
                'task': task.get('query', '')
            })
        
        return {
            'result': current_data,
            'method': 'StaticPipeline',
            'agents_used': agents_used,
            'num_messages': len(self.agents) - 1  # n-1 messages in pipeline
        }
