"""Full Broadcast baseline"""
import numpy as np
from typing import List, Dict, Any, Set
from ada_coalition.agents import BaseAgent


class FullBroadcast:
    """
    Baseline: Broadcast all messages to all agents
    
    Maximum communication - every agent receives every message.
    """
    
    def __init__(self, agents: List[BaseAgent]):
        self.agents = agents
        self.agent_dict = {a.agent_id: a for a in agents}
    
    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process task with full broadcast communication
        
        Args:
            task: Task data
            
        Returns:
            Aggregated result from all agents
        """
        messages = []
        agents_used = []
        
        # Initial round: all agents process the task
        for agent in self.agents:
            result = agent.forward(task)
            
            message = {
                'sender': agent.agent_id,
                'content': result.get('response', ''),
                'confidence': result.get('confidence', 0.5)
            }
            messages.append(message)
            agents_used.append(agent.agent_id)
        
        # Broadcast all messages to all agents
        for agent in self.agents:
            for message in messages:
                if message['sender'] != agent.agent_id:
                    agent.update_context(message)
        
        # Final aggregation: highest confidence response
        best_message = max(messages, key=lambda m: m['confidence'])
        
        return {
            'result': best_message['content'],
            'method': 'FullBroadcast',
            'agents_used': agents_used,
            'num_messages': len(self.agents) * (len(self.agents) - 1),  # Full mesh
            'best_agent': best_message['sender']
        }
