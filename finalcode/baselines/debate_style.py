"""Multi-Agent Debate style baseline"""
import numpy as np
from typing import List, Dict, Any
from ada_coalition.agents import BaseAgent


class DebateStyle:
    """
    Mimics Multi-Agent Debate strategy.
    Multiple agents debate over several rounds, refining answers through adversarial discussion.
    
    Reference: Du et al. "Improving Factuality and Reasoning in Language Models through 
    Multiagent Debate" (2023) - agents iteratively critique and refine each other's responses.
    """
    
    def __init__(self, agents: List[BaseAgent], num_rounds: int = 3):
        """
        Args:
            agents: List of agents (debaters)
            num_rounds: Number of debate rounds (default 3)
        """
        self.agents = agents
        self.num_rounds = num_rounds
        self.agent_dict = {a.agent_id: a for a in agents}
    
    def process_task(self, task: Dict[str, Any], task_requirement: np.ndarray = None) -> Dict[str, Any]:
        """
        Multi-round debate process:
        1. Each agent generates initial response
        2. For each round: agents see others' responses and refine their own
        3. Final aggregation through voting/averaging
        
        Args:
            task: Task data
            task_requirement: Optional requirement vector (unused in debate)
            
        Returns:
            Task result with debate metadata
        """
        query = task.get('query', task.get('text', ''))
        
        # Round 0: Initial responses
        responses = []
        for agent in self.agents:
            resp = agent.forward({
                'query': query,
                'context': 'Provide your initial answer to this question.',
                'round': 0
            })
            responses.append({
                'agent_id': agent.agent_id,
                'response': resp.get('response', ''),
                'confidence': resp.get('confidence', 0.5)
            })
        
        # Debate rounds
        for round_num in range(1, self.num_rounds + 1):
            new_responses = []
            
            for i, agent in enumerate(self.agents):
                # Each agent sees others' previous responses
                others_responses = [r for j, r in enumerate(responses) if j != i]
                
                # Build debate context
                debate_context = f"Round {round_num} of debate.\n"
                debate_context += f"Your previous answer: {responses[i]['response']}\n\n"
                debate_context += "Other agents' answers:\n"
                for idx, other in enumerate(others_responses):
                    debate_context += f"Agent {idx+1}: {other['response']}\n"
                debate_context += "\nConsider the other answers and refine your response."
                
                debate_input = {
                    'query': query,
                    'context': debate_context,
                    'round': round_num,
                    'role': 'debater'
                }
                
                refined = agent.forward(debate_input)
                new_responses.append({
                    'agent_id': agent.agent_id,
                    'response': refined.get('response', ''),
                    'confidence': refined.get('confidence', 0.5)
                })
            
            responses = new_responses
        
        # Final aggregation: weighted by confidence
        confidences = [r['confidence'] for r in responses]
        weights = np.array(confidences) / (np.sum(confidences) + 1e-8)
        
        # Select response with highest confidence
        best_idx = np.argmax(confidences)
        final_response = responses[best_idx]['response']
        avg_confidence = np.mean(confidences)
        
        # Calculate total messages: 
        # Each agent sees all others' responses each round
        # Round 0: n agents produce n responses (no messages, just generation)
        # Rounds 1-k: each agent sees (n-1) others' responses
        # Total messages = k * n * (n-1)
        num_messages = self.num_rounds * len(self.agents) * (len(self.agents) - 1)
        
        return {
            'response': final_response,
            'confidence': avg_confidence,
            'coalition_size': len(self.agents),
            'agents_used': [a.agent_id for a in self.agents],
            'method': 'Debate-Style',
            'num_rounds': self.num_rounds,
            'all_confidences': confidences,
            'num_messages': num_messages
        }
