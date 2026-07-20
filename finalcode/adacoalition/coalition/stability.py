"""Coalition stability analysis"""
import numpy as np
from typing import List, Set, Dict
from ..agents import BaseAgent


class StabilityAnalyzer:
    """Analyze coalition stability using core-based metrics"""
    
    def __init__(self):
        pass
    
    def is_core_stable(self, 
                      coalitions: Dict[str, Set[str]], 
                      agents: List[BaseAgent]) -> bool:
        """
        Check if coalition structure is in the core
        
        A coalition structure is core-stable if no subset of agents
        can improve their payoff by deviating.
        """
        agent_dict = {a.agent_id: a for a in agents}
        
        # Compute current payoffs
        current_payoffs = self._compute_payoffs(coalitions, agents)
        
        # Check all possible deviating coalitions
        all_agent_ids = set(agent_dict.keys())
        
        for size in range(2, len(all_agent_ids) + 1):
            # Sample subsets (exhaustive check is exponential)
            for _ in range(min(100, 2**size)):
                subset = set(np.random.choice(list(all_agent_ids), 
                                             size=min(size, len(all_agent_ids)), 
                                             replace=False))
                
                # Compute potential payoff if this subset forms a coalition
                potential_payoff = self._coalition_value(subset, agents)
                avg_potential = potential_payoff / len(subset)
                
                # Check if any member would benefit
                current_avg = np.mean([current_payoffs[aid] for aid in subset])
                
                if avg_potential > current_avg + 0.01:  # Epsilon for numerical stability
                    return False
        
        return True
    
    def compute_stability_score(self, 
                               coalitions: Dict[str, Set[str]], 
                               agents: List[BaseAgent]) -> float:
        """
        Compute stability score in [0, 1]
        
        Higher score means more stable coalition structure.
        """
        agent_dict = {a.agent_id: a for a in agents}
        payoffs = self._compute_payoffs(coalitions, agents)
        
        # Measure variance in payoffs (lower is better)
        payoff_values = list(payoffs.values())
        payoff_variance = np.var(payoff_values)
        
        # Measure intra-coalition similarity (higher is better)
        similarities = []
        for coalition in coalitions.values():
            if len(coalition) > 1:
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
        
        avg_similarity = np.mean(similarities) if similarities else 0.5
        
        # Combine metrics
        stability = (1.0 / (1.0 + payoff_variance)) * 0.5 + avg_similarity * 0.5
        
        return stability
    
    def _compute_payoffs(self, 
                        coalitions: Dict[str, Set[str]], 
                        agents: List[BaseAgent]) -> Dict[str, float]:
        """Compute payoff for each agent using Shapley value approximation"""
        payoffs = {}
        
        for coalition in coalitions.values():
            coalition_value = self._coalition_value(coalition, agents)
            
            # Equal split as simple payoff division
            per_agent = coalition_value / len(coalition) if coalition else 0.0
            
            for agent_id in coalition:
                payoffs[agent_id] = per_agent
        
        return payoffs
    
    def _coalition_value(self, coalition: Set[str], agents: List[BaseAgent]) -> float:
        """Compute value function v(S)"""
        if not coalition:
            return 0.0
        
        agent_dict = {a.agent_id: a for a in agents}
        
        # Aggregate capabilities
        capabilities = np.array([agent_dict[aid].capability_profile 
                                for aid in coalition])
        
        # Value = coverage + diversity bonus
        coverage = np.max(capabilities, axis=0).sum()
        
        if len(coalition) > 1:
            diversity = np.std(capabilities)
        else:
            diversity = 0.0
        
        return coverage + 0.1 * diversity
