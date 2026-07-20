"""Hierarchical multi-agent baseline (SciAgent/HiveMind style)"""
import numpy as np
from typing import List, Dict, Any
from ada_coalition.agents import BaseAgent


class HierarchicalStyle:
    """
    Mimics SciAgent/HiveMind hierarchical structure.
    Three-layer architecture: Coordinator → Specialists → Executors
    
    Reference: Hierarchical multi-agent systems where a coordinator delegates
    to specialists who further delegate to executors.
    """
    
    def __init__(self, agents: List[BaseAgent]):
        """
        Args:
            agents: List of agents (need at least 5 for 3-layer hierarchy)
        """
        if len(agents) < 5:
            raise ValueError("Hierarchical-Style requires at least 5 agents")
        
        # Layer 1: Top coordinator (1 agent)
        self.coordinator = agents[0]
        
        # Layer 2: Specialists (2-3 agents)
        num_specialists = min(3, len(agents) - 2)
        self.specialists = agents[1:1+num_specialists]
        
        # Layer 3: Executors (remaining agents)
        self.executors = agents[1+num_specialists:]
        
        if len(self.executors) == 0:
            self.executors = [agents[-1]]  # At least one executor
        
        self.agent_dict = {a.agent_id: a for a in agents}
    
    def process_task(self, task: Dict[str, Any], task_requirement: np.ndarray = None) -> Dict[str, Any]:
        """
        Hierarchical processing:
        1. Coordinator creates high-level plan
        2. Specialists develop detailed strategies
        3. Executors implement solutions
        4. Coordinator integrates results
        
        Args:
            task: Task data
            task_requirement: Optional requirement vector (unused in fixed hierarchy)
            
        Returns:
            Task result with hierarchy metadata
        """
        query = task.get('query', task.get('text', ''))
        
        # Layer 1: Coordinator planning
        coord_input = {
            'query': query,
            'context': 'You are the Coordinator. Create a high-level plan and decompose this task.',
            'role': 'coordinator',
            'stage': 'planning'
        }
        plan = self.coordinator.forward(coord_input)
        
        # Layer 2: Specialists processing
        specialist_results = []
        for i, specialist in enumerate(self.specialists):
            spec_input = {
                'query': query,
                'context': f"Coordinator's plan: {plan.get('response', '')}. You are Specialist {i+1}. Develop a detailed strategy for your part.",
                'role': f'specialist_{i}',
                'stage': 'strategy'
            }
            spec_result = specialist.forward(spec_input)
            specialist_results.append(spec_result)
        
        # Layer 3: Executors implementation
        executor_results = []
        for i, executor in enumerate(self.executors):
            # Each executor works on specialist's strategy (round-robin assignment)
            spec_idx = i % len(specialist_results)
            exec_input = {
                'query': query,
                'context': f"Plan: {plan.get('response', '')}. Strategy: {specialist_results[spec_idx].get('response', '')}. You are Executor {i+1}. Implement this solution.",
                'role': f'executor_{i}',
                'stage': 'execution'
            }
            exec_result = executor.forward(exec_input)
            executor_results.append(exec_result)
        
        # Layer 1: Coordinator integration
        integration_context = "Integration phase. Specialist strategies:\n"
        for i, sr in enumerate(specialist_results):
            integration_context += f"Specialist {i+1}: {sr.get('response', '')}\n"
        integration_context += "\nExecutor results:\n"
        for i, er in enumerate(executor_results):
            integration_context += f"Executor {i+1}: {er.get('response', '')}\n"
        integration_context += "\nIntegrate all results into a final answer."
        
        integration_input = {
            'query': query,
            'context': integration_context,
            'role': 'coordinator',
            'stage': 'integration'
        }
        final_result = self.coordinator.forward(integration_input)
        
        # Calculate average confidence
        all_confidences = (
            [plan.get('confidence', 0.5)] +
            [r.get('confidence', 0.5) for r in specialist_results] +
            [r.get('confidence', 0.5) for r in executor_results] +
            [final_result.get('confidence', 0.5)]
        )
        
        # Calculate messages:
        # Coordinator -> all specialists: len(specialists) messages
        # Each specialist -> assigned executors: len(executors) messages
        # All executors -> coordinator: len(executors) messages
        num_messages = len(self.specialists) + len(self.executors) + len(self.executors)
        
        return {
            'response': final_result.get('response', ''),
            'confidence': np.mean(all_confidences),
            'coalition_size': 1 + len(self.specialists) + len(self.executors),
            'agents_used': (
                [self.coordinator.agent_id] +
                [s.agent_id for s in self.specialists] +
                [e.agent_id for e in self.executors]
            ),
            'method': 'Hierarchical-Style',
            'hierarchy': f'1 coord + {len(self.specialists)} spec + {len(self.executors)} exec',
            'num_messages': num_messages
        }
