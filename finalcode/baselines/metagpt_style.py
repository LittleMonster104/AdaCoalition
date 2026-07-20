"""MetaGPT-style baseline: Fixed role-based workflow"""
import numpy as np
from typing import List, Dict, Any
from ada_coalition.agents import BaseAgent


class MetaGPTStyle:
    """
    Mimics MetaGPT's fixed role assignment strategy.
    Agents are assigned fixed roles: Planner → Architect → Executor → Verifier
    
    Reference: MetaGPT uses Software Development Life Cycle (SDLC) roles with
    fixed sequential workflow.
    """
    
    def __init__(self, agents: List[BaseAgent]):
        """
        Args:
            agents: List of agents (need at least 4)
        """
        if len(agents) < 4:
            raise ValueError("MetaGPT-Style requires at least 4 agents")
        
        # Fixed role assignment
        self.planner = agents[0]      # Product Manager role
        self.architect = agents[1]     # Architect role
        self.executor = agents[2]      # Engineer role
        self.verifier = agents[3]      # QA role
        
        self.roles = ['Planner', 'Architect', 'Executor', 'Verifier']
        self.agent_dict = {a.agent_id: a for a in agents}
    
    def process_task(self, task: Dict[str, Any], task_requirement: np.ndarray = None) -> Dict[str, Any]:
        """
        Process task through fixed workflow: Plan → Design → Execute → Verify
        
        Args:
            task: Task data
            task_requirement: Optional requirement vector (unused in fixed workflow)
            
        Returns:
            Task result with method metadata
        """
        # Stage 1: Planning
        plan_input = {
            'query': task.get('query', task.get('text', '')),
            'context': 'You are the Planner. Create a high-level plan for this task.',
            'role': 'planner'
        }
        plan_result = self.planner.forward(plan_input)
        
        # Stage 2: Architecture design
        arch_input = {
            'query': task.get('query', task.get('text', '')),
            'context': f"Previous plan: {plan_result.get('response', '')}. You are the Architect. Design the solution architecture.",
            'role': 'architect'
        }
        arch_result = self.architect.forward(arch_input)
        
        # Stage 3: Execution
        exec_input = {
            'query': task.get('query', task.get('text', '')),
            'context': f"Plan: {plan_result.get('response', '')}. Architecture: {arch_result.get('response', '')}. You are the Executor. Implement the solution.",
            'role': 'executor'
        }
        exec_result = self.executor.forward(exec_input)
        
        # Stage 4: Verification
        verify_input = {
            'query': task.get('query', task.get('text', '')),
            'context': f"Implementation: {exec_result.get('response', '')}. You are the Verifier. Check and validate the solution.",
            'role': 'verifier'
        }
        verify_result = self.verifier.forward(verify_input)
        
        # Calculate average confidence
        confidences = [
            plan_result.get('confidence', 0.5),
            arch_result.get('confidence', 0.5),
            exec_result.get('confidence', 0.5),
            verify_result.get('confidence', 0.5)
        ]
        
        return {
            'response': verify_result.get('response', ''),
            'confidence': np.mean(confidences),
            'coalition_size': 4,
            'agents_used': [self.planner.agent_id, self.architect.agent_id, 
                           self.executor.agent_id, self.verifier.agent_id],
            'method': 'MetaGPT-Style',
            'workflow': ' → '.join(self.roles),
            'num_messages': 3  # 3 handoffs between 4 agents
        }
