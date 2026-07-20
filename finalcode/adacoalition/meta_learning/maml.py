"""MAML-based meta-learning for cross-domain transfer"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Tuple
from copy import deepcopy


class MAML:
    """
    Model-Agnostic Meta-Learning for coalition strategies
    
    Enables fast adaptation to new domains with few examples.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 inner_lr: float = 0.01,
                 outer_lr: float = 0.001,
                 num_inner_steps: int = 5):
        """
        Args:
            model: Base model to meta-learn
            inner_lr: Learning rate for inner loop (task adaptation)
            outer_lr: Learning rate for outer loop (meta-update)
            num_inner_steps: Number of gradient steps in inner loop
        """
        self.model = model
        self.inner_lr = inner_lr
        self.num_inner_steps = num_inner_steps
        
        self.meta_optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=outer_lr
        )
    
    def inner_loop(self, 
                   support_data: List[Tuple[torch.Tensor, torch.Tensor]],
                   model: nn.Module) -> nn.Module:
        """
        Adapt model to a specific task using support set
        
        Args:
            support_data: List of (input, target) pairs
            model: Model to adapt
            
        Returns:
            Adapted model
        """
        adapted_model = deepcopy(model)
        optimizer = torch.optim.SGD(adapted_model.parameters(), lr=self.inner_lr)
        
        for _ in range(self.num_inner_steps):
            total_loss = 0.0
            
            for inputs, targets in support_data:
                outputs = adapted_model(inputs)
                loss = nn.functional.mse_loss(outputs, targets)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
        
        return adapted_model
    
    def outer_loop(self, 
                   tasks: List[Dict[str, Any]]) -> float:
        """
        Meta-training across multiple tasks
        
        Args:
            tasks: List of task dictionaries, each containing:
                - 'support': support set data
                - 'query': query set data
                
        Returns:
            Average meta-loss
        """
        meta_loss = 0.0
        
        self.meta_optimizer.zero_grad()
        
        for task in tasks:
            support_data = task['support']
            query_data = task['query']
            
            # Inner loop: adapt to this task
            adapted_model = self.inner_loop(support_data, self.model)
            
            # Evaluate on query set
            task_loss = 0.0
            for inputs, targets in query_data:
                outputs = adapted_model(inputs)
                loss = nn.functional.mse_loss(outputs, targets)
                task_loss += loss
            
            task_loss = task_loss / len(query_data)
            meta_loss += task_loss
        
        # Meta-update
        meta_loss = meta_loss / len(tasks)
        meta_loss.backward()
        self.meta_optimizer.step()
        
        return meta_loss.item()
    
    def adapt_to_domain(self, 
                       domain_data: List[Tuple[torch.Tensor, torch.Tensor]],
                       num_steps: int = None) -> nn.Module:
        """
        Quick adaptation to a new domain
        
        Args:
            domain_data: Few-shot data from the new domain
            num_steps: Number of adaptation steps (default: self.num_inner_steps)
            
        Returns:
            Domain-adapted model
        """
        if num_steps is None:
            num_steps = self.num_inner_steps
        
        adapted_model = deepcopy(self.model)
        optimizer = torch.optim.SGD(adapted_model.parameters(), lr=self.inner_lr)
        
        for step in range(num_steps):
            for inputs, targets in domain_data:
                outputs = adapted_model(inputs)
                loss = nn.functional.mse_loss(outputs, targets)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        return adapted_model


class MetaCoalitionLearner:
    """
    Meta-learner for coalition formation strategies
    
    Learns how to quickly form effective coalitions in new domains.
    """
    
    def __init__(self, capability_dim: int = 5):
        self.capability_dim = capability_dim
        
        # Simple coalition scoring model
        self.model = nn.Sequential(
            nn.Linear(capability_dim * 2, 64),  # coalition + requirement
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.maml = MAML(self.model, inner_lr=0.01, outer_lr=0.001)
    
    def meta_train(self, domain_tasks: Dict[str, List[Dict]]) -> List[float]:
        """
        Meta-train across multiple domains
        
        Args:
            domain_tasks: Dict mapping domain_name to list of task episodes
            
        Returns:
            List of meta-losses over training
        """
        losses = []
        
        # Convert domain tasks to MAML format
        maml_tasks = []
        for domain_name, episodes in domain_tasks.items():
            for episode in episodes:
                maml_tasks.append({
                    'support': episode['support'],
                    'query': episode['query']
                })
        
        # Meta-training loop
        num_epochs = 100
        batch_size = 4
        
        for epoch in range(num_epochs):
            # Sample batch of tasks
            task_batch = np.random.choice(
                maml_tasks, 
                size=min(batch_size, len(maml_tasks)),
                replace=False
            ).tolist()
            
            loss = self.maml.outer_loop(task_batch)
            losses.append(loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Meta-Loss: {loss:.4f}")
        
        return losses
    
    def adapt_to_new_domain(self, 
                           few_shot_examples: List[Dict]) -> nn.Module:
        """
        Adapt to a new domain with few examples
        
        Args:
            few_shot_examples: Few labeled examples from new domain
            
        Returns:
            Adapted coalition scoring model
        """
        # Convert to (input, target) format
        domain_data = []
        for example in few_shot_examples:
            inputs = torch.tensor(example['input'], dtype=torch.float32)
            target = torch.tensor([example['score']], dtype=torch.float32)
            domain_data.append((inputs, target))
        
        adapted_model = self.maml.adapt_to_domain(domain_data)
        
        return adapted_model
