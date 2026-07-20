"""Coalition formation algorithm (Algorithm 1 from paper)"""
import numpy as np
from typing import List, Set, Dict, Tuple
from ..agents import BaseAgent


class CoalitionFormation:
    """Dynamic coalition formation using merge-split algorithm"""
    
    def __init__(self, 
                 theta_merge: float = 0.1,
                 theta_split: float = 0.95,
                 max_iterations: int = 20,
                 verbose: bool = False,
                 diversity_weight: float = 0.3,
                 max_coalition_size: int = 5,
                 min_coalition_size: int = 1,
                 use_adaptive: bool = False,
                 domain: str = 'general'):
        self.theta_merge = theta_merge
        self.theta_split = theta_split
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.diversity_weight = diversity_weight
        self.max_coalition_size = max_coalition_size
        self.min_coalition_size = min_coalition_size
        self.use_adaptive = use_adaptive
        self.domain = domain
    
    def estimate_task_complexity(self, task: Dict, requirement: np.ndarray) -> float:
        """
        Estimate task complexity to adapt coalition strategy
        
        Returns:
            float: complexity score 0-1 (0=simple, 1=complex)
        """
        complexity = 0.5  # Default
        
        # Domain-specific heuristics
        if self.domain == 'talent':
            # Talent tasks are generally simple matching
            complexity = 0.3
            
            # Check number of candidates
            if 'candidates' in task:
                num_candidates = len(task['candidates'])
                if num_candidates > 10:
                    complexity += 0.1
            
            # Check required skills complexity
            if 'required_skills' in task:
                num_skills = len(task['required_skills'])
                if num_skills > 5:
                    complexity += 0.1
        
        elif self.domain == 'education':
            # Education tasks are medium complexity
            complexity = 0.5
            
            # Check student history length
            if 'student_id' in task:
                complexity += 0.1
        
        elif self.domain == 'science':
            # Science tasks are complex
            complexity = 0.8
            
            # Check abstract/text length
            if 'abstract' in task:
                text_len = len(str(task.get('abstract', '')))
                if text_len > 500:
                    complexity = 0.9
        
        # Check requirement vector complexity
        if requirement is not None and len(requirement) > 0:
            req_variance = np.var(requirement)
            if req_variance > 0.05:
                complexity += 0.1
        
        return max(0.1, min(1.0, complexity))
    
    def form_coalitions(self, 
                       agents: List[BaseAgent],
                       task_requirements: Dict[str, np.ndarray],
                       task: Dict = None) -> Dict[str, Set[str]]:
        """
        Algorithm 1: Merge-Split Coalition Formation
        
        Args:
            agents: List of available agents
            task_requirements: Dict mapping subtask_id to requirement vector
            task: Optional task dict for complexity estimation
            
        Returns:
            Dict mapping subtask_id to set of agent_ids
        """
        # Adaptive coalition parameters based on task complexity
        if self.use_adaptive and task is not None:
            requirement = list(task_requirements.values())[0] if task_requirements else None
            complexity = self.estimate_task_complexity(task, requirement)
            
            # Adjust parameters based on complexity
            if complexity < 0.4:  # Simple task
                self.theta_merge = 0.15  # Slightly more conservative
                self.theta_split = 0.85  # Easier to split
                adaptive_max_size = 2
            elif complexity < 0.6:  # Medium task
                self.theta_merge = 0.12
                self.theta_split = 0.90
                adaptive_max_size = 3
            else:  # Complex task
                self.theta_merge = 0.1  # Original (easy to merge)
                self.theta_split = 0.95
                adaptive_max_size = 5
            
            # Override max coalition size if adaptive
            max_coalition_size = min(self.max_coalition_size, adaptive_max_size)
            
            if self.verbose:
                print(f"Task complexity: {complexity:.2f}, max_size: {adaptive_max_size}, theta_merge: {self.theta_merge}")
        else:
            max_coalition_size = self.max_coalition_size
        
        # Step 1: Initialize singleton coalitions
        coalitions = {agent.agent_id: {agent.agent_id} for agent in agents}
        
        if self.verbose:
            print(f"\n=== Coalition Formation Debug ===")
            print(f"Starting with {len(agents)} agents")
            agent_caps = np.array([a.capability_profile for a in agents])
            print(f"Agent capability variance: {np.var(agent_caps, axis=0)}")
            print(f"Task requirements count: {len(task_requirements)}")
            if task_requirements:
                req_array = np.array(list(task_requirements.values()))
                print(f"Task requirement variance: {np.var(req_array, axis=0)}")
        
        for iteration in range(self.max_iterations):
            changed = False
            
            # Merge phase - use list() to avoid modification during iteration
            coalition_ids = list(coalitions.keys())
            merge_happened = False
            for i in range(len(coalition_ids)):
                if merge_happened:
                    break
                for j in range(i + 1, len(coalition_ids)):
                    # Check if both coalitions still exist
                    if coalition_ids[i] not in coalitions or coalition_ids[j] not in coalitions:
                        continue
                        
                    c_i = coalitions[coalition_ids[i]]
                    c_j = coalitions[coalition_ids[j]]
                    
                    comp = self._compute_complementarity(c_i, c_j, agents)
                    if self.verbose and iteration == 0 and i < 3 and j < 3:
                        print(f"  Complementarity({list(c_i)}, {list(c_j)}) = {comp:.4f} (threshold={self.theta_merge})")
                    
                    # Check if merge would exceed max coalition size
                    merged_size = len(c_i) + len(c_j)
                    
                    if comp > self.theta_merge and merged_size <= max_coalition_size:
                        # Merge coalitions
                        merged = c_i.union(c_j)
                        new_id = f"coalition_{iteration}_{i}_{j}"
                        coalitions[new_id] = merged
                        del coalitions[coalition_ids[i]]
                        del coalitions[coalition_ids[j]]
                        changed = True
                        merge_happened = True
                        if self.verbose:
                            print(f"  → Merged {list(c_i)} + {list(c_j)} → {list(merged)}")
                        break
            
            # Split phase
            split_happened = False
            for coalition_id in list(coalitions.keys()):
                if split_happened:
                    break
                if coalition_id not in coalitions:
                    continue
                coalition = coalitions[coalition_id]
                if len(coalition) > self.min_coalition_size:
                    conflict = self._compute_conflict(coalition, agents)
                    if self.verbose and iteration == 0:
                        print(f"  Conflict({list(coalition)}) = {conflict:.4f} (threshold={self.theta_split})")
                    if conflict > self.theta_split:
                        # Split coalition
                        split1, split2 = self._find_best_split(coalition, agents)
                        coalitions[f"{coalition_id}_s1"] = split1
                        coalitions[f"{coalition_id}_s2"] = split2
                        del coalitions[coalition_id]
                        changed = True
                        split_happened = True
                        if self.verbose:
                            print(f"  → Split {list(coalition)} into {list(split1)} and {list(split2)}")
                        break
            
            if not changed:
                break
        
        if self.verbose:
            coalition_sizes = [len(c) for c in coalitions.values()]
            print(f"\nFinal coalitions after {iteration+1} iterations:")
            print(f"  Total coalitions: {len(coalitions)}")
            print(f"  Coalition sizes: {coalition_sizes}")
            print(f"  Size distribution: {dict(zip(*np.unique(coalition_sizes, return_counts=True)))}")
        
        # Assignment phase: assign coalitions to subtasks
        assignment = {}
        for subtask_id, requirement in task_requirements.items():
            best_coalition = max(coalitions.values(),
                                key=lambda c: self._match_score(c, requirement, agents))
            assignment[subtask_id] = best_coalition
            if self.verbose:
                match = self._match_score(best_coalition, requirement, agents)
                print(f"  Task {subtask_id}: assigned coalition of size {len(best_coalition)} (match={match:.3f})")
        
        return assignment
    
    def _compute_complementarity(self, c1: Set[str], c2: Set[str], 
                                 agents: List[BaseAgent]) -> float:
        """Compute Comp(S_i, S_j) from paper"""
        perf_merged = self._coalition_performance(c1.union(c2), agents)
        perf_c1 = self._coalition_performance(c1, agents)
        perf_c2 = self._coalition_performance(c2, agents)
        
        numerator = perf_merged - max(perf_c1, perf_c2)
        denominator = perf_c1 + perf_c2 + 1e-8
        return numerator / denominator
    
    def _compute_conflict(self, coalition: Set[str], agents: List[BaseAgent]) -> float:
        """Compute Conflict(S) from paper"""
        if len(coalition) <= 1:
            return 0.0
        
        agent_dict = {a.agent_id: a for a in agents}
        conflicts = []
        
        for aid1 in coalition:
            for aid2 in coalition:
                if aid1 != aid2:
                    # Measure capability dissimilarity
                    a1 = agent_dict[aid1]
                    a2 = agent_dict[aid2]
                    dissim = 1.0 - np.dot(a1.capability_profile, a2.capability_profile) / (
                        np.linalg.norm(a1.capability_profile) * 
                        np.linalg.norm(a2.capability_profile) + 1e-8
                    )
                    conflicts.append(dissim)
        
        return np.mean(conflicts) if conflicts else 0.0
    
    def _coalition_performance(self, coalition: Set[str], agents: List[BaseAgent]) -> float:
        """Estimate coalition performance v(S)"""
        if not coalition:
            return 0.0
        
        agent_dict = {a.agent_id: a for a in agents}
        
        # Coverage: max capability across agents
        capabilities = np.array([agent_dict[aid].capability_profile 
                                for aid in coalition])
        coverage = np.max(capabilities, axis=0).mean()
        
        # Diversity bonus (increased from 0.1 to 0.3 to encourage merging)
        if len(coalition) > 1:
            diversity = np.std(capabilities)
        else:
            diversity = 0.0
        
        return coverage + self.diversity_weight * diversity
    
    def _match_score(self, coalition: Set[str], requirement: np.ndarray,
                    agents: List[BaseAgent]) -> float:
        """Compute match(S, r_j) from paper"""
        agent_dict = {a.agent_id: a for a in agents}
        
        # Max individual match
        matches = [agent_dict[aid].get_capability_match(requirement) 
                  for aid in coalition]
        max_match = max(matches) if matches else 0.0
        
        # Diversity term
        if len(coalition) > 1:
            capabilities = np.array([agent_dict[aid].capability_profile 
                                    for aid in coalition])
            diversity = np.mean([np.linalg.norm(capabilities[i] - capabilities[j])
                                for i in range(len(capabilities))
                                for j in range(i+1, len(capabilities))])
        else:
            diversity = 0.0
        
        lambda_div = 0.1
        return max_match + lambda_div * diversity
    
    def _find_best_split(self, coalition: Set[str], agents: List[BaseAgent]) -> Tuple[Set[str], Set[str]]:
        """Find best bipartition of coalition"""
        coalition_list = list(coalition)
        n = len(coalition_list)
        
        best_split = (set(coalition_list[:n//2]), set(coalition_list[n//2:]))
        best_score = (self._coalition_performance(best_split[0], agents) +
                     self._coalition_performance(best_split[1], agents))
        
        # Try a few random splits
        for _ in range(5):
            np.random.shuffle(coalition_list)
            split1 = set(coalition_list[:n//2])
            split2 = set(coalition_list[n//2:])
            score = (self._coalition_performance(split1, agents) +
                    self._coalition_performance(split2, agents))
            if score > best_score:
                best_score = score
                best_split = (split1, split2)
        
        return best_split
