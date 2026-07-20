"""Run ablation study experiments"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ada_coalition import CoalitionFormation, InformationRouter
from ada_coalition.agents import LLMAgent, CapabilityProfile
from ada_coalition.coalition import CoalitionMetrics
from ada_coalition.utils import ExperimentLogger
from datasets import EdNetDataset


def create_agents(num_agents: int = 5) -> list:
    """Create agents"""
    agents = []
    dimensions = ['reasoning', 'knowledge_retrieval', 'data_analysis', 'text_generation', 'domain_expertise']
    
    for i, dim in enumerate(dimensions):
        profile = CapabilityProfile.specialist_profile(dim, strength=0.85)
        agent = LLMAgent(
            agent_id=f"agent_{i}_{dim}",
            capability_profile=profile
        )
        agents.append(agent)
    
    return agents


def ablation_no_coalition(tasks, agents, logger):
    """Ablation: Remove coalition formation (use all agents)"""
    logger.info("Running ablation: No coalition formation")
    
    results = []
    for i, task in enumerate(tasks):
        # Use all agents
        result = {
            'task_id': i,
            'coalition_size': len(agents),
            'agents_used': [a.agent_id for a in agents]
        }
        results.append(result)
    
    return results


def ablation_no_routing(tasks, agents, logger):
    """Ablation: Remove information routing (broadcast all)"""
    logger.info("Running ablation: No information routing")
    
    coalition_former = CoalitionFormation()
    
    results = []
    total_messages = 0
    
    for i, task in enumerate(tasks):
        requirement = task.get('requirement', np.ones(5))
        task_requirements = {f"subtask_{i}": requirement}
        coalitions = coalition_former.form_coalitions(agents, task_requirements)
        
        coalition = coalitions.get(f"subtask_{i}", set())
        
        # Broadcast all messages (no routing)
        num_messages = len(coalition) * (len(coalition) - 1)  # Full broadcast
        total_messages += num_messages
        
        results.append({
            'task_id': i,
            'coalition_size': len(coalition),
            'num_messages': num_messages
        })
    
    logger.log_metric("total_messages_no_routing", total_messages)
    return results


def ablation_no_meta_learning(tasks, agents, logger):
    """Ablation: Remove meta-learning (no cross-domain transfer)"""
    logger.info("Running ablation: No meta-learning")
    
    # Coalition formation without meta-learned initialization
    coalition_former = CoalitionFormation(theta_merge=0.3, theta_split=0.7)
    
    results = []
    for i, task in enumerate(tasks):
        requirement = task.get('requirement', np.ones(5))
        task_requirements = {f"subtask_{i}": requirement}
        coalitions = coalition_former.form_coalitions(agents, task_requirements)
        
        results.append({
            'task_id': i,
            'coalition': list(coalitions.get(f"subtask_{i}", set()))
        })
    
    return results


def main():
    """Run all ablation studies"""
    logger = ExperimentLogger("ablation_study", "/Users/jiazhu/Documents/ZJNU/EvoScientist/AAAI2027-2/artifacts")
    
    config = {
        'study': 'ablation',
        'num_agents': 5,
        'num_tasks': 50
    }
    logger.log_config(config)
    
    # Load data
    dataset = EdNetDataset("/Users/jiazhu/Documents/ZJNU/EvoScientist/AAAI2027-2/data/ednet")
    dataset.load()
    
    tasks = dataset.get_task_batch(batch_size=config['num_tasks'])
    requirement = dataset.get_requirement_vector()
    for task in tasks:
        task['requirement'] = requirement
    
    agents = create_agents(config['num_agents'])
    
    # Run ablations
    ablation_results = {
        'no_coalition': ablation_no_coalition(tasks, agents, logger),
        'no_routing': ablation_no_routing(tasks, agents, logger),
        'no_meta_learning': ablation_no_meta_learning(tasks, agents, logger)
    }
    
    logger.save_results({'config': config, 'ablations': ablation_results})
    logger.info("Ablation studies completed!")


if __name__ == "__main__":
    main()
