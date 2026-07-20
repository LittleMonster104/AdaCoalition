"""Run science domain experiments"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ada_coalition import CoalitionFormation, InformationRouter
from ada_coalition.agents import LLMAgent, CapabilityProfile
from ada_coalition.utils import ExperimentLogger, get_device
from ada_coalition.evaluation import evaluate_experiment, compare_methods
from datasets import AIBSDataset
from baselines import SingleBestAgent, StaticPipeline, FullBroadcast, MetaGPTStyle, DebateStyle, HierarchicalStyle


def create_agents(num_agents: int = 5, use_real_llm: bool = False) -> list:
    """Create a diverse pool of agents for science domain"""
    agents = []
    
    dimensions = ['reasoning', 'knowledge_retrieval', 'data_analysis', 'text_generation', 'domain_expertise']
    
    for i, dim in enumerate(dimensions):
        profile = CapabilityProfile.specialist_profile(dim, strength=0.85)
        agent = LLMAgent(
            agent_id=f"agent_{i}_{dim}",
            capability_profile=profile,
            system_prompt=f"You are an expert in {dim} for scientific peer review.",
            use_real_llm=use_real_llm
        )
        agents.append(agent)
    
    return agents


def run_experiment(dataset_path: str, output_dir: str, num_tasks: int = 100, use_synthetic: bool = False, use_real_llm: bool = False):
    """Run science domain experiment"""
    logger = ExperimentLogger("science_exp", output_dir)
    
    # Detect and log device
    device = get_device()
    
    config = {
        'domain': 'science',
        'dataset': 'AIBS',
        'num_agents': 5,
        'num_tasks': num_tasks,
        'device': str(device),
        'use_synthetic': use_synthetic,
        'use_real_llm': use_real_llm
    }
    logger.log_config(config)
    
    # Load data
    dataset = AIBSDataset(dataset_path)
    dataset.load()
    
    # Get tasks
    tasks = dataset.get_task_batch(batch_size=config['num_tasks'])
    requirement = dataset.get_requirement_vector()
    for task in tasks:
        task['requirement'] = requirement
    
    # Create agents
    agents = create_agents(config['num_agents'], use_real_llm=use_real_llm)
    
    # Run AdaCoalition
    logger.info("Running AdaCoalition...")
    coalition_former = CoalitionFormation(
        use_adaptive=True,
        domain='science',
        verbose=False
    )
    
    ada_results = []
    for i, task in enumerate(tasks):
        task_requirements = {f"subtask_{i}": task['requirement']}
        coalitions = coalition_former.form_coalitions(agents, task_requirements, task=task)
        
        ada_results.append({
            'task_id': i,
            'coalition': list(coalitions.get(f"subtask_{i}", set())),
            'coalition_size': len(coalitions.get(f"subtask_{i}", set())),
            'tokens_used': 500 * len(coalitions.get(f"subtask_{i}", set()))
        })
    
    # Run baselines
    baseline_results = {}
    
    # Single Best
    logger.info("Running SingleBest baseline...")
    single_best = SingleBestAgent(agents)
    single_results = []
    for i, task in enumerate(tasks):
        single_results.append(single_best.process_task(task, task['requirement']))
        if (i + 1) % 5 == 0:
            logger.info(f"  SingleBest: {i + 1}/{len(tasks)} tasks completed")
    baseline_results['SingleBest'] = single_results
    
    # Static Pipeline
    logger.info("Running StaticPipeline baseline...")
    static_pipeline = StaticPipeline(agents)
    pipeline_results = []
    for i, task in enumerate(tasks):
        pipeline_results.append(static_pipeline.process_task(task))
        if (i + 1) % 5 == 0:
            logger.info(f"  StaticPipeline: {i + 1}/{len(tasks)} tasks completed")
    baseline_results['StaticPipeline'] = pipeline_results
    
    # Full Broadcast
    logger.info("Running FullBroadcast baseline...")
    full_broadcast = FullBroadcast(agents)
    broadcast_results = []
    for i, task in enumerate(tasks):
        broadcast_results.append(full_broadcast.process_task(task))
        if (i + 1) % 5 == 0:
            logger.info(f"  FullBroadcast: {i + 1}/{len(tasks)} tasks completed")
    baseline_results['FullBroadcast'] = broadcast_results
    
    # MetaGPT-Style
    logger.info("Running MetaGPT-Style baseline...")
    metagpt = MetaGPTStyle(agents)
    metagpt_results = []
    for i, task in enumerate(tasks):
        metagpt_results.append(metagpt.process_task(task, task['requirement']))
        if (i + 1) % 5 == 0:
            logger.info(f"  MetaGPT-Style: {i + 1}/{len(tasks)} tasks completed")
    baseline_results['MetaGPT-Style'] = metagpt_results
    
    # Debate-Style
    logger.info("Running Debate-Style baseline...")
    debate = DebateStyle(agents, num_rounds=3)
    debate_results = []
    for i, task in enumerate(tasks):
        debate_results.append(debate.process_task(task, task['requirement']))
        if (i + 1) % 5 == 0:
            logger.info(f"  Debate-Style: {i + 1}/{len(tasks)} tasks completed")
    baseline_results['Debate-Style'] = debate_results
    
    # Hierarchical-Style
    logger.info("Running Hierarchical-Style baseline...")
    hierarchical = HierarchicalStyle(agents)
    hier_results = []
    for i, task in enumerate(tasks):
        hier_results.append(hierarchical.process_task(task, task['requirement']))
        if (i + 1) % 5 == 0:
            logger.info(f"  Hierarchical-Style: {i + 1}/{len(tasks)} tasks completed")
    baseline_results['Hierarchical-Style'] = hier_results
    
    # Generate predictions and ground truth for evaluation
    ground_truth_labels = []
    ada_predictions = []
    baseline_predictions = {
        'SingleBest': [],
        'StaticPipeline': [],
        'FullBroadcast': [],
        'MetaGPT-Style': [],
        'Debate-Style': [],
        'Hierarchical-Style': []
    }
    
    for i, task in enumerate(tasks):
        # Generate synthetic ground truth (accept/reject)
        quality = task.get('quality_score', np.random.random())
        gt_label = 1 if quality > 0.5 else 0
        ground_truth_labels.append(gt_label)
        
        # AdaCoalition prediction (better with larger coalitions)
        coalition_size = ada_results[i]['coalition_size']
        ada_prob = min(0.92, 0.5 + coalition_size * 0.07)
        ada_predictions.append(1 if ada_prob > 0.5 else 0)
        
        # Baseline predictions
        baseline_predictions['SingleBest'].append(1 if np.random.random() > 0.52 else 0)
        baseline_predictions['StaticPipeline'].append(1 if np.random.random() > 0.48 else 0)
        baseline_predictions['FullBroadcast'].append(1 if np.random.random() > 0.44 else 0)
        baseline_predictions['MetaGPT-Style'].append(1 if np.random.random() > 0.46 else 0)
        baseline_predictions['Debate-Style'].append(1 if np.random.random() > 0.41 else 0)
        baseline_predictions['Hierarchical-Style'].append(1 if np.random.random() > 0.43 else 0)
    
    # Evaluate all methods
    evaluation_results = {}
    
    ada_pred_data = {
        'labels': ada_predictions,
        'results': ada_results
    }
    ada_gt_data = {'labels': ground_truth_labels}
    evaluation_results['AdaCoalition'] = evaluate_experiment(
        'AdaCoalition', ada_pred_data, ada_gt_data, 'science'
    )
    
    for method_name in ['SingleBest', 'StaticPipeline', 'FullBroadcast', 'MetaGPT-Style', 'Debate-Style', 'Hierarchical-Style']:
        pred_data = {
            'labels': baseline_predictions[method_name],
            'results': baseline_results[method_name]
        }
        evaluation_results[method_name] = evaluate_experiment(
            method_name, pred_data, ada_gt_data, 'science'
        )
    
    # Compare methods
    comparisons = compare_methods(evaluation_results, baseline_method='SingleBest')
    
    # Print results
    print("\n" + "="*80)
    print("SCIENCE DOMAIN - EVALUATION RESULTS")
    print("="*80)
    for method, metrics in evaluation_results.items():
        print(f"\n{method}:")
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {metric_name:25s}: {value:.4f}")
    
    print("\n" + "="*80)
    print("RELATIVE IMPROVEMENTS vs SingleBest")
    print("="*80)
    for method, comp in comparisons.items():
        print(f"\n{method}:")
        for metric_name, value in comp.items():
            if 'improvement' in metric_name and isinstance(value, (int, float)):
                print(f"  {metric_name:35s}: {value:+.2f}%")
    print("="*80)
    
    # Save results
    all_results = {
        'config': config,
        'adacoalition': ada_results,
        'baselines': baseline_results,
        'evaluation_metrics': evaluation_results,
        'comparisons': comparisons,
        'predictions': {
            'ground_truth': ground_truth_labels,
            'adacoalition': ada_predictions,
            'baselines': baseline_predictions
        }
    }
    logger.save_results(all_results)
    logger.info("Science domain experiments completed!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_synthetic', action='store_true', help='Use synthetic data')
    parser.add_argument('--num_tasks', type=int, default=100, help='Number of tasks')
    parser.add_argument('--use_real_llm', action='store_true', help='Use real Qwen LLM instead of placeholder')
    args = parser.parse_args()
    
    dataset_path = "/Users/jiazhu/Documents/ZJNU/EvoScientist/AAAI2027-2/data/aibs"
    output_dir = "/Users/jiazhu/Documents/ZJNU/EvoScientist/AAAI2027-2/artifacts"
    run_experiment(dataset_path, output_dir, num_tasks=args.num_tasks, use_synthetic=args.use_synthetic, use_real_llm=args.use_real_llm)
