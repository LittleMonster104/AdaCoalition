#!/usr/bin/env python3
"""
AdaCoalition - Main Experiment Runner
Runs all three domains with AdaCoalition and baseline methods
"""

import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime

# Add paths
sys.path.insert(0, str(Path(__file__).parent))

from adacoalition.domain_agents import create_agent_pool
from adacoalition.coalition.formation import CoalitionFormation
from baselines.single_best import SingleBest
from baselines.static_pipeline import StaticPipeline
from baselines.metagpt_style import MetaGPTStyle
from baselines.debate_style import DebateStyle
from baselines.hierarchical_style import HierarchicalStyle
from baselines.full_broadcast import FullBroadcast

from datasets.ednet import EdNetDataset
from datasets.resume import ResumeDataset
from datasets.aibs import AIBSDataset

def run_domain(domain_name, dataset, methods, output_dir):
    """Run all methods on a domain"""
    print(f"\n{'='*70}")
    print(f"Running {domain_name} Domain")
    print(f"{'='*70}\n")
    
    results = {}
    tasks = dataset.get_task_batch(batch_size=30)
    
    for method_name, method in methods.items():
        print(f"\nRunning {method_name}...")
        start_time = time.time()
        
        predictions = []
        for i, task in enumerate(tasks):
            pred = method.predict(task)
            predictions.append(pred)
            
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/30")
        
        elapsed = time.time() - start_time
        
        # Calculate metrics
        from adacoalition.evaluation.metrics import compute_accuracy, compute_f1_score, compute_cohens_kappa
        ground_truth = dataset.get_ground_truth()
        
        acc = compute_accuracy(predictions, ground_truth)
        f1 = compute_f1_score(predictions, ground_truth)
        kappa = compute_cohens_kappa(predictions, ground_truth)
        
        results[method_name] = {
            'accuracy': float(acc),
            'f1_score': float(f1),
            'cohens_kappa': float(kappa),
            'time': float(elapsed),
            'predictions': [int(p) for p in predictions]
        }
        
        print(f"  ✓ Acc={acc:.4f}, F1={f1:.4f}, Kappa={kappa:.4f}, Time={elapsed:.1f}s")
    
    # Save results
    output_file = output_dir / f"{domain_name.lower()}_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'domain': domain_name,
            'timestamp': datetime.now().isoformat(),
            'results': results
        }, f, indent=2)
    
    print(f"\n✓ Saved results to {output_file}")
    return results

def main():
    """Main experiment runner"""
    print("="*70)
    print("AdaCoalition: Domain-Agnostic Multi-Agent Collaboration")
    print("Running all experiments")
    print("="*70)
    
    # Create output directory
    output_dir = Path("results") / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize methods
    print("\n1. Initializing methods...")
    agent_pool = create_agent_pool(domain="general", model_name="qwen2.5-vl:3b")
    
    methods = {
        'AdaCoalition': CoalitionFormation(agents=agent_pool),
        'MetaGPT-Style': MetaGPTStyle(agents=agent_pool),
        'Debate-Style': DebateStyle(agents=agent_pool),
        'Hierarchical-Style': HierarchicalStyle(agents=agent_pool),
        'SingleBest': SingleBest(agents=agent_pool),
        'StaticPipeline': StaticPipeline(agents=agent_pool),
        'FullBroadcast': FullBroadcast(agents=agent_pool),
    }
    print(f"   ✓ Initialized {len(methods)} methods")
    
    # Run domains
    all_results = {}
    
    # Education Domain
    print("\n2. Loading Education dataset (EdNet)...")
    ednet = EdNetDataset(data_dir="data/ednet", num_users=50)
    ednet.load()
    all_results['education'] = run_domain('Education', ednet, methods, output_dir)
    
    # Science Domain  
    print("\n3. Loading Science dataset (AIBS + PeerRead)...")
    aibs = AIBSDataset(data_dir="data/aibs")
    aibs.load()
    all_results['science'] = run_domain('Science', aibs, methods, output_dir)
    
    # Talent Domain
    print("\n4. Loading Talent dataset (Resume)...")
    resume = ResumeDataset(data_dir="data/resume")
    resume.load()
    all_results['talent'] = run_domain('Talent', resume, methods, output_dir)
    
    # Summary
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    
    for domain, results in all_results.items():
        print(f"\n{domain.upper()} Domain:")
        sorted_methods = sorted(results.items(), key=lambda x: x[1]['f1_score'], reverse=True)
        for i, (method, metrics) in enumerate(sorted_methods, 1):
            rank = "🏆" if i == 1 else f"#{i}"
            print(f"  {rank} {method:20s} F1={metrics['f1_score']:.4f}  Kappa={metrics['cohens_kappa']:.4f}")
    
    print(f"\n✓ All results saved to: {output_dir}")
    print("="*70)

if __name__ == "__main__":
    main()
