"""Performance metrics implementation"""
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    cohen_kappa_score
)
from scipy.stats import spearmanr
from typing import List, Dict, Any

def compute_auc(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """
    Compute AUC-ROC for knowledge tracing (Education domain)
    
    Args:
        y_true: True binary labels (0/1)
        y_pred_proba: Predicted probabilities (0-1)
    
    Returns:
        AUC score (0-1)
    """
    if len(np.unique(y_true)) < 2:
        return 0.5  # Random baseline if only one class
    return roc_auc_score(y_true, y_pred_proba)

def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute classification accuracy (Science & Talent domains)
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        Accuracy (0-1)
    """
    return accuracy_score(y_true, y_pred)

def compute_f1_score(y_true: np.ndarray, y_pred: np.ndarray, 
                     average: str = 'binary') -> float:
    """
    Compute F1 score (Science domain)
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: 'binary', 'micro', 'macro', 'weighted'
    
    Returns:
        F1 score (0-1)
    """
    return f1_score(y_true, y_pred, average=average, zero_division=0)

def compute_cohens_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Cohen's Kappa (Talent domain - HR agreement)
    
    Args:
        y_true: True labels (HR annotations)
        y_pred: Predicted labels
    
    Returns:
        Cohen's Kappa (-1 to 1)
    """
    return cohen_kappa_score(y_true, y_pred)

def compute_spearman_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Spearman correlation (Science domain - impact prediction)
    
    Args:
        y_true: True values (e.g., citation counts)
        y_pred: Predicted values
    
    Returns:
        Spearman correlation coefficient (-1 to 1)
    """
    if len(y_true) < 2:
        return 0.0
    corr, _ = spearmanr(y_true, y_pred)
    return corr if not np.isnan(corr) else 0.0

def compute_token_efficiency(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute token usage efficiency metrics
    
    Args:
        results: List of experiment results with 'tokens_used' field
    
    Returns:
        Dict with total_tokens, avg_tokens_per_task, efficiency_ratio
    """
    total_tokens = sum(r.get('tokens_used', 0) for r in results)
    num_tasks = len(results)
    avg_tokens = total_tokens / num_tasks if num_tasks > 0 else 0
    
    return {
        'total_tokens': total_tokens,
        'avg_tokens_per_task': avg_tokens,
        'num_tasks': num_tasks
    }

def compute_cross_domain_bias(predictions: Dict[str, np.ndarray],
                              ground_truth: Dict[str, np.ndarray]) -> float:
    """
    Compute cross-domain bias (Science domain - interdisciplinary proposals)
    
    Args:
        predictions: Dict mapping domain to predictions
        ground_truth: Dict mapping domain to true labels
    
    Returns:
        Bias metric (0-1, lower is better)
    """
    domain_errors = {}
    for domain in predictions:
        if domain in ground_truth:
            preds = predictions[domain]
            truth = ground_truth[domain]
            errors = np.mean(preds != truth)
            domain_errors[domain] = errors
    
    if len(domain_errors) < 2:
        return 0.0
    
    # Compute variance in error rates across domains
    error_values = list(domain_errors.values())
    bias = np.std(error_values)  # Higher variance = higher bias
    return bias

def evaluate_experiment(method_name: str,
                       predictions: Dict[str, Any],
                       ground_truth: Dict[str, Any],
                       domain: str) -> Dict[str, float]:
    """
    Comprehensive evaluation for a single method
    
    Args:
        method_name: Name of method (e.g., 'AdaCoalition', 'BERT')
        predictions: Predictions from the method
        ground_truth: Ground truth labels
        domain: 'education', 'science', or 'talent'
    
    Returns:
        Dict of metric name -> value
    """
    metrics = {}
    
    y_true = np.array(ground_truth['labels'])
    y_pred = np.array(predictions['labels'])
    
    if domain == 'education':
        # Knowledge tracing metrics
        y_pred_proba = np.array(predictions.get('probabilities', y_pred))
        metrics['auc'] = compute_auc(y_true, y_pred_proba)
        metrics['accuracy'] = compute_accuracy(y_true, (y_pred_proba > 0.5).astype(int))
        
    elif domain == 'science':
        # Research evaluation metrics
        metrics['accuracy'] = compute_accuracy(y_true, y_pred)
        metrics['f1_score'] = compute_f1_score(y_true, y_pred)
        
        # Impact prediction if available
        if 'impact_true' in ground_truth and 'impact_pred' in predictions:
            metrics['impact_correlation'] = compute_spearman_correlation(
                np.array(ground_truth['impact_true']),
                np.array(predictions['impact_pred'])
            )
        
        # Cross-domain bias if available
        if 'domain_predictions' in predictions:
            metrics['cross_domain_bias'] = compute_cross_domain_bias(
                predictions['domain_predictions'],
                ground_truth.get('domain_labels', {})
            )
    
    elif domain == 'talent':
        # Talent matching metrics
        metrics['accuracy'] = compute_accuracy(y_true, y_pred)
        metrics['cohens_kappa'] = compute_cohens_kappa(y_true, y_pred)
        
        # Precision@5 if available
        if 'top5_correct' in predictions:
            metrics['precision_at_5'] = np.mean(predictions['top5_correct'])
    
    # Token efficiency (all domains)
    if 'results' in predictions:
        token_metrics = compute_token_efficiency(predictions['results'])
        metrics.update(token_metrics)
    
    return metrics

def compare_methods(method_results: Dict[str, Dict[str, float]],
                   baseline_method: str = 'SingleBest') -> Dict[str, Dict[str, float]]:
    """
    Compare all methods against a baseline
    
    Args:
        method_results: Dict mapping method name to metrics dict
        baseline_method: Name of baseline method
    
    Returns:
        Dict with relative improvements for each method
    """
    if baseline_method not in method_results:
        return {}
    
    baseline_metrics = method_results[baseline_method]
    comparisons = {}
    
    for method_name, metrics in method_results.items():
        if method_name == baseline_method:
            continue
        
        comparison = {}
        for metric_name, value in metrics.items():
            baseline_value = baseline_metrics.get(metric_name, 0)
            if baseline_value > 0:
                improvement = ((value - baseline_value) / baseline_value) * 100
                comparison[f'{metric_name}_improvement_%'] = improvement
            comparison[f'{metric_name}_absolute'] = value
        
        comparisons[method_name] = comparison
    
    return comparisons
