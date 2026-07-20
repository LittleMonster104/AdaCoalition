"""
Utility functions for evaluation metrics
"""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

def compute_accuracy(predictions, ground_truth):
    """Compute accuracy score"""
    return accuracy_score(ground_truth, predictions)

def compute_f1_score(predictions, ground_truth):
    """Compute F1 score"""
    return f1_score(ground_truth, predictions, average='weighted', zero_division=0)

def compute_cohens_kappa(predictions, ground_truth):
    """Compute Cohen's Kappa for inter-rater agreement"""
    return cohen_kappa_score(ground_truth, predictions)

def compute_precision_recall(predictions, ground_truth):
    """Compute precision and recall"""
    from sklearn.metrics import precision_score, recall_score
    precision = precision_score(ground_truth, predictions, average='weighted', zero_division=0)
    recall = recall_score(ground_truth, predictions, average='weighted', zero_division=0)
    return precision, recall
