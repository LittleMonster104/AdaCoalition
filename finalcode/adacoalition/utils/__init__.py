"""Utility functions"""
import torch
from .logger import setup_logger, ExperimentLogger


def get_device():
    """Detect best available device (MPS for Mac M-series, CUDA, or CPU)"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✓ Using MPS (Metal Performance Shaders) - Mac GPU acceleration")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("✓ Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("⚠ Using CPU (slower)")
    return device


__all__ = ['setup_logger', 'ExperimentLogger', 'get_device']
