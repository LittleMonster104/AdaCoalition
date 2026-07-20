"""Dataset loaders"""
from .ednet import EdNetDataset
from .aibs import AIBSDataset
from .resume import ResumeDataset

__all__ = ['EdNetDataset', 'AIBSDataset', 'ResumeDataset']
