"""Baseline implementations for comparison"""
from .single_best import SingleBestAgent
from .static_pipeline import StaticPipeline
from .full_broadcast import FullBroadcast
from .metagpt_style import MetaGPTStyle
from .debate_style import DebateStyle
from .hierarchical_style import HierarchicalStyle

__all__ = [
    'SingleBestAgent', 
    'StaticPipeline', 
    'FullBroadcast',
    'MetaGPTStyle',
    'DebateStyle',
    'HierarchicalStyle'
]
