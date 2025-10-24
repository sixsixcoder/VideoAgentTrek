"""
VideoAgentTrek - Video2Action Module

This module provides a complete pipeline for extracting agent trajectories from
unlabeled screen-recorded videos.

Usage:
    from video2action import Video2ActionPipeline
    
    pipeline = Video2ActionPipeline(
        model_path="/path/to/model",
        api_key="your-openai-key"
    )
    
    trajectory = pipeline.process_video("input_video.mp4", output_dir="./output")
"""

from .pipeline import Video2ActionPipeline
from .config import PipelineConfig

__version__ = "1.0.0"
__all__ = ["Video2ActionPipeline", "PipelineConfig"]

