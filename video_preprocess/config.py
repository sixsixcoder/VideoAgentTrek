"""
Configuration for video preprocessing pipeline
"""

import os
from dataclasses import dataclass, field
from typing import List
from pathlib import Path


@dataclass
class PreprocessConfig:
    """Configuration for cursor detection and video preprocessing"""
    
    # Cursor Detection
    cursor_threshold: float = 0.50  # Keep videos with >=50% cursor presence
    yolo_model_path: str = "cursor_model.pt"  # Path to YOLO model weights
    confidence_threshold: float = 0.3  # YOLO confidence threshold
    detection_stride: int = 30  # Process every Nth frame (for speed)
    
    # Segment Filtering
    min_segment_duration: int = 20  # Minimum segment duration in seconds
    cursor_timeout: int = 2  # Seconds without cursor before ending segment
    pixel_diff_threshold: int = 10  # Threshold for frame similarity
    
    # Video Settings
    support_video_formats: List[str] = field(default_factory=lambda: ['.mp4', '.webm', '.mkv'])
    
    # Output Settings
    output_dir: str = "preprocessed_data"
    save_segments: bool = True
    save_detection_details: bool = True
    save_visualizations: bool = False
    overwrite: bool = False
    
    # Processing
    num_gpus: int = 1
    max_workers: int = 4
    
    # Logging
    log_level: str = "INFO"
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if not 0 <= self.cursor_threshold <= 1:
            raise ValueError("cursor_threshold must be between 0 and 1")
        
        if self.detection_stride < 1:
            raise ValueError("detection_stride must be >= 1")
        
        if self.min_segment_duration < 0:
            raise ValueError("min_segment_duration must be >= 0")
        
        # Create output directory if it doesn't exist
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_dict(cls, config_dict: dict):
        """Create config from dictionary"""
        return cls(**config_dict)
    
    def to_dict(self) -> dict:
        """Convert config to dictionary"""
        return {
            'cursor_threshold': self.cursor_threshold,
            'yolo_model_path': self.yolo_model_path,
            'confidence_threshold': self.confidence_threshold,
            'detection_stride': self.detection_stride,
            'min_segment_duration': self.min_segment_duration,
            'cursor_timeout': self.cursor_timeout,
            'pixel_diff_threshold': self.pixel_diff_threshold,
            'support_video_formats': self.support_video_formats,
            'output_dir': self.output_dir,
            'save_segments': self.save_segments,
            'save_detection_details': self.save_detection_details,
            'save_visualizations': self.save_visualizations,
            'overwrite': self.overwrite,
            'num_gpus': self.num_gpus,
            'max_workers': self.max_workers,
            'log_level': self.log_level,
        }

