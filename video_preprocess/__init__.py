"""
Video preprocessing module - Cursor detection and quality filtering

This module provides tools to:
1. Detect cursor presence in screen recordings using YOLO-v8
2. Calculate cursor presence percentage
3. Filter videos based on cursor activity threshold (default: 50%)
4. Extract active segments for further processing
"""

from .config import PreprocessConfig
from .cursor_detector import CursorDetector
from .pipeline import PreprocessPipeline

__all__ = [
    'PreprocessConfig',
    'CursorDetector',
    'PreprocessPipeline',
]

__version__ = '1.0.0'

