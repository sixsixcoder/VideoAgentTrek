"""
Configuration management for Video2Action pipeline
"""
from dataclasses import dataclass, field
from typing import Optional, List
import os


@dataclass
class PipelineConfig:
    """Configuration for the Video2Action pipeline"""
    
    # Model paths (Qwen2.5-VL for keyframe detection and action identification)
    # Set these to your local model paths or use environment variables
    model_path: str = os.getenv("KEYFRAME_MODEL_PATH", "/path/to/keyframe/detection/model")
    action_model_path: str = os.getenv("ACTION_MODEL_PATH", "/path/to/action/identification/model")
    
    # OpenAI API configuration
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None  # For DashScope or other compatible APIs
    openai_model: str = "gpt-4o"  # Model for both validation and inner monologue
    validation_model: str = "gpt-4o"  # Deprecated: use openai_model
    validation_reasoning_effort: Optional[str] = None
    
    # Inner monologue generation
    enable_inner_monologue: bool = False  # Requires OpenAI API key
    inner_monologue_skip_existing: bool = True
    
    # Video splitting parameters
    segment_duration: int = 10  # seconds
    
    # Keyframe detection parameters
    keyframe_fps: float = 1.0
    similarity_threshold: float = 0.9999
    
    # GPU configuration
    available_gpus: List[int] = field(default_factory=lambda: [0])
    processes_per_gpu: int = 2
    cpu_workers: int = 32
    
    # Action identification parameters
    action_max_frames: int = 20
    
    # Validation parameters
    enable_validation: bool = True
    validation_workers: int = 4
    
    # GPT content restoration for invalid type/key actions
    enable_gpt_restore: bool = False  # Restore invalid content using GPT
    gpt_restore_workers: int = 4
    
    # Image format for trajectory
    image_format: str = "JPEG"  # or "PNG"
    
    # Processing parameters
    skip_existing: bool = True
    clean_intermediate: bool = False  # Whether to clean up intermediate files
    
    # Transcript parameters (optional)
    transcript_dir: Optional[str] = None
    
    def __post_init__(self):
        """Validate and set defaults"""
        if self.openai_api_key is None:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "PipelineConfig":
        """Create config from dictionary"""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__annotations__})

