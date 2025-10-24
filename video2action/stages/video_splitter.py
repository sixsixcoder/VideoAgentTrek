"""
Video splitting stage - Split long videos into segments
"""

import os
import subprocess
import math
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class VideoSplitter:
    """
    Split videos into fixed-duration segments for model processing.
    
    Adapted from src/video_clip.py
    """
    
    def __init__(self, config):
        """
        Initialize video splitter with configuration.
        
        Args:
            config: PipelineConfig instance
        """
        self.config = config
    
    def split_video(self, video_path: str, output_dir: str, video_id: str):
        """
        Split a video into segments of specified duration.
        
        Args:
            video_path: Path to input video file
            output_dir: Directory to save output clips
            video_id: Video identifier (used for logging)
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        
        # Check if input video exists
        if not video_path.exists():
            raise FileNotFoundError(f"Input video not found: {video_path}")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get video duration
        duration = self._get_video_duration(str(video_path))
        if duration is None:
            raise ValueError(f"Could not determine video duration for {video_path}")
        
        # Calculate number of segments
        num_segments = math.ceil(duration / self.config.segment_duration)
        
        logger.info(f"Splitting {video_id}: {duration:.1f}s → {num_segments} segments")
        
        # Process each segment
        segments_created = 0
        segments_skipped = 0
        
        for i in range(num_segments):
            start_time = i * self.config.segment_duration
            output_file = output_dir / f"{i+1}.mp4"
            
            # Skip if file already exists and skip_existing is enabled
            if output_file.exists() and self.config.skip_existing:
                segments_skipped += 1
                continue
            
            # Clip the segment
            success = self._clip_segment(
                str(video_path),
                str(output_file),
                start_time,
                self.config.segment_duration
            )
            
            if success:
                segments_created += 1
            else:
                logger.warning(f"Failed to create segment {i+1}")
        
        logger.info(f"✓ Created {segments_created} segments, skipped {segments_skipped}")
    
    def _get_video_duration(self, video_path: str) -> float:
        """
        Get video duration using ffprobe.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Duration in seconds, or None if failed
        """
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=30
            )
            duration_str = result.stdout.strip()
            
            if not duration_str:
                return None
            
            return float(duration_str)
            
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError) as e:
            logger.error(f"Failed to get video duration: {e}")
            return None
    
    def _clip_segment(
        self,
        input_path: str,
        output_path: str,
        start_time: float,
        duration: float
    ) -> bool:
        """
        Clip a single segment using FFmpeg with re-encoding.
        
        Args:
            input_path: Input video path
            output_path: Output video path
            start_time: Start time in seconds
            duration: Duration in seconds
            
        Returns:
            True if successful, False otherwise
        """
        # FFmpeg command with re-encoding for smooth motion
        cmd = [
            "ffmpeg",
            "-ss", str(start_time),  # Seek to start time
            "-i", input_path,
            "-t", str(duration),  # Duration
            "-c:v", "libx264",  # Re-encode video
            "-c:a", "aac",  # Re-encode audio
            "-preset", "ultrafast",  # Fast encoding
            "-crf", "23",  # Good quality
            "-g", "30",  # Keyframe every 30 frames (~1 second at 30fps)
            "-force_key_frames", "expr:gte(t,n_forced*1)",  # Force keyframes
            "-avoid_negative_ts", "make_zero",  # Handle timestamp issues
            "-y",  # Overwrite output files
            "-loglevel", "quiet",  # Suppress FFmpeg output
            "-nostdin",  # Prevent FFmpeg from reading stdin
            output_path
        ]
        
        try:
            subprocess.run(
                cmd,
                check=True,
                timeout=120,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            return True
            
        except subprocess.TimeoutExpired:
            logger.error(f"FFmpeg timeout for segment starting at {start_time}s")
            
            # Try fallback with stream copying
            return self._clip_segment_fallback(input_path, output_path, start_time, duration)
            
        except subprocess.CalledProcessError:
            logger.warning(f"FFmpeg failed for segment starting at {start_time}s, trying fallback")
            
            # Try fallback with stream copying
            return self._clip_segment_fallback(input_path, output_path, start_time, duration)
    
    def _clip_segment_fallback(
        self,
        input_path: str,
        output_path: str,
        start_time: float,
        duration: float
    ) -> bool:
        """
        Fallback method using stream copying (faster but may have issues).
        
        Args:
            input_path: Input video path
            output_path: Output video path
            start_time: Start time in seconds
            duration: Duration in seconds
            
        Returns:
            True if successful, False otherwise
        """
        cmd = [
            "ffmpeg",
            "-ss", str(start_time),
            "-i", input_path,
            "-t", str(duration),
            "-c:v", "copy",  # Copy video codec
            "-c:a", "copy",  # Copy audio codec
            "-avoid_negative_ts", "make_zero",
            "-y",
            "-loglevel", "quiet",
            "-nostdin",
            output_path
        ]
        
        try:
            subprocess.run(
                cmd,
                check=True,
                timeout=60,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            return True
            
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
            logger.error(f"Fallback also failed for segment at {start_time}s")
            return False

