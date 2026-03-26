"""
Cursor detection using YOLO-v8

Adapted from cursor-detection-main/src/main.py
"""

import cv2
import json
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)


class CursorDetector:
    """
    Detect cursor presence in videos using YOLO-v8 model.
    
    Adapted from cursor-detection-main
    """
    
    def __init__(self, config):
        """
        Initialize cursor detector.
        
        Args:
            config: PreprocessConfig instance
        """
        self.config = config
        self.model = None
        
    def _load_model(self, gpu_id: int = 0):
        """Load YOLO model"""
        try:
            from ultralytics import YOLO
            
            if self.model is None:
                import os
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                self.model = YOLO(self.config.yolo_model_path)
                logger.info(f"Loaded YOLO model from {self.config.yolo_model_path}")
        except ImportError:
            logger.error("ultralytics package not installed. Install with: pip install ultralytics")
            raise
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Process a single frame using YOLO model.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            Dictionary containing detection results
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")
        
        try:
            results = self.model.predict(
                frame,
                max_det=1,
                conf=self.config.confidence_threshold,
                verbose=False
            )
            
            res_json_str = results[0].to_json()
            res_list = json.loads(res_json_str)
            if res_list:
                return res_list[0]
        except Exception as e:
            logger.debug(f"Frame processing error: {e}")
        
        return {"name": None, "class": None, "confidence": None, "box": None}
    
    def get_video_metadata(self, cap: cv2.VideoCapture, video_path: str) -> Dict[str, Any]:
        """
        Extract metadata from video capture object.
        
        Args:
            cap: OpenCV VideoCapture object
            video_path: Path to the video file
            
        Returns:
            Dictionary containing video metadata
        """
        return {
            "video_path": str(video_path),
            "frame_width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "frame_height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": int(cap.get(cv2.CAP_PROP_FPS)),
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
    
    def update_segments(
        self,
        frame_idx: int,
        segments: List[Dict[str, int]],
        segment_started: bool
    ) -> Tuple[List[Dict[str, int]], bool]:
        """
        Update segment information when cursor is detected.
        
        Args:
            frame_idx: Current frame index
            segments: List of existing segments
            segment_started: Flag indicating if a segment is in progress
            
        Returns:
            Updated segments list and segment_started flag
        """
        if segment_started:
            segments[-1]["end_frame"] = frame_idx
        else:
            segments.append({"start_frame": frame_idx, "end_frame": frame_idx})
            segment_started = True
        
        return segments, segment_started
    
    def check_frame_difference(
        self,
        frame: np.ndarray,
        last_valid_frame: np.ndarray,
        frame_idx: int,
        segments: List[Dict[str, int]],
    ) -> bool:
        """
        Check pixel differences between current and last valid frame.
        
        Args:
            frame: Current frame
            last_valid_frame: Last frame with cursor detected
            frame_idx: Current frame index
            segments: List of segments
            
        Returns:
            True if frames are similar, False otherwise
        """
        diff = cv2.absdiff(frame, last_valid_frame)
        diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        if np.mean(diff) < self.config.pixel_diff_threshold:
            segments[-1]["end_frame"] = frame_idx
            return True
        return False
    
    def filter_short_segments(
        self,
        segments: List[Dict[str, int]],
        fps: int
    ) -> List[Dict[str, int]]:
        """
        Remove segments shorter than minimum duration.
        
        Args:
            segments: List of segments to filter
            fps: Frames per second of the video
            
        Returns:
            Filtered list of segments
        """
        return [
            seg
            for seg in segments
            if (seg["end_frame"] - seg["start_frame"]) / fps >= self.config.min_segment_duration
        ]
    
    def detect_cursor_in_video(
        self,
        video_path: str,
        gpu_id: int = 0
    ) -> Dict[str, Any]:
        """
        Detect cursor presence in a video and return analysis results.
        
        Args:
            video_path: Path to the video file
            gpu_id: GPU device ID to use
            
        Returns:
            Dictionary containing detection results and analysis
        """
        video_path = Path(video_path)
        transcript_path = Path(video_path.parent, video_path.stem + "_transcript.json")
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Load model if not already loaded
        if self.model is None:
            self._load_model(gpu_id)
        
        # Initialize video capture
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        
        metadata = self.get_video_metadata(cap, video_path)
        logger.info(f"Processing {video_path.name}: {metadata['total_frames']} frames @ {metadata['fps']} fps")
        
        # Initialize tracking variables
        segments = []
        frame_idx = 0
        frames_processed = 0
        frames_with_cursor = 0
        no_cursor_count = 0
        last_valid_frame = None
        segment_started = False
        detection_results = []
        
        # Process video
        while cap.isOpened():
            success, frame = cap.read()
            
            if not success:
                break
            
            # Process every Nth frame based on stride
            if frame_idx % self.config.detection_stride == 0:
                frames_processed += 1
                
                # Check frame difference if segment active and we have a reference
                if (segment_started and last_valid_frame is not None and
                    self.check_frame_difference(frame, last_valid_frame, frame_idx, segments)):
                    # Frame is similar to last valid frame
                    segments, segment_started = self.update_segments(frame_idx, segments, segment_started)
                else:
                    # Run YOLO detection
                    res_json = self.process_frame(frame)
                    detection_results.append({"frame_idx": frame_idx, **res_json})
                    
                    if res_json["box"] is not None:
                        # Cursor detected
                        frames_with_cursor += 1
                        segments, segment_started = self.update_segments(frame_idx, segments, segment_started)
                        no_cursor_count = 0
                        last_valid_frame = frame.copy()
                    else:
                        # No cursor detected
                        no_cursor_count += 1
                        if no_cursor_count * self.config.detection_stride / metadata["fps"] > self.config.cursor_timeout:
                            no_cursor_count = 0
                            segment_started = False
                            last_valid_frame = None
            
            frame_idx += 1
        
        cap.release()
        
        # Filter short segments
        segments = self.filter_short_segments(segments, metadata["fps"])
        
        # Calculate statistics
        total_cursor_frames = sum(seg["end_frame"] - seg["start_frame"] for seg in segments)
        cursor_percentage = (total_cursor_frames / metadata["total_frames"] * 100) if metadata["total_frames"] > 0 else 0
        
        # Make decision
        keep = cursor_percentage >= (self.config.cursor_threshold * 100)
        
        result = {
            "video_id": video_path.stem,
            "video_path": str(video_path),
            "transcript_path": str(transcript_path),
            "metadata": metadata,
            "analysis": {
                "total_frames": metadata["total_frames"],
                "frames_processed": frames_processed,
                "frames_with_cursor": frames_with_cursor,
                "cursor_frames_in_segments": total_cursor_frames,
                "cursor_percentage": round(cursor_percentage, 2),
                "active_segments": len(segments),
                "segments": segments
            },
            "decision": {
                "keep": keep,
                "reason": f"Cursor detected in {cursor_percentage:.1f}% of frames (threshold: {self.config.cursor_threshold * 100}%)",
                "recommended_for_trajectory": keep
            },
            "detection_results": detection_results if self.config.save_detection_details else []
        }
        
        logger.info(f"✓ {video_path.name}: {cursor_percentage:.1f}% cursor presence - {'KEEP' if keep else 'REJECT'}")
        
        return result

