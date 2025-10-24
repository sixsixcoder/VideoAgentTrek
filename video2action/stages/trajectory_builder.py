"""
Trajectory building stage - Build raw trajectory with keyframes
"""

import os
import json
import cv2
import base64
import subprocess
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

logger = logging.getLogger(__name__)


class TrajectoryBuilder:
    """
    Build raw trajectory by combining keyframes, action details, and transcripts.
    
    Adapted from src/extract_raw.py
    """
    
    def __init__(self, config):
        self.config = config
    
    def build_trajectory(
        self,
        clips_dir: str,
        keyframes_dir: str,
        action_clips_dir: str,
        output_dir: str,
        video_id: str,
        transcript_file: str = None,
        action_results_dir: str = None  # New parameter for Stage 4 output
    ):
        """
        Build raw trajectory from all components.
        
        Args:
            clips_dir: Directory with video clips
            keyframes_dir: Directory with keyframe results
            action_clips_dir: Directory with action clips (Stage 3 output)
            output_dir: Directory to save trajectory
            video_id: Video identifier
            transcript_file: Optional transcript file
            action_results_dir: Directory with action identification results (Stage 4 output).
                              If None, looks in action_clips_dir (for backward compatibility)
        """
        clips_dir = Path(clips_dir)
        keyframes_dir = Path(keyframes_dir)
        action_clips_dir = Path(action_clips_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use action_results_dir if provided, otherwise fall back to action_clips_dir
        results_dir = Path(action_results_dir) if action_results_dir else action_clips_dir
        is_video_specific = action_results_dir is not None  # If explicit results dir, it's video-specific
        
        # Load transcript if available
        transcript_data = self._load_transcript(transcript_file) if transcript_file else []
        
        # Load action identification results
        action_identification_index = self._load_action_identification(results_dir, video_id, is_video_specific)
        
        # Find all keyframe JSON files
        keyframe_files = sorted(keyframes_dir.glob("*_output.json"))
        
        if not keyframe_files:
            logger.warning(f"No keyframe files found in {keyframes_dir}")
            return
        
        logger.info(f"Building trajectory from {len(keyframe_files)} keyframe files")
        
        # Process each clip
        total_actions = 0
        for keyframe_file in keyframe_files:
            try:
                # Extract clip number
                basename = keyframe_file.stem.replace("_output", "")
                clip_number = int(basename)
                
                # Find corresponding video clip
                video_clip = clips_dir / f"{clip_number}.mp4"
                if not video_clip.exists():
                    continue
                
                # Load keyframe data
                with open(keyframe_file) as f:
                    keyframe_data = json.load(f)
                
                if not isinstance(keyframe_data, list) or not keyframe_data:
                    continue
                
                # Process each action in this clip
                for action_num, action in enumerate(keyframe_data, 1):
                    try:
                        action_data = self._process_single_action(
                            video_clip=str(video_clip),
                            action=action,
                            video_id=video_id,
                            clip_number=clip_number,
                            action_num=action_num,
                            transcript_data=transcript_data,
                            action_identification_index=action_identification_index,
                            action_clips_dir=action_clips_dir
                        )
                        
                        if action_data:
                            # Save action to separate file
                            action_output_file = output_dir / f"{video_id}_{clip_number}_{action_num}.json"
                            with open(action_output_file, 'w') as f:
                                json.dump(action_data, f, indent=2)
                            total_actions += 1
                            
                    except Exception as e:
                        logger.debug(f"Failed to process action {action_num} in clip {clip_number}: {e}")
                        continue
                        
            except Exception as e:
                logger.warning(f"Failed to process {keyframe_file}: {e}")
                continue
        
        logger.info(f"✓ Built trajectory with {total_actions} actions")
    
    def _process_single_action(
        self,
        video_clip: str,
        action: Dict[str, Any],
        video_id: str,
        clip_number: int,
        action_num: int,
        transcript_data: List[Dict[str, Any]],
        action_identification_index: Dict[Tuple[str, int], Dict[str, Any]],
        action_clips_dir: Path
    ) -> Optional[Dict[str, Any]]:
        """Process a single action and build trajectory data"""
        
        action_type = action.get("action_type", "unknown")
        start_time = float(action.get("start_time", 0))
        end_time = float(action.get("end_time", 0))
        
        # Calculate global timestamps (each clip is 10 seconds)
        global_start_time = 10 * (clip_number - 1) + start_time
        global_end_time = 10 * (clip_number - 1) + end_time
        
        # Try to find action clip
        action_clip_path = self._find_action_clip(
            action_clips_dir, video_id, clip_number, action_num
        )
        
        # Extract keyframes
        if action_clip_path and action_clip_path.exists():
            # Use action clip for keyframes
            start_frame, end_frame = self._extract_first_last_frames(str(action_clip_path))
            end_time_delayed = end_time
            used_source = "action_clip"
        else:
            # Use original clip with timestamps
            start_frame = self._extract_frame_at_time(video_clip, start_time)
            
            # Get video duration
            video_duration = self._get_video_duration(video_clip)
            end_time_delayed = min(end_time + 0.5, video_duration - 0.1)
            
            end_frame = self._extract_frame_at_time(video_clip, end_time_delayed)
            used_source = "original_clip"
        
        # Convert frames to base64
        start_frame_b64 = self._frame_to_base64(start_frame)
        end_frame_b64 = self._frame_to_base64(end_frame)
        
        # Get transcript segments
        transcript_segments = self._get_transcript_segments(
            transcript_data, global_start_time, global_end_time
        )
        
        # Get action identification if available
        clip_key = f"{video_id}_clip{clip_number}"
        action_identification = action_identification_index.get((clip_key, action_num))
        
        # Build action data
        action_data = {
            "video_id": video_id,
            "clip_number": str(clip_number),
            "video_path": video_clip,
            "actions": [{
                "action_type": action_type,
                "start_time": start_time,
                "end_time": end_time,
                "end_time_delayed": end_time_delayed,
                "global_start_time": global_start_time,
                "global_end_time": global_end_time,
                "global_end_time_delayed": 10 * (clip_number - 1) + end_time_delayed,
                "duration": end_time - start_time,
                "keyframes": {
                    "source": used_source,
                    "start_frame": {
                        "base64": start_frame_b64,
                        "format": self.config.image_format
                    },
                    "end_frame": {
                        "base64": end_frame_b64,
                        "format": self.config.image_format
                    }
                },
                "transcripts": transcript_segments,
                "action_identification": action_identification
            }]
        }
        
        return action_data
    
    def _find_action_clip(
        self, action_clips_dir: Path, video_id: str, clip_number: int, action_num: int
    ) -> Optional[Path]:
        """Find the action clip file"""
        video_action_dir = action_clips_dir / video_id
        if not video_action_dir.exists():
            return None
        
        # Look for action clip matching pattern
        pattern = f"{video_id}_clip{clip_number}_action{action_num}_*.mp4"
        matches = list(video_action_dir.glob(pattern))
        
        return matches[0] if matches else None
    
    def _load_transcript(self, transcript_file: str) -> List[Dict[str, Any]]:
        """Load transcript data"""
        try:
            with open(transcript_file) as f:
                return json.load(f)
        except Exception as e:
            logger.debug(f"Could not load transcript: {e}")
            return []
    
    def _load_action_identification(
        self, results_dir: Path, video_id: str, is_video_specific: bool = False
    ) -> Dict[Tuple[str, int], Dict[str, Any]]:
        """Load action identification results
        
        Args:
            results_dir: Directory containing results
            video_id: Video identifier
            is_video_specific: If True, results_dir already points to video-specific dir.
                             If False, will append video_id to results_dir.
        """
        index = {}
        
        if is_video_specific:
            results_file = results_dir / "action_identification_results_s2.json"
        else:
            results_file = results_dir / video_id / "action_identification_results_s2.json"
            
        if not results_file.exists():
            return index
        
        try:
            with open(results_file) as f:
                results = json.load(f)
            
            if not isinstance(results, list):
                return index
            
            for entry in results:
                try:
                    basename = entry.get("video_basename", "")
                    # Parse: "videoId_clipXX_actionY_...*.mp4"
                    clip_part, rest = basename.split("_action", 1)
                    action_idx_str = rest.split("_", 1)[0]
                    action_idx = int(action_idx_str)
                    index[(clip_part, action_idx)] = entry
                except Exception:
                    continue
                    
        except Exception as e:
            logger.debug(f"Could not load action identification: {e}")
        
        return index
    
    def _get_transcript_segments(
        self, transcript_data: List[Dict[str, Any]], start_time: float, end_time: float
    ) -> Dict[str, Optional[str]]:
        """Extract transcript segments around action"""
        if not transcript_data:
            return {"before": None, "during": None, "after": None}
        
        # Define time ranges
        before_start = max(0, start_time - 120.0)
        before_end = start_time - 15.0
        during_start = start_time - 15.0
        during_end = end_time + 15.0
        after_start = end_time + 15.0
        after_end = end_time + 120.0
        
        before_texts = []
        during_texts = []
        after_texts = []
        
        for segment in transcript_data:
            seg_start = segment.get('start', 0)
            seg_duration = segment.get('duration', 0)
            seg_end = seg_start + seg_duration
            text = segment.get('text', '').strip()
            
            if not text:
                continue
            
            if before_end >= 0 and seg_start < before_end and seg_end > before_start:
                before_texts.append(text)
            elif seg_start < during_end and seg_end > during_start:
                during_texts.append(text)
            elif seg_start < after_end and seg_end > after_start:
                after_texts.append(text)
        
        return {
            "before": " ".join(before_texts).strip() if before_texts else None,
            "during": " ".join(during_texts).strip() if during_texts else None,
            "after": " ".join(after_texts).strip() if after_texts else None
        }
    
    def _extract_frame_at_time(self, video_path: str, time_seconds: float) -> np.ndarray:
        """Extract frame at specific time"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            # Try FFmpeg fallback
            return self._extract_frame_ffmpeg(video_path, time_seconds)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            cap.release()
            return self._extract_frame_ffmpeg(video_path, time_seconds)
        
        frame_number = int(time_seconds * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return self._extract_frame_ffmpeg(video_path, time_seconds)
        
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    def _extract_first_last_frames(self, video_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Extract first and last frames"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return self._extract_first_last_ffmpeg(video_path)
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            cap.release()
            return self._extract_first_last_ffmpeg(video_path)
        
        # First frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret_first, first = cap.read()
        
        # Last frame
        last_index = max(0, frame_count - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, last_index)
        ret_last, last = cap.read()
        
        cap.release()
        
        if not ret_first or not ret_last:
            return self._extract_first_last_ffmpeg(video_path)
        
        return (cv2.cvtColor(first, cv2.COLOR_BGR2RGB), cv2.cvtColor(last, cv2.COLOR_BGR2RGB))
    
    def _extract_frame_ffmpeg(self, video_path: str, time_seconds: float) -> np.ndarray:
        """Extract frame using FFmpeg"""
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-ss", f"{float(time_seconds):.3f}",
            "-i", video_path,
            "-frames:v", "1",
            "-f", "image2pipe",
            "-vcodec", "mjpeg",
            "pipe:1"
        ]
        
        try:
            proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            data = np.frombuffer(proc.stdout, dtype=np.uint8)
            bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Exception:
            # Return black frame as fallback
            return np.zeros((480, 640, 3), dtype=np.uint8)
    
    def _extract_first_last_ffmpeg(self, video_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Extract first and last frames using FFmpeg"""
        first = self._extract_frame_ffmpeg(video_path, 0.0)
        last = self._extract_frame_ffmpeg(video_path, 0.0)  # Approximate
        return (first, last)
    
    def _frame_to_base64(self, frame: np.ndarray) -> str:
        """Convert frame to base64 string"""
        if self.config.image_format.upper() == 'JPEG':
            success, encoded = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            prefix = 'data:image/jpeg;base64,'
        else:  # PNG
            success, encoded = cv2.imencode('.png', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            prefix = 'data:image/png;base64,'
        
        if not success:
            raise ValueError("Failed to encode frame")
        
        base64_string = base64.b64encode(encoded.tobytes()).decode('utf-8')
        return prefix + base64_string
    
    def _get_video_duration(self, video_path: str) -> float:
        """Get video duration"""
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = frame_count / fps if fps > 0 else float('inf')
            cap.release()
            return duration
        except Exception:
            return float('inf')
