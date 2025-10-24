"""
Action clipping stage - Extract action clips from timestamps
"""

import os
import json
import glob
import cv2
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

logger = logging.getLogger(__name__)


class ActionClipper:
    """
    Extract individual action clips based on keyframe timestamps.
    
    Adapted from src/action_clip.py
    """
    
    def __init__(self, config):
        self.config = config
        self._video_metadata_cache = {}
    
    def extract_action_clips(
        self,
        keyframes_dir: str,
        clips_dir: str,
        output_dir: str,
        video_id: str
    ):
        """
        Extract action clips from keyframe timestamps.
        
        Args:
            keyframes_dir: Directory with keyframe detection results
            clips_dir: Directory with original video clips
            output_dir: Directory to save action clips
            video_id: Video identifier
        """
        keyframes_dir = Path(keyframes_dir)
        clips_dir = Path(clips_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all keyframe output JSON files
        json_files = sorted(keyframes_dir.glob("*_output.json"))
        
        if not json_files:
            logger.warning(f"No keyframe files found in {keyframes_dir}")
            return
        
        # Collect all action tasks
        action_tasks = []
        for json_file in json_files:
            # Extract clip number from filename (e.g., "1_output.json" -> 1)
            basename = json_file.stem.replace("_output", "")
            try:
                clip_number = int(basename)
            except ValueError:
                continue
            
            # Find corresponding video clip
            video_file = clips_dir / f"{clip_number}.mp4"
            if not video_file.exists():
                logger.warning(f"Video clip not found: {video_file}")
                continue
            
            # Load actions from keyframe file
            try:
                with open(json_file) as f:
                    actions = json.load(f)
                
                if isinstance(actions, list) and actions:
                    for action_num, action_data in enumerate(actions, 1):
                        action_tasks.append({
                            'json_file': str(json_file),
                            'video_file': str(video_file),
                            'output_dir': str(output_dir),
                            'video_id': video_id,
                            'clip_number': clip_number,
                            'action_num': action_num,
                            'action_data': action_data
                        })
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to load {json_file}: {e}")
                continue
        
        if not action_tasks:
            logger.info("No actions found to clip")
            return
        
        logger.info(f"Found {len(action_tasks)} actions to clip")
        
        # Process actions in parallel
        max_workers = min(self.config.cpu_workers, len(action_tasks))
        successful = 0
        skipped = 0
        failed = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._process_single_action, task): task
                for task in action_tasks
            }
            
            for future in as_completed(futures):
                try:
                    status = future.result()
                    if status == 'success':
                        successful += 1
                    elif status == 'skipped':
                        skipped += 1
                    else:
                        failed += 1
                except Exception as e:
                    logger.error(f"Action processing failed: {e}")
                    failed += 1
        
        logger.info(f"✓ Created {successful} action clips, skipped {skipped}, failed {failed}")
    
    def _process_single_action(self, task):
        """Process a single action clip"""
        try:
            action_data = task['action_data']
            action_type = action_data.get('action_type', 'unknown')
            start_time = float(action_data.get('start_time', 0))
            end_time = float(action_data.get('end_time', 0))
            
            # Get video duration
            video_duration = self._get_video_duration(task['video_file'])
            
            # Add 0.5s delay to end time but don't exceed video duration
            end_time_delayed = min(end_time + 0.5, video_duration - 0.1)
            
            if start_time >= end_time_delayed:
                return 'skipped'
            
            # Construct output filename
            time_str = f"{start_time:.1f}-{end_time_delayed:.1f}s"
            output_filename = (
                f"{task['video_id']}_clip{task['clip_number']}_"
                f"action{task['action_num']}_{action_type}_{time_str}.mp4"
            )
            
            # Create output directory
            video_output_dir = Path(task['output_dir']) / task['video_id']
            video_output_dir.mkdir(parents=True, exist_ok=True)
            output_video = video_output_dir / output_filename
            
            # Skip if already exists
            if output_video.exists() and self.config.skip_existing:
                return 'skipped'
            
            # Clip the video
            success = self._clip_video(
                task['video_file'],
                str(output_video),
                start_time,
                end_time_delayed
            )
            
            return 'success' if success else 'failed'
            
        except Exception as e:
            logger.error(f"Error processing action: {e}")
            return 'failed'
    
    def _get_video_duration(self, video_file: str) -> float:
        """Get video duration with caching"""
        if video_file in self._video_metadata_cache:
            return self._video_metadata_cache[video_file]
        
        try:
            cap = cv2.VideoCapture(video_file)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = frame_count / fps if fps > 0 else float('inf')
            cap.release()
            
            self._video_metadata_cache[video_file] = duration
            return duration
        except Exception:
            return float('inf')
    
    def _clip_video(self, input_path: str, output_path: str, start_time: float, end_time: float) -> bool:
        """Clip video segment using FFmpeg"""
        duration = end_time - start_time
        
        cmd = [
            "ffmpeg",
            "-threads", "1",
            "-ss", str(start_time),
            "-i", input_path,
            "-t", str(duration),
            "-c:v", "libx264",
            "-c:a", "aac",
            "-preset", "ultrafast",
            "-crf", "23",
            "-g", "30",
            "-force_key_frames", "expr:gte(t,n_forced*1)",
            "-avoid_negative_ts", "make_zero",
            "-y",
            "-loglevel", "quiet",
            "-nostdin",
            output_path
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=300)
            
            # Verify output
            if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
                return True
            return False
            
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            # Try fallback with stream copying
            return self._clip_video_fallback(input_path, output_path, start_time, duration)
    
    def _clip_video_fallback(self, input_path: str, output_path: str, start_time: float, duration: float) -> bool:
        """Fallback clipping with stream copying"""
        cmd = [
            "ffmpeg",
            "-threads", "1",
            "-ss", str(start_time),
            "-i", input_path,
            "-t", str(duration),
            "-c:v", "copy",
            "-c:a", "copy",
            "-avoid_negative_ts", "make_zero",
            "-y",
            "-loglevel", "quiet",
            "-nostdin",
            output_path
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=180)
            
            if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
                return True
            return False
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return False

