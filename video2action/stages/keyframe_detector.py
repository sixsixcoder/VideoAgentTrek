"""
Keyframe detection stage - Detect action timestamps in videos
"""

import os
import json
import glob
import cv2
import numpy as np
import re
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import multiprocessing as mp
from multiprocessing import Manager
import queue
import time
import sys

logger = logging.getLogger(__name__)

# Optional imports for GPU inference
try:
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    import torch
    from qwen_vl_utils import process_vision_info
    QWEN_AVAILABLE = True
except ImportError as e:
    QWEN_AVAILABLE = False
    logger.warning(f"Qwen2.5-VL not available - GPU inference will be disabled: {e}")


class KeyframeDetector:
    """
    Detect keyframes (action timestamps) using two-stage approach:
    1. CPU-based frame comparison (fast filtering)
    2. GPU model inference (accurate detection)
    
    Adapted from src/keyframe_detect_s1.py + src/keyframe_detect_s2.py
    """
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.processor = None
    
    def detect_keyframes(self, clips_dir: str, output_dir: str, video_id: str):
        """
        Detect keyframes in video clips using two-stage approach.
        
        Args:
            clips_dir: Directory containing video clips
            output_dir: Directory to save keyframe detection results
            video_id: Video identifier
        """
        clips_dir = Path(clips_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all video clips
        video_clips = sorted(clips_dir.glob("*.mp4"))
        
        if not video_clips:
            logger.warning(f"No video clips found in {clips_dir}")
            return
        
        logger.info(f"Detecting keyframes in {len(video_clips)} clips")
        
        # Stage 1: CPU-based filtering (fast)
        logger.info("Stage 1: CPU filtering for static videos...")
        static_videos, keyframe_results = self._stage1_cpu_filter(video_clips, output_dir)
        
        # Stage 2: GPU model inference (only non-static videos)
        videos_with_changes = [v for v in video_clips if v not in static_videos]
        if videos_with_changes:
            logger.info(f"Stage 2: GPU inference on {len(videos_with_changes)} videos with changes...")
            self._stage2_gpu_inference(videos_with_changes, output_dir, keyframe_results)
        else:
            logger.info("All videos are static - skipping GPU inference")
        
        logger.info(f"✓ Keyframe detection complete")
    
    def _stage1_cpu_filter(
        self, video_clips: List[Path], output_dir: Path
    ) -> Tuple[set, Dict[Path, Dict[str, Any]]]:
        """Stage 1: CPU-based frame comparison to filter static videos"""
        
        static_videos = set()
        keyframe_results = {}
        
        for video_clip in video_clips:
            try:
                # Detect keyframes using frame comparison
                keyframes, has_changes, video_info = self._detect_keyframes_cpu(
                    str(video_clip),
                    fps=self.config.keyframe_fps,
                    similarity_threshold=self.config.similarity_threshold
                )
                
                # Save keyframe analysis
                keyframe_file = output_dir / f"{video_clip.stem}_keyframe_analysis.json"
                result = {
                    "video_path": str(video_clip),
                    "video_basename": video_clip.name,
                    "keyframes": keyframes,
                    "has_changes_detected": has_changes,
                    "total_keyframes": len(keyframes),
                    "changes_detected": sum(1 for kf in keyframes if kf.get('change_detected')),
                    "video_info": video_info
                }
                
                with open(keyframe_file, 'w') as f:
                    json.dump(result, f, indent=2)
                
                keyframe_results[video_clip] = result
                
                if not has_changes:
                    static_videos.add(video_clip)
                    # Create empty output for static videos
                    output_file = output_dir / f"{video_clip.stem}_output.json"
                    with open(output_file, 'w') as f:
                        json.dump([], f)
                
            except Exception as e:
                logger.debug(f"Failed to process {video_clip.name}: {e}")
                continue
        
        logger.info(f"Found {len(static_videos)} static videos, {len(video_clips) - len(static_videos)} with changes")
        return static_videos, keyframe_results
    
    def _load_model(self, gpu_id: int = 0):
        """Load Qwen2.5-VL model for GPU inference"""
        if not QWEN_AVAILABLE:
            logger.warning("Qwen2.5-VL not available - skipping model loading")
            return None, None
        
        if not Path(self.config.model_path).exists():
            logger.warning(f"Model not found at {self.config.model_path}")
            return None, None
        
        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            
            logger.info(f"Loading model from {self.config.model_path}")
            processor = AutoProcessor.from_pretrained(self.config.model_path)
            processor.tokenizer.padding_side = 'left'
            
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.config.model_path,
                torch_dtype="auto",
                device_map="cuda"
            )
            
            logger.info("Model loaded successfully")
            return model, processor
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None, None
    
    def _stage2_gpu_inference(
        self, video_clips: List[Path], output_dir: Path, keyframe_results: Dict[Path, Dict[str, Any]]
    ):
        """Stage 2: GPU model inference for videos with changes"""
        
        if not QWEN_AVAILABLE:
            raise RuntimeError(
                "Qwen2.5-VL is not available. Please install required packages:\n"
                "  pip install transformers torch qwen-vl-utils\n"
                "  pip install 'numpy<2.0'"
            )
        
        # Load model
        model, processor = self._load_model(gpu_id=0)
        
        if model is None or processor is None:
            raise RuntimeError(
                f"Failed to load Qwen2.5-VL model from {self.config.model_path}. "
                "Please check the model path and ensure the model is properly downloaded."
            )
        
        logger.info(f"Processing {len(video_clips)} videos with Qwen2.5-VL model")
        
        for i, video_clip in enumerate(video_clips, 1):
            try:
                output_file = output_dir / f"{video_clip.stem}_output.json"
                
                # Skip if already exists
                if output_file.exists() and self.config.skip_existing:
                    logger.debug(f"Skipping {video_clip.name} - already exists")
                    continue
                
                # Get video info from keyframe analysis
                analysis = keyframe_results.get(video_clip, {})
                video_info = analysis.get("video_info", {})
                
                if video_info.get("error"):
                    # Keyframe detection failed, skip
                    logger.debug(f"Skipping {video_clip.name} - keyframe detection failed")
                    with open(output_file, 'w') as f:
                        json.dump([], f)
                    continue
                
                resized_height = video_info.get("resized_height", 672)
                resized_width = video_info.get("resized_width", 896)
                
                # Prepare messages (matching reference keyframe_detect_s2.py)
                messages = [
                    {
                        "role": "system",
                        "content": [
                            {"text": "You are a helpful assistant. you can identify all user interface interactions in this video, providing their action types and timestamps and output in JSON format."}
                        ]
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": f"file://{video_clip.absolute()}", "fps": 2.0, "resized_height": resized_height, "resized_width": resized_width},
                            {"type": "text", "text": "Identify all user interface interactions in this video, providing their action types and timestamps. Please output in JSON format."}
                        ]
                    }
                ]
                
                # Process with model (matching reference)
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                images, videos, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
                inputs = processor(text=text, images=images, videos=videos, padding=True, return_tensors="pt", **video_kwargs)
                
                # Move to GPU
                inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
                
                # Generate
                generated_ids = model.generate(**inputs, max_new_tokens=256)
                
                # Decode
                response = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                
                # Extract JSON from response (matching reference)
                json_match = re.search(r'\[\s*{.*}\s*\]', response, re.DOTALL)
                
                if json_match:
                    json_output = json_match.group(0)
                    with open(output_file, 'w') as f:
                        f.write(json_output)
                    logger.info(f"[{i}/{len(video_clips)}] {video_clip.name} - PROCESSED")
                else:
                    # No JSON found
                    with open(output_file, 'w') as f:
                        json.dump({"error": "No JSON output found in the response"}, f)
                    logger.warning(f"[{i}/{len(video_clips)}] {video_clip.name} - NO JSON")
                
                # Cleanup
                del inputs, generated_ids
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"Failed to process {video_clip.name}: {e}")
                try:
                    with open(output_file, 'w') as f:
                        json.dump({"error": str(e)}, f)
                except:
                    pass
                continue
        
        # Cleanup model
        del model, processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info(f"GPU inference complete")
    
    def _detect_keyframes_cpu(
        self, video_path: str, fps: float = 1.0, similarity_threshold: float = 0.9999
    ) -> Tuple[List[Dict[str, Any]], bool, Dict[str, Any]]:
        """Detect keyframes using CPU-based frame comparison"""
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return [], True, {"error": "video_open_failed", "fallback": True}
        
        try:
            # Get video properties
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / video_fps if video_fps > 0 else 10.0
            original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Calculate resized dimensions
            resized_height, resized_width = self._calculate_optimal_resize(
                original_height, original_width
            )
            
            video_info = {
                "duration": duration,
                "fps": video_fps,
                "total_frames": total_frames,
                "original_width": original_width,
                "original_height": original_height,
                "resized_width": resized_width,
                "resized_height": resized_height
            }
            
            # Calculate frame interval
            frame_interval = int(video_fps / fps) if video_fps > 0 else 1
            
            keyframes = []
            prev_frame = None
            any_changes_detected = False
            
            # Extract frames at specified FPS
            max_frames_to_check = min(10, int(duration * fps))
            
            for i in range(0, min(total_frames, max_frames_to_check * frame_interval), frame_interval):
                try:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ret, frame = cap.read()
                    
                    if not ret:
                        break
                    
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    timestamp = i / video_fps
                    
                    if prev_frame is not None:
                        # Calculate MSE
                        if gray_frame.shape != prev_frame.shape:
                            gray_frame = cv2.resize(gray_frame, (prev_frame.shape[1], prev_frame.shape[0]))
                        
                        mse = np.mean((gray_frame.astype(float) - prev_frame.astype(float)) ** 2)
                        max_possible_mse = 255 ** 2
                        similarity = 1 - (mse / max_possible_mse)
                        
                        if similarity >= similarity_threshold:
                            keyframes.append({
                                'timestamp': timestamp,
                                'similarity': similarity,
                                'change_detected': False
                            })
                        else:
                            keyframes.append({
                                'timestamp': timestamp,
                                'similarity': similarity,
                                'change_detected': True
                            })
                            any_changes_detected = True
                    else:
                        keyframes.append({
                            'timestamp': timestamp,
                            'similarity': 1.0,
                            'change_detected': False
                        })
                    
                    prev_frame = gray_frame.copy()
                    
                except Exception:
                    continue
            
            cap.release()
            return keyframes, any_changes_detected, video_info
            
        except Exception as e:
            cap.release()
            return [], True, {"error": str(e), "fallback": True}
    
    def _calculate_optimal_resize(self, original_height: int, original_width: int) -> Tuple[int, int]:
        """Calculate optimal resized dimensions"""
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from utils.qwen_vl_utils import smart_resize
            
            patch_size = 14
            merge_base = 2
            min_tokens = 1
            max_tokens = 1280
            
            pixels_per_token = patch_size * patch_size * merge_base * merge_base
            resized_height, resized_width = smart_resize(
                original_height,
                original_width,
                factor=merge_base * patch_size,
                min_pixels=pixels_per_token * min_tokens,
                max_pixels=pixels_per_token * max_tokens,
                max_long_side=50000,
            )
            
            return resized_height, resized_width
            
        except Exception as e:
            logger.debug(f"Failed to calculate resize: {e}")
            return 672, 896  # Default
