"""
Action identification stage - Identify action parameters
"""

import os
import json
import re
import glob
import cv2
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
from multiprocessing import Manager
import queue
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


class ActionIdentifier:
    """
    Identify detailed action parameters (coordinates, keys, text) using two-stage approach:
    1. Metadata extraction (fast, CPU)
    2. Model inference (accurate, GPU)
    
    Adapted from src/action_identify_s1.py + src/action_identify_s2.py
    """
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.processor = None
    
    def _load_model(self, gpu_id: int = 0):
        """Load Qwen2.5-VL model for action identification"""
        if not QWEN_AVAILABLE:
            return None, None
        
        try:
            logger.info(f"Loading Qwen2.5-VL model from {self.config.action_model_path}")
            
            processor = AutoProcessor.from_pretrained(self.config.action_model_path)
            processor.tokenizer.padding_side = 'left'
            
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.config.action_model_path,
                torch_dtype="auto",
                device_map=f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
            )
            
            logger.info(f"Model loaded successfully on GPU {gpu_id}")
            return model, processor
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None, None
    
    def identify_actions(self, action_clips_dir: str, output_dir: str, video_id: str):
        """
        Identify action parameters from action clips.
        
        Args:
            action_clips_dir: Directory containing action clips
            output_dir: Directory to save identification results
            video_id: Video identifier
        """
        action_clips_dir = Path(action_clips_dir)
        output_dir = Path(output_dir)
        video_action_dir = action_clips_dir / video_id
        video_output_dir = output_dir / video_id
        
        if not video_action_dir.exists():
            logger.warning(f"Action clips directory not found: {video_action_dir}")
            return
        
        # Create output directory
        video_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all action clips
        action_clips = sorted(video_action_dir.glob("*.mp4"))
        
        if not action_clips:
            logger.warning(f"No action clips found in {video_action_dir}")
            return
        
        logger.info(f"Identifying actions in {len(action_clips)} clips")
        
        # Stage 1: Extract metadata (fast, CPU)
        logger.info("Stage 1: Extracting metadata...")
        metadata_files = self._stage1_extract_metadata(action_clips, video_output_dir)
        
        # Stage 2: Model inference (GPU)
        logger.info("Stage 2: Running model inference...")
        self._stage2_model_inference(metadata_files, video_output_dir)
        
        logger.info(f"✓ Action identification complete")
        logger.info(f"✓ Results saved to: {video_output_dir}")
    
    def _stage1_extract_metadata(self, action_clips: List[Path], output_dir: Path) -> List[Path]:
        """Stage 1: Extract video metadata (CPU, parallel)"""
        metadata_files = []
        
        def process_clip(clip_path):
            # Save metadata to output directory, not with the clip
            metadata_file = output_dir / f"{clip_path.stem}.metadata.json"
            
            # Skip if already exists
            if metadata_file.exists() and self.config.skip_existing:
                return metadata_file
            
            try:
                video_info = self._get_video_info(str(clip_path))
                if video_info is None:
                    return None
                
                duration = video_info["duration"]
                original_fps = video_info["fps"]
                
                # Calculate optimal FPS
                optimal_fps, _ = self._get_dynamic_fps(0.0, duration, original_fps)
                
                # Calculate resized dimensions (using ORIGINAL fps, not optimal fps)
                # This matches the reference implementation in action_identify_s1.py
                resized_height, resized_width = self._calculate_optimal_resize(
                    video_info["original_height"],
                    video_info["original_width"],
                    duration,
                    original_fps  # Use original_fps, not optimal_fps!
                )
                
                metadata = {
                    "video_file": str(clip_path.absolute()),
                    "video_id": clip_path.parent.name,
                    "video_basename": clip_path.name,
                    "video_info": video_info,
                    "optimal_fps": optimal_fps,
                    "resized_height": resized_height,
                    "resized_width": resized_width
                }
                
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                return metadata_file
                
            except Exception as e:
                logger.debug(f"Failed to process {clip_path.name}: {e}")
                return None
        
        # Process in parallel with threads
        max_workers = min(self.config.cpu_workers, len(action_clips))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_clip, clip) for clip in action_clips]
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        metadata_files.append(result)
                except Exception as e:
                    logger.debug(f"Metadata extraction failed: {e}")
        
        logger.info(f"Extracted metadata for {len(metadata_files)} clips")
        return metadata_files
    
    def _stage2_model_inference(self, metadata_files: List[Path], output_dir: Path):
        """Stage 2: Run model inference on action clips (GPU)"""
        if not metadata_files:
            logger.warning("No metadata files to process")
            return
        
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
                f"Failed to load Qwen2.5-VL model from {self.config.action_model_path}. "
                "Please check the model path and ensure the model is properly downloaded."
            )
        
        logger.info(f"Processing {len(metadata_files)} action clips with Qwen2.5-VL model")
        
        results = []
        for i, metadata_file in enumerate(metadata_files, 1):
            try:
                with open(metadata_file) as f:
                    metadata = json.load(f)
                
                video_path = metadata["video_file"]
                resized_height = metadata["resized_height"]
                resized_width = metadata["resized_width"]
                optimal_fps = metadata["optimal_fps"]
                
                # Skip if already exists
                output_file = output_dir / f"{Path(video_path).stem}_output.json"
                if output_file.exists() and self.config.skip_existing:
                    logger.debug(f"Skipping {Path(video_path).name} - already exists")
                    continue
                
                # Prepare messages for action identification (matching reference action_identify_s2.py)
                messages = [
                    {
                        "role": "system",
                        "content": [
                            {"text": "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{\"type\": \"function\", \"function\": {\"name_for_human\": \"computer_use\", \"name\": \"computer_use\", \"description\": \"Use a mouse and keyboard to interact with a computer, and take screenshots.\\n* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.\\n* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn't open, try wait and taking another screenshot.\\n* The screen's resolution is 168x224.\\n* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.\\n* If you tried clicking on a program or link but it failed to load, even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.\\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.\", \"parameters\": {\"properties\": {\"action\": {\"description\": \"The action to perform. The available actions are:\\n* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.\\n* `type`: Type a string of text on the keyboard.\\n* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.\\n* `left_click`: Click the left mouse button.\\n* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.\\n* `right_click`: Click the right mouse button.\\n* `middle_click`: Click the middle mouse button.\\n* `double_click`: Double-click the left mouse button.\\n* `scroll`: Performs a scroll of the mouse scroll wheel.\\n* `wait`: Wait specified seconds for the change to happen.\\n* `terminate`: Terminate the current task and report its completion status.\", \"enum\": [\"key\", \"type\", \"mouse_move\", \"left_click\", \"left_click_drag\", \"right_click\", \"middle_click\", \"double_click\", \"scroll\", \"wait\", \"terminate\"], \"type\": \"string\"}, \"keys\": {\"description\": \"Required only by `action=key`.\", \"type\": \"array\"}, \"text\": {\"description\": \"Required only by `action=type`.\", \"type\": \"string\"}, \"coordinate\": {\"description\": \"(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=mouse_move` and `action=left_click_drag`.\", \"type\": \"array\"}, \"pixels\": {\"description\": \"The amount of scrolling to perform. Positive values scroll up, negative values scroll down. Required only by `action=scroll`.\", \"type\": \"number\"}, \"time\": {\"description\": \"The seconds to wait. Required only by `action=wait`.\", \"type\": \"number\"}, \"status\": {\"description\": \"The status of the task. Required only by `action=terminate`.\", \"type\": \"string\", \"enum\": [\"success\", \"failure\"]}}, \"required\": [\"action\"], \"type\": \"object\"}, \"args_format\": \"Format the arguments as a JSON object.\"}}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>"}
                        ]
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": f"file://{video_path}", "fps": optimal_fps, "resized_height": resized_height, "resized_width": resized_width},
                            {"type": "text", "text": "Here is a video clip of a GUI interaction, please identify the interaction"}
                        ]
                    }
                ]
                
                # Process with model
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                images, videos, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
                inputs = processor(text=text, images=images, videos=videos, padding=True, return_tensors="pt", **video_kwargs)
                
                # Move to GPU
                inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
                
                # Generate
                generated_ids = model.generate(**inputs, max_new_tokens=256)
                
                # Decode the full response
                full_response = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                
                # Extract only the model's generated response (after the last "assistant" token)
                # This matches the reference implementation
                if "assistant" in full_response:
                    model_response = full_response.split("assistant")[-1].strip()
                else:
                    model_response = full_response.split(text)[-1].strip() if text in full_response else full_response.strip()
                
                # Parse response for action parameters
                parsed_args = self._parse_action_response(model_response)
                
                result = {
                    "video_path": video_path,
                    "video_id": metadata["video_id"],
                    "video_basename": metadata["video_basename"],
                    "video_info": metadata["video_info"],
                    "optimal_fps": optimal_fps,
                    "resized_width": resized_width,
                    "resized_height": resized_height,
                    "response": model_response,
                    "parsed_args": parsed_args
                }
                
                # Save individual result
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2)
                
                results.append(result)
                logger.info(f"[{i}/{len(metadata_files)}] Processed {Path(video_path).name}")
                
                # Clean up
                del inputs, generated_ids
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"Failed to process {metadata_file}: {e}")
                continue
        
        # Save combined results
        if results:
            output_file = output_dir / "action_identification_results_s2.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Saved {len(results)} results to {output_file}")
        
        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _parse_action_response(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse tool call response and extract structured arguments.
        Matches the reference implementation in action_identify_s2.py
        """
        try:
            # Find all tool calls in the response
            tool_call_pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
            matches = re.findall(tool_call_pattern, response, re.DOTALL)
            
            if not matches:
                return []
                
            parsed_calls = []
            for match in matches:
                try:
                    # Parse the JSON content
                    tool_call_json = json.loads(match)
                    
                    # Extract the arguments
                    if 'arguments' in tool_call_json:
                        parsed_calls.append(tool_call_json['arguments'])
                    else:
                        # If no arguments field, append None
                        parsed_calls.append(None)
                        
                except json.JSONDecodeError:
                    # If JSON parsing fails for this call, append None
                    parsed_calls.append(None)
            
            # Return the list of parsed calls, or empty if all failed
            return parsed_calls if any(call is not None for call in parsed_calls) else []
            
        except Exception as e:
            logger.debug(f"Failed to parse tool call: {e}")
            return []
    
    def _get_video_info(self, video_path: str) -> Optional[Dict[str, Any]]:
        """Extract video information"""
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                return None
            
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / video_fps if video_fps > 0 else 0
            original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            cap.release()
            
            return {
                "duration": duration,
                "fps": video_fps,
                "total_frames": total_frames,
                "original_width": original_width,
                "original_height": original_height
            }
            
        except Exception as e:
            logger.debug(f"Failed to get video info: {e}")
            return None
    
    def _calculate_optimal_resize(
        self, original_height: int, original_width: int, duration: float, fps: float
    ) -> tuple:
        """Calculate optimal resized dimensions"""
        try:
            # Import smart_resize from utils
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from utils.qwen_vl_utils import smart_resize
            
            temporal_patch_size = 2
            num_frames = int(duration * fps)
            
            max_video_seq_len = 20000
            min_pixels = 16 * 28 * 28
            max_pixels = max_video_seq_len / num_frames * temporal_patch_size * 28 * 28
            
            resized_height, resized_width = smart_resize(
                int(original_height),
                int(original_width),
                min_pixels=min_pixels,
                max_pixels=max_pixels
            )
            
            return resized_height, resized_width
            
        except Exception as e:
            logger.debug(f"Failed to calculate resize: {e}")
            # Return default dimensions
            return 672, 896
    
    def _get_dynamic_fps(
        self, start_time: float, end_time: float, original_fps: float, max_frames: int = 20
    ) -> tuple:
        """Determine optimal FPS"""
        duration = end_time - start_time
        fps_options = [24, 12, 8, 6, 4, 3, 2, 1]
        
        for fps in fps_options:
            effective_fps = min(fps, original_fps)
            num_frames = int(duration * effective_fps)
            if num_frames <= max_frames:
                return float(effective_fps), end_time
        
        # Cap duration if needed
        effective_fps = min(1.0, original_fps)
        max_duration = max_frames / effective_fps
        capped_end_time = start_time + max_duration
        return effective_fps, capped_end_time
