"""
Action validation stage - Validate actions using GPT
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Validation prompt template (matches reference action_validation.py)
VALIDATION_USER_GUIDE = (
    "Task: Detect whether a GUI action truly occurred in the provided time window. "
    "You are given: declared action_type, start_time, end_time, transcripts (before/during/after), and two images: start_frame and end_frame. "
    "Decide if any GUI action occurred that caused the scene to change (valid=true), or if changes are due to ordinary video scene cuts/edits (valid=false). "
    "Additionally, infer the most likely action type among the allowed set if any GUI action is evident.\n\n"
    "Common failure modes to detect: \n"
    "1) Scene changed due to a video edit/cut, not due to a GUI action → valid=false.\n"
    "2) Minor subtype differences within the same family (e.g., single vs double click; left vs right click) are acceptable and should NOT cause invalidation.\n"
    "3) Gross mismatches across action families (e.g., keyboard 'key' vs 'type', or keyboard vs mouse/scroll) should cause valid=false.\n\n"
    "Allowed action types (detected_action):\n"
    "- key: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.\n"
    "- type: Type a string of text on the keyboard.\n"
    "- mouse_move: Move the cursor to a specified (x, y) pixel coordinate on the screen.\n"
    "- left_click: Click the left mouse button.\n"
    "- left_click_drag: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.\n"
    "- right_click: Click the right mouse button.\n"
    "- middle_click: Click the middle mouse button.\n"
    "- double_click: Double-click the left mouse button.\n"
    "- scroll: Performs a scroll of the mouse scroll wheel.\n\n"
    "Evidence guidelines (prioritize visual evidence between start and end frames; transcripts are weak hints):\n"
    "- key/type: Look for caret/focus changes, insertion point movement, text changes in inputs/editors, focused element highlight changes.\n"
    "- left_click/right_click/middle_click/double_click: Look for menus opening/closing, button state toggles, selection changes, focus rings, dialog pop-ups.\n"
    "- left_click_drag: Look for dragged object repositioning, selection marquee, scrollbar thumb continuous movement, slider/thumb moving with a path between frames.\n"
    "- scroll: Look for viewport/content shift (list position, page position, scrollbar thumb movement).\n"
    "- mouse_move: Only cursor relocation. If cursor is not visible and no UI state change is evident, prefer detected_action=null (no conclusive GUI action).\n\n"
    "Decision:\n"
    "- valid=true if a GUI-caused change is visible between frames consistent with any allowed action.\n"
    "- valid=false ONLY if (a) the change is attributable to a scene edit/cut or there is no GUI-driven change, OR (b) the declared action belongs to a different action family than the visual evidence (e.g., declared 'key' but only a mouse click/drag/scroll is evident).\n"
    "- Do NOT mark invalid for click subtype differences (single vs double; left vs right); treat these as the same click family.\n"
    "- detected_action: If you infer a click, you may output 'left_click' as the generic click type when unsure.\n"
    "- reason should briefly explain the key visual cues (or lack thereof) that justify the decision and the chosen detected_action.\n\n"
    "Content validation: If the action_identification provides concrete content (e.g., typed text, keys, coordinates, drag path), also judge whether that content is consistent with the observed change.\n"
    "- For type: Compare the typed text to any visible text change.\n"
    "- For key: Compare specific keys (e.g., Enter, Ctrl+C) to visible UI changes.\n"
    "- For mouse actions: If coordinates are provided, use them as hints for where to look and check if the click/drag location aligns with the UI element that changed.\n"
    "- Click family consistency: Treat single vs double click and left vs right click as consistent for content validation. Do NOT set content_valid=false merely due to these subtype differences.\n"
    "Visualization note: When coordinates are identified, a marker may be overlaid on the start_frame image only in debug outputs. This overlay is for visualization only; do not let it bias your decision. Base validity on the visual change between start and end frames and the consistency with any provided content.\n\n"
    "Output format (STRICT) — return EXACTLY these five lines, and nothing else (no bullets, no markdown, no code fences):\n"
    "valid: <true|false>\n"
    "detected_action: <key|type|mouse_move|left_click|left_click_drag|right_click|middle_click|double_click|scroll|null>\n"
    "content_valid: <true|false|null>\n"
    "content_details: <brief description of content match/mismatch or null>\n"
    "reason: <brief one-sentence justification>"
)


class ActionValidator:
    """
    Validate actions using GPT to ensure they are real GUI interactions.
    
    Adapted from src/post_process/action_validation.py
    """
    
    def __init__(self, config):
        self.config = config
        self.client = None
    
    def validate_actions(self, trajectory_dir: str, video_id: str, output_dir: str = None):
        """
        Validate all actions in trajectory.
        
        Args:
            trajectory_dir: Directory with trajectory files
            video_id: Video identifier
            output_dir: Directory to save validation results (if None, write back to source)
        """
        if not self.config.enable_validation:
            logger.info("Validation disabled - skipping")
            return
        
        trajectory_dir = Path(trajectory_dir)
        
        # Find all trajectory JSON files
        json_files = list(trajectory_dir.glob("*.json"))
        
        if not json_files:
            logger.warning(f"No trajectory files found in {trajectory_dir}")
            return
        
        logger.info(f"Validating {len(json_files)} action files")
        
        # Create output directory if specified
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving validation results to: {output_dir}")
        
        # Initialize OpenAI client
        self._ensure_client()
        
        if self.client is None:
            logger.warning("OpenAI client not available - skipping validation")
            return
        
        # Process files sequentially
        validated = 0
        failed = 0
        
        for i, json_file in enumerate(json_files, 1):
            try:
                logger.info(f"[{i}/{len(json_files)}] Validating {json_file.name}")
                
                if output_dir:
                    # Save to separate output directory
                    output_file = output_dir / json_file.name
                    success = self._validate_file_to_output(
                        str(json_file),
                        str(output_file),
                        self.config.validation_model,
                        self.config.openai_api_key,
                        self.config.openai_base_url
                    )
                else:
                    # Write back to source file
                    success = self._validate_file(
                        str(json_file),
                        self.config.validation_model,
                        self.config.openai_api_key,
                        self.config.openai_base_url
                    )
                
                if success:
                    validated += 1
                else:
                    failed += 1
            except Exception as e:
                logger.error(f"Validation failed for {json_file}: {e}")
                failed += 1
        
        logger.info(f"✓ Validated {validated} files, {failed} failed")
    
    def _ensure_client(self):
        """Initialize OpenAI client (supports DashScope via base_url)"""
        if self.client is not None:
            return
        
        try:
            from openai import OpenAI
            
            if self.config.openai_api_key:
                # Support custom base_url for DashScope or other OpenAI-compatible APIs
                client_kwargs = {"api_key": self.config.openai_api_key}
                if self.config.openai_base_url:
                    client_kwargs["base_url"] = self.config.openai_base_url
                self.client = OpenAI(**client_kwargs)
            else:
                logger.warning("No OpenAI API key configured")
                self.client = None
        except ImportError:
            logger.warning("openai package not installed")
            self.client = None
    
    @staticmethod
    def _validate_file(file_path: str, model: str, api_key: str, base_url: Optional[str] = None) -> bool:
        """Validate a single trajectory file and write back to source (supports DashScope via base_url)"""
        try:
            from openai import OpenAI
            
            with open(file_path) as f:
                data = json.load(f)
            
            # Check if already validated
            if data.get("action_validation"):
                return True
            
            actions = data.get("actions")
            if not isinstance(actions, list) or not actions:
                return False
            
            video_id = data.get("video_id")
            clip_number = data.get("clip_number")
            
            # Initialize client (support DashScope or other OpenAI-compatible APIs)
            client_kwargs = {"api_key": api_key} if api_key else {}
            if base_url:
                client_kwargs["base_url"] = base_url
            client = OpenAI(**client_kwargs) if client_kwargs else OpenAI()
            
            # Validate each action
            evaluations = []
            
            for idx, action in enumerate(actions, start=1):
                try:
                    evaluation = ActionValidator._validate_action(
                        client,
                        action,
                        video_id,
                        clip_number,
                        idx,
                        model
                    )
                    evaluations.append(evaluation)
                except Exception as e:
                    logger.debug(f"Failed to validate action {idx}: {e}")
                    continue
            
            if not evaluations:
                return False
            
            # Write back to file
            data["action_validation"] = {
                "file": file_path,
                "video_id": video_id,
                "clip_number": clip_number,
                "evaluations": evaluations
            }
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to validate {file_path}: {e}")
            return False
    
    @staticmethod
    def _validate_file_to_output(input_path: str, output_path: str, model: str, api_key: str, base_url: Optional[str] = None) -> bool:
        """Validate a trajectory file and save to separate output file (supports DashScope via base_url)"""
        try:
            from openai import OpenAI
            
            with open(input_path) as f:
                data = json.load(f)
            
            actions = data.get("actions")
            if not isinstance(actions, list) or not actions:
                return False
            
            video_id = data.get("video_id")
            clip_number = data.get("clip_number")
            
            # Initialize client (support DashScope or other OpenAI-compatible APIs)
            client_kwargs = {"api_key": api_key} if api_key else {}
            if base_url:
                client_kwargs["base_url"] = base_url
            client = OpenAI(**client_kwargs) if client_kwargs else OpenAI()
            
            # Validate each action
            evaluations = []
            
            for idx, action in enumerate(actions, start=1):
                try:
                    evaluation = ActionValidator._validate_action(
                        client,
                        action,
                        video_id,
                        clip_number,
                        idx,
                        model
                    )
                    evaluations.append(evaluation)
                except Exception as e:
                    logger.debug(f"Failed to validate action {idx}: {e}")
                    continue
            
            if not evaluations:
                return False
            
            # Add validation results
            data["action_validation"] = {
                "file": output_path,
                "video_id": video_id,
                "clip_number": clip_number,
                "evaluations": evaluations
            }
            
            # Save to output file
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to validate {input_path}: {e}")
            return False
    
    @staticmethod
    def _extract_coords_and_dims(action: Dict[str, Any]) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
        """Extract coordinates and dimensions from action identification"""
        ai_block = action.get("action_identification") or {}
        x: Optional[int] = None
        y: Optional[int] = None
        
        # Prefer parsed_args
        parsed_args = ai_block.get("parsed_args")
        if isinstance(parsed_args, list):
            for arg in parsed_args:
                if isinstance(arg, dict) and isinstance(arg.get("coordinate"), list) and len(arg.get("coordinate")) == 2:
                    try:
                        cx, cy = arg.get("coordinate")
                        x = int(cx)
                        y = int(cy)
                        break
                    except Exception:
                        pass
        
        # Fallback to parsed_steps
        if x is None or y is None:
            steps = ai_block.get("parsed_steps")
            if isinstance(steps, list):
                for s in steps:
                    if isinstance(s, dict) and isinstance(s.get("coordinate"), list) and len(s.get("coordinate")) == 2:
                        try:
                            cx, cy = s.get("coordinate")
                            x = int(cx)
                            y = int(cy)
                            break
                        except Exception:
                            pass
        
        # Dimensions
        resized_w = ai_block.get("resized_width")
        resized_h = ai_block.get("resized_height")
        vi = ai_block.get("video_info") or {}
        if resized_w is None:
            resized_w = vi.get("resized_width")
        if resized_h is None:
            resized_h = vi.get("resized_height")
        
        try:
            rw = int(resized_w) if resized_w is not None else None
        except Exception:
            rw = None
        
        try:
            rh = int(resized_h) if resized_h is not None else None
        except Exception:
            rh = None
        
        return x, y, rw, rh
    
    @staticmethod
    def _validate_action(
        client,
        action: Dict[str, Any],
        video_id: str,
        clip_number: str,
        action_index: int,
        model: str
    ) -> Dict[str, Any]:
        """Validate a single action using GPT (matches reference action_validation.py)"""
        action_type = action.get("action_type")
        start_time = action.get("start_time")
        end_time = action.get("end_time")
        transcripts = action.get("transcripts", {}) or {}
        keyframes = action.get("keyframes", {}) or {}
        
        start_img = ((keyframes.get("start_frame") or {}).get("base64"))
        end_img = ((keyframes.get("end_frame") or {}).get("base64"))
        
        # Build prompt (matching reference's build_user_content)
        text_parts = [
            "Judge if the declared action occurred and is valid.",
            f"video_id: {video_id}",
            f"clip_number: {clip_number}",
            f"declared_action_type: {action_type}",
            f"start_time: {start_time}",
            f"end_time: {end_time}",
            "Transcripts:",
            f"before: {transcripts.get('before')}",
            f"during: {transcripts.get('during')}",
            f"after: {transcripts.get('after')}",
        ]
        
        # Include action_identification content (parsed_args and parsed_steps) for content validation
        ai_block = action.get("action_identification") or {}
        if isinstance(ai_block, dict):
            parsed_args = ai_block.get("parsed_args")
            if isinstance(parsed_args, list) and parsed_args:
                text_parts.append("identified_parsed_args:")
                for a in parsed_args:
                    try:
                        text_parts.append(f"- {a}")
                    except Exception:
                        continue
            steps = ai_block.get("parsed_steps")
            if isinstance(steps, list) and steps:
                text_parts.append("identified_steps:")
                for s in steps:
                    try:
                        idx = s.get("index")
                        act = s.get("action")
                        extra = {k: v for k, v in s.items() if k not in ("index", "action")}
                        text_parts.append(f"- step {idx}: {act} {extra}")
                    except Exception:
                        continue
        
        content = [
            {"type": "text", "text": VALIDATION_USER_GUIDE},
            {"type": "text", "text": "\n".join(text_parts)},
        ]
        
        if start_img:
            content.append({"type": "image_url", "image_url": {"url": start_img, "detail": "high"}})
        if end_img:
            content.append({"type": "image_url", "image_url": {"url": end_img, "detail": "high"}})
        
        # Call OpenAI API with retry
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": content}],
                )
                
                text = response.choices[0].message.content.strip()
                
                # Parse response
                valid, detected_action, content_valid, content_details, reason = ActionValidator._parse_strict_output(text)
                
                evaluation = {
                    "action_index": action_index,
                    "declared_action_type": action_type,
                    "model": model,
                    "valid": bool(valid) if isinstance(valid, bool) else False,
                    "detected_action": detected_action,
                    "content_valid": content_valid,
                    "content_details": content_details,
                    "reason": reason,
                }
                
                # Add coord_relative for coordinate-based actions
                x, y, rw, rh = ActionValidator._extract_coords_and_dims(action)
                if x is not None and y is not None and rw and rh and rw > 0 and rh > 0:
                    try:
                        rel_x = float(x) / float(rw)
                        rel_y = float(y) / float(rh)
                        evaluation["coord_relative"] = [rel_x, rel_y]
                    except Exception:
                        evaluation["coord_relative"] = None
                else:
                    evaluation["coord_relative"] = None
                
                return evaluation
                
            except Exception as e:
                if attempt < max_attempts - 1:
                    time.sleep(1.5 * (2 ** attempt))  # Exponential backoff
                    continue
                else:
                    raise
        
        raise Exception("Max attempts exceeded")
    
    @staticmethod
    def _parse_strict_output(text: str) -> Tuple[Optional[bool], Optional[str], Optional[bool], Optional[str], Optional[str]]:
        """Parse the strict 5-line output format"""
        valid_match = re.search(r"(?im)^\s*valid\s*:\s*(true|false)\s*$", text)
        action_match = re.search(r"(?im)^\s*detected_action\s*:\s*([a-z_]+|null|none)\s*$", text)
        content_valid_match = re.search(r"(?im)^\s*content_valid\s*:\s*(true|false|null|none)\s*$", text)
        content_details_match = re.search(r"(?im)^\s*content_details\s*:\s*(.+)$", text)
        reason_match = re.search(r"(?im)^\s*reason\s*:\s*(.+)$", text)
        
        valid_value = True if valid_match and valid_match.group(1).lower() == "true" else (False if valid_match else None)
        
        detected_value = None
        if action_match:
            candidate = action_match.group(1).lower()
            detected_value = None if candidate in {"null", "none"} else candidate
        
        content_valid_value = None
        if content_valid_match:
            v = content_valid_match.group(1).lower()
            if v not in {"null", "none"}:
                content_valid_value = True if v == "true" else False
        
        content_details_value = None
        if content_details_match:
            cd = content_details_match.group(1).strip()
            content_details_value = None if cd.lower() in {"null", "none"} else cd
        
        reason_value = reason_match.group(1).strip() if reason_match else None
        
        return (valid_value, detected_value, content_valid_value, content_details_value, reason_value)

