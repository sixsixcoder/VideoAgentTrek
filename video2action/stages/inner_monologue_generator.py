"""
Stage 8: Inner Monologue Generator

Generate inner-monologue annotations for trajectory actions using GPT.
Adapted from src/post_process/gen_im.py
"""

import os
import json
import base64
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from io import BytesIO

logger = logging.getLogger(__name__)

try:
    import backoff
except ImportError:
    backoff = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from PIL import Image, ImageDraw
except ImportError:
    Image = None
    ImageDraw = None


SYSTEM_PROMPT = (
    "You are generating inner-monologue annotations for a dataset of GUI agent trajectories built from in-the-wild screen recordings.\n\n"
    "End-to-end setting we are building:\n"
    "1) Source: Real GUI videos (screen recordings) from the wild.\n"
    "2) Extraction: We automatically detect each GUI interaction (an action) from video/audio.\n"
    "3) For every detected action, we provide three kinds of evidence to you:\n"
    "   a. Action details: {action_type} and {action_content}.\n"
    "      - Common action_type values (with definitions):\n"
    "        * key: Performs key down presses on the arguments in order, then releases in reverse order.\n"
    "        * type: Type a string of text on the keyboard.\n"
    "        * mouse_move: Move the cursor to a specified on-screen location.\n"
    "        * left_click: Click the left mouse button.\n"
    "        * left_click_drag: Click and drag the cursor to another on-screen location.\n"
    "        * right_click: Click the right mouse button.\n"
    "        * middle_click: Click the middle mouse button.\n"
    "        * double_click: Double-click the left mouse button.\n"
    "        * scroll: Perform a scroll of the mouse wheel.\n"
    "      - action_content may contain: coordinates (absolute or normalized) and/or a bbox; typed text; pressed keys; scroll amount/direction; drag start/end; and similar specifics.\n"
    "   b. Keyframes: a 'start' screenshot and optionally an 'end' screenshot right after the action executes.\n"
    "   c. Surrounding transcripts: short snippets of narration or speech immediately before, during, and after the action.\n"
    "   d. Action validation (optional): A brief validator-provided action description summarizing what occurred.\n\n"
    "Your job for each action is to output EXACTLY ONE JSON object with two fields: 'action_description' and 'thought'.\n\n"
    "Field definitions (strict):\n"
    "- action_description: A concise natural-language description of WHAT I do in the UI at this step.\n"
    "  - Name the target UI element if inferable (e.g., button/menu/tab/field); otherwise describe by role/label/relative position.\n"
    "  - Mention immediate visible outcome only if it is clearly observable from the provided evidence.\n"
    "  - FORBIDDEN: raw coordinates, code, function/method names, automation tokens, or key-value argument lists.\n"
    "- thought: My FIRST-PERSON inner monologue (4–8 sentences) AS THE DEMONSTRATOR (use 'I', 'me', 'my').\n"
    "  - Provide substantive reasoning, not just a short remark.\n"
    "  - Include: (a) what I aim to accomplish and why now, (b) how the transcripts inform my intent (weave naturally), (c) a brief summary of what likely changes from start to end (if both frames exist), (d) a short breakdown of the atomic actions in this step (e.g., type + press) and why each is needed, and (e) what I expect to verify or do next.\n"
    "  - Prefer present tense when natural (e.g., 'I open...', 'I select...').\n"
    "  - Avoid meta-references to data sources: do NOT say 'the transcript says' or 'in the frame'.\n\n"
    "Additional guidance per action type (if applicable):\n"
    "- Mouse click (left/right/double): Identify the clicked UI element by name/role or appearance; avoid coordinates.\n"
    "- Mouse move: Describe the UI region or element I move toward (purpose), not numeric positions.\n"
    "- Drag (left_click_drag): Describe source and destination in terms of UI elements/regions (not numeric points).\n"
    "- Scroll: State what I am trying to reveal or navigate to.\n"
    "- Type (type): State exactly what I type and why (e.g., search, form fill).\n"
    "- Key (key): State the hotkey combo and purpose (e.g., copy, open search, new tab) without code-like syntax.\n"
    "General rules:\n"
    "- The thought MUST be in first-person (I/me/my). Never switch to third-person.\n"
    "- Evidence priority: Prefer visual evidence between the start/end keyframes; treat transcripts as weak hints to infer WHY I act. If they conflict, prefer visuals.\n"
    "- Use transcripts to inform intent; weave evidence naturally without naming 'transcripts' or 'frames'.\n"
    "- For coordinate-based actions, the first image may include a hollow red circle marking the interaction point to help you localize the target. Do NOT mention the marker explicitly in the output; describe the target element in words instead.\n"
    "- If only a start keyframe is available, focus on WHY I perform the action (intent). If an end keyframe is available, you may include the immediate visible result.\n"
    "- When this step bundles multiple atomic actions (e.g., type + press), reason across them as one coherent operation.\n"
    "- Keep action_description concise; let the thought carry the richer details; avoid hedging and boilerplate.\n"
    "- OUTPUT FORMAT: Exactly one valid JSON object with ONLY 'action_description' and 'thought'. No markdown fences, no extra keys, no commentary."
)


class InnerMonologueGenerator:
    """
    Stage 8: Generate inner monologue annotations for trajectory actions.
    
    Uses OpenAI GPT to generate natural language action descriptions and
    first-person thought processes for each action in the trajectory.
    """
    
    def __init__(self, config):
        """
        Initialize inner monologue generator.
        
        Args:
            config: PipelineConfig instance
        """
        self.config = config
        self.client = None
    
    def _ensure_client(self) -> Any:
        """Initialize OpenAI client (supports DashScope via base_url)."""
        if self.client is not None:
            return self.client
        
        # Use config API key, fallback to environment variables
        api_key = self.config.openai_api_key or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Missing API key for inner monologue generation. "
                "Set openai_api_key in config or API_KEY/OPENAI_API_KEY environment variable."
            )
        
        if OpenAI is None:
            raise RuntimeError(
                "openai package not available. "
                "Install with: pip install openai>=1.0"
            )
        
        # Support custom base_url for DashScope or other OpenAI-compatible APIs
        client_kwargs = {"api_key": api_key}
        if self.config.openai_base_url:
            client_kwargs["base_url"] = self.config.openai_base_url
        
        self.client = OpenAI(**client_kwargs)
        
        return self.client
    
    def _maybe_b64_image_str(self, value: Optional[str]) -> Optional[str]:
        """Convert base64 string to data URL if needed."""
        if not value or not isinstance(value, str):
            return None
        
        if value.startswith("data:image/"):
            return value
        
        try:
            base64.b64decode(value, validate=True)
            return f"data:image/jpeg;base64,{value}"
        except Exception:
            return None
    
    def _decode_base64_image_to_pil(self, b64_str: Optional[str]) -> Optional["Image.Image"]:
        """Decode base64 string to PIL Image."""
        if not b64_str or Image is None:
            return None
        
        try:
            if b64_str.startswith("data:"):
                _header, encoded = b64_str.split(",", 1)
            else:
                encoded = b64_str
            
            img_bytes = base64.b64decode(encoded)
            return Image.open(BytesIO(img_bytes)).convert("RGB")
        except Exception:
            return None
    
    def _encode_pil_to_data_url(self, img: Optional["Image.Image"], fmt: str = "JPEG") -> Optional[str]:
        """Encode PIL Image to data URL."""
        if img is None or Image is None:
            return None
        
        try:
            buff = BytesIO()
            img.save(buff, format=fmt)
            encoded = base64.b64encode(buff.getvalue()).decode()
            mime = "image/jpeg" if fmt.upper() == "JPEG" else f"image/{fmt.lower()}"
            return f"data:{mime};base64,{encoded}"
        except Exception:
            return None
    
    def _draw_hollow_circle(
        self,
        image: "Image.Image",
        rel_x: float,
        rel_y: float,
        color: Tuple[int, int, int] = (255, 0, 0)
    ) -> "Image.Image":
        """Draw a hollow circle on image at relative coordinates."""
        if ImageDraw is None:
            return image
        
        w, h = image.size
        cx = max(0, min(w - 1, int(rel_x * w)))
        cy = max(0, min(h - 1, int(rel_y * h)))
        radius = max(4, int(min(w, h) * 0.02))
        
        draw = ImageDraw.Draw(image)
        bbox = [cx - radius, cy - radius, cx + radius, cy + radius]
        draw.ellipse(bbox, outline=color, width=max(2, radius // 3))
        
        return image
    
    def _overlay_coord_on_start_frame(self, action: Dict[str, Any]) -> Optional[str]:
        """Overlay coordinate marker on start frame for coord-based actions."""
        if Image is None:
            return None
        
        action_type = (action.get("action_type") or "").lower()
        coord_actions = {
            "left_click", "right_click", "middle_click", "double_click",
            "mouse_move", "left_click_drag", "drag", "scroll"
        }
        
        if action_type not in coord_actions:
            return None
        
        # Get start frame
        keyframes = action.get("keyframes", {}) or {}
        start_frame = keyframes.get("start_frame", {})
        start_b64 = start_frame.get("base64") or start_frame.get("b64")
        
        if not isinstance(start_b64, str):
            return None
        
        # Get coordinate
        parsed_actions = action.get("parsed_actions", [])
        coordinate = None
        resized_width = action.get("resized_width")
        resized_height = action.get("resized_height")
        
        for pa in parsed_actions:
            if isinstance(pa, dict) and pa.get("coordinate"):
                coordinate = pa["coordinate"]
                break
        
        if not coordinate or len(coordinate) != 2:
            return None
        
        try:
            x_val, y_val = float(coordinate[0]), float(coordinate[1])
        except Exception:
            return None
        
        # Decode image
        img = self._decode_base64_image_to_pil(start_b64)
        if img is None:
            return None
        
        # Calculate relative coordinates
        rel_x, rel_y = None, None
        
        if resized_width and resized_height:
            try:
                rel_x = x_val / float(resized_width)
                rel_y = y_val / float(resized_height)
            except Exception:
                pass
        
        if rel_x is None or rel_y is None:
            w, h = img.size
            if w > 0 and h > 0:
                rel_x = x_val / float(w)
                rel_y = y_val / float(h)
        
        if rel_x is None or rel_y is None:
            return None
        
        # Draw marker
        overlay = self._draw_hollow_circle(img.copy(), rel_x, rel_y)
        return self._encode_pil_to_data_url(overlay)
    
    def _build_messages(self, action: Dict[str, Any], model: str) -> List[Dict[str, Any]]:
        """Build messages for GPT API call."""
        user_text_parts = []
        user_text_parts.append("Context: We extract GUI interactions from real-world videos.")
        
        # Action metadata
        user_text_parts.append("\nStep metadata:")
        user_text_parts.append(f"- type: {action.get('action_type')}")
        
        # Content (parsed actions)
        parsed_actions = action.get("parsed_actions", [])
        if parsed_actions:
            user_text_parts.append(f"- content: {json.dumps(parsed_actions)}")
        
        # Transcripts
        transcripts = action.get("transcripts", {})
        before = transcripts.get("before")
        during = transcripts.get("during")
        after = transcripts.get("after")
        
        if any([before, during, after]):
            user_text_parts.append("\nTranscripts (surrounding speech):")
            if before:
                user_text_parts.append(f"- before: {before}")
            if during:
                user_text_parts.append(f"- during: {during}")
            if after:
                user_text_parts.append(f"- after: {after}")
        
        # Validation hint
        action_validation = action.get("action_validation", {})
        if isinstance(action_validation, dict):
            hint = action_validation.get("content_details")
            if hint and isinstance(hint, str):
                user_text_parts.append("\nValidator hint (brief action description):")
                user_text_parts.append(f"- {hint.strip()}")
        
        user_text_parts.append(
            "\nTask: Generate the action_description and thought. "
            "Return exactly one JSON object with keys 'action_description' and 'thought'."
        )
        
        content_list = [{"type": "text", "text": "\n".join(user_text_parts)}]
        
        # Add images
        keyframes = action.get("keyframes", {})
        start_frame = keyframes.get("start_frame", {})
        end_frame = keyframes.get("end_frame", {})
        
        start_b64 = start_frame.get("base64") or start_frame.get("b64")
        end_b64 = end_frame.get("base64") or end_frame.get("b64")
        
        start_url = self._maybe_b64_image_str(start_b64 if isinstance(start_b64, str) else None)
        end_url = self._maybe_b64_image_str(end_b64 if isinstance(end_b64, str) else None)
        
        # Try to overlay coordinate marker on start frame
        overlay_start = self._overlay_coord_on_start_frame(action)
        if overlay_start:
            content_list.append({"type": "image_url", "image_url": {"url": overlay_start, "detail": "high"}})
        elif start_url:
            content_list.append({"type": "image_url", "image_url": {"url": start_url, "detail": "high"}})
        
        if end_url:
            content_list.append({"type": "image_url", "image_url": {"url": end_url, "detail": "high"}})
        
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content_list}
        ]
    
    def _call_llm(
        self,
        client: Any,
        messages: List[Dict[str, Any]],
        model: str
    ) -> str:
        """Call LLM with retry logic."""
        if backoff is not None:
            # Use backoff if available
            @backoff.on_exception(
                backoff.expo,
                (Exception,),
                max_time=180,
                max_tries=3,
                jitter=backoff.full_jitter
            )
            def call_with_backoff():
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.2
                )
                return response.choices[0].message.content
            
            return call_with_backoff()
        else:
            # Simple retry without backoff
            for attempt in range(3):
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=0.2
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    if attempt == 2:
                        raise
                    import time
                    time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
    
    def _extract_json_object(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON object from LLM response."""
        try:
            return json.loads(text)
        except Exception:
            pass
        
        # Try extracting from markdown code block
        if "```" in text:
            try:
                inner = text.split("```", 1)[1]
                inner = inner.split("```", 1)[0]
                # Remove potential language identifier
                if inner.startswith("json\n"):
                    inner = inner[5:]
                return json.loads(inner)
            except Exception:
                return None
        
        return None
    
    def generate_for_action(
        self,
        action: Dict[str, Any],
        model: str = "gpt-4o"
    ) -> Dict[str, Any]:
        """
        Generate inner monologue for a single action.
        
        Args:
            action: Action dictionary from trajectory
            model: OpenAI model name
            
        Returns:
            Dictionary with action_type and model_response
        """
        client = self._ensure_client()
        
        messages = self._build_messages(action, model)
        response_text = self._call_llm(client, messages, model)
        
        parsed = self._extract_json_object(response_text) or {"raw": response_text}
        
        return {
            "action_type": action.get("action_type"),
            "model_response": parsed
        }
    
    def generate_for_trajectory(
        self,
        trajectory_path: str,
        model: str = "gpt-4o",
        skip_existing: bool = True
    ) -> Dict[str, Any]:
        """
        Generate inner monologue for all actions in a trajectory file.
        
        Args:
            trajectory_path: Path to trajectory JSON file
            model: OpenAI model name
            skip_existing: Skip if inner_monologue already exists
            
        Returns:
            Dictionary with generation results
        """
        trajectory_path = Path(trajectory_path)
        
        if not trajectory_path.exists():
            raise FileNotFoundError(f"Trajectory file not found: {trajectory_path}")
        
        # Load trajectory
        with open(trajectory_path, 'r') as f:
            trajectory = json.load(f)
        
        # Skip if already has inner monologue
        if skip_existing and trajectory.get("inner_monologue"):
            logger.info(f"Skipping {trajectory_path.name} - already has inner monologue")
            return {
                "trajectory_file": str(trajectory_path),
                "video_id": trajectory.get("video_id"),
                "skipped": True,
                "reason": "inner_monologue already exists"
            }
        
        # Get valid actions
        valid_actions = trajectory.get("valid_actions", [])
        if not valid_actions:
            logger.warning(f"No valid actions in {trajectory_path.name}")
            return {
                "trajectory_file": str(trajectory_path),
                "video_id": trajectory.get("video_id"),
                "skipped": True,
                "reason": "no valid_actions"
            }
        
        logger.info(f"Generating inner monologue for {len(valid_actions)} actions in {trajectory_path.name}")
        
        # Generate for each action
        results = []
        for i, action in enumerate(valid_actions):
            try:
                logger.info(f"  Action {i+1}/{len(valid_actions)}: {action.get('action_type')}")
                result = self.generate_for_action(action, model)
                results.append(result)
                
                # Add inner monologue to action
                if "model_response" in result and isinstance(result["model_response"], dict):
                    inner_monologue = {}
                    if "action_description" in result["model_response"]:
                        inner_monologue["action_description"] = result["model_response"]["action_description"]
                    if "thought" in result["model_response"]:
                        inner_monologue["thought"] = result["model_response"]["thought"]
                    
                    action["inner_monologue"] = inner_monologue
                
            except Exception as e:
                logger.error(f"  Failed to generate for action {i+1}: {e}")
                results.append({
                    "action_type": action.get("action_type"),
                    "error": str(e)
                })
        
        # Save updated trajectory
        with open(trajectory_path, 'w') as f:
            json.dump(trajectory, f, indent=2)
        
        logger.info(f"✓ Saved trajectory with inner monologue: {trajectory_path}")
        
        return {
            "trajectory_file": str(trajectory_path),
            "video_id": trajectory.get("video_id"),
            "actions_processed": len(results),
            "successful": sum(1 for r in results if "error" not in r),
            "failed": sum(1 for r in results if "error" in r),
            "results": results
        }

