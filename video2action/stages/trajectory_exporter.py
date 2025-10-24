"""
Trajectory export stage - Export validated trajectory
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class TrajectoryExporter:
    """
    Export final validated trajectory by filtering only valid actions.
    
    Supports GPT-based content restoration for invalid 'type' and 'key' actions.
    Adapted from src/raw_to_valid.py
    """
    
    def __init__(self, config):
        self.config = config
        self.client = None
        self.use_gpt_restore = getattr(config, 'enable_gpt_restore', False)
        self.gpt_workers = getattr(config, 'gpt_restore_workers', 4)
    
    def export_trajectory(
        self,
        validated_dir: str,
        output_dir: str,
        video_id: str
    ) -> Dict[str, Any]:
        """
        Export final validated trajectory.
        
        Args:
            validated_dir: Directory with validated actions
            output_dir: Directory to save final trajectory
            video_id: Video identifier
            
        Returns:
            Trajectory data
        """
        validated_dir = Path(validated_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all JSON files in validated directory
        json_files = list(validated_dir.rglob("*.json"))
        
        if not json_files:
            logger.warning(f"No trajectory files found in {validated_dir}")
            return {"video_id": video_id, "valid_actions": []}
        
        logger.info(f"Processing {len(json_files)} trajectory files")
        
        # Process each file and collect valid actions
        valid_actions = []
        total_actions = 0
        
        for json_file in json_files:
            try:
                # Count total actions
                with open(json_file) as f:
                    data = json.load(f)
                    actions = data.get("actions")
                    if isinstance(actions, list):
                        total_actions += len(actions)
                
                # Extract valid actions
                doc = self._build_valid_actions(json_file)
                if doc:
                    valid_actions.extend(doc)
            except Exception as e:
                logger.warning(f"Failed to process {json_file}: {e}")
                continue
        
        # Count restored actions
        num_restored = sum(1 for action in valid_actions if action.get("_gpt_restored"))
        
        logger.info(f"✓ Exported {len(valid_actions)} valid actions out of {total_actions} total")
        if num_restored > 0:
            logger.info(f"  → {num_restored} actions had content restored by GPT")
        
        # Create final trajectory
        trajectory = {
            "video_id": video_id,
            "num_total_actions": total_actions,
            "num_valid_actions": len(valid_actions),
            "num_gpt_restored": num_restored,
            "valid_actions": valid_actions
        }
        
        # Save to file
        output_file = output_dir / f"{video_id}_trajectory.json"
        with open(output_file, 'w') as f:
            json.dump(trajectory, f, indent=2)
        
        logger.info(f"Saved final trajectory to: {output_file}")
        
        return trajectory
    
    def _ensure_client(self):
        """Initialize OpenAI client for GPT restoration"""
        if self.client is not None:
            return self.client
        
        if not self.use_gpt_restore:
            return None
        
        try:
            from openai import OpenAI
            
            api_key = self.config.openai_api_key
            if not api_key:
                logger.warning("GPT restore enabled but no API key configured")
                return None
            
            client_kwargs = {"api_key": api_key}
            if self.config.openai_base_url:
                client_kwargs["base_url"] = self.config.openai_base_url
            
            self.client = OpenAI(**client_kwargs)
            return self.client
            
        except ImportError:
            logger.warning("openai package not installed - GPT restore disabled")
            return None
    
    def _restore_content_with_gpt(
        self,
        action_type: Optional[str],
        reason_text: Optional[str],
        parsed_steps: List[Dict[str, Any]],
        content_kind: Optional[str],
        content_value: Optional[Any]
    ) -> Optional[List[Dict[str, Any]]]:
        """Use GPT to restore missing/invalid content for type/key actions"""
        client = self._ensure_client()
        if not client:
            return None
        
        try:
            sys_prompt = (
                "You are an assistant that infers missing user input for UI actions. "
                "Given an action type ('type' or 'key') and a natural-language reason describing what changed, "
                "return STRICT JSON with a 'parsed_steps' array of one step. "
                "Base your answer only on the given reason text. "
                "For 'type': {index:1, action:'type', text:string}. "
                "For 'key': {index:1, action:'key', keys:[string,...]}."
            )
            
            user_prompt = {
                "action_type": action_type,
                "reason": reason_text or "",
                "parsed_steps": parsed_steps or [],
                "content_kind": content_kind,
                "content_value": content_value,
            }
            
            resp = client.chat.completions.create(
                model=self.config.openai_model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            
            text = resp.choices[0].message.content.strip()
            restored = json.loads(text)
            
            if isinstance(restored, dict) and isinstance(restored.get("parsed_steps"), list):
                steps = restored.get("parsed_steps")
                if steps:
                    # Extract readable summary
                    summary = []
                    for s in steps:
                        if s.get("action") == "type":
                            summary.append(f"type '{s.get('text', '')}'")
                        elif s.get("action") == "key":
                            summary.append(f"key {s.get('keys', [])}")
                    logger.info(f"✓ GPT restored {action_type}: {' + '.join(summary) if summary else steps}")
                    return steps
                    
        except Exception as e:
            logger.debug(f"GPT restoration failed: {e}")
        
        return None
    
    def _restore_batch_with_gpt(
        self,
        tasks: List[Dict[str, Any]]
    ) -> Dict[int, Optional[List[Dict[str, Any]]]]:
        """Run content restoration in parallel threads"""
        results: Dict[int, Optional[List[Dict[str, Any]]]] = {}
        
        if not tasks:
            return results
        
        def _one(task: Dict[str, Any]) -> Tuple[int, Optional[List[Dict[str, Any]]]]:
            idx = int(task.get("idx"))
            steps = self._restore_content_with_gpt(
                task.get("action_type"),
                task.get("reason_text"),
                task.get("parsed_steps") or [],
                task.get("content_kind"),
                task.get("content_value")
            )
            return (idx, steps)
        
        workers = max(1, min(int(self.gpt_workers), 8))
        
        if workers == 1:
            for t in tasks:
                k, v = _one(t)
                results[k] = v
            return results
        
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_one, t) for t in tasks]
            for fut in as_completed(futures):
                try:
                    k, v = fut.result()
                    results[k] = v
                except Exception:
                    continue
        
        return results
    
    def _build_valid_actions(self, json_file: Path) -> List[Dict[str, Any]]:
        """Extract valid actions from a trajectory file (with GPT restoration support)"""
        try:
            with open(json_file) as f:
                data = json.load(f)
        except Exception as e:
            logger.debug(f"Failed to load {json_file}: {e}")
            return []
        
        actions = data.get("actions")
        validation = data.get("action_validation") or {}
        evaluations = validation.get("evaluations") if isinstance(validation, dict) else None
        
        if not isinstance(actions, list) or not isinstance(evaluations, list):
            return []
        
        # First pass: collect valid actions and candidates for restoration
        candidates = []
        per_action_static = {}
        
        for ev in evaluations:
            try:
                # Check if action is valid
                if not isinstance(ev, dict) or not ev.get("valid"):
                    continue
                
                # Check coordinate-based content validity
                coord_rel = ev.get("coord_relative")
                is_coord_based = isinstance(coord_rel, list) and len(coord_rel) == 2
                content_valid = ev.get("content_valid")
                
                if is_coord_based and content_valid is not True:
                    continue
                
                # Get corresponding action
                action_index = ev.get("action_index")
                if not isinstance(action_index, int) or action_index < 1 or action_index > len(actions):
                    continue
                
                action = actions[action_index - 1]
                keyframes = action.get("keyframes") or {}
                ai = action.get("action_identification") or {}
                parsed_steps = ai.get("parsed_steps") or []
                action_type = action.get("action_type")
                
                # Find primary content
                content_kind, content_value = self._find_primary_content(parsed_steps)
                
                # Store static data
                per_action_static[action_index] = {
                    "action": action,
                    "ev": ev,
                    "keyframes": keyframes,
                    "ai": ai,
                    "parsed_steps": parsed_steps,
                    "action_type": action_type,
                }
                
                # Candidate for GPT restoration
                if self.use_gpt_restore and action_type in ("type", "key") and content_valid is not True:
                    reason_text = ev.get("reason", "") or ""
                    candidates.append({
                        "idx": action_index,
                        "action_type": action_type,
                        "reason_text": reason_text,
                        "parsed_steps": parsed_steps,
                        "content_kind": content_kind,
                        "content_value": content_value,
                    })
                    
            except Exception as e:
                logger.debug(f"Failed to process evaluation: {e}")
                continue
        
        # Run GPT restoration in parallel for candidates
        restored_map = {}
        if candidates:
            logger.info(f"Attempting GPT restoration for {len(candidates)} actions")
            restored_map = self._restore_batch_with_gpt(candidates)
        
        # Second pass: build outputs using restored content where available
        valid_items = []
        
        for action_index, static in per_action_static.items():
            try:
                action = static.get("action") or {}
                ev = static.get("ev") or {}
                keyframes = static.get("keyframes") or {}
                ai = static.get("ai") or {}
                parsed_steps = static.get("parsed_steps") or []
                action_type = static.get("action_type")
                
                # Use restored steps if available
                restored_steps = restored_map.get(action_index)
                final_steps = restored_steps if restored_steps is not None else parsed_steps
                was_restored = restored_steps is not None
                
                # For actions without parsed steps, try to use parsed_args or create default steps
                if not isinstance(final_steps, list) or len(final_steps) == 0:
                    # Try to get parsed_args instead
                    parsed_args = ai.get("parsed_args")
                    if isinstance(parsed_args, list) and parsed_args:
                        # Use parsed_args as final_steps
                        final_steps = []
                        for idx, arg in enumerate(parsed_args, 1):
                            if isinstance(arg, dict):
                                step = {"index": idx}
                                step.update(arg)
                                final_steps.append(step)
                    elif action_type:
                        # Create a default step for each action type
                        final_steps = [{"index": 1, "action": action_type}]
                    else:
                        continue
                
                # Extract dimensions
                resized_w, resized_h = self._extract_dimensions(ai)
                
                item = {
                    "action_type": action_type,
                    "keyframes": {
                        "start_frame": keyframes.get("start_frame"),
                        "end_frame": keyframes.get("end_frame"),
                    },
                    "transcripts": action.get("transcripts") or {},
                    "parsed_actions": final_steps,
                    "resized_width": resized_w,
                    "resized_height": resized_h,
                    "action_validation": ev,
                }
                
                # Mark if this action was restored by GPT
                if was_restored:
                    item["_gpt_restored"] = True
                
                valid_items.append(item)
                
            except Exception as e:
                logger.debug(f"Failed to build action item: {e}")
                continue
        
        return valid_items
    
    def _find_primary_content(self, parsed_steps: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[Any]]:
        """Find primary content from parsed steps"""
        if not isinstance(parsed_steps, list):
            return (None, None)
        
        for step in parsed_steps:
            if not isinstance(step, dict):
                continue
            
            action_name = step.get("action")
            if action_name == "type" and "text" in step:
                return ("text", step.get("text"))
            if action_name == "key" and "keys" in step:
                return ("keys", step.get("keys"))
        
        return (None, None)
    
    def _extract_dimensions(self, ai_block: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
        """Extract resized dimensions from action identification"""
        resized_w = ai_block.get("resized_width") if isinstance(ai_block, dict) else None
        resized_h = ai_block.get("resized_height") if isinstance(ai_block, dict) else None
        
        if resized_w is None or resized_h is None:
            vi = ai_block.get("video_info") if isinstance(ai_block, dict) else None
            if isinstance(vi, dict):
                resized_w = resized_w if resized_w is not None else vi.get("resized_width")
                resized_h = resized_h if resized_h is not None else vi.get("resized_height")
        
        try:
            resized_w = int(resized_w) if resized_w is not None else None
        except (ValueError, TypeError):
            resized_w = None
        
        try:
            resized_h = int(resized_h) if resized_h is not None else None
        except (ValueError, TypeError):
            resized_h = None
        
        return (resized_w, resized_h)

