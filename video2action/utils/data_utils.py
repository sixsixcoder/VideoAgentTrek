"""
Data utilities for handling raw_data structure

Structure:
    raw_data/
    ├── video_id_1/
    │   ├── video_id_1.mp4
    │   └── video_id_1_transcript.json
    └── video_id_2/
        ├── video_id_2.mp4
        └── video_id_2_transcript.json
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple


def get_preprocessed_videos(preprocess_dir: str, keep_only: bool = True) -> List[Dict[str, Any]]:
    """
    Get list of videos from preprocessing results.
    
    Args:
        preprocess_dir: Directory containing preprocessing results (e.g., "preprocessed_data")
        keep_only: If True, only return videos marked as "keep"
        
    Returns:
        List of video info dictionaries with preprocessing decisions
    """
    preprocess_dir = Path(preprocess_dir)
    
    if not preprocess_dir.exists():
        return []
    
    videos = []
    
    # Each subdirectory represents a video
    for video_dir in preprocess_dir.iterdir():
        if not video_dir.is_dir():
            continue
        
        decision_file = video_dir / "decision.json"
        if not decision_file.exists():
            continue
        
        try:
            with open(decision_file) as f:
                decision = json.load(f)
            
            # Check if video should be kept
            keep = decision.get("decision", {}).get("keep", False)
            
            if keep_only and not keep:
                continue
            
            video_info = {
                "video_id": decision.get("video_id"),
                "video_path": decision.get("video_path"),
                "keep": keep,
                "cursor_percentage": decision.get("analysis_summary", {}).get("cursor_percentage", 0.0),
                "preprocessing_dir": str(video_dir)
            }
            
            videos.append(video_info)
            
        except Exception as e:
            continue
    
    return videos


def find_raw_videos(raw_data_dir: str = "raw_data") -> List[Dict[str, str]]:
    """
    Find all videos in raw_data directory structure.
    
    Args:
        raw_data_dir: Path to raw_data directory
        
    Returns:
        List of dictionaries containing video information:
        [
            {
                "video_id": "S8Kbt1xKRcs",
                "video_path": "raw_data/S8Kbt1xKRcs/S8Kbt1xKRcs.mp4",
                "transcript_path": "raw_data/S8Kbt1xKRcs/S8Kbt1xKRcs_transcript.json",
                "video_dir": "raw_data/S8Kbt1xKRcs"
            },
            ...
        ]
    """
    raw_data_path = Path(raw_data_dir)
    
    if not raw_data_path.exists():
        raise FileNotFoundError(f"raw_data directory not found: {raw_data_dir}")
    
    videos = []
    
    # Iterate through subdirectories
    for subdir in raw_data_path.iterdir():
        if not subdir.is_dir():
            continue
        
        video_id = subdir.name
        
        # Look for video file (mp4, webm, mkv)
        video_file = None
        for ext in ['.mp4', '.webm', '.mkv']:
            candidate = subdir / f"{video_id}{ext}"
            if candidate.exists():
                video_file = candidate
                break
        
        if video_file is None:
            continue
        
        # Look for transcript
        transcript_file = subdir / f"{video_id}_transcript.json"
        
        videos.append({
            "video_id": video_id,
            "video_path": str(video_file),
            "transcript_path": str(transcript_file) if transcript_file.exists() else None,
            "video_dir": str(subdir)
        })
    
    return sorted(videos, key=lambda x: x['video_id'])


def get_video_info(video_id: str, raw_data_dir: str = "raw_data") -> Optional[Dict[str, str]]:
    """
    Get information for a specific video.
    
    Args:
        video_id: Video identifier
        raw_data_dir: Path to raw_data directory
        
    Returns:
        Dictionary with video information or None if not found
    """
    videos = find_raw_videos(raw_data_dir)
    
    for video in videos:
        if video['video_id'] == video_id:
            return video
    
    return None


def get_transcript_path(video_id: str, raw_data_dir: str = "raw_data") -> Optional[str]:
    """
    Get transcript path for a specific video.
    
    Args:
        video_id: Video identifier
        raw_data_dir: Path to raw_data directory
        
    Returns:
        Path to transcript JSON file or None if not found
    """
    video_info = get_video_info(video_id, raw_data_dir)
    
    if video_info:
        return video_info['transcript_path']
    
    return None


def validate_raw_data_structure(raw_data_dir: str = "raw_data") -> Tuple[List[str], List[str]]:
    """
    Validate raw_data directory structure.
    
    Args:
        raw_data_dir: Path to raw_data directory
        
    Returns:
        Tuple of (valid_videos, issues)
    """
    raw_data_path = Path(raw_data_dir)
    
    if not raw_data_path.exists():
        return [], [f"raw_data directory not found: {raw_data_dir}"]
    
    valid_videos = []
    issues = []
    
    for subdir in raw_data_path.iterdir():
        if not subdir.is_dir():
            issues.append(f"Non-directory item in raw_data: {subdir.name}")
            continue
        
        video_id = subdir.name
        
        # Check for video file
        video_found = False
        for ext in ['.mp4', '.webm', '.mkv']:
            if (subdir / f"{video_id}{ext}").exists():
                video_found = True
                break
        
        if not video_found:
            issues.append(f"{video_id}: Video file not found")
            continue
        
        # Check for transcript (optional but warn if missing)
        transcript_file = subdir / f"{video_id}_transcript.json"
        if not transcript_file.exists():
            issues.append(f"{video_id}: Transcript not found (optional)")
        
        valid_videos.append(video_id)
    
    return valid_videos, issues

