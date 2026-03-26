#!/usr/bin/env python3
"""
Run the complete 8-stage Video2Action pipeline on videos that passed preprocessing.

Usage:
    python video2action.py                    # Process all videos that passed preprocessing
    python video2action.py VIDEO_ID           # Process specific video by ID

This will process videos through all 8 stages:
    Stage 1: Video Splitting
    Stage 2: Keyframe Detection (Qwen2.5-VL)
    Stage 3: Action Clipping
    Stage 4: Action Identification (Qwen2.5-VL)
    Stage 5: Trajectory Building
    Stage 6: Action Validation (GPT/DashScope)
    Stage 7: Trajectory Export (with GPT content restoration)
    Stage 8: Inner Monologue Generation (GPT/DashScope)
"""

import sys
import json
from pathlib import Path

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent))

from video2action import Video2ActionPipeline
from video2action.config import PipelineConfig
from video2action.utils.data_utils import get_preprocessed_videos, find_raw_videos
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def get_videos_to_process(video_id=None):
    """Get list of videos to process"""
    preprocess_dir = Path("preprocessed_data")
    raw_data_dir = Path("raw_data")
    
    # If specific video ID provided, process only that one
    if video_id:
        video_dir = raw_data_dir / video_id
        video_path = video_dir / f"{video_id}.mp4"
        transcript_path = video_dir / f"{video_id}_transcript.json"
        if not video_path.exists():
            logger.error(f"Video not found: {video_path}")
            return []
        if not transcript_path.exists():
            logger.warning(f"Transcript not found for {video_id}, proceeding without it")
        return [{
            'video_id': video_id,
            'video_path': str(video_path),
            'transcript_path': str(video_dir / f"{video_id}_transcript.json"),
            'video_dir': str(video_dir)
        }]
    
    # Otherwise, get all videos that passed preprocessing
    if preprocess_dir.exists():
        videos = get_preprocessed_videos(str(preprocess_dir), keep_only=True)
        if videos:
            logger.info(f"Found {len(videos)} videos that passed preprocessing")
            return videos
    
    # Fallback: process all videos in raw_data/
    logger.warning("No preprocessing results found, loading all videos from raw_data/")
    raw_videos = find_raw_videos(str(raw_data_dir))
    return raw_videos


def main():
    """Run the full pipeline on videos"""
    
    # Check for command line argument (specific video ID)
    video_id = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Get videos to process
    videos = get_videos_to_process(video_id)
    
    if not videos:
        logger.error("No videos to process")
        return 1
    
    # Output directory
    output_dir = Path("output_video2action")
    output_dir.mkdir(exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("  VIDEO2ACTION COMPLETE PIPELINE (8 STAGES)")
    logger.info("=" * 80)
    logger.info(f"Processing {len(videos)} video(s)")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 80)
    
    # Pipeline configuration
    import os
    config = PipelineConfig(
        # Qwen2.5-VL model paths (local models for Stage 2 & 4)
        # Set via environment variables: KEYFRAME_MODEL_PATH and ACTION_MODEL_PATH
        model_path=os.getenv("KEYFRAME_MODEL_PATH", "/path/to/keyframe/detection/model"),
        action_model_path=os.getenv("ACTION_MODEL_PATH", "/path/to/action/identification/model"),
        
        # OpenAI/DashScope API configuration (for Stage 6 & 8)
        # Set via environment variables: OPENAI_API_KEY and OPENAI_BASE_URL
        openai_api_key=os.getenv("OPENAI_API_KEY"),  # Required for validation and inner monologue
        openai_base_url=os.getenv("OPENAI_BASE_URL"),  # Optional: for DashScope or custom endpoints
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o"),  # or "qwen-plus", "qwen3-vl-plus", etc.
        validation_model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        
        # Enable all stages
        enable_validation=True,
        enable_inner_monologue=True,
        
        # Stage 7: GPT content restoration
        enable_gpt_restore=True,
        gpt_restore_workers=4,
        
        # Video processing parameters
        segment_duration=10,
        keyframe_fps=1.0,
        similarity_threshold=0.9999,
        
        # GPU configuration
        available_gpus=[0],
        processes_per_gpu=2,
        cpu_workers=32,
        
        # File management
        skip_existing=False,  # Force reprocess
        clean_intermediate=False,  # Keep intermediate files for inspection
    )
    
    logger.info("\n⚙️  Pipeline Configuration:")
    logger.info(f"   Keyframe model: {config.model_path.split('/')[-1]}")
    logger.info(f"   Action model: {config.action_model_path.split('/')[-1]}")
    logger.info(f"   GPT API: {config.openai_base_url}")
    logger.info(f"   GPT model: {config.openai_model}")
    logger.info(f"   Validation: {'Enabled' if config.enable_validation else 'Disabled'}")
    logger.info(f"   Inner monologue: {'Enabled' if config.enable_inner_monologue else 'Disabled'}")
    logger.info(f"   GPT content restoration: {'Enabled' if config.enable_gpt_restore else 'Disabled'}")
    
    # Initialize pipeline
    logger.info("\n🔧 Initializing pipeline...")
    pipeline = Video2ActionPipeline(config=config)
    
    # Process videos
    results = []
    for i, video_info in enumerate(videos, 1):
        video_id = video_info['video_id']
        video_path = video_info['video_path']
        transcript_path = video_info.get('transcript_path')
        
        logger.info(f"\n{'='*80}")
        logger.info(f"  Video {i}/{len(videos)}: {video_id}")
        logger.info(f"{'='*80}")
        
        try:
            result = pipeline.process_video(
                video_path=str(video_path),
                output_dir=str(output_dir),
                video_id=video_id,
                transcript_file=transcript_path if transcript_path and Path(transcript_path).exists() else None,
                keep_intermediate=True
            )
            
            results.append({
                'video_id': video_id,
                'success': True,
                'output_file': result['output_file'],
                'num_actions': result['num_actions']
            })
            
            logger.info(f"\n✓ Successfully processed {video_id}")
            logger.info(f"   Actions: {result['num_actions']}")
            logger.info(f"   Output: {result['output_file']}")
            
            if 'inner_monologue_stats' in result:
                stats = result['inner_monologue_stats']
                logger.info(f"   Inner monologue: {stats.get('num_generated', 0)} generated")
            
        except Exception as e:
            logger.error(f"\n✗ Failed to process {video_id}: {e}")
            results.append({
                'video_id': video_id,
                'success': False,
                'error': str(e)
            })
    
    # Summary
    successful = sum(1 for r in results if r.get('success', False))
    failed = len(results) - successful
    
    logger.info("\n" + "=" * 80)
    logger.info("  PIPELINE COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total videos: {len(results)}")
    logger.info(f"✓ Successful: {successful}")
    logger.info(f"✗ Failed: {failed}")
    
    if successful > 0:
        total_actions = sum(r.get('num_actions', 0) for r in results if r.get('success'))
        logger.info(f"\nTotal actions extracted: {total_actions}")
        logger.info(f"Output location: {output_dir}/trajectories/")
    
    logger.info("=" * 80)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(main())

