"""
Main Video2Action Pipeline Orchestrator

This module provides a unified interface for processing videos through the entire
Video2Action pipeline.
"""

import os
import json
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

from .config import PipelineConfig
from .stages import (
    VideoSplitter,
    KeyframeDetector,
    ActionClipper,
    ActionIdentifier,
    TrajectoryBuilder,
    ActionValidator,
    TrajectoryExporter,
    InnerMonologueGenerator
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class Video2ActionPipeline:
    """
    Complete pipeline for extracting agent trajectories from screen-recorded videos.
    
    Usage:
        pipeline = Video2ActionPipeline(config=PipelineConfig())
        trajectory = pipeline.process_video("video.mp4", output_dir="./output")
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the pipeline with configuration.
        
        Args:
            config: Pipeline configuration. If None, uses default config.
        """
        self.config = config or PipelineConfig()
        self.stages = self._initialize_stages()
        
    def _initialize_stages(self) -> Dict[str, Any]:
        """Initialize all pipeline stages"""
        logger.info("Initializing pipeline stages...")
        
        stages = {
            'splitter': VideoSplitter(self.config),
            'keyframe_detector': KeyframeDetector(self.config),
            'action_clipper': ActionClipper(self.config),
            'action_identifier': ActionIdentifier(self.config),
            'trajectory_builder': TrajectoryBuilder(self.config),
        }
        
        # Optional stages
        if self.config.enable_validation:
            stages['validator'] = ActionValidator(self.config)
            stages['exporter'] = TrajectoryExporter(self.config)
        
        if self.config.enable_inner_monologue:
            stages['inner_monologue'] = InnerMonologueGenerator(self.config)
        
        logger.info("✓ Pipeline stages initialized")
        return stages
    
    def process_video(
        self,
        video_path: str,
        output_dir: str,
        video_id: Optional[str] = None,
        transcript_file: Optional[str] = None,
        keep_intermediate: bool = False
    ) -> Dict[str, Any]:
        """
        Process a single video through the complete pipeline.
        
        Args:
            video_path: Path to input video file
            output_dir: Directory for output files
            video_id: Optional video ID (defaults to video filename)
            transcript_file: Optional transcript JSON file
            keep_intermediate: Whether to keep intermediate files
            
        Returns:
            Dictionary containing trajectory data and metadata
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Generate video ID
        if video_id is None:
            video_id = video_path.stem
        
        logger.info("=" * 70)
        logger.info(f"Processing video: {video_id}")
        logger.info(f"Input: {video_path}")
        logger.info(f"Output: {output_dir}")
        logger.info("=" * 70)
        
        # Create workspace directory
        workspace_dir = output_dir / f"workspace_{video_id}"
        workspace_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Stage 1: Split video into segments
            logger.info("\n[Stage 1/8] Splitting video...")
            clips_dir = self._run_stage_1(video_path, workspace_dir, video_id)
            
            # Stage 2: Detect keyframes (action timestamps)
            logger.info("\n[Stage 2/8] Detecting keyframes...")
            keyframes_dir = self._run_stage_2(clips_dir, workspace_dir, video_id)
            
            # Stage 3: Extract action clips
            logger.info("\n[Stage 3/8] Extracting action clips...")
            action_clips_dir = self._run_stage_3(keyframes_dir, clips_dir, workspace_dir, video_id)
            
            # Stage 4: Identify action parameters
            logger.info("\n[Stage 4/8] Identifying action parameters...")
            action_results_dir = self._run_stage_4(action_clips_dir, workspace_dir, video_id)
            
            # Stage 5: Build raw trajectory
            logger.info("\n[Stage 5/8] Building raw trajectory...")
            raw_trajectory_dir = self._run_stage_5(
                clips_dir, keyframes_dir, action_clips_dir, action_results_dir,
                workspace_dir, video_id, transcript_file
            )
            
            # Stage 6 & 7: Validation and export (if enabled)
            if self.config.enable_validation:
                logger.info("\n[Stage 6/8] Validating actions...")
                validated_dir = self._run_stage_6(raw_trajectory_dir, workspace_dir, video_id)
                
                logger.info("\n[Stage 7/8] Exporting final trajectory...")
                final_trajectory = self._run_stage_7(validated_dir, output_dir, video_id)
            else:
                logger.info("\n[Stage 6-7/8] Skipping validation (disabled)")
                final_trajectory = self._export_raw_trajectory(raw_trajectory_dir, output_dir, video_id)
            
            # Stage 8: Generate inner monologue (if enabled)
            if self.config.enable_inner_monologue:
                logger.info("\n[Stage 8/8] Generating inner monologue...")
                final_trajectory = self._run_stage_8(final_trajectory, output_dir, video_id)
            else:
                logger.info("\n[Stage 8/8] Skipping inner monologue (disabled)")
            
            # Cleanup intermediate files if requested
            if not keep_intermediate and self.config.clean_intermediate:
                logger.info("\nCleaning up intermediate files...")
                shutil.rmtree(workspace_dir, ignore_errors=True)
                logger.info("✓ Cleanup complete")
            
            logger.info("\n" + "=" * 70)
            logger.info("✓ PIPELINE COMPLETE")
            logger.info(f"Final trajectory: {final_trajectory['output_file']}")
            logger.info("=" * 70)
            
            return final_trajectory
            
        except Exception as e:
            logger.error(f"\n✗ Pipeline failed: {str(e)}")
            raise
    
    def _run_stage_1(self, video_path: Path, workspace_dir: Path, video_id: str) -> Path:
        """Stage 1: Split video into segments"""
        clips_dir = workspace_dir / "clips"
        clips_dir.mkdir(exist_ok=True)
        
        self.stages['splitter'].split_video(
            video_path=str(video_path),
            output_dir=str(clips_dir),
            video_id=video_id
        )
        
        num_clips = len(list(clips_dir.glob("*.mp4")))
        logger.info(f"✓ Created {num_clips} video segments")
        return clips_dir
    
    def _run_stage_2(self, clips_dir: Path, workspace_dir: Path, video_id: str) -> Path:
        """Stage 2: Detect keyframes"""
        keyframes_dir = workspace_dir / "keyframes"
        keyframes_dir.mkdir(exist_ok=True)
        
        self.stages['keyframe_detector'].detect_keyframes(
            clips_dir=str(clips_dir),
            output_dir=str(keyframes_dir),
            video_id=video_id
        )
        
        num_keyframes = len(list(keyframes_dir.glob("*_output.json")))
        logger.info(f"✓ Detected keyframes in {num_keyframes} clips")
        return keyframes_dir
    
    def _run_stage_3(self, keyframes_dir: Path, clips_dir: Path, workspace_dir: Path, video_id: str) -> Path:
        """Stage 3: Extract action clips"""
        action_clips_dir = workspace_dir / "action_clips"
        action_clips_dir.mkdir(exist_ok=True)
        
        self.stages['action_clipper'].extract_action_clips(
            keyframes_dir=str(keyframes_dir),
            clips_dir=str(clips_dir),
            output_dir=str(action_clips_dir),
            video_id=video_id
        )
        
        num_actions = len(list(action_clips_dir.glob("*.mp4")))
        logger.info(f"✓ Extracted {num_actions} action clips")
        return action_clips_dir
    
    def _run_stage_4(self, action_clips_dir: Path, workspace_dir: Path, video_id: str) -> Path:
        """Stage 4: Identify action parameters"""
        # Create separate output directory for Stage 4
        action_results_dir = workspace_dir / "stage4_action_identification"
        action_results_dir.mkdir(parents=True, exist_ok=True)
        
        self.stages['action_identifier'].identify_actions(
            action_clips_dir=str(action_clips_dir),
            output_dir=str(action_results_dir),
            video_id=video_id
        )
        
        video_results_dir = action_results_dir / video_id
        results_file = video_results_dir / "action_identification_results_s2.json"
        if results_file.exists():
            with open(results_file) as f:
                results = json.load(f)
            logger.info(f"✓ Identified {len(results)} actions")
        else:
            logger.warning("⚠ No action identification results found")
        
        return video_results_dir
    
    def _run_stage_5(
        self,
        clips_dir: Path,
        keyframes_dir: Path,
        action_clips_dir: Path,
        action_results_dir: Path,
        workspace_dir: Path,
        video_id: str,
        transcript_file: Optional[str]
    ) -> Path:
        """Stage 5: Build raw trajectory"""
        raw_trajectory_dir = workspace_dir / "raw_trajectory"
        raw_trajectory_dir.mkdir(exist_ok=True)
        
        self.stages['trajectory_builder'].build_trajectory(
            clips_dir=str(clips_dir),
            keyframes_dir=str(keyframes_dir),
            action_clips_dir=str(action_clips_dir),
            action_results_dir=str(action_results_dir),
            output_dir=str(raw_trajectory_dir),
            video_id=video_id,
            transcript_file=transcript_file
        )
        
        num_actions = len(list(raw_trajectory_dir.glob("*.json")))
        logger.info(f"✓ Built raw trajectory with {num_actions} action records")
        return raw_trajectory_dir
    
    def _run_stage_6(self, raw_trajectory_dir: Path, workspace_dir: Path, video_id: str) -> Path:
        """Stage 6: Validate actions"""
        # Create separate output directory for validated trajectories
        validated_dir = workspace_dir / "stage6_validated_trajectory"
        validated_dir.mkdir(parents=True, exist_ok=True)
        
        self.stages['validator'].validate_actions(
            trajectory_dir=str(raw_trajectory_dir),
            video_id=video_id,
            output_dir=str(validated_dir)
        )
        
        logger.info("✓ Action validation complete")
        return validated_dir
    
    def _run_stage_7(self, validated_dir: Path, output_dir: Path, video_id: str) -> Dict[str, Any]:
        """Stage 7: Export final trajectory"""
        final_output_dir = output_dir / "trajectories"
        final_output_dir.mkdir(parents=True, exist_ok=True)
        
        trajectory = self.stages['exporter'].export_trajectory(
            validated_dir=str(validated_dir),
            output_dir=str(final_output_dir),
            video_id=video_id
        )
        
        output_file = final_output_dir / f"{video_id}_trajectory.json"
        with open(output_file, 'w') as f:
            json.dump(trajectory, f, indent=2)
        
        logger.info(f"✓ Exported trajectory: {output_file}")
        
        return {
            'video_id': video_id,
            'output_file': str(output_file),
            'num_actions': len(trajectory.get('valid_actions', [])),
            'trajectory': trajectory
        }
    
    def _export_raw_trajectory(self, raw_dir: Path, output_dir: Path, video_id: str) -> Dict[str, Any]:
        """Export raw trajectory without validation"""
        final_output_dir = output_dir / "trajectories"
        final_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect all raw action files
        action_files = sorted(raw_dir.glob("*.json"))
        actions = []
        
        for action_file in action_files:
            with open(action_file) as f:
                action_data = json.load(f)
                if 'actions' in action_data and action_data['actions']:
                    actions.extend(action_data['actions'])
        
        trajectory = {
            'video_id': video_id,
            'actions': actions,
            'num_actions': len(actions)
        }
        
        output_file = final_output_dir / f"{video_id}_trajectory.json"
        with open(output_file, 'w') as f:
            json.dump(trajectory, f, indent=2)
        
        logger.info(f"✓ Exported raw trajectory: {output_file}")
        
        return {
            'video_id': video_id,
            'output_file': str(output_file),
            'num_actions': len(actions),
            'trajectory': trajectory
        }
    
    def _run_stage_8(self, trajectory_result: Dict[str, Any], output_dir: Path, video_id: str) -> Dict[str, Any]:
        """Stage 8: Generate inner monologue"""
        trajectory_file = Path(trajectory_result['output_file'])
        
        # Generate inner monologue
        im_result = self.stages['inner_monologue'].generate_for_trajectory(
            trajectory_path=str(trajectory_file),
            model=self.config.openai_model,
            skip_existing=False
        )
        
        logger.info(f"✓ Generated inner monologue for {im_result.get('num_generated', 0)} actions")
        
        # Reload the trajectory with inner monologue
        with open(trajectory_file) as f:
            updated_trajectory = json.load(f)
        
        # Update the result
        trajectory_result['trajectory'] = updated_trajectory
        trajectory_result['inner_monologue_stats'] = {
            'num_generated': im_result.get('num_generated', 0),
            'num_skipped': im_result.get('num_skipped', 0),
            'num_failed': im_result.get('num_failed', 0)
        }
        
        return trajectory_result
    
    def batch_process(
        self,
        video_paths: List[str],
        output_dir: str,
        keep_intermediate: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Process multiple videos through the pipeline.
        
        Args:
            video_paths: List of video file paths
            output_dir: Directory for output files
            keep_intermediate: Whether to keep intermediate files
            
        Returns:
            List of trajectory results
        """
        results = []
        
        logger.info(f"\nBatch processing {len(video_paths)} videos...")
        
        for i, video_path in enumerate(video_paths, 1):
            logger.info(f"\n{'='*70}")
            logger.info(f"Video {i}/{len(video_paths)}")
            logger.info(f"{'='*70}")
            
            try:
                result = self.process_video(
                    video_path=video_path,
                    output_dir=output_dir,
                    keep_intermediate=keep_intermediate
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {video_path}: {e}")
                results.append({
                    'video_path': video_path,
                    'error': str(e),
                    'success': False
                })
        
        # Summary
        successful = sum(1 for r in results if 'error' not in r)
        logger.info(f"\n{'='*70}")
        logger.info(f"Batch processing complete: {successful}/{len(video_paths)} successful")
        logger.info(f"{'='*70}")
        
        return results

