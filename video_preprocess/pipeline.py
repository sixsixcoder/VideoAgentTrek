"""
Preprocessing pipeline orchestrator
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

from .config import PreprocessConfig
from .cursor_detector import CursorDetector

logger = logging.getLogger(__name__)


class PreprocessPipeline:
    """
    Main preprocessing pipeline for cursor detection and video filtering.
    
    Usage:
        config = PreprocessConfig(cursor_threshold=0.5)
        pipeline = PreprocessPipeline(config)
        results = pipeline.process_folder("sample_data/")
    """
    
    def __init__(self, config: Optional[PreprocessConfig] = None):
        """
        Initialize preprocessing pipeline.
        
        Args:
            config: PreprocessConfig instance. If None, uses default config.
        """
        self.config = config or PreprocessConfig()
        self.detector = CursorDetector(self.config)
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def find_videos(self, input_dir: str) -> List[Path]:
        """
        Find all supported video files in directory.
        
        Supports both structures:
        - Flat: input_dir/*.mp4
        - Nested: input_dir/video_id/video_id.mp4
        
        Args:
            input_dir: Directory to search for videos
            
        Returns:
            List of video file paths
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        videos = []
        
        # Check for nested structure (raw_data/video_id/video_id.mp4)
        for subdir in input_path.iterdir():
            if not subdir.is_dir():
                continue
            
            video_id = subdir.name
            for ext in self.config.support_video_formats:
                video_file = subdir / f"{video_id}{ext}"
                if video_file.exists():
                    videos.append(video_file)
                    break
        
        # Fallback: also check for flat structure
        if not videos:
            for ext in self.config.support_video_formats:
                videos.extend(input_path.glob(f"*{ext}"))
        
        logger.info(f"Found {len(videos)} videos in {input_dir}")
        return sorted(videos)
    
    def process_single_video(
        self,
        video_path: str,
        gpu_id: int = 0
    ) -> Dict[str, Any]:
        """
        Process a single video through cursor detection.
        
        Args:
            video_path: Path to video file
            gpu_id: GPU device ID to use
            
        Returns:
            Dictionary containing analysis results
        """
        video_path = Path(video_path)
        video_id = video_path.stem
        transcript_path = Path(video_path.parent, video_path.stem + "_transcript.json")
        
        # Create output directory for this video
        output_dir = Path(self.config.output_dir) / video_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if already processed
        decision_file = output_dir / "decision.json"
        if decision_file.exists() and not self.config.overwrite:
            logger.info(f"Skipping {video_id} - already processed")
            with open(decision_file) as f:
                return json.load(f)
        
        try:
            # Run cursor detection
            logger.info(f"Processing {video_id}...")
            result = self.detector.detect_cursor_in_video(str(video_path), gpu_id)
            
            # Save results
            self._save_results(output_dir, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to process {video_id}: {e}")
            error_result = {
                "video_id": video_id,
                "video_path": str(video_path),
                "transcript_path": str(transcript_path),
                "error": str(e),
                "decision": {
                    "keep": False,
                    "reason": f"Processing error: {str(e)}",
                    "recommended_for_trajectory": False
                }
            }
            
            # Save error result
            with open(output_dir / "decision.json", 'w') as f:
                json.dump(error_result, f, indent=2)
            
            return error_result
    
    def _save_results(self, output_dir: Path, result: Dict[str, Any]):
        """
        Save processing results to disk.
        
        Args:
            output_dir: Directory to save results
            result: Detection results dictionary
        """
        # Save decision (always)
        with open(output_dir / "decision.json", 'w') as f:
            json.dump({
                "video_id": result["video_id"],
                "video_path": result["video_path"],
                "transcript_path": result["transcript_path"],
                "decision": result["decision"],
                "analysis_summary": {
                    "cursor_percentage": result["analysis"]["cursor_percentage"],
                    "active_segments": result["analysis"]["active_segments"],
                }
            }, f, indent=2)
        
        # Save full analysis
        with open(output_dir / "cursor_analysis.json", 'w') as f:
            json.dump({
                "video_id": result["video_id"],
                "metadata": result["metadata"],
                "analysis": result["analysis"],
            }, f, indent=2)
        
        # Save segments if enabled
        if self.config.save_segments:
            with open(output_dir / "segments.json", 'w') as f:
                json.dump(result["analysis"]["segments"], f, indent=2)
        
        # Save detection details if enabled
        if self.config.save_detection_details:
            with open(output_dir / "detection_results.json", 'w') as f:
                json.dump(result["detection_results"], f, indent=2)
        
        logger.info(f"✓ Saved results to {output_dir}")
    
    def process_folder(
        self,
        input_dir: str,
        parallel: bool = False
    ) -> Dict[str, List[str]]:
        """
        Process all videos in a folder.
        
        Args:
            input_dir: Directory containing videos
            parallel: Whether to use parallel processing (experimental)
            
        Returns:
            Dictionary with 'kept' and 'rejected' video IDs
        """
        videos = self.find_videos(input_dir)
        
        if not videos:
            logger.warning(f"No videos found in {input_dir}")
            return {"kept": [], "rejected": []}
        
        logger.info(f"Processing {len(videos)} videos...")
        logger.info(f"Cursor threshold: {self.config.cursor_threshold * 100}%")
        logger.info(f"Output directory: {self.config.output_dir}")
        
        kept_videos = []
        rejected_videos = []
        
        if parallel and len(videos) > 1:
            # Parallel processing (experimental)
            logger.info(f"Using parallel processing with {self.config.max_workers} workers")
            with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = {
                    executor.submit(self.process_single_video, str(video), i % self.config.num_gpus): video
                    for i, video in enumerate(videos)
                }
                
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result["decision"]["keep"]:
                            kept_videos.append(result["video_id"])
                        else:
                            rejected_videos.append(result["video_id"])
                    except Exception as e:
                        logger.error(f"Processing error: {e}")
        else:
            # Sequential processing
            for video in videos:
                result = self.process_single_video(str(video))
                if result["decision"]["keep"]:
                    kept_videos.append(result["video_id"])
                else:
                    rejected_videos.append(result["video_id"])
        
        # Save summary
        summary = {
            "total_videos": len(videos),
            "kept": kept_videos,
            "rejected": rejected_videos,
            "kept_count": len(kept_videos),
            "rejected_count": len(rejected_videos),
            "config": self.config.to_dict()
        }
        
        summary_file = Path(self.config.output_dir) / "preprocessing_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"PREPROCESSING COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Total videos: {len(videos)}")
        logger.info(f"✓ Kept: {len(kept_videos)} ({len(kept_videos)/len(videos)*100:.1f}%)")
        logger.info(f"✗ Rejected: {len(rejected_videos)} ({len(rejected_videos)/len(videos)*100:.1f}%)")
        logger.info(f"Summary saved to: {summary_file}")
        
        return summary

