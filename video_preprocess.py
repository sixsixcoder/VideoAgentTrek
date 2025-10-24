#!/usr/bin/env python3
"""
Test video preprocessing (cursor detection)
"""

import sys
from pathlib import Path

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent))

from video_preprocess import PreprocessConfig, PreprocessPipeline


def main():
    print("\n" + "=" * 80)
    print("  VIDEO PREPROCESSING - CURSOR DETECTION TEST")
    print("=" * 80)
    
    # Check if YOLO model exists
    import os
    model_path = Path(os.getenv("YOLO_MODEL_PATH", "/path/to/yolov8x-cursor-model/best.pt"))
    if not model_path.exists():
        print(f"\n⚠️  YOLO model not found: {model_path}")
        print("   Please set YOLO_MODEL_PATH environment variable or update the path")
        print("   Example: export YOLO_MODEL_PATH=/path/to/yolov8x-cursor-model/best.pt")
        return 1
    
    print(f"\n✓ Model found: {model_path}")
    
    # Configure preprocessing
    config = PreprocessConfig(
        cursor_threshold=0.30,  # Keep if >=30% cursor presence (adjusted based on testing)
        yolo_model_path=str(model_path),
        detection_stride=10,  # Process every 10th frame (balanced speed/accuracy)
        confidence_threshold=0.3,  # Keep default (most detections are 0.76-0.86)
        min_segment_duration=6,  # Minimum 6s segments
        output_dir="preprocessed_data",
        save_segments=True,
        save_detection_details=True,  # Save details for analysis
        overwrite=True,  # Overwrite previous results
        num_gpus=1
    )
    
    print(f"\n⚙️  Configuration:")
    print(f"   Cursor threshold: {config.cursor_threshold * 100}%")
    print(f"   Detection stride: every {config.detection_stride} frames (improved from 30)")
    print(f"   Confidence threshold: {config.confidence_threshold}")
    print(f"   Min segment duration: {config.min_segment_duration}s")
    print(f"   Output directory: {config.output_dir}")
    
    # Initialize pipeline
    pipeline = PreprocessPipeline(config)
    
    # Check input directory (new raw_data structure)
    input_dir = Path("raw_data")
    if not input_dir.exists():
        print(f"\n✗ ERROR: Input directory not found: {input_dir}")
        print(f"   Expected structure: raw_data/video_id/video_id.mp4")
        return 1
    
    videos = pipeline.find_videos(str(input_dir))
    if not videos:
        print(f"\n✗ ERROR: No videos found in {input_dir}")
        return 1
    
    print(f"\n📹 Found {len(videos)} video(s):")
    for video in videos:
        size_mb = video.stat().st_size / 1024 / 1024
        print(f"   - {video.name} ({size_mb:.1f} MB)")
    
    # Process videos
    print(f"\n🚀 Starting cursor detection...")
    print(f"   (This may take several minutes depending on video length)")
    
    try:
        results = pipeline.process_folder(str(input_dir), parallel=False)
        
        print(f"\n" + "=" * 80)
        print(f"  RESULTS")
        print(f"=" * 80)
        
        print(f"\n✓ Kept videos ({results['kept_count']}):")
        for video_id in results['kept']:
            output_dir = Path(config.output_dir) / video_id
            decision_file = output_dir / "decision.json"
            if decision_file.exists():
                import json
                with open(decision_file) as f:
                    data = json.load(f)
                cursor_pct = data['analysis_summary']['cursor_percentage']
                print(f"   {video_id}: {cursor_pct:.1f}% cursor presence")
        
        print(f"\n✗ Rejected videos ({results['rejected_count']}):")
        for video_id in results['rejected']:
            output_dir = Path(config.output_dir) / video_id
            decision_file = output_dir / "decision.json"
            if decision_file.exists():
                import json
                with open(decision_file) as f:
                    data = json.load(f)
                if 'analysis_summary' in data:
                    cursor_pct = data['analysis_summary']['cursor_percentage']
                    print(f"   {video_id}: {cursor_pct:.1f}% cursor presence (below threshold)")
                else:
                    reason = data['decision']['reason']
                    print(f"   {video_id}: {reason}")
        
        print(f"\n📁 Output location: {config.output_dir}/")
        print(f"   Each video has:")
        print(f"   - decision.json        (keep/reject decision)")
        print(f"   - cursor_analysis.json (detailed analysis)")
        print(f"   - segments.json        (cursor-active segments)")
        
        if results['kept_count'] > 0:
            print(f"\n🎯 Next step:")
            print(f"   Run video2action pipeline on kept videos:")
            for video_id in results['kept']:
                print(f"   python test_stage1_only.py  # Or full pipeline")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

