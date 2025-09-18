#!/usr/bin/env python3
"""
Run HaWoR with automatic hand visualization
"""

import os
import sys
import argparse
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from advanced_hawor import AdvancedHaWoR
from create_improved_visualization import create_improved_visualization

def main():
    """Main function to run HaWoR with visualization"""
    
    parser = argparse.ArgumentParser(description='Run HaWoR with hand visualization')
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--output', type=str, help='Output directory')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cuda/mps/cpu/auto)')
    parser.add_argument('--focal', type=float, help='Camera focal length')
    parser.add_argument('--no-vis', action='store_true', help='Skip visualization')
    
    args = parser.parse_args()
    
    # Setup output directory
    if args.output is None:
        args.output = f"hawor_output_{Path(args.video).stem}"
    
    print("🤖 HaWoR with Hand Visualization")
    print("=" * 50)
    print(f"Video: {args.video}")
    print(f"Output: {args.output}")
    print(f"Device: {args.device}")
    print()
    
    # Initialize HaWoR pipeline
    print("🔧 Initializing HaWoR pipeline...")
    pipeline = AdvancedHaWoR(device=args.device)
    
    # Process video
    print("📹 Processing video...")
    results = pipeline.process_video_with_hawor(
        video_path=args.video,
        output_dir=args.output,
        img_focal=args.focal
    )
    
    # Create visualization if not disabled
    if not args.no_vis:
        print("🎨 Creating hand visualization...")
        results_path = os.path.join(args.output, 'hawor_results.npz')
        
        if os.path.exists(results_path):
            create_improved_visualization(results_path, args.output, args.video)
            print("✅ Hand visualization created successfully!")
        else:
            print("⚠️  No results file found for visualization")
    
    print()
    print("🎉 Processing complete!")
    print(f"📁 Results saved to: {args.output}")
    
    if not args.no_vis:
        print(f"🎬 Visualization video: {args.output}/improved_hand_visualization.mp4")
        print(f"🖼️  Frame images: {args.output}/improved_visualizations/")

if __name__ == '__main__':
    main()
