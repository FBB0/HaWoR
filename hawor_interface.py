#!/usr/bin/env python3
"""
HaWoR User Interface - Easy-to-use interface for processing egocentric videos
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import time
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from advanced_hawor import AdvancedHaWoR

class HaWoRInterface:
    """
    User-friendly interface for HaWoR processing
    """

    def __init__(self, device: str = 'auto'):
        """Initialize the interface"""
        self.device = device
        self.pipeline = None
        self.processing_history = []

        print("🤖 HaWoR Interface - Hand Motion Reconstruction from Egocentric Videos")
        print("=" * 70)

    def initialize_pipeline(self):
        """Initialize the HaWoR pipeline"""
        print("🔧 Initializing HaWoR pipeline...")

        try:
            self.pipeline = AdvancedHaWoR(device=self.device)
            print("✅ Pipeline initialized successfully!")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize pipeline: {e}")
            return False

    def check_system_status(self):
        """Check system status and available models"""
        print("\n📊 System Status:")
        print("-" * 30)

        if self.pipeline is None:
            print("❌ Pipeline not initialized")
            return False

        status = {
            'device': str(self.pipeline.device),
            'models_available': self.pipeline.models_available
        }

        print(f"🖥️  Device: {status['device']}")
        print(f"📦 Models Available:")

        for model, available in status['models_available'].items():
            status_icon = "✅" if available else "❌"
            print(f"   {status_icon} {model}: {available}")

        return True

    def process_single_video(self,
                           video_path: str,
                           output_dir: Optional[str] = None,
                           mode: str = 'auto',
                           visualize: bool = True) -> Dict[str, Any]:
        """Process a single video"""

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        video_name = Path(video_path).stem
        if output_dir is None:
            output_dir = f"hawor_output_{video_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        print(f"\n🎬 Processing Video: {video_name}")
        print(f"📁 Output Directory: {output_dir}")
        print(f"⚙️  Mode: {mode}")

        start_time = time.time()

        try:
            # Determine processing mode
            if mode == 'auto':
                if (self.pipeline.models_available['hawor_modules'] and
                    self.pipeline.models_available['checkpoint']):
                    actual_mode = 'hawor'
                else:
                    actual_mode = 'simplified'
            else:
                actual_mode = mode

            print(f"🔄 Using {actual_mode} processing mode...")

            # Process video
            if actual_mode == 'hawor':
                results = self.pipeline.process_video_with_hawor(
                    video_path=video_path,
                    output_dir=output_dir
                )
            else:
                results = self.pipeline.process_video_simplified(
                    video_path=video_path,
                    output_dir=output_dir,
                    visualize=visualize
                )

            processing_time = time.time() - start_time

            # Create summary report
            self.pipeline.create_summary_report(results, output_dir)

            # Add to processing history
            history_entry = {
                'video_path': video_path,
                'output_dir': output_dir,
                'mode': actual_mode,
                'processing_time': processing_time,
                'timestamp': datetime.now().isoformat(),
                'success': True
            }
            self.processing_history.append(history_entry)

            print(f"✅ Processing completed in {processing_time:.1f} seconds")
            print(f"📂 Results saved to: {output_dir}")

            return results

        except Exception as e:
            print(f"❌ Processing failed: {e}")

            history_entry = {
                'video_path': video_path,
                'output_dir': output_dir,
                'mode': mode,
                'processing_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat(),
                'success': False,
                'error': str(e)
            }
            self.processing_history.append(history_entry)

            raise e

    def process_multiple_videos(self,
                              video_paths: List[str],
                              output_base_dir: str = 'hawor_batch_output',
                              mode: str = 'auto',
                              visualize: bool = True) -> List[Dict[str, Any]]:
        """Process multiple videos in batch"""

        print(f"\n🎬 Batch Processing: {len(video_paths)} videos")
        print(f"📁 Base Output Directory: {output_base_dir}")

        os.makedirs(output_base_dir, exist_ok=True)
        results = []

        for i, video_path in enumerate(video_paths):
            print(f"\n--- Processing {i+1}/{len(video_paths)} ---")

            try:
                video_name = Path(video_path).stem
                output_dir = os.path.join(output_base_dir, f"{video_name}_output")

                result = self.process_single_video(
                    video_path=video_path,
                    output_dir=output_dir,
                    mode=mode,
                    visualize=visualize
                )
                results.append(result)

            except Exception as e:
                print(f"⚠️  Failed to process {video_path}: {e}")
                results.append({'error': str(e), 'video_path': video_path})

        # Create batch summary
        self._create_batch_summary(results, output_base_dir)

        print(f"\n✅ Batch processing completed!")
        print(f"📊 Successfully processed: {sum(1 for r in results if 'error' not in r)}/{len(video_paths)}")

        return results

    def _create_batch_summary(self, results: List[Dict[str, Any]], output_dir: str):
        """Create a summary of batch processing results"""

        summary_path = os.path.join(output_dir, 'batch_summary.json')

        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_videos': len(results),
            'successful': sum(1 for r in results if 'error' not in r),
            'failed': sum(1 for r in results if 'error' in r),
            'processing_history': self.processing_history,
            'results': results
        }

        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"📊 Batch summary saved to: {summary_path}")

    def list_example_videos(self) -> List[str]:
        """List available example videos"""
        example_dir = Path('example')

        if not example_dir.exists():
            print("❌ Example directory not found")
            return []

        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        example_videos = []

        for ext in video_extensions:
            example_videos.extend(list(example_dir.glob(f'*{ext}')))

        if example_videos:
            print("\n📹 Available Example Videos:")
            for i, video in enumerate(example_videos):
                print(f"  {i+1}. {video.name}")
        else:
            print("❌ No example videos found")

        return [str(v) for v in example_videos]

    def interactive_mode(self):
        """Run in interactive mode"""
        print("\n🎮 Interactive Mode")
        print("Type 'help' for available commands")

        while True:
            try:
                command = input("\nhawor> ").strip().lower()

                if command == 'help':
                    self._show_help()
                elif command == 'status':
                    self.check_system_status()
                elif command == 'examples':
                    self.list_example_videos()
                elif command.startswith('process '):
                    video_path = command[8:].strip()
                    if video_path:
                        try:
                            self.process_single_video(video_path)
                        except Exception as e:
                            print(f"❌ Error: {e}")
                    else:
                        print("❌ Please provide a video path")
                elif command == 'history':
                    self._show_history()
                elif command in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                else:
                    print("❌ Unknown command. Type 'help' for available commands")

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

    def _show_help(self):
        """Show help information"""
        print("""
📚 Available Commands:
  help      - Show this help message
  status    - Check system status and available models
  examples  - List available example videos
  process <video_path> - Process a single video
  history   - Show processing history
  quit/exit/q - Exit interactive mode

📖 Examples:
  process example/video_0.mp4
  process /path/to/my/video.mp4
        """)

    def _show_history(self):
        """Show processing history"""
        if not self.processing_history:
            print("📝 No processing history available")
            return

        print("\n📝 Processing History:")
        print("-" * 50)

        for i, entry in enumerate(self.processing_history[-10:], 1):  # Show last 10
            status = "✅" if entry['success'] else "❌"
            video_name = Path(entry['video_path']).name
            timestamp = entry['timestamp'][:19]  # Remove microseconds

            print(f"{i}. {status} {video_name} ({timestamp})")
            if not entry['success']:
                print(f"   Error: {entry.get('error', 'Unknown error')}")


def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(
        description='HaWoR Interface - Easy hand motion reconstruction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python hawor_interface.py --interactive

  # Process single video
  python hawor_interface.py --video example/video_0.mp4

  # Process multiple videos
  python hawor_interface.py --videos video1.mp4 video2.mp4 video3.mp4

  # Batch process all examples
  python hawor_interface.py --examples
        """
    )

    parser.add_argument('--video', type=str, help='Path to input video')
    parser.add_argument('--videos', nargs='+', help='Paths to multiple videos')
    parser.add_argument('--output', type=str, help='Output directory')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/mps/cpu/auto)')
    parser.add_argument('--mode', type=str, default='auto',
                       choices=['auto', 'hawor', 'simplified'],
                       help='Processing mode')
    parser.add_argument('--no-vis', action='store_true', help='Skip visualization')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    parser.add_argument('--examples', action='store_true',
                       help='Process all example videos')

    args = parser.parse_args()

    # Initialize interface
    interface = HaWoRInterface(device=args.device)

    if not interface.initialize_pipeline():
        print("❌ Failed to initialize pipeline. Please check your setup.")
        return 1

    # Check system status
    interface.check_system_status()

    # Interactive mode
    if args.interactive:
        interface.interactive_mode()
        return 0

    # Process example videos
    if args.examples:
        example_videos = interface.list_example_videos()
        if example_videos:
            interface.process_multiple_videos(
                video_paths=example_videos,
                output_base_dir='example_outputs',
                mode=args.mode,
                visualize=not args.no_vis
            )
        return 0

    # Process single video
    if args.video:
        try:
            interface.process_single_video(
                video_path=args.video,
                output_dir=args.output,
                mode=args.mode,
                visualize=not args.no_vis
            )
        except Exception as e:
            print(f"❌ Processing failed: {e}")
            return 1
        return 0

    # Process multiple videos
    if args.videos:
        try:
            interface.process_multiple_videos(
                video_paths=args.videos,
                output_base_dir=args.output or 'batch_output',
                mode=args.mode,
                visualize=not args.no_vis
            )
        except Exception as e:
            print(f"❌ Batch processing failed: {e}")
            return 1
        return 0

    # If no specific action, show help
    parser.print_help()
    return 0


if __name__ == '__main__':
    exit(main())