#!/usr/bin/env python3
"""
Main script to train HaWoR with real model architecture and loss functions
This replaces the previous simulation-based training with actual neural network training
"""

import os
import sys
import argparse
from pathlib import Path
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.training.real_hawor_trainer import RealHaWoRTrainer


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Real HaWoR Training Pipeline")

    parser.add_argument('--config', type=str, default='arctic_training_config.yaml',
                        help='Path to training configuration file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--test-only', action='store_true',
                        help='Run test pipeline only (no training)')
    parser.add_argument('--eval-only', action='store_true',
                        help='Run evaluation only (requires trained model)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with verbose output')

    args = parser.parse_args()

    print("🤖 Real HaWoR Training Pipeline")
    print("=" * 50)
    print("🎯 This pipeline implements ACTUAL model training with:")
    print("  - Real neural network forward/backward passes")
    print("  - Actual loss computation and gradients")
    print("  - Real optimizer updates")
    print("  - Comprehensive MANO-based loss functions")
    print("  - Vision Transformer backbone")
    print("  - Temporal modeling with LSTM")
    print("  - Camera pose estimation (SLAM)")
    print("=" * 50)

    # Set debug level
    if args.debug:
        torch.autograd.set_detect_anomaly(True)
        print("🐛 Debug mode enabled")

    # Test mode
    if args.test_only:
        print("🧪 Running test pipeline...")
        from test_real_training import main as test_main
        success = test_main()
        return success

    # Evaluation mode
    if args.eval_only:
        print("📊 Running evaluation only...")
        from src.evaluation.hawor_evaluation import evaluate_hawor_model

        # Check for trained model
        model_path = "outputs/real_hawor_training/best_checkpoint.pth"
        if not os.path.exists(model_path):
            print(f"❌ No trained model found at {model_path}")
            print("   Please train a model first or specify correct path")
            return False

        try:
            metrics = evaluate_hawor_model(
                model_path=model_path,
                config_path=args.config,
                data_root="thirdparty/arctic",
                output_dir="evaluation_results"
            )

            print("✅ Evaluation completed!")
            print(f"📊 Key Results:")
            print(f"  - MPJPE: {metrics.get('mpjpe', 0):.2f} mm")
            print(f"  - PA-MPJPE: {metrics.get('pa_mpjpe', 0):.2f} mm")
            print(f"  - Camera tracking error: {metrics.get('camera_trans_error', 0):.4f} m")

            return True

        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            return False

    # Check configuration file
    if not os.path.exists(args.config):
        print(f"❌ Configuration file not found: {args.config}")
        print("   Please provide a valid configuration file")
        return False

    # Check for ARCTIC data
    arctic_data_path = Path("thirdparty/arctic")
    if not arctic_data_path.exists():
        print(f"⚠️  ARCTIC data directory not found: {arctic_data_path}")
        print("   Training will proceed but may fail if data is not available")
        print("   Please ensure ARCTIC dataset is properly downloaded and organized")

    # Initialize trainer
    print(f"🚀 Initializing trainer with config: {args.config}")

    try:
        trainer = RealHaWoRTrainer(args.config)
    except Exception as e:
        print(f"❌ Failed to initialize trainer: {e}")
        return False

    # Resume from checkpoint if specified
    if args.resume:
        if os.path.exists(args.resume):
            print(f"📥 Resuming from checkpoint: {args.resume}")
            if not trainer.load_checkpoint(args.resume):
                print("❌ Failed to load checkpoint")
                return False
        else:
            print(f"❌ Checkpoint file not found: {args.resume}")
            return False

    # Start training
    print("\n🚀 Starting real HaWoR training...")
    print("⚡ This will perform actual neural network training with:")
    print("  - Real gradient computation and backpropagation")
    print("  - Actual loss function optimization")
    print("  - Model parameter updates")
    print("  - Checkpoint saving and validation")

    try:
        success = trainer.train()

        if success:
            print("\n🎉 Training completed successfully!")
            print("📁 Results saved to:")
            print(f"  - Checkpoints: {trainer.output_dir}/best_checkpoint.pth")
            print(f"  - Training logs: {trainer.output_dir}/real_training_report.json")
            print(f"  - Visualizations: {trainer.output_dir}/")

            print("\n🚀 Next steps:")
            print("  1. Run evaluation: python train_real_hawor.py --eval-only")
            print("  2. Check training visualizations in output directory")
            print("  3. Use trained model for inference on new videos")

            return True
        else:
            print("\n❌ Training failed!")
            print("   Check the error messages above for details")
            return False

    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        print("   Partial training results may be saved in output directory")
        return False
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        if args.debug:
            traceback.print_exc()
        return False


def check_requirements():
    """Check if all requirements are available"""
    print("🔍 Checking requirements...")

    # Check PyTorch
    try:
        import torch
        print(f"  ✅ PyTorch: {torch.__version__}")

        # Check device availability
        if torch.cuda.is_available():
            device = "CUDA"
            device_name = torch.cuda.get_device_name()
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "MPS (Apple Silicon)"
            device_name = "Apple Silicon GPU"
        else:
            device = "CPU"
            device_name = "CPU only"

        print(f"  🖥️  Device: {device} ({device_name})")

    except ImportError:
        print("  ❌ PyTorch not available")
        return False

    # Check other requirements
    required_packages = ['numpy', 'opencv-python', 'matplotlib', 'tqdm', 'yaml']
    missing_packages = []

    for package in required_packages:
        try:
            if package == 'opencv-python':
                import cv2
            elif package == 'yaml':
                import yaml
            else:
                __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package}")
            missing_packages.append(package)

    if missing_packages:
        print(f"\n⚠️  Missing packages: {missing_packages}")
        print("   Install with: pip install " + " ".join(missing_packages))
        return False

    print("  ✅ All requirements satisfied!")
    return True


if __name__ == "__main__":
    print("🤖 HaWoR Real Training Pipeline")
    print("🎯 Training actual neural networks for hand pose estimation")

    # Check requirements first
    if not check_requirements():
        print("\n❌ Requirements check failed!")
        sys.exit(1)

    # Run main function
    success = main()

    if success:
        print("\n✅ Pipeline completed successfully!")
        exit_code = 0
    else:
        print("\n❌ Pipeline failed!")
        exit_code = 1

    sys.exit(exit_code)