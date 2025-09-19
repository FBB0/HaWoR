#!/usr/bin/env python3
"""
Production Training Pipeline Setup for HaWoR
Configures data paths, validates environment, and sets up training infrastructure
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional

class TrainingPipelineSetup:
    """Setup and configuration for HaWoR training pipeline"""

    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.config = {}
        self.data_paths = {}
        self.validation_results = {}

    def validate_environment(self) -> Dict[str, bool]:
        """Validate the training environment"""
        print("🔍 Validating training environment...")

        checks = {
            "base_directory": self.base_dir.exists(),
            "weights_directory": (self.base_dir / "weights").exists(),
            "data_directory": (self.base_dir / "_DATA").exists(),
            "configs_directory": (self.base_dir / "configs").exists(),
            "enhanced_training_script": (self.base_dir / "enhanced_training_pipeline.py").exists(),
            "evaluation_framework": (self.base_dir / "enhanced_training_evaluation.py").exists(),
            "arctic_data": (self.base_dir / "thirdparty" / "arctic" / "unpack" / "arctic_data" / "data").exists()
        }

        # Check for MANO data
        mano_paths = [
            self.base_dir / "_DATA" / "data" / "mano" / "MANO_RIGHT.pkl",
            self.base_dir / "_DATA" / "data_left" / "mano_left" / "MANO_LEFT.pkl"
        ]
        checks["mano_models"] = all(p.exists() for p in mano_paths)

        # Check for model weights
        model_weights = [
            self.base_dir / "weights" / "hawor" / "checkpoints" / "hawor.ckpt",
            self.base_dir / "weights" / "hawor" / "checkpoints" / "infiller.pt",
            self.base_dir / "weights" / "hawor" / "model_config.yaml"
        ]
        checks["model_weights"] = all(p.exists() for p in model_weights)

        # Display results
        for check_name, passed in checks.items():
            status = "✅" if passed else "❌"
            print(f"   {status} {check_name.replace('_', ' ').title()}")

        self.validation_results = checks
        return checks

    def setup_data_paths(self) -> Dict[str, str]:
        """Setup and validate data paths for training"""
        print("\n📁 Setting up data paths...")

        # Define standard data paths
        paths = {
            # Input data
            "arctic_data_root": str(self.base_dir / "thirdparty" / "arctic" / "unpack" / "arctic_data" / "data"),
            "mano_data_dir": str(self.base_dir / "_DATA" / "data"),
            "mano_model_path": str(self.base_dir / "_DATA" / "data" / "mano"),
            "mano_left_path": str(self.base_dir / "_DATA" / "data_left" / "mano_left"),

            # Model weights
            "pretrained_weights": str(self.base_dir / "weights" / "hawor" / "checkpoints" / "hawor.ckpt"),
            "infiller_weights": str(self.base_dir / "weights" / "hawor" / "checkpoints" / "infiller.pt"),
            "model_config": str(self.base_dir / "weights" / "hawor" / "model_config.yaml"),

            # Training data (to be created)
            "training_data_dir": str(self.base_dir / "training_data"),
            "validation_data_dir": str(self.base_dir / "validation_data"),

            # Output directories
            "output_dir": str(self.base_dir / "training_output"),
            "log_dir": str(self.base_dir / "training_logs"),
            "checkpoint_dir": str(self.base_dir / "checkpoints"),
            "report_dir": str(self.base_dir / "reports"),

            # Tensorboard logs
            "tensorboard_log_dir": str(self.base_dir / "logs" / "tensorboard"),
        }

        # Create output directories
        output_dirs = [
            "training_data_dir", "validation_data_dir", "output_dir",
            "log_dir", "checkpoint_dir", "report_dir", "tensorboard_log_dir"
        ]

        for dir_key in output_dirs:
            dir_path = Path(paths[dir_key])
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"   📂 Created: {dir_path}")

        self.data_paths = paths
        return paths

    def create_training_config(self) -> Dict:
        """Create optimized training configuration"""
        print("\n⚙️  Creating training configuration...")

        config = {
            # Data configuration
            "data": {
                "arctic_data_root": self.data_paths["arctic_data_root"],
                "training_data_dir": self.data_paths["training_data_dir"],
                "validation_data_dir": self.data_paths["validation_data_dir"],
                "mano_data_dir": self.data_paths["mano_data_dir"],
                "mano_model_path": self.data_paths["mano_model_path"],
                "image_size": 256,
                "num_workers": 8,
                "pin_memory": True,
                "prefetch_factor": 2
            },

            # Model configuration
            "model": {
                "backbone_type": "vit",
                "pretrained_weights": self.data_paths["pretrained_weights"],
                "torch_compile": 1,
                "mano_gender": "neutral",
                "num_hand_joints": 15,
                "st_module": True,
                "motion_module": True,
                "st_hdim": 512,
                "motion_hdim": 384,
                "st_nlayer": 6,
                "motion_nlayer": 6
            },

            # Training configuration
            "training": {
                "max_epochs": 100,
                "batch_size": 8,
                "learning_rate": 1e-5,
                "weight_decay": 1e-4,
                "grad_clip_val": 1.0,
                "warmup_epochs": 5,
                "early_stopping_patience": 15,
                "accumulate_grad_batches": 1
            },

            # Loss configuration
            "loss": {
                "use_enhanced_loss": True,
                "use_adaptive_weights": True,
                "temporal_window": 5,
                "weights": {
                    "KEYPOINTS_3D": 0.1,
                    "KEYPOINTS_2D": 0.05,
                    "GLOBAL_ORIENT": 0.01,
                    "HAND_POSE": 0.01,
                    "BETAS": 0.005,
                    "MESH_VERTICES": 0.02,
                    "MESH_FACES": 0.01,
                    "TEMPORAL_CONSISTENCY": 0.005,
                    "OCCLUSION_ROBUSTNESS": 0.01
                }
            },

            # Evaluation configuration
            "evaluation": {
                "eval_every_n_epochs": 1,
                "val_check_interval": 1.0,
                "eval_metrics": [
                    "mpjpe_3d", "mpjpe_2d", "pck_3d_15mm",
                    "pck_2d_10px", "temporal_consistency", "mesh_quality"
                ]
            },

            # Logging configuration
            "logging": {
                "use_tensorboard": True,
                "use_wandb": False,
                "tensorboard_log_dir": self.data_paths["tensorboard_log_dir"],
                "log_every_n_steps": 50,
                "log_images_every_n_steps": 500,
                "render_frequency": 1000
            },

            # Hardware configuration
            "hardware": {
                "devices": 1,
                "accelerator": "auto",
                "precision": 16,
                "find_unused_parameters": False
            },

            # Output configuration
            "output": {
                "output_dir": self.data_paths["output_dir"],
                "log_dir": self.data_paths["log_dir"],
                "checkpoint_dir": self.data_paths["checkpoint_dir"],
                "report_dir": self.data_paths["report_dir"],
                "save_top_k": 3,
                "save_every_n_epochs": 10,
                "monitor": "val/loss",
                "mode": "min"
            },

            # ARCTIC dataset configuration
            "arctic": {
                "subjects": ["s01", "s02", "s04", "s05", "s06", "s07", "s08", "s09", "s10"],
                "max_sequences_per_subject": 20,
                "min_sequence_length": 10,
                "max_sequence_length": 100,
                "data_filtering": {
                    "min_confidence": 0.5,
                    "max_occlusion_level": 0.8,
                    "min_keypoint_visibility": 0.7
                }
            }
        }

        self.config = config
        return config

    def save_configuration(self, config_file: str = "production_training_config.json") -> str:
        """Save training configuration to file"""
        config_path = self.base_dir / "configs" / config_file

        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

        print(f"💾 Configuration saved to: {config_path}")
        return str(config_path)

    def create_data_preparation_script(self) -> str:
        """Create a data preparation script for the pipeline"""
        script_content = f'''#!/usr/bin/env python3
"""
Data Preparation Script for HaWoR Training
Auto-generated by setup_training_pipeline.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from training_data_preparation import ArcticDataConverter

def main():
    """Prepare training data from ARCTIC dataset"""
    print("🚀 Starting data preparation for HaWoR training...")

    # Configuration
    arctic_root = "{self.data_paths['arctic_data_root']}"
    training_output = "{self.data_paths['training_data_dir']}"
    validation_output = "{self.data_paths['validation_data_dir']}"

    # Initialize converter
    converter = ArcticDataConverter(
        arctic_root=arctic_root,
        output_dir=training_output,
        target_resolution=(256, 256),
        num_workers=8
    )

    # Convert training data (first 80% of sequences)
    print("\\n1️⃣  Converting training data...")
    train_subjects = ["s01", "s02", "s04", "s05", "s06", "s07"]
    train_stats = converter.convert_dataset(
        subjects=train_subjects,
        max_sequences_per_subject=15,
        output_split="train"
    )
    print(f"Training data conversion completed: {{train_stats}}")

    # Convert validation data (remaining 20%)
    print("\\n2️⃣  Converting validation data...")
    converter.output_dir = validation_output
    val_subjects = ["s08", "s09", "s10"]
    val_stats = converter.convert_dataset(
        subjects=val_subjects,
        max_sequences_per_subject=5,
        output_split="val"
    )
    print(f"Validation data conversion completed: {{val_stats}}")

    print("\\n✅ Data preparation completed successfully!")
    print(f"Training data: {{training_output}}")
    print(f"Validation data: {{validation_output}}")

if __name__ == "__main__":
    main()
'''

        script_path = self.base_dir / "prepare_training_data.py"
        with open(script_path, 'w') as f:
            f.write(script_content)

        # Make executable
        os.chmod(script_path, 0o755)
        print(f"📝 Data preparation script created: {script_path}")
        return str(script_path)

    def create_training_launcher(self) -> str:
        """Create a training launcher script"""
        launcher_content = f'''#!/usr/bin/env python3
"""
HaWoR Training Launcher
Auto-generated by setup_training_pipeline.py
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from enhanced_training_pipeline import TrainingPipeline

def main():
    """Launch HaWoR training"""
    parser = argparse.ArgumentParser(description="Launch HaWoR Training")
    parser.add_argument("--config", default="configs/production_training_config.json",
                       help="Path to training configuration file")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--dry-run", action="store_true",
                       help="Perform dry run without actual training")

    args = parser.parse_args()

    print("🚀 HaWoR Training Pipeline")
    print("=" * 50)

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Configuration file not found: {{config_path}}")
        return 1

    print(f"📄 Using configuration: {{config_path}}")

    # Initialize training pipeline
    try:
        pipeline = TrainingPipeline(str(config_path))

        if args.dry_run:
            print("🔍 Running dry run...")
            pipeline.validate_setup()
            print("✅ Dry run completed successfully")
            return 0

        # Start training
        print("🏃 Starting training...")
        trainer, model = pipeline.train()

        print("✅ Training completed successfully!")
        print(f"📊 Logs saved to: {{pipeline.config.get('output', {{}}).get('log_dir', 'logs')}}")
        print(f"💾 Checkpoints saved to: {{pipeline.config.get('output', {{}}).get('checkpoint_dir', 'checkpoints')}}")

    except Exception as e:
        print(f"❌ Training failed: {{e}}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
'''

        launcher_path = self.base_dir / "launch_training.py"
        with open(launcher_path, 'w') as f:
            f.write(launcher_content)

        # Make executable
        os.chmod(launcher_path, 0o755)
        print(f"🚀 Training launcher created: {launcher_path}")
        return str(launcher_path)

    def create_monitoring_dashboard(self) -> str:
        """Create a simple monitoring dashboard script"""
        dashboard_content = '''#!/usr/bin/env python3
"""
HaWoR Training Monitoring Dashboard
Simple monitoring for training progress
"""

import json
import time
from pathlib import Path
from datetime import datetime

def monitor_training(log_dir: str = "./training_logs"):
    """Monitor training progress"""
    log_path = Path(log_dir)

    print("📊 HaWoR Training Monitor")
    print("=" * 40)
    print("Press Ctrl+C to stop monitoring\\n")

    try:
        while True:
            # Check for latest metrics
            metrics_files = list(log_path.glob("**/metrics.json"))

            if metrics_files:
                latest_file = max(metrics_files, key=lambda x: x.stat().st_mtime)

                try:
                    with open(latest_file, 'r') as f:
                        metrics = json.load(f)

                    print(f"\\r🕐 {datetime.now().strftime('%H:%M:%S')} | "
                          f"Epoch: {metrics.get('epoch', 'N/A')} | "
                          f"Loss: {metrics.get('train_loss', 'N/A'):.4f} | "
                          f"Val Loss: {metrics.get('val_loss', 'N/A'):.4f}", end="")

                except Exception:
                    print(f"\\r🕐 {datetime.now().strftime('%H:%M:%S')} | "
                          f"Monitoring... (no metrics yet)", end="")
            else:
                print(f"\\r🕐 {datetime.now().strftime('%H:%M:%S')} | "
                      f"Waiting for training to start...", end="")

            time.sleep(5)

    except KeyboardInterrupt:
        print("\\n\\n📊 Monitoring stopped")

if __name__ == "__main__":
    import sys
    log_dir = sys.argv[1] if len(sys.argv) > 1 else "./training_logs"
    monitor_training(log_dir)
'''

        dashboard_path = self.base_dir / "monitor_training.py"
        with open(dashboard_path, 'w') as f:
            f.write(dashboard_content)

        # Make executable
        os.chmod(dashboard_path, 0o755)
        print(f"📊 Monitoring dashboard created: {dashboard_path}")
        return str(dashboard_path)

    def generate_setup_report(self) -> str:
        """Generate comprehensive setup report"""
        report = {
            "setup_summary": {
                "timestamp": datetime.now().isoformat(),
                "base_directory": str(self.base_dir),
                "environment_validation": self.validation_results,
                "data_paths": self.data_paths,
                "configuration": self.config
            },
            "next_steps": [
                "1. Run data preparation: python prepare_training_data.py",
                "2. Start training: python launch_training.py",
                "3. Monitor progress: python monitor_training.py",
                "4. View logs: tensorboard --logdir=logs/tensorboard"
            ],
            "recommendations": [
                "Ensure sufficient disk space for training data",
                "Monitor GPU memory usage during training",
                "Run validation on small dataset first",
                "Set up regular checkpoint backups"
            ]
        }

        # Check environment status
        all_checks_passed = all(self.validation_results.values())
        report["status"] = "READY" if all_checks_passed else "NEEDS_SETUP"

        if not all_checks_passed:
            failed_checks = [k for k, v in self.validation_results.items() if not v]
            report["required_actions"] = [
                f"Fix missing component: {check.replace('_', ' ')}"
                for check in failed_checks
            ]

        # Save report
        report_path = self.base_dir / "training_setup_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"📋 Setup report saved to: {report_path}")
        return str(report_path)

    def run_complete_setup(self) -> bool:
        """Run complete training pipeline setup"""
        print("🔧 HaWoR Production Training Pipeline Setup")
        print("=" * 50)

        # Step 1: Validate environment
        env_checks = self.validate_environment()

        # Step 2: Setup data paths
        self.setup_data_paths()

        # Step 3: Create configuration
        self.create_training_config()

        # Step 4: Save configuration
        config_path = self.save_configuration()

        # Step 5: Create utility scripts
        data_script = self.create_data_preparation_script()
        launcher_script = self.create_training_launcher()
        monitor_script = self.create_monitoring_dashboard()

        # Step 6: Generate report
        report_path = self.generate_setup_report()

        # Summary
        print("\\n" + "=" * 50)
        all_checks_passed = all(env_checks.values())

        if all_checks_passed:
            print("✅ Training pipeline setup completed successfully!")
            print("\\n🚀 Ready to start training!")
            print("\\nNext steps:")
            print("1. Prepare data:    python prepare_training_data.py")
            print("2. Start training:  python launch_training.py")
            print("3. Monitor:         python monitor_training.py")
            print("4. View logs:       tensorboard --logdir=logs/tensorboard")
        else:
            print("⚠️  Training pipeline setup completed with warnings")
            print("\\n🔧 Please fix the following issues:")
            failed_checks = [k for k, v in env_checks.items() if not v]
            for check in failed_checks:
                print(f"   ❌ {check.replace('_', ' ').title()}")

        print(f"\\n📋 Detailed report: {report_path}")
        return all_checks_passed

def main():
    """Main setup function"""
    import argparse

    parser = argparse.ArgumentParser(description="Setup HaWoR Training Pipeline")
    parser.add_argument("--base-dir", default=".", help="Base directory for setup")
    args = parser.parse_args()

    setup = TrainingPipelineSetup(args.base_dir)
    success = setup.run_complete_setup()

    return 0 if success else 1

if __name__ == "__main__":
    import sys
    from datetime import datetime
    sys.exit(main())