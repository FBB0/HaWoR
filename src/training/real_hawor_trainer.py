#!/usr/bin/env python3
"""
Real HaWoR Training Pipeline
Implements actual model training with real forward/backward passes and loss computation
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import time
import yaml
import json
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.hawor_model import HaWoRModel, create_hawor_model
from src.training.hawor_losses import HaWoRLossFunction, create_joint_weights
from src.datasets.arctic_dataset_real import create_arctic_dataloaders
from src.training.visualization import TrainingVisualizer


class RealHaWoRTrainer:
    """
    Real HaWoR trainer with actual model training, gradients, and optimization
    """

    def __init__(self, config_path: str):
        """Initialize real training pipeline"""

        self.config = self._load_config(config_path)
        self.device = self._setup_device()
        self.output_dir = Path(self.config.get('output_dir', 'outputs/real_hawor_training'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.loss_function = None
        self.train_loader = None
        self.val_loader = None

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_metrics = {
            'train_loss': [],
            'val_loss': [],
            'train_keypoint_error': [],
            'val_keypoint_error': [],
            'learning_rate': [],
            'epoch_times': []
        }

        # Setup logging
        self.visualizer = TrainingVisualizer(str(self.output_dir))

        print(f"🚀 Real HaWoR Trainer initialized")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"💻 Device: {self.device}")

    def _load_config(self, config_path: str) -> Dict:
        """Load training configuration"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"✅ Loaded config from: {config_path}")
            return config
        except Exception as e:
            print(f"❌ Error loading config: {e}")
            return {}

    def _setup_device(self) -> torch.device:
        """Setup computation device"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"🔥 Using CUDA: {torch.cuda.get_device_name()}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            print(f"🍎 Using Apple Silicon MPS")
        else:
            device = torch.device('cpu')
            print(f"💻 Using CPU")

        return device

    def setup_model(self) -> bool:
        """Initialize the HaWoR model"""
        try:
            print("🤖 Setting up HaWoR model...")

            # Create model
            self.model = create_hawor_model(self.config)
            self.model.to(self.device)

            # Count parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

            print(f"  📊 Total parameters: {total_params:,}")
            print(f"  🎯 Trainable parameters: {trainable_params:,}")

            # Load pretrained weights if specified
            pretrained_path = self.config.get('model', {}).get('pretrained_weights')
            if pretrained_path and os.path.exists(pretrained_path):
                print(f"  📥 Loading pretrained weights from: {pretrained_path}")
                checkpoint = torch.load(pretrained_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)

            return True

        except Exception as e:
            print(f"❌ Error setting up model: {e}")
            return False

    def setup_loss_function(self) -> bool:
        """Initialize loss function"""
        try:
            print("📏 Setting up loss function...")

            # Loss weights from config
            loss_weights = self.config.get('loss_weights', {})

            self.loss_function = HaWoRLossFunction(
                lambda_keypoint=loss_weights.get('keypoint', 10.0),
                lambda_mano_pose=loss_weights.get('mano_pose', 1.0),
                lambda_mano_shape=loss_weights.get('mano_shape', 0.1),
                lambda_temporal=loss_weights.get('temporal', 5.0),
                lambda_camera=loss_weights.get('camera', 1.0),
                lambda_reprojection=loss_weights.get('reprojection', 100.0),
                lambda_consistency=loss_weights.get('consistency', 2.0)
            )

            print("  ✅ Loss function initialized")
            return True

        except Exception as e:
            print(f"❌ Error setting up loss function: {e}")
            return False

    def setup_optimizer_and_scheduler(self) -> bool:
        """Setup optimizer and learning rate scheduler"""
        try:
            print("⚙️ Setting up optimizer and scheduler...")

            training_config = self.config.get('training', {})

            # Optimizer
            learning_rate = training_config.get('learning_rate', 5e-5)
            weight_decay = training_config.get('weight_decay', 1e-4)

            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )

            # Learning rate scheduler
            num_epochs = training_config.get('max_epochs', 10)
            warmup_epochs = training_config.get('warmup_epochs', 1)

            # Cosine annealing with warmup
            def lr_lambda(epoch):
                if epoch < warmup_epochs:
                    return epoch / warmup_epochs
                else:
                    progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
                    return 0.5 * (1 + np.cos(np.pi * progress))

            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

            print(f"  📈 Optimizer: AdamW (lr={learning_rate}, wd={weight_decay})")
            print(f"  📊 Scheduler: Cosine annealing with {warmup_epochs} warmup epochs")
            return True

        except Exception as e:
            print(f"❌ Error setting up optimizer: {e}")
            return False

    def setup_data_loaders(self) -> bool:
        """Setup data loaders"""
        try:
            print("📊 Setting up data loaders...")

            self.train_loader, self.val_loader = create_arctic_dataloaders(self.config)

            print(f"  🏋️ Train samples: {len(self.train_loader.dataset)}")
            print(f"  🧪 Val samples: {len(self.val_loader.dataset)}")
            print(f"  📦 Batch size: {self.train_loader.batch_size}")

            return True

        except Exception as e:
            print(f"❌ Error setting up data loaders: {e}")
            return False

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""

        self.model.train()
        epoch_losses = []
        epoch_keypoint_errors = []

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch}",
            leave=False
        )

        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                # Zero gradients
                self.optimizer.zero_grad()

                # Forward pass
                images = batch['images']  # [B, T, C, H, W]
                outputs = self.model(images)

                # Prepare targets for loss computation
                targets = {
                    'hand_pose': batch['hand_pose'],        # [B, T, 45]
                    'hand_shape': batch['hand_shape'],      # [B, T, 10]
                    'hand_trans': batch['hand_trans'],      # [B, T, 3]
                    'hand_rot': batch['hand_rot'],          # [B, T, 3]
                    'hand_joints': batch['hand_joints'],    # [B, T, 21, 3]
                    'keypoints_2d': batch['keypoints_2d'],  # [B, T, 21, 2]
                    'camera_pose': batch['camera_pose']     # [B, T, 6]
                }

                # Get predictions for loss (we'll use left hand for now)
                predictions = outputs  # The model outputs both hands
                pred_for_loss = {
                    'hand_pose': predictions['left_hand_pose'],
                    'hand_shape': predictions['left_hand_shape'],
                    'hand_trans': predictions['left_hand_trans'],
                    'hand_rot': predictions['left_hand_rot'],
                    'hand_joints': predictions['left_hand_joints'],
                    'camera_pose': predictions['camera_pose']
                }

                # Camera parameters (simplified)
                camera_params = {
                    'fx': 500.0,
                    'fy': 500.0,
                    'cx': 128.0,  # Half of img_size
                    'cy': 128.0
                }

                # Compute loss
                loss_dict = self.loss_function(pred_for_loss, targets, camera_params)
                total_loss = loss_dict['total']

                # Backward pass
                total_loss.backward()

                # Gradient clipping
                grad_clip = self.config.get('training', {}).get('grad_clip_val', 1.0)
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

                # Optimizer step
                self.optimizer.step()

                # Compute metrics
                with torch.no_grad():
                    keypoint_error = self._compute_keypoint_error(
                        pred_for_loss['hand_joints'],
                        targets['hand_joints']
                    )

                # Log metrics
                epoch_losses.append(total_loss.item())
                epoch_keypoint_errors.append(keypoint_error.item())

                # Update progress bar
                progress_bar.set_postfix({
                    'Loss': f"{total_loss.item():.4f}",
                    'KeypointErr': f"{keypoint_error.item():.2f}mm"
                })

                # Print detailed loss breakdown occasionally
                if batch_idx % 10 == 0:
                    loss_breakdown = {k: v.item() for k, v in loss_dict.items() if k != 'total'}
                    print(f"\n    Batch {batch_idx} loss breakdown: {loss_breakdown}")

            except Exception as e:
                print(f"❌ Error in training batch {batch_idx}: {e}")
                continue

        # Compute epoch metrics
        epoch_metrics = {
            'train_loss': np.mean(epoch_losses) if epoch_losses else 0.0,
            'train_keypoint_error': np.mean(epoch_keypoint_errors) if epoch_keypoint_errors else 0.0,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }

        return epoch_metrics

    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""

        self.model.eval()
        val_losses = []
        val_keypoint_errors = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Validation", leave=False)):
                try:
                    # Move batch to device
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                            for k, v in batch.items()}

                    # Forward pass
                    images = batch['images']
                    outputs = self.model(images)

                    # Prepare targets
                    targets = {
                        'hand_pose': batch['hand_pose'],
                        'hand_shape': batch['hand_shape'],
                        'hand_trans': batch['hand_trans'],
                        'hand_rot': batch['hand_rot'],
                        'hand_joints': batch['hand_joints'],
                        'keypoints_2d': batch['keypoints_2d'],
                        'camera_pose': batch['camera_pose']
                    }

                    # Get predictions for loss
                    pred_for_loss = {
                        'hand_pose': outputs['left_hand_pose'],
                        'hand_shape': outputs['left_hand_shape'],
                        'hand_trans': outputs['left_hand_trans'],
                        'hand_rot': outputs['left_hand_rot'],
                        'hand_joints': outputs['left_hand_joints'],
                        'camera_pose': outputs['camera_pose']
                    }

                    # Camera parameters
                    camera_params = {
                        'fx': 500.0,
                        'fy': 500.0,
                        'cx': 128.0,
                        'cy': 128.0
                    }

                    # Compute loss
                    loss_dict = self.loss_function(pred_for_loss, targets, camera_params)
                    total_loss = loss_dict['total']

                    # Compute keypoint error
                    keypoint_error = self._compute_keypoint_error(
                        pred_for_loss['hand_joints'],
                        targets['hand_joints']
                    )

                    val_losses.append(total_loss.item())
                    val_keypoint_errors.append(keypoint_error.item())

                except Exception as e:
                    print(f"❌ Error in validation batch {batch_idx}: {e}")
                    continue

        # Compute validation metrics
        val_metrics = {
            'val_loss': np.mean(val_losses) if val_losses else 0.0,
            'val_keypoint_error': np.mean(val_keypoint_errors) if val_keypoint_errors else 0.0
        }

        return val_metrics

    def _compute_keypoint_error(self, pred_joints: torch.Tensor, gt_joints: torch.Tensor) -> torch.Tensor:
        """Compute mean keypoint error in millimeters"""
        # L2 distance between predicted and ground truth joints
        errors = torch.norm(pred_joints - gt_joints, dim=-1)  # [B, T, 21]
        mean_error = torch.mean(errors) * 1000  # Convert to mm
        return mean_error

    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'training_metrics': self.training_metrics,
            'config': self.config
        }

        # Save latest checkpoint
        checkpoint_path = self.output_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / 'best_checkpoint.pth'
            torch.save(checkpoint, best_path)
            print(f"  💾 Saved best checkpoint: {best_path}")

    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """Load model checkpoint"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.current_epoch = checkpoint['epoch']
            self.best_val_loss = checkpoint['best_val_loss']
            self.training_metrics = checkpoint['training_metrics']

            print(f"✅ Loaded checkpoint from epoch {self.current_epoch}")
            return True

        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            return False

    def train(self) -> bool:
        """Main training loop"""
        print("🚀 Starting HaWoR training...")

        # Setup all components
        if not self.setup_model():
            return False
        if not self.setup_loss_function():
            return False
        if not self.setup_optimizer_and_scheduler():
            return False
        if not self.setup_data_loaders():
            return False

        # Training configuration
        num_epochs = self.config.get('training', {}).get('max_epochs', 10)
        early_stopping_patience = self.config.get('training', {}).get('early_stopping_patience', 5)

        print(f"\n📊 Training Configuration:")
        print(f"  - Epochs: {num_epochs}")
        print(f"  - Learning rate: {self.optimizer.param_groups[0]['lr']}")
        print(f"  - Device: {self.device}")
        print(f"  - Output dir: {self.output_dir}")

        # Training loop
        patience_counter = 0

        for epoch in range(self.current_epoch + 1, num_epochs + 1):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            print(f"\n📈 Epoch {epoch}/{num_epochs}")
            print("=" * 50)

            # Train epoch
            train_metrics = self.train_epoch()

            # Validate epoch
            val_metrics = self.validate_epoch()

            # Update scheduler
            self.scheduler.step()

            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            epoch_time = time.time() - epoch_start_time

            # Log metrics
            self.training_metrics['train_loss'].append(train_metrics['train_loss'])
            self.training_metrics['val_loss'].append(val_metrics['val_loss'])
            self.training_metrics['train_keypoint_error'].append(train_metrics['train_keypoint_error'])
            self.training_metrics['val_keypoint_error'].append(val_metrics['val_keypoint_error'])
            self.training_metrics['learning_rate'].append(train_metrics['learning_rate'])
            self.training_metrics['epoch_times'].append(epoch_time)

            # Print epoch summary
            print(f"\n📊 Epoch {epoch} Summary:")
            print(f"  🏋️  Train Loss: {train_metrics['train_loss']:.4f}")
            print(f"  🧪 Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"  📐 Train Keypoint Error: {train_metrics['train_keypoint_error']:.2f}mm")
            print(f"  📐 Val Keypoint Error: {val_metrics['val_keypoint_error']:.2f}mm")
            print(f"  📈 Learning Rate: {train_metrics['learning_rate']:.2e}")
            print(f"  ⏱️  Epoch Time: {epoch_time:.1f}s")

            # Check for best model
            is_best = val_metrics['val_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['val_loss']
                patience_counter = 0
                print(f"  🎉 New best validation loss: {self.best_val_loss:.4f}")
            else:
                patience_counter += 1

            # Save checkpoint
            self.save_checkpoint(is_best)

            # Generate visualizations
            if epoch % 2 == 0 or epoch == num_epochs:
                print("  🎨 Generating visualizations...")
                self.generate_visualizations()

            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\n⏹️  Early stopping triggered after {patience_counter} epochs without improvement")
                break

        print(f"\n🎉 Training completed!")
        print(f"  📊 Best validation loss: {self.best_val_loss:.4f}")
        print(f"  💾 Checkpoints saved to: {self.output_dir}")

        return True

    def generate_visualizations(self):
        """Generate training visualizations"""
        try:
            # Plot training curves
            self.visualizer.plot_training_metrics(
                self.training_metrics,
                "real_training_progress.png"
            )

            # Save training report
            report_data = {
                'current_epoch': self.current_epoch,
                'best_val_loss': self.best_val_loss,
                'metrics': self.training_metrics,
                'config': self.config
            }

            report_path = self.output_dir / "real_training_report.json"
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)

            print(f"    📈 Visualizations saved")

        except Exception as e:
            print(f"    ❌ Error generating visualizations: {e}")


def main():
    """Main training function"""
    import argparse

    parser = argparse.ArgumentParser(description="Real HaWoR Training")
    parser.add_argument('--config', type=str, default='arctic_training_config.yaml',
                        help='Path to training configuration')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')

    args = parser.parse_args()

    print("🤖 Real HaWoR Training Pipeline")
    print("=" * 50)

    # Initialize trainer
    trainer = RealHaWoRTrainer(args.config)

    # Resume from checkpoint if provided
    if args.resume and os.path.exists(args.resume):
        trainer.load_checkpoint(args.resume)

    # Start training
    success = trainer.train()

    if success:
        print("\n✅ Training completed successfully!")
    else:
        print("\n❌ Training failed!")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)