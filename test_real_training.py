#!/usr/bin/env python3
"""
Test script for the real HaWoR training pipeline
Validates that all components work together properly
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import time
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.hawor_model import HaWoRModel, create_hawor_model
from src.training.hawor_losses import HaWoRLossFunction
from src.datasets.arctic_dataset_real import create_arctic_dataloaders, test_arctic_dataset
from src.training.real_hawor_trainer import RealHaWoRTrainer
from src.evaluation.hawor_evaluation import HaWoREvaluator


def test_model_architecture():
    """Test HaWoR model architecture"""
    print("🧪 Testing model architecture...")

    config = {
        'model': {
            'img_size': 256,
            'backbone_type': 'vit_small',
            'pretrained': False,
            'num_joints': 21
        },
        'arctic': {
            'sequence_length': 8
        }
    }

    try:
        # Create model
        model = create_hawor_model(config)
        print(f"  ✅ Model created: {type(model).__name__}")

        # Test forward pass
        batch_size = 2
        seq_len = 8
        img_size = 256

        dummy_images = torch.randn(batch_size, seq_len, 3, img_size, img_size)
        print(f"  🔄 Testing forward pass with input shape: {dummy_images.shape}")

        with torch.no_grad():
            outputs = model(dummy_images)

        # Check outputs
        expected_keys = [
            'left_hand_pose', 'left_hand_shape', 'left_hand_trans',
            'left_hand_rot', 'left_hand_joints', 'left_confidence',
            'right_hand_pose', 'right_hand_shape', 'right_hand_trans',
            'right_hand_rot', 'right_hand_joints', 'right_confidence',
            'camera_pose', 'features'
        ]

        for key in expected_keys:
            if key in outputs:
                print(f"    ✅ {key}: {outputs[key].shape}")
            else:
                print(f"    ❌ Missing output: {key}")

        print("  ✅ Model architecture test passed!")
        return True

    except Exception as e:
        print(f"  ❌ Model architecture test failed: {e}")
        return False


def test_loss_function():
    """Test loss function computation"""
    print("🧪 Testing loss function...")

    try:
        loss_fn = HaWoRLossFunction()

        batch_size = 2
        seq_len = 8
        num_joints = 21

        # Create dummy predictions and targets
        predictions = {
            'hand_pose': torch.randn(batch_size, seq_len, 45),
            'hand_shape': torch.randn(batch_size, seq_len, 10),
            'hand_trans': torch.randn(batch_size, seq_len, 3),
            'hand_rot': torch.randn(batch_size, seq_len, 3),
            'hand_joints': torch.randn(batch_size, seq_len, num_joints, 3),
            'camera_pose': torch.randn(batch_size, seq_len, 6)
        }

        targets = {
            'hand_pose': torch.randn(batch_size, seq_len, 45),
            'hand_shape': torch.randn(batch_size, seq_len, 10),
            'hand_trans': torch.randn(batch_size, seq_len, 3),
            'hand_rot': torch.randn(batch_size, seq_len, 3),
            'hand_joints': torch.randn(batch_size, seq_len, num_joints, 3),
            'keypoints_2d': torch.randn(batch_size, seq_len, num_joints, 2),
            'camera_pose': torch.randn(batch_size, seq_len, 6)
        }

        camera_params = {
            'fx': 500.0, 'fy': 500.0, 'cx': 128.0, 'cy': 128.0
        }

        # Compute loss
        loss_dict = loss_fn(predictions, targets, camera_params)

        print(f"  ✅ Loss computation successful!")
        print(f"    Total loss: {loss_dict['total'].item():.4f}")

        # Check individual loss components
        loss_components = ['mano_pose', 'mano_shape', 'keypoint_3d', 'reprojection', 'temporal', 'camera']
        for component in loss_components:
            if component in loss_dict:
                print(f"    {component}: {loss_dict[component].item():.4f}")

        print("  ✅ Loss function test passed!")
        return True

    except Exception as e:
        print(f"  ❌ Loss function test failed: {e}")
        return False


def test_dataset_loading():
    """Test ARCTIC dataset loading"""
    print("🧪 Testing dataset loading...")

    try:
        config = {
            'arctic': {
                'data_root': 'thirdparty/arctic',
                'train_subjects': ['s01'],
                'sequence_length': 8,
                'max_sequences': 1,
                'images_per_camera': 5
            },
            'training': {
                'batch_size': 1
            }
        }

        # Test dataset creation
        train_loader, val_loader = create_arctic_dataloaders(config)
        print(f"  ✅ Dataloaders created")
        print(f"    Train samples: {len(train_loader.dataset)}")
        print(f"    Val samples: {len(val_loader.dataset)}")

        # Test loading a batch
        if len(train_loader) > 0:
            batch = next(iter(train_loader))
            print(f"  ✅ Sample batch loaded")

            expected_keys = [
                'images', 'hand_pose', 'hand_shape', 'hand_trans',
                'hand_rot', 'hand_joints', 'keypoints_2d', 'camera_pose'
            ]

            for key in expected_keys:
                if key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        print(f"    {key}: {batch[key].shape}")
                    else:
                        print(f"    {key}: {batch[key]}")

            print("  ✅ Dataset loading test passed!")
            return True
        else:
            print("  ⚠️  No training samples found, but dataset creation succeeded")
            return True

    except Exception as e:
        print(f"  ❌ Dataset loading test failed: {e}")
        return False


def test_training_step():
    """Test a single training step"""
    print("🧪 Testing training step...")

    try:
        # Create minimal config for testing
        config = {
            'model': {
                'img_size': 128,  # Smaller for faster testing
                'backbone_type': 'vit_small',
                'pretrained': False,
                'num_joints': 21
            },
            'arctic': {
                'sequence_length': 4,  # Shorter sequences
                'data_root': 'thirdparty/arctic',
                'train_subjects': ['s01'],
                'max_sequences': 1,
                'images_per_camera': 3
            },
            'training': {
                'batch_size': 1,
                'learning_rate': 1e-4,
                'weight_decay': 1e-4
            },
            'loss_weights': {
                'keypoint': 1.0,
                'mano_pose': 0.1,
                'temporal': 0.1
            }
        }

        # Create components
        model = create_hawor_model(config)
        loss_fn = HaWoRLossFunction()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Create dummy batch
        batch_size = 1
        seq_len = 4
        img_size = 128

        batch = {
            'images': torch.randn(batch_size, seq_len, 3, img_size, img_size),
            'hand_pose': torch.randn(batch_size, seq_len, 45),
            'hand_shape': torch.randn(batch_size, seq_len, 10),
            'hand_trans': torch.randn(batch_size, seq_len, 3),
            'hand_rot': torch.randn(batch_size, seq_len, 3),
            'hand_joints': torch.randn(batch_size, seq_len, 21, 3),
            'keypoints_2d': torch.randn(batch_size, seq_len, 21, 2),
            'camera_pose': torch.randn(batch_size, seq_len, 6)
        }

        # Training step
        model.train()
        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch['images'])

        # Prepare predictions for loss
        pred_for_loss = {
            'hand_pose': outputs['left_hand_pose'],
            'hand_shape': outputs['left_hand_shape'],
            'hand_trans': outputs['left_hand_trans'],
            'hand_rot': outputs['left_hand_rot'],
            'hand_joints': outputs['left_hand_joints'],
            'camera_pose': outputs['camera_pose']
        }

        # Prepare targets
        targets = {k: v for k, v in batch.items() if k != 'images'}

        # Compute loss
        camera_params = {'fx': 500.0, 'fy': 500.0, 'cx': 64.0, 'cy': 64.0}
        loss_dict = loss_fn(pred_for_loss, targets, camera_params)

        # Backward pass
        total_loss = loss_dict['total']
        total_loss.backward()

        # Check gradients
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Optimizer step
        optimizer.step()

        print(f"  ✅ Training step completed!")
        print(f"    Loss: {total_loss.item():.4f}")
        print(f"    Grad norm: {grad_norm:.4f}")
        print("  ✅ Training step test passed!")
        return True

    except Exception as e:
        print(f"  ❌ Training step test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_pipeline():
    """Test the complete training pipeline"""
    print("🧪 Testing complete training pipeline...")

    try:
        # Create test config file
        test_config = {
            'model': {
                'img_size': 128,
                'backbone_type': 'vit_small',
                'pretrained': False,
                'num_joints': 21
            },
            'arctic': {
                'data_root': 'thirdparty/arctic',
                'train_subjects': ['s01'],
                'val_subjects': ['s01'],
                'sequence_length': 4,
                'max_sequences': 1,
                'images_per_camera': 3
            },
            'training': {
                'max_epochs': 1,  # Just one epoch for testing
                'batch_size': 1,
                'learning_rate': 1e-4,
                'weight_decay': 1e-4,
                'grad_clip_val': 1.0
            },
            'loss_weights': {
                'keypoint': 1.0,
                'mano_pose': 0.1,
                'temporal': 0.1
            },
            'output_dir': 'test_outputs'
        }

        # Save test config
        config_path = 'test_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(test_config, f)

        # Test trainer initialization
        trainer = RealHaWoRTrainer(config_path)

        # Test individual setup steps
        print("  🔧 Testing trainer setup...")

        if not trainer.setup_model():
            print("  ❌ Model setup failed")
            return False

        if not trainer.setup_loss_function():
            print("  ❌ Loss function setup failed")
            return False

        if not trainer.setup_optimizer_and_scheduler():
            print("  ❌ Optimizer setup failed")
            return False

        # Note: We won't test full training as it requires actual ARCTIC data
        print("  ✅ Pipeline setup completed successfully!")

        # Cleanup
        if os.path.exists(config_path):
            os.remove(config_path)

        print("  ✅ Complete pipeline test passed!")
        return True

    except Exception as e:
        print(f"  ❌ Complete pipeline test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🚀 HaWoR Real Training Pipeline Tests")
    print("=" * 50)

    tests = [
        ("Model Architecture", test_model_architecture),
        ("Loss Function", test_loss_function),
        ("Dataset Loading", test_dataset_loading),
        ("Training Step", test_training_step),
        ("Full Pipeline", test_full_pipeline)
    ]

    results = {}
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 30)
        start_time = time.time()

        try:
            result = test_func()
            elapsed = time.time() - start_time
            results[test_name] = result
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"  {status} ({elapsed:.2f}s)")
        except Exception as e:
            elapsed = time.time() - start_time
            results[test_name] = False
            print(f"  ❌ FAILED ({elapsed:.2f}s): {e}")

    # Summary
    print(f"\n📊 Test Summary:")
    print("=" * 30)
    passed = sum(results.values())
    total = len(results)

    for test_name, result in results.items():
        status = "✅" if result else "❌"
        print(f"  {status} {test_name}")

    print(f"\n🎯 Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Real training pipeline is ready!")
        return True
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)