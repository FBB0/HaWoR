#!/usr/bin/env python3
"""
Test script for MANO hand visualization system
Demonstrates the comprehensive validation visualization
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.visualization.hand_visualization import MANOHandVisualizer


def test_visualization():
    """Test the MANO hand visualization system"""

    print("🎨 Testing MANO Hand Visualization System")
    print("=" * 50)

    # Initialize visualizer
    visualizer = MANOHandVisualizer("test_visualization_output")

    # Create synthetic test data
    batch_size = 1
    seq_len = 4
    img_size = 256

    # Synthetic input data
    images = torch.rand(batch_size, seq_len, 3, img_size, img_size)

    # Create realistic hand poses (center around reasonable values)
    pred_joints = torch.randn(batch_size, seq_len, 21, 3) * 0.05  # 5cm variation
    gt_joints = torch.randn(batch_size, seq_len, 21, 3) * 0.05

    # Add some structure to make it look like hands
    for b in range(batch_size):
        for t in range(seq_len):
            # Wrist at origin
            pred_joints[b, t, 0] = torch.tensor([0.0, 0.0, -0.3])
            gt_joints[b, t, 0] = torch.tensor([0.0, 0.0, -0.3])

            # Fingers extending from wrist
            for i in range(1, 21):
                finger_id = (i - 1) // 4
                joint_in_finger = (i - 1) % 4

                # Base position for each finger
                finger_base_x = (finger_id - 2) * 0.03  # Spread fingers
                finger_base_y = 0.0
                finger_base_z = -0.25

                # Extend along finger
                extension = joint_in_finger * 0.025

                pred_joints[b, t, i] = torch.tensor([
                    finger_base_x + extension * 0.1,
                    finger_base_y + extension,
                    finger_base_z
                ]) + torch.randn(3) * 0.01

                gt_joints[b, t, i] = torch.tensor([
                    finger_base_x + extension * 0.1,
                    finger_base_y + extension,
                    finger_base_z
                ]) + torch.randn(3) * 0.005  # GT is slightly more accurate

    # Create MANO pose parameters
    pred_pose = torch.randn(batch_size, seq_len, 45) * 0.2
    gt_pose = torch.randn(batch_size, seq_len, 45) * 0.2

    # Create test metrics
    test_metrics = {
        'train_loss': 850000.0,
        'val_loss': 620000.0,
        'lr': 2.5e-5,
        'epoch': 5,
        'batch_idx': 0
    }

    print("🖼️ Creating comprehensive validation visualization...")

    # Test the visualization
    try:
        visualizer.visualize_validation_batch(
            images=images,
            pred_joints=pred_joints,
            gt_joints=gt_joints,
            pred_pose=pred_pose,
            gt_pose=gt_pose,
            batch_idx=0,
            epoch=5,
            metrics=test_metrics
        )

        print("✅ Validation visualization created successfully!")

    except Exception as e:
        print(f"❌ Error creating validation visualization: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test training progress visualization
    print("\n📈 Creating training progress visualization...")

    # Create synthetic training metrics
    epoch_metrics = []
    for epoch in range(1, 6):
        epoch_metrics.append({
            'train_loss': 1000000 * (0.8 ** epoch) + np.random.normal(0, 50000),
            'val_loss': 900000 * (0.85 ** epoch) + np.random.normal(0, 40000),
            'lr': 5e-5 * (0.9 ** epoch)
        })

    try:
        visualizer.create_training_progress_visualization(epoch_metrics, epoch=5)
        print("✅ Training progress visualization created successfully!")

    except Exception as e:
        print(f"❌ Error creating training progress visualization: {e}")
        import traceback
        traceback.print_exc()
        return False

    print(f"\n🎉 Visualization test completed!")
    print(f"📁 Check output directory: {visualizer.output_dir}")
    print(f"📊 Visualizations demonstrate:")
    print(f"  - Input images with 2D joint projections")
    print(f"  - 3D hand pose comparisons (predicted vs ground truth)")
    print(f"  - Per-joint error analysis")
    print(f"  - MANO pose parameter comparisons")
    print(f"  - Multiple viewing angles")
    print(f"  - Finger-wise error breakdown")
    print(f"  - Comprehensive validation metrics")
    print(f"  - Training progress charts")

    return True


if __name__ == "__main__":
    success = test_visualization()
    if success:
        print("\n✅ Visualization system test passed!")
    else:
        print("\n❌ Visualization system test failed!")
    sys.exit(0 if success else 1)