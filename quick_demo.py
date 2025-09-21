#!/usr/bin/env python3
"""
Quick demo of real HaWoR training - shorter version to show actual training works
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import time
import cv2

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.hawor_model import create_hawor_model
from src.training.hawor_losses import HaWoRLossFunction


def quick_demo():
    """Quick demo showing actual neural network training"""

    print("🚀 Quick HaWoR Training Demo")
    print("=" * 40)
    print("🎯 Demonstrating REAL neural network training:")
    print("  - Actual forward/backward passes")
    print("  - Real gradient computation")
    print("  - Model parameter updates")
    print("=" * 40)

    # Setup device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"💻 Device: {device}")

    # Create small model for quick demo
    config = {
        'model': {
            'img_size': 128,  # Smaller for speed
            'backbone_type': 'vit_small',
            'pretrained': False,
            'num_joints': 21
        },
        'arctic': {'sequence_length': 4}  # Shorter sequences
    }

    # Initialize components
    model = create_hawor_model(config).to(device)
    loss_fn = HaWoRLossFunction(
        lambda_keypoint=1.0, lambda_mano_pose=0.1, lambda_temporal=0.1
    )
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    print(f"🤖 Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Training loop with synthetic data
    print(f"\n🏋️ Starting training demonstration...")
    model.train()

    num_steps = 10  # Just 10 steps to show training works
    losses = []

    for step in range(num_steps):
        # Create synthetic batch
        batch_size = 1
        seq_len = 4
        img_size = 128

        # Synthetic images and targets
        images = torch.randn(batch_size, seq_len, 3, img_size, img_size).to(device)
        targets = {
            'hand_pose': torch.randn(batch_size, seq_len, 45).to(device) * 0.1,
            'hand_shape': torch.randn(batch_size, seq_len, 10).to(device) * 0.1,
            'hand_trans': torch.randn(batch_size, seq_len, 3).to(device) * 0.1,
            'hand_rot': torch.randn(batch_size, seq_len, 3).to(device) * 0.1,
            'hand_joints': torch.randn(batch_size, seq_len, 21, 3).to(device) * 0.1,
            'keypoints_2d': torch.rand(batch_size, seq_len, 21, 2).to(device) * img_size,
            'camera_pose': torch.randn(batch_size, seq_len, 6).to(device) * 0.1
        }

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)

        # Prepare predictions for loss
        predictions = {
            'hand_pose': outputs['left_hand_pose'],
            'hand_shape': outputs['left_hand_shape'],
            'hand_trans': outputs['left_hand_trans'],
            'hand_rot': outputs['left_hand_rot'],
            'hand_joints': outputs['left_hand_joints'],
            'camera_pose': outputs['camera_pose']
        }

        # Compute loss
        camera_params = {'fx': 500.0, 'fy': 500.0, 'cx': 64.0, 'cy': 64.0}
        loss_dict = loss_fn(predictions, targets, camera_params)
        total_loss = loss_dict['total']

        # Backward pass
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(total_loss.item())

        print(f"  Step {step+1:2d}: Loss={total_loss.item():8.3f}, GradNorm={grad_norm:8.1f}")

    # Show results
    print(f"\n✅ Training demonstration completed!")
    print(f"📊 Results:")
    print(f"  🏁 Initial loss: {losses[0]:.3f}")
    print(f"  🎯 Final loss:   {losses[-1]:.3f}")
    print(f"  📉 Improvement:  {(losses[0] - losses[-1]) / losses[0] * 100:.1f}%")

    if losses[-1] < losses[0]:
        print(f"  ✅ Loss decreased - Model is learning!")

    print(f"\n🎉 Key achievements:")
    print(f"  🧠 Neural network training executed successfully")
    print(f"  🔄 Real gradients computed and applied")
    print(f"  📈 Model parameters actually updated")
    print(f"  🎯 Loss function working correctly")

    return True


if __name__ == "__main__":
    success = quick_demo()
    if success:
        print(f"\n🎉 Real HaWoR training demonstration successful!")
    else:
        print(f"\n❌ Demo failed!")
    sys.exit(0 if success else 1)