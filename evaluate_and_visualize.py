#!/usr/bin/env python3
"""
HaWoR Evaluation and Visualization Script
Runs inference and visualizes mesh predictions
"""

import sys
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import cv2

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from training_data_preparation import TrainingDataset
from enhanced_training_pipeline import EnhancedHaWoRTrainer
import trimesh


def load_model(checkpoint_path, config_path):
    """Load trained model from checkpoint"""

    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Create model
    model = EnhancedHaWoRTrainer(
        config=config,
        training_data_dir=config['data']['training_data_dir'],
        validation_data_dir=config['data']['validation_data_dir'],
        use_enhanced_loss=config['loss']['use_enhanced_loss'],
        use_adaptive_weights=config['loss']['use_adaptive_weights']
    )

    # Load checkpoint if provided
    if checkpoint_path and Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        print(f"✅ Loaded checkpoint from {checkpoint_path}")
    else:
        print("⚠️  Using randomly initialized model (no checkpoint provided)")

    model.eval()
    return model


def run_inference(model, sample):
    """Run inference on a single sample"""

    with torch.no_grad():
        # Add batch dimension
        batch = {}
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.unsqueeze(0)
            elif isinstance(value, dict):
                batch[key] = {}
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, torch.Tensor):
                        batch[key][sub_key] = sub_value.unsqueeze(0)
                    else:
                        batch[key][sub_key] = sub_value
            else:
                batch[key] = value

        # Run forward pass
        output = model.forward(batch)

        # Remove batch dimension from outputs
        for key, value in output.items():
            if isinstance(value, torch.Tensor) and value.dim() > 0:
                output[key] = value.squeeze(0)

        return output


def visualize_mesh(vertices, faces, title="Hand Mesh"):
    """Visualize 3D mesh using trimesh"""

    try:
        # Convert to numpy if needed
        if isinstance(vertices, torch.Tensor):
            vertices = vertices.detach().cpu().numpy()
        if isinstance(faces, torch.Tensor):
            faces = faces.detach().cpu().numpy()

        # Create mesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        # Show mesh
        mesh.show(title=title)

        return mesh

    except Exception as e:
        print(f"❌ Error visualizing mesh: {e}")
        return None


def visualize_2d_overlay(image, keypoints_2d, pred_keypoints_2d=None, title="2D Keypoint Overlay"):
    """Visualize 2D keypoints overlaid on image"""

    plt.figure(figsize=(12, 8))

    # Convert image to numpy if needed
    if isinstance(image, torch.Tensor):
        if image.dim() == 3 and image.shape[0] == 3:  # CHW format
            image = image.permute(1, 2, 0)
        image = image.detach().cpu().numpy()

    # Ensure image is in [0, 1] range
    if image.max() > 1.0:
        image = image / 255.0

    plt.imshow(image)

    # Convert keypoints to numpy if needed
    if isinstance(keypoints_2d, torch.Tensor):
        keypoints_2d = keypoints_2d.detach().cpu().numpy()

    # Plot ground truth keypoints
    if keypoints_2d.shape[0] > 0:
        valid_kpts = keypoints_2d[np.isfinite(keypoints_2d).all(axis=1)]
        if len(valid_kpts) > 0:
            plt.scatter(valid_kpts[:, 0], valid_kpts[:, 1],
                       c='red', s=50, alpha=0.7, label='Ground Truth')

    # Plot predicted keypoints if provided
    if pred_keypoints_2d is not None:
        if isinstance(pred_keypoints_2d, torch.Tensor):
            pred_keypoints_2d = pred_keypoints_2d.detach().cpu().numpy()

        if pred_keypoints_2d.shape[0] > 0:
            valid_pred_kpts = pred_keypoints_2d[np.isfinite(pred_keypoints_2d).all(axis=1)]
            if len(valid_pred_kpts) > 0:
                plt.scatter(valid_pred_kpts[:, 0], valid_pred_kpts[:, 1],
                           c='blue', s=50, alpha=0.7, label='Predicted')

    plt.title(title)
    plt.legend()
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def evaluate_sample(model, dataset, idx):
    """Evaluate a single sample and visualize results"""

    print(f"\n🔍 Evaluating sample {idx}")

    # Get sample
    sample = dataset[idx]

    # Run inference
    print("Running inference...")
    output = run_inference(model, sample)

    # Print available outputs
    print("Available outputs:", list(output.keys()))

    # Visualize image with 2D keypoints
    if 'img' in sample and 'keypoints_2d' in sample:
        pred_kpts_2d = output.get('pred_keypoints_2d', output.get('pred_j2d', None))
        visualize_2d_overlay(
            sample['img'],
            sample['keypoints_2d'],
            pred_kpts_2d,
            f"Sample {idx}: 2D Keypoint Overlay"
        )

    # Visualize 3D mesh if available
    if 'vertices' in sample:
        print("Visualizing ground truth mesh...")
        gt_mesh = visualize_mesh(
            sample['vertices'],
            sample['faces'],
            f"Sample {idx}: Ground Truth Mesh"
        )

    # Visualize predicted mesh if available
    pred_vertices = output.get('pred_vertices', output.get('pred_mesh_vertices', None))
    if pred_vertices is not None:
        print("Visualizing predicted mesh...")
        faces = sample.get('faces', output.get('pred_faces', None))
        if faces is not None:
            pred_mesh = visualize_mesh(
                pred_vertices,
                faces,
                f"Sample {idx}: Predicted Mesh"
            )

    # Compute and display metrics
    print("\n📊 Evaluation Metrics:")

    # 3D keypoint error (MPJPE)
    if 'keypoints_3d' in sample:
        pred_kpts_3d = output.get('pred_keypoints_3d', output.get('pred_j3d', None))
        if pred_kpts_3d is not None:
            gt_kpts_3d = sample['keypoints_3d']
            if isinstance(gt_kpts_3d, torch.Tensor):
                gt_kpts_3d = gt_kpts_3d.detach().cpu().numpy()
            if isinstance(pred_kpts_3d, torch.Tensor):
                pred_kpts_3d = pred_kpts_3d.detach().cpu().numpy()

            # Compute MPJPE (Mean Per Joint Position Error)
            errors = np.linalg.norm(gt_kpts_3d - pred_kpts_3d, axis=1)
            mpjpe = np.mean(errors)
            print(f"  MPJPE (3D): {mpjpe:.2f} mm")

    # 2D keypoint error
    if 'keypoints_2d' in sample and pred_kpts_2d is not None:
        gt_kpts_2d = sample['keypoints_2d']
        if isinstance(gt_kpts_2d, torch.Tensor):
            gt_kpts_2d = gt_kpts_2d.detach().cpu().numpy()
        if isinstance(pred_kpts_2d, torch.Tensor):
            pred_kpts_2d = pred_kpts_2d.detach().cpu().numpy()

        # Compute 2D error for valid keypoints
        valid_mask = np.isfinite(gt_kpts_2d).all(axis=1) & np.isfinite(pred_kpts_2d).all(axis=1)
        if valid_mask.sum() > 0:
            errors_2d = np.linalg.norm(gt_kpts_2d[valid_mask] - pred_kpts_2d[valid_mask], axis=1)
            mpjpe_2d = np.mean(errors_2d)
            print(f"  MPJPE (2D): {mpjpe_2d:.2f} pixels")

    return output


def main():
    """Main evaluation function"""

    parser = argparse.ArgumentParser(description="HaWoR Evaluation and Visualization")
    parser.add_argument("--config", default="configs/macbook_training_config.json",
                       help="Path to training configuration file")
    parser.add_argument("--checkpoint", default=None,
                       help="Path to model checkpoint (optional)")
    parser.add_argument("--data-dir", default="test_output",
                       help="Path to test data directory")
    parser.add_argument("--sample-idx", type=int, default=0,
                       help="Index of sample to evaluate")
    parser.add_argument("--num-samples", type=int, default=5,
                       help="Number of samples to evaluate")

    args = parser.parse_args()

    print("🎯 HaWoR Evaluation and Visualization")
    print("=" * 50)

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        return 1

    print(f"📄 Using configuration: {config_path}")

    # Load model
    print("🤖 Loading model...")
    model = load_model(args.checkpoint, args.config)

    # Create dataset
    print("📚 Loading dataset...")
    dataset = TrainingDataset(
        data_dir=args.data_dir,
        split='val',  # Use validation split for evaluation
        target_resolution=(256, 256)
    )

    print(f"Dataset size: {len(dataset)} samples")

    # Evaluate samples
    print(f"\n🔬 Evaluating {args.num_samples} samples starting from index {args.sample_idx}")

    for i in range(args.num_samples):
        sample_idx = args.sample_idx + i
        if sample_idx >= len(dataset):
            print(f"⚠️  Sample index {sample_idx} out of range")
            break

        try:
            output = evaluate_sample(model, dataset, sample_idx)
        except Exception as e:
            print(f"❌ Error evaluating sample {sample_idx}: {e}")
            continue

    print("\n✅ Evaluation completed!")

    return 0


if __name__ == "__main__":
    sys.exit(main())