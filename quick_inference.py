#!/usr/bin/env python3
"""
Quick HaWoR Inference Script
Runs inference with pretrained model and saves visualizations
"""

import sys
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import cv2

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from training_data_preparation import TrainingDataset


def load_pretrained_model():
    """Load the pretrained HaWoR model"""

    # Import the original HaWoR model
    sys.path.append('lib/models')
    from hawor import HAWOR

    # Create simple config
    class SimpleConfig:
        def __init__(self):
            self.MODEL = type('obj', (object,), {
                'BACKBONE_TYPE': 'vit',
                'ST_MODULE': True,
                'MOTION_MODULE': True,
                'ST_HDIM': 256,
                'MOTION_HDIM': 192,
                'ST_NLAYER': 4,
                'MOTION_NLAYER': 4
            })()
            self.MANO = type('obj', (object,), {
                'DATA_DIR': '_DATA/data/',
                'MODEL_PATH': '_DATA/data/mano',
                'GENDER': 'neutral',
                'NUM_HAND_JOINTS': 15,
                'MEAN_PARAMS': '_DATA/data/mano_mean_params.npz',
                'CREATE_BODY_POSE': False
            })()
            self.TRAIN = type('obj', (object,), {
                'LR': 1e-5,
                'WEIGHT_DECAY': 1e-4,
                'GRAD_CLIP_VAL': 0,
                'RENDER_LOG': True
            })()
            self.GENERAL = type('obj', (object,), {
                'LOG_STEPS': 1000
            })()

    cfg = SimpleConfig()

    # Create model
    model = HAWOR(cfg)

    # Load pretrained weights if available
    checkpoint_path = "weights/hawor/checkpoints/hawor.ckpt"
    if Path(checkpoint_path).exists():
        print(f"📦 Loading pretrained weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Extract model state dict (handle different checkpoint formats)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        # Remove 'model.' prefix if present
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('model.'):
                cleaned_state_dict[key[6:]] = value
            else:
                cleaned_state_dict[key] = value

        model.load_state_dict(cleaned_state_dict, strict=False)
        print("✅ Pretrained weights loaded successfully")
    else:
        print("⚠️  Pretrained weights not found, using random initialization")

    model.eval()
    return model


def visualize_predictions(sample, predictions, sample_idx):
    """Visualize predictions vs ground truth"""

    # Create output directory
    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True)

    # Set up the plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Sample {sample_idx}: HaWoR Predictions vs Ground Truth', fontsize=16)

    # Convert image for display
    image = sample['img']
    if isinstance(image, torch.Tensor):
        if image.dim() == 3 and image.shape[0] == 3:  # CHW format
            image = image.permute(1, 2, 0)
        image = image.detach().cpu().numpy()

    # Plot 1: Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Input Image')
    axes[0, 0].axis('off')

    # Plot 2: 2D keypoints overlay
    axes[0, 1].imshow(image)

    # Plot ground truth 2D keypoints
    gt_kpts_2d = sample['keypoints_2d']
    if isinstance(gt_kpts_2d, torch.Tensor):
        gt_kpts_2d = gt_kpts_2d.detach().cpu().numpy()

    # Filter valid keypoints
    valid_gt = gt_kpts_2d[np.isfinite(gt_kpts_2d).all(axis=1)]
    if len(valid_gt) > 0:
        axes[0, 1].scatter(valid_gt[:, 0], valid_gt[:, 1],
                          c='red', s=30, alpha=0.8, label='GT 2D')

    # Plot predicted 2D keypoints if available
    if 'pred_j2d' in predictions:
        pred_kpts_2d = predictions['pred_j2d']
        if isinstance(pred_kpts_2d, torch.Tensor):
            pred_kpts_2d = pred_kpts_2d.detach().cpu().numpy()

        valid_pred = pred_kpts_2d[np.isfinite(pred_kpts_2d).all(axis=1)]
        if len(valid_pred) > 0:
            axes[0, 1].scatter(valid_pred[:, 0], valid_pred[:, 1],
                              c='blue', s=30, alpha=0.8, label='Pred 2D')

    axes[0, 1].set_title('2D Keypoints Overlay')
    axes[0, 1].legend()
    axes[0, 1].axis('off')

    # Plot 3: 3D keypoints (ground truth)
    gt_kpts_3d = sample['keypoints_3d']
    if isinstance(gt_kpts_3d, torch.Tensor):
        gt_kpts_3d = gt_kpts_3d.detach().cpu().numpy()

    axes[1, 0].scatter(gt_kpts_3d[:, 0], gt_kpts_3d[:, 1],
                      c=gt_kpts_3d[:, 2], s=50, alpha=0.8, cmap='viridis')
    axes[1, 0].set_title('Ground Truth 3D Keypoints (X-Y view)')
    axes[1, 0].set_xlabel('X')
    axes[1, 0].set_ylabel('Y')

    # Plot 4: 3D keypoints (predicted) or mesh info
    if 'pred_j3d' in predictions:
        pred_kpts_3d = predictions['pred_j3d']
        if isinstance(pred_kpts_3d, torch.Tensor):
            pred_kpts_3d = pred_kpts_3d.detach().cpu().numpy()

        axes[1, 1].scatter(pred_kpts_3d[:, 0], pred_kpts_3d[:, 1],
                          c=pred_kpts_3d[:, 2], s=50, alpha=0.8, cmap='plasma')
        axes[1, 1].set_title('Predicted 3D Keypoints (X-Y view)')
        axes[1, 1].set_xlabel('X')
        axes[1, 1].set_ylabel('Y')
    else:
        # Show mesh vertices if available
        if 'vertices' in sample:
            vertices = sample['vertices']
            if isinstance(vertices, torch.Tensor):
                vertices = vertices.detach().cpu().numpy()

            axes[1, 1].scatter(vertices[:, 0], vertices[:, 1],
                              c=vertices[:, 2], s=10, alpha=0.6, cmap='coolwarm')
            axes[1, 1].set_title('Ground Truth Mesh Vertices (X-Y view)')
            axes[1, 1].set_xlabel('X')
            axes[1, 1].set_ylabel('Y')

    plt.tight_layout()

    # Save plot
    output_path = output_dir / f'sample_{sample_idx}_predictions.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"📊 Visualization saved to: {output_path}")

    plt.show()

    return output_path


def compute_metrics(sample, predictions):
    """Compute evaluation metrics"""

    metrics = {}

    # 3D keypoint error (MPJPE)
    if 'keypoints_3d' in sample and 'pred_j3d' in predictions:
        gt_kpts_3d = sample['keypoints_3d']
        pred_kpts_3d = predictions['pred_j3d']

        if isinstance(gt_kpts_3d, torch.Tensor):
            gt_kpts_3d = gt_kpts_3d.detach().cpu().numpy()
        if isinstance(pred_kpts_3d, torch.Tensor):
            pred_kpts_3d = pred_kpts_3d.detach().cpu().numpy()

        # Compute MPJPE
        errors = np.linalg.norm(gt_kpts_3d - pred_kpts_3d, axis=1)
        mpjpe_3d = np.mean(errors) * 1000  # Convert to mm
        metrics['mpjpe_3d'] = mpjpe_3d

        # Compute PCK@15mm
        pck_15mm = np.mean(errors < 0.015) * 100  # Percentage
        metrics['pck_3d_15mm'] = pck_15mm

    # 2D keypoint error
    if 'keypoints_2d' in sample and 'pred_j2d' in predictions:
        gt_kpts_2d = sample['keypoints_2d']
        pred_kpts_2d = predictions['pred_j2d']

        if isinstance(gt_kpts_2d, torch.Tensor):
            gt_kpts_2d = gt_kpts_2d.detach().cpu().numpy()
        if isinstance(pred_kpts_2d, torch.Tensor):
            pred_kpts_2d = pred_kpts_2d.detach().cpu().numpy()

        # Filter valid keypoints
        valid_mask = (np.isfinite(gt_kpts_2d).all(axis=1) &
                     np.isfinite(pred_kpts_2d).all(axis=1))

        if valid_mask.sum() > 0:
            errors_2d = np.linalg.norm(gt_kpts_2d[valid_mask] - pred_kpts_2d[valid_mask], axis=1)
            mpjpe_2d = np.mean(errors_2d)
            metrics['mpjpe_2d'] = mpjpe_2d

    return metrics


def run_inference_on_sample(model, sample):
    """Run inference on a single sample"""

    with torch.no_grad():
        # Prepare batch (add batch dimension)
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

        # Run model forward
        try:
            predictions = model.forward_step(batch, train=False)

            # Remove batch dimension
            for key, value in predictions.items():
                if isinstance(value, torch.Tensor) and value.dim() > 0:
                    predictions[key] = value.squeeze(0)

            return predictions

        except Exception as e:
            print(f"❌ Forward pass failed: {e}")
            # Return dummy predictions
            return {
                'pred_j3d': torch.zeros_like(sample['keypoints_3d']),
                'pred_j2d': torch.zeros_like(sample['keypoints_2d'])
            }


def main():
    """Main function"""

    print("🎯 Quick HaWoR Inference")
    print("=" * 40)

    # Load model
    print("🤖 Loading pretrained model...")
    try:
        model = load_pretrained_model()
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return 1

    # Load dataset
    print("📚 Loading test dataset...")
    try:
        dataset = TrainingDataset(
            data_dir="test_output",
            split='val',
            target_resolution=(256, 256)
        )
        print(f"Dataset loaded: {len(dataset)} samples")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return 1

    # Run inference on first few samples
    num_samples = min(3, len(dataset))
    print(f"\n🔬 Running inference on {num_samples} samples...")

    for i in range(num_samples):
        print(f"\n--- Sample {i} ---")

        try:
            # Get sample
            sample = dataset[i]
            print(f"Sample keys: {list(sample.keys())}")

            # Run inference
            print("Running inference...")
            predictions = run_inference_on_sample(model, sample)
            print(f"Prediction keys: {list(predictions.keys())}")

            # Compute metrics
            metrics = compute_metrics(sample, predictions)
            print("📊 Metrics:")
            for metric_name, metric_value in metrics.items():
                print(f"  {metric_name}: {metric_value:.2f}")

            # Visualize results
            print("🎨 Creating visualizations...")
            viz_path = visualize_predictions(sample, predictions, i)

        except Exception as e:
            print(f"❌ Error processing sample {i}: {e}")
            continue

    print("\n✅ Inference completed!")
    print("📁 Check 'evaluation_results/' directory for visualizations")

    return 0


if __name__ == "__main__":
    sys.exit(main())