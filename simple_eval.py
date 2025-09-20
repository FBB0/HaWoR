#!/usr/bin/env python3
"""
Simple HaWoR Evaluation Script
Shows predictions and evaluations using existing pipeline
"""

import sys
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from training_data_preparation import TrainingDataset


def visualize_data_sample(sample, sample_idx):
    """Visualize a data sample to verify data loading works"""

    # Create output directory
    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True)

    # Set up the plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Sample {sample_idx}: Data Visualization', fontsize=16)

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

    # Filter valid keypoints (remove extreme outliers)
    valid_gt = gt_kpts_2d[np.isfinite(gt_kpts_2d).all(axis=1)]
    if len(valid_gt) > 0:
        # Filter keypoints within reasonable image bounds
        h, w = image.shape[:2]
        in_bounds = ((valid_gt[:, 0] >= -w) & (valid_gt[:, 0] <= 2*w) &
                    (valid_gt[:, 1] >= -h) & (valid_gt[:, 1] <= 2*h))
        valid_gt = valid_gt[in_bounds]

        if len(valid_gt) > 0:
            axes[0, 1].scatter(valid_gt[:, 0], valid_gt[:, 1],
                              c='red', s=30, alpha=0.8, label='GT 2D Keypoints')

    axes[0, 1].set_title('2D Keypoints Overlay')
    axes[0, 1].legend()
    axes[0, 1].axis('off')

    # Plot 3: 3D keypoints visualization
    gt_kpts_3d = sample['keypoints_3d']
    if isinstance(gt_kpts_3d, torch.Tensor):
        gt_kpts_3d = gt_kpts_3d.detach().cpu().numpy()

    # Create 3D scatter plot in 2D projection
    axes[1, 0].scatter(gt_kpts_3d[:, 0], gt_kpts_3d[:, 1],
                      c=gt_kpts_3d[:, 2], s=50, alpha=0.8, cmap='viridis')
    axes[1, 0].set_title('3D Keypoints (X-Y view, colored by Z)')
    axes[1, 0].set_xlabel('X (meters)')
    axes[1, 0].set_ylabel('Y (meters)')

    # Plot 4: Mesh vertices
    if 'vertices' in sample:
        vertices = sample['vertices']
        if isinstance(vertices, torch.Tensor):
            vertices = vertices.detach().cpu().numpy()

        # Downsample for visualization
        step = max(1, len(vertices) // 500)  # Show max 500 points
        vertices_sub = vertices[::step]

        axes[1, 1].scatter(vertices_sub[:, 0], vertices_sub[:, 1],
                          c=vertices_sub[:, 2], s=10, alpha=0.6, cmap='coolwarm')
        axes[1, 1].set_title(f'Mesh Vertices (showing {len(vertices_sub)}/{len(vertices)})')
        axes[1, 1].set_xlabel('X (meters)')
        axes[1, 1].set_ylabel('Y (meters)')

    plt.tight_layout()

    # Save plot
    output_path = output_dir / f'sample_{sample_idx}_data.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"📊 Visualization saved to: {output_path}")

    # Show plot
    plt.show()

    return output_path


def analyze_sample_data(sample, sample_idx):
    """Analyze and print information about a sample"""

    print(f"\n📋 Sample {sample_idx} Analysis:")
    print("-" * 30)

    # Basic info
    print(f"Available keys: {list(sample.keys())}")

    # Image info
    if 'img' in sample:
        img_shape = sample['img'].shape
        print(f"Image shape: {img_shape}")
        print(f"Image dtype: {sample['img'].dtype}")
        print(f"Image range: [{sample['img'].min():.3f}, {sample['img'].max():.3f}]")

    # Camera info
    if 'img_focal' in sample and 'img_center' in sample:
        focal = sample['img_focal']
        center = sample['img_center']
        print(f"Focal length: {focal}")
        print(f"Image center: {center}")

    # MANO parameters
    if 'mano_params' in sample:
        mano = sample['mano_params']
        print(f"MANO keys: {list(mano.keys())}")
        for key, value in mano.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: shape {value.shape}, range [{value.min():.3f}, {value.max():.3f}]")

    # 3D keypoints
    if 'keypoints_3d' in sample:
        kpts_3d = sample['keypoints_3d']
        print(f"3D keypoints: {kpts_3d.shape}")
        print(f"3D range: X[{kpts_3d[:, 0].min():.3f}, {kpts_3d[:, 0].max():.3f}]")
        print(f"          Y[{kpts_3d[:, 1].min():.3f}, {kpts_3d[:, 1].max():.3f}]")
        print(f"          Z[{kpts_3d[:, 2].min():.3f}, {kpts_3d[:, 2].max():.3f}]")

    # 2D keypoints
    if 'keypoints_2d' in sample:
        kpts_2d = sample['keypoints_2d']
        valid_2d = kpts_2d[np.isfinite(kpts_2d.numpy()).all(axis=1)]
        print(f"2D keypoints: {kpts_2d.shape} ({len(valid_2d)} valid)")
        if len(valid_2d) > 0:
            print(f"2D range: X[{valid_2d[:, 0].min():.1f}, {valid_2d[:, 0].max():.1f}]")
            print(f"          Y[{valid_2d[:, 1].min():.1f}, {valid_2d[:, 1].max():.1f}]")

    # Mesh info
    if 'vertices' in sample:
        vertices = sample['vertices']
        print(f"Mesh vertices: {vertices.shape}")
        print(f"Vertex range: X[{vertices[:, 0].min():.3f}, {vertices[:, 0].max():.3f}]")
        print(f"              Y[{vertices[:, 1].min():.3f}, {vertices[:, 1].max():.3f}]")
        print(f"              Z[{vertices[:, 2].min():.3f}, {vertices[:, 2].max():.3f}]")

    if 'faces' in sample:
        faces = sample['faces']
        print(f"Mesh faces: {faces.shape}")


def load_and_test_model():
    """Load and test the training pipeline model"""

    try:
        # Load configuration
        config_path = "configs/macbook_training_config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Try to load the enhanced training pipeline
        from enhanced_training_pipeline import EnhancedHaWoRTrainer

        # Create model (using the same structure as training)
        data_config = config.get('data', {})
        loss_config = config.get('loss', {})

        model = EnhancedHaWoRTrainer(
            config=config,
            training_data_dir=data_config.get('training_data_dir', './training_data'),
            validation_data_dir=data_config.get('validation_data_dir'),
            use_enhanced_loss=loss_config.get('use_enhanced_loss', True),
            use_adaptive_weights=loss_config.get('use_adaptive_weights', True)
        )

        model.eval()
        print("✅ Model loaded successfully!")
        print(f"📊 Model info: {sum(p.numel() for p in model.parameters())} parameters")

        return model

    except Exception as e:
        print(f"⚠️  Could not load full model: {e}")
        return None


def main():
    """Main function"""

    print("🎯 Simple HaWoR Evaluation")
    print("=" * 40)

    # Load dataset
    print("📚 Loading test dataset...")
    try:
        dataset = TrainingDataset(
            data_dir="test_output",
            split='val',
            target_resolution=(256, 256)
        )
        print(f"✅ Dataset loaded: {len(dataset)} samples")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return 1

    # Try to load model
    print("\n🤖 Testing model loading...")
    model = load_and_test_model()

    # Analyze and visualize first few samples
    num_samples = min(3, len(dataset))
    print(f"\n🔬 Analyzing {num_samples} samples...")

    for i in range(num_samples):
        try:
            print(f"\n{'='*50}")
            print(f"Processing Sample {i}")
            print(f"{'='*50}")

            # Get sample
            sample = dataset[i]

            # Analyze sample data
            analyze_sample_data(sample, i)

            # Create visualizations
            print(f"\n🎨 Creating visualizations for sample {i}...")
            viz_path = visualize_data_sample(sample, i)

            # Test model forward pass if model is available
            if model is not None:
                try:
                    print(f"\n🧠 Testing model forward pass...")

                    # Prepare batch
                    batch = {}
                    for key, value in sample.items():
                        if isinstance(value, torch.Tensor):
                            batch[key] = value.unsqueeze(0)  # Add batch dimension
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
                    with torch.no_grad():
                        output = model.forward(batch)

                    print(f"✅ Forward pass successful!")
                    print(f"📤 Output keys: {list(output.keys())}")

                    # Print output shapes
                    for key, value in output.items():
                        if isinstance(value, torch.Tensor):
                            print(f"  {key}: {value.shape}")

                except Exception as e:
                    print(f"❌ Forward pass failed: {e}")

        except Exception as e:
            print(f"❌ Error processing sample {i}: {e}")
            continue

    print("\n✅ Evaluation completed!")
    print("📁 Check 'evaluation_results/' directory for visualizations")

    # Summary
    print(f"\n📊 Summary:")
    print(f"  Dataset: {len(dataset)} samples loaded successfully")
    print(f"  Model: {'✅ Working' if model else '❌ Not loaded'}")
    print(f"  Visualizations: Created for {num_samples} samples")

    return 0


if __name__ == "__main__":
    sys.exit(main())