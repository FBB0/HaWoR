#!/usr/bin/env python3
"""
View Ground Truth Hand Meshes
Visualizes the ground truth hand meshes from the dataset
"""

import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from training_data_preparation import TrainingDataset


def plot_3d_mesh(vertices, faces, title="Hand Mesh", save_path=None):
    """Plot 3D mesh using matplotlib"""

    if isinstance(vertices, torch.Tensor):
        vertices = vertices.detach().cpu().numpy()
    if isinstance(faces, torch.Tensor):
        faces = faces.detach().cpu().numpy()

    fig = plt.figure(figsize=(15, 5))

    # Multiple views
    views = [
        (30, 45, "3/4 View"),
        (0, 0, "Front View"),
        (0, 90, "Side View")
    ]

    for i, (elev, azim, view_title) in enumerate(views):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')

        # Plot mesh surface using triangulation
        ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                       triangles=faces, alpha=0.8, cmap='viridis')

        # Set equal aspect ratio
        max_range = np.array([vertices[:, 0].max()-vertices[:, 0].min(),
                             vertices[:, 1].max()-vertices[:, 1].min(),
                             vertices[:, 2].max()-vertices[:, 2].min()]).max() / 2.0

        mid_x = (vertices[:, 0].max()+vertices[:, 0].min()) * 0.5
        mid_y = (vertices[:, 1].max()+vertices[:, 1].min()) * 0.5
        mid_z = (vertices[:, 2].max()+vertices[:, 2].min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"{view_title}")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"🖼️  Mesh visualization saved to: {save_path}")

    plt.show()


def plot_keypoints_on_mesh(vertices, faces, keypoints_3d, title="Mesh with Keypoints", save_path=None):
    """Plot mesh with 3D keypoints overlaid"""

    if isinstance(vertices, torch.Tensor):
        vertices = vertices.detach().cpu().numpy()
    if isinstance(faces, torch.Tensor):
        faces = faces.detach().cpu().numpy()
    if isinstance(keypoints_3d, torch.Tensor):
        keypoints_3d = keypoints_3d.detach().cpu().numpy()

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot mesh surface
    ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                   triangles=faces, alpha=0.6, cmap='viridis', linewidth=0)

    # Plot keypoints
    ax.scatter(keypoints_3d[:, 0], keypoints_3d[:, 1], keypoints_3d[:, 2],
              c='red', s=100, alpha=0.9, marker='o')

    # Add keypoint labels
    for i, kpt in enumerate(keypoints_3d):
        ax.text(kpt[0], kpt[1], kpt[2], str(i), fontsize=8)

    # Set equal aspect ratio
    max_range = np.array([vertices[:, 0].max()-vertices[:, 0].min(),
                         vertices[:, 1].max()-vertices[:, 1].min(),
                         vertices[:, 2].max()-vertices[:, 2].min()]).max() / 2.0

    mid_x = (vertices[:, 0].max()+vertices[:, 0].min()) * 0.5
    mid_y = (vertices[:, 1].max()+vertices[:, 1].min()) * 0.5
    mid_z = (vertices[:, 2].max()+vertices[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_title(title)
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"🖼️  Keypoints visualization saved to: {save_path}")

    plt.show()


def analyze_mesh_quality(vertices, faces):
    """Analyze mesh quality and statistics"""

    if isinstance(vertices, torch.Tensor):
        vertices = vertices.detach().cpu().numpy()
    if isinstance(faces, torch.Tensor):
        faces = faces.detach().cpu().numpy()

    print("\n📊 Mesh Quality Analysis:")
    print("-" * 30)
    print(f"Vertices: {len(vertices)}")
    print(f"Faces: {len(faces)}")

    # Bounding box
    min_coords = vertices.min(axis=0)
    max_coords = vertices.max(axis=0)
    size = max_coords - min_coords

    print(f"Bounding box:")
    print(f"  X: [{min_coords[0]:.3f}, {max_coords[0]:.3f}] (size: {size[0]:.3f})")
    print(f"  Y: [{min_coords[1]:.3f}, {max_coords[1]:.3f}] (size: {size[1]:.3f})")
    print(f"  Z: [{min_coords[2]:.3f}, {max_coords[2]:.3f}] (size: {size[2]:.3f})")

    # Face analysis
    if len(faces) > 0:
        # Check for valid face indices
        max_vertex_idx = len(vertices) - 1
        invalid_faces = faces.max() > max_vertex_idx
        if invalid_faces:
            print(f"⚠️  Warning: Found faces with invalid vertex indices")
        else:
            print(f"✅ All face indices are valid")

        # Triangle edge lengths (sample)
        sample_faces = faces[:min(100, len(faces))]
        edge_lengths = []
        for face in sample_faces:
            v1, v2, v3 = vertices[face]
            edges = [
                np.linalg.norm(v2 - v1),
                np.linalg.norm(v3 - v2),
                np.linalg.norm(v1 - v3)
            ]
            edge_lengths.extend(edges)

        edge_lengths = np.array(edge_lengths)
        print(f"Edge lengths (sampled): mean={edge_lengths.mean():.4f}, std={edge_lengths.std():.4f}")


def main():
    """Main function"""

    print("🎯 Hand Mesh Viewer")
    print("=" * 30)

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

    # Create output directory
    output_dir = Path("mesh_visualizations")
    output_dir.mkdir(exist_ok=True)

    # Process first few samples
    num_samples = min(3, len(dataset))
    print(f"\n🖼️  Creating mesh visualizations for {num_samples} samples...")

    for i in range(num_samples):
        print(f"\n{'='*50}")
        print(f"Processing Sample {i}")
        print(f"{'='*50}")

        try:
            # Get sample
            sample = dataset[i]

            # Check if mesh data is available
            if 'vertices' not in sample or 'faces' not in sample:
                print(f"❌ Sample {i} missing mesh data")
                continue

            vertices = sample['vertices']
            faces = sample['faces']
            keypoints_3d = sample.get('keypoints_3d', None)

            print(f"📊 Sample {i} mesh info:")
            print(f"  Vertices: {vertices.shape}")
            print(f"  Faces: {faces.shape}")
            if keypoints_3d is not None:
                print(f"  Keypoints: {keypoints_3d.shape}")

            # Analyze mesh quality
            analyze_mesh_quality(vertices, faces)

            # Create mesh visualization
            print(f"\n🎨 Creating mesh visualization...")
            mesh_save_path = output_dir / f'sample_{i}_mesh.png'
            plot_3d_mesh(
                vertices, faces,
                title=f"Sample {i}: Ground Truth Hand Mesh",
                save_path=mesh_save_path
            )

            # Create mesh + keypoints visualization
            if keypoints_3d is not None:
                print(f"\n🎯 Creating mesh + keypoints visualization...")
                keypts_save_path = output_dir / f'sample_{i}_mesh_keypoints.png'
                plot_keypoints_on_mesh(
                    vertices, faces, keypoints_3d,
                    title=f"Sample {i}: Mesh with 3D Keypoints",
                    save_path=keypts_save_path
                )

        except Exception as e:
            print(f"❌ Error processing sample {i}: {e}")
            continue

    print(f"\n✅ Mesh visualization completed!")
    print(f"📁 Check '{output_dir}/' directory for mesh visualizations")

    return 0


if __name__ == "__main__":
    sys.exit(main())