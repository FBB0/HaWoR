#!/usr/bin/env python3
"""
MANO Hand Visualization for HaWoR Training
Shows predicted vs ground truth hand poses during validation
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import cv2


class MANOHandVisualizer:
    """Visualizer for MANO hand poses and comparisons"""

    def __init__(self, output_dir: str = "validation_visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # MANO hand topology (21 joints)
        self.joint_names = [
            'wrist',
            'thumb_mcp', 'thumb_pip', 'thumb_dip', 'thumb_tip',
            'index_mcp', 'index_pip', 'index_dip', 'index_tip',
            'middle_mcp', 'middle_pip', 'middle_dip', 'middle_tip',
            'ring_mcp', 'ring_pip', 'ring_dip', 'ring_tip',
            'pinky_mcp', 'pinky_pip', 'pinky_dip', 'pinky_tip'
        ]

        # Hand skeleton connections
        self.connections = [
            # Thumb
            [0, 1], [1, 2], [2, 3], [3, 4],
            # Index finger
            [0, 5], [5, 6], [6, 7], [7, 8],
            # Middle finger
            [0, 9], [9, 10], [10, 11], [11, 12],
            # Ring finger
            [0, 13], [13, 14], [14, 15], [15, 16],
            # Pinky finger
            [0, 17], [17, 18], [18, 19], [19, 20]
        ]

        # Colors for different parts
        self.colors = {
            'gt': '#2E86AB',      # Blue for ground truth
            'pred': '#A23B72',    # Purple for predictions
            'wrist': '#F18F01',   # Orange for wrist
            'thumb': '#C73E1D',   # Red for thumb
            'fingers': '#2E86AB'  # Blue for fingers
        }

        print(f"📊 MANO Hand Visualizer initialized")
        print(f"📁 Output directory: {self.output_dir}")

    def visualize_validation_batch(self,
                                  images: torch.Tensor,
                                  pred_joints: torch.Tensor,
                                  gt_joints: torch.Tensor,
                                  pred_pose: torch.Tensor,
                                  gt_pose: torch.Tensor,
                                  batch_idx: int,
                                  epoch: int,
                                  metrics: Dict[str, float]) -> None:
        """
        Visualize a validation batch with predictions vs ground truth

        Args:
            images: Input images [B, T, C, H, W]
            pred_joints: Predicted 3D joints [B, T, 21, 3]
            gt_joints: Ground truth 3D joints [B, T, 21, 3]
            pred_pose: Predicted MANO pose [B, T, 45]
            gt_pose: Ground truth MANO pose [B, T, 45]
            batch_idx: Batch index
            epoch: Current epoch
            metrics: Computed metrics for this batch
        """

        batch_size = images.shape[0]
        seq_len = images.shape[1]

        for b in range(min(batch_size, 2)):  # Visualize first 2 samples
            for t in range(min(seq_len, 4)):  # Visualize first 4 frames

                # Extract data for this sample and frame
                img = images[b, t].cpu().numpy().transpose(1, 2, 0)  # [H, W, C]
                pred_joints_frame = pred_joints[b, t].cpu().numpy()  # [21, 3]
                gt_joints_frame = gt_joints[b, t].cpu().numpy()      # [21, 3]
                pred_pose_frame = pred_pose[b, t].cpu().numpy()      # [45]
                gt_pose_frame = gt_pose[b, t].cpu().numpy()          # [45]

                # Create comprehensive visualization
                self._create_frame_visualization(
                    img, pred_joints_frame, gt_joints_frame,
                    pred_pose_frame, gt_pose_frame,
                    batch_idx, epoch, b, t, metrics
                )

    def _create_frame_visualization(self,
                                  image: np.ndarray,
                                  pred_joints: np.ndarray,
                                  gt_joints: np.ndarray,
                                  pred_pose: np.ndarray,
                                  gt_pose: np.ndarray,
                                  batch_idx: int,
                                  epoch: int,
                                  sample_idx: int,
                                  frame_idx: int,
                                  metrics: Dict[str, float]) -> None:
        """Create comprehensive visualization for a single frame"""

        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))

        # 1. Input image with 2D projections
        ax1 = plt.subplot(2, 4, 1)
        self._plot_image_with_2d_joints(ax1, image, pred_joints, gt_joints)
        ax1.set_title(f'Input Image + 2D Projections\nEpoch {epoch}, Batch {batch_idx}')

        # 2. 3D hand pose comparison
        ax2 = plt.subplot(2, 4, 2, projection='3d')
        self._plot_3d_hand_comparison(ax2, pred_joints, gt_joints)
        ax2.set_title('3D Hand Pose Comparison')

        # 3. Joint-wise error visualization
        ax3 = plt.subplot(2, 4, 3)
        self._plot_joint_errors(ax3, pred_joints, gt_joints)
        ax3.set_title('Per-Joint Errors')

        # 4. MANO pose parameter comparison
        ax4 = plt.subplot(2, 4, 4)
        self._plot_pose_parameters(ax4, pred_pose, gt_pose)
        ax4.set_title('MANO Pose Parameters')

        # 5. Hand skeleton from different views
        ax5 = plt.subplot(2, 4, 5, projection='3d')
        self._plot_hand_skeleton(ax5, pred_joints, gt_joints, view='front')
        ax5.set_title('Front View')

        ax6 = plt.subplot(2, 4, 6, projection='3d')
        self._plot_hand_skeleton(ax6, pred_joints, gt_joints, view='side')
        ax6.set_title('Side View')

        # 6. Finger-wise analysis
        ax7 = plt.subplot(2, 4, 7)
        self._plot_finger_analysis(ax7, pred_joints, gt_joints)
        ax7.set_title('Finger-wise Analysis')

        # 7. Metrics summary
        ax8 = plt.subplot(2, 4, 8)
        self._plot_metrics_summary(ax8, metrics, pred_joints, gt_joints)
        ax8.set_title('Validation Metrics')

        plt.tight_layout()

        # Save the visualization
        filename = f"validation_epoch{epoch:02d}_batch{batch_idx:02d}_sample{sample_idx}_frame{frame_idx}.png"
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  📊 Saved validation visualization: {filename}")

    def _plot_image_with_2d_joints(self, ax, image: np.ndarray, pred_joints: np.ndarray, gt_joints: np.ndarray):
        """Plot input image with 2D joint projections"""

        # Normalize image for display
        img_display = np.clip(image, 0, 1)
        ax.imshow(img_display)

        # Simple perspective projection (assuming centered camera)
        fx, fy = 500.0, 500.0
        cx, cy = image.shape[1] // 2, image.shape[0] // 2

        # Project 3D joints to 2D
        def project_joints(joints_3d):
            x, y, z = joints_3d[:, 0], joints_3d[:, 1], joints_3d[:, 2]
            z = np.clip(z, 0.1, None)  # Avoid division by zero
            u = (fx * x / z) + cx
            v = (fy * y / z) + cy
            return np.stack([u, v], axis=1)

        pred_2d = project_joints(pred_joints)
        gt_2d = project_joints(gt_joints)

        # Plot ground truth joints
        ax.scatter(gt_2d[:, 0], gt_2d[:, 1], c=self.colors['gt'], s=50, alpha=0.8, label='GT', marker='o')

        # Plot predicted joints
        ax.scatter(pred_2d[:, 0], pred_2d[:, 1], c=self.colors['pred'], s=50, alpha=0.8, label='Pred', marker='x')

        # Draw hand skeleton for ground truth
        for connection in self.connections:
            if len(gt_2d) > max(connection):
                ax.plot([gt_2d[connection[0], 0], gt_2d[connection[1], 0]],
                       [gt_2d[connection[0], 1], gt_2d[connection[1], 1]],
                       color=self.colors['gt'], alpha=0.6, linewidth=1)

        # Draw hand skeleton for predictions
        for connection in self.connections:
            if len(pred_2d) > max(connection):
                ax.plot([pred_2d[connection[0], 0], pred_2d[connection[1], 0]],
                       [pred_2d[connection[0], 1], pred_2d[connection[1], 1]],
                       color=self.colors['pred'], alpha=0.6, linewidth=1, linestyle='--')

        ax.legend()
        ax.set_xlim(0, image.shape[1])
        ax.set_ylim(image.shape[0], 0)

    def _plot_3d_hand_comparison(self, ax, pred_joints: np.ndarray, gt_joints: np.ndarray):
        """Plot 3D hand pose comparison"""

        # Plot ground truth
        ax.scatter(gt_joints[:, 0], gt_joints[:, 1], gt_joints[:, 2],
                  c=self.colors['gt'], s=60, alpha=0.8, label='Ground Truth')

        # Plot predictions
        ax.scatter(pred_joints[:, 0], pred_joints[:, 1], pred_joints[:, 2],
                  c=self.colors['pred'], s=60, alpha=0.8, label='Predicted')

        # Draw skeleton connections for ground truth
        for connection in self.connections:
            if len(gt_joints) > max(connection):
                ax.plot([gt_joints[connection[0], 0], gt_joints[connection[1], 0]],
                       [gt_joints[connection[0], 1], gt_joints[connection[1], 1]],
                       [gt_joints[connection[0], 2], gt_joints[connection[1], 2]],
                       color=self.colors['gt'], alpha=0.6, linewidth=2)

        # Draw skeleton connections for predictions
        for connection in self.connections:
            if len(pred_joints) > max(connection):
                ax.plot([pred_joints[connection[0], 0], pred_joints[connection[1], 0]],
                       [pred_joints[connection[0], 1], pred_joints[connection[1], 1]],
                       [pred_joints[connection[0], 2], pred_joints[connection[1], 2]],
                       color=self.colors['pred'], alpha=0.6, linewidth=2, linestyle='--')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()

        # Set equal aspect ratio
        max_range = np.array([gt_joints.max()-gt_joints.min(),
                             pred_joints.max()-pred_joints.min()]).max() / 2.0
        mid_x = (gt_joints[:, 0].max()+gt_joints[:, 0].min()) * 0.5
        mid_y = (gt_joints[:, 1].max()+gt_joints[:, 1].min()) * 0.5
        mid_z = (gt_joints[:, 2].max()+gt_joints[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    def _plot_joint_errors(self, ax, pred_joints: np.ndarray, gt_joints: np.ndarray):
        """Plot per-joint errors"""

        # Compute per-joint errors in mm
        joint_errors = np.linalg.norm(pred_joints - gt_joints, axis=1) * 1000

        # Create bar plot
        x_pos = np.arange(len(self.joint_names))
        bars = ax.bar(x_pos, joint_errors, color='skyblue', alpha=0.7)

        # Color bars by finger
        finger_colors = {
            'wrist': '#F18F01',
            'thumb': '#C73E1D',
            'index': '#2E86AB',
            'middle': '#A23B72',
            'ring': '#3A86FF',
            'pinky': '#06FFA5'
        }

        for i, (bar, joint_name) in enumerate(zip(bars, self.joint_names)):
            if 'wrist' in joint_name:
                bar.set_color(finger_colors['wrist'])
            elif 'thumb' in joint_name:
                bar.set_color(finger_colors['thumb'])
            elif 'index' in joint_name:
                bar.set_color(finger_colors['index'])
            elif 'middle' in joint_name:
                bar.set_color(finger_colors['middle'])
            elif 'ring' in joint_name:
                bar.set_color(finger_colors['ring'])
            elif 'pinky' in joint_name:
                bar.set_color(finger_colors['pinky'])

        ax.set_xlabel('Joint')
        ax.set_ylabel('Error (mm)')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([name.replace('_', '\n') for name in self.joint_names], rotation=45, ha='right')

        # Add error statistics
        mean_error = np.mean(joint_errors)
        max_error = np.max(joint_errors)
        ax.axhline(y=mean_error, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_error:.1f}mm')
        ax.legend()

        # Add text with statistics
        ax.text(0.02, 0.98, f'Max: {max_error:.1f}mm\nMean: {mean_error:.1f}mm',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    def _plot_pose_parameters(self, ax, pred_pose: np.ndarray, gt_pose: np.ndarray):
        """Plot MANO pose parameter comparison"""

        # Show first 15 pose parameters (global rotation + first few finger joints)
        n_params = min(15, len(pred_pose))
        x_pos = np.arange(n_params)

        width = 0.35
        ax.bar(x_pos - width/2, gt_pose[:n_params], width, label='Ground Truth', alpha=0.7, color=self.colors['gt'])
        ax.bar(x_pos + width/2, pred_pose[:n_params], width, label='Predicted', alpha=0.7, color=self.colors['pred'])

        ax.set_xlabel('Pose Parameter Index')
        ax.set_ylabel('Parameter Value')
        ax.set_xticks(x_pos)
        ax.legend()

        # Add parameter error statistics
        param_errors = np.abs(pred_pose - gt_pose)
        mean_param_error = np.mean(param_errors)
        ax.text(0.02, 0.98, f'Mean Param Error: {mean_param_error:.3f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    def _plot_hand_skeleton(self, ax, pred_joints: np.ndarray, gt_joints: np.ndarray, view: str = 'front'):
        """Plot hand skeleton from specific viewpoint"""

        # Plot both skeletons
        self._plot_3d_hand_comparison(ax, pred_joints, gt_joints)

        # Set viewpoint
        if view == 'front':
            ax.view_init(elev=0, azim=0)
        elif view == 'side':
            ax.view_init(elev=0, azim=90)
        elif view == 'top':
            ax.view_init(elev=90, azim=0)

        ax.set_title(f'{view.capitalize()} View')

    def _plot_finger_analysis(self, ax, pred_joints: np.ndarray, gt_joints: np.ndarray):
        """Plot finger-wise error analysis"""

        # Define finger groups
        finger_groups = {
            'Thumb': [1, 2, 3, 4],
            'Index': [5, 6, 7, 8],
            'Middle': [9, 10, 11, 12],
            'Ring': [13, 14, 15, 16],
            'Pinky': [17, 18, 19, 20]
        }

        finger_errors = []
        finger_names = []

        for finger_name, joint_indices in finger_groups.items():
            finger_joints_pred = pred_joints[joint_indices]
            finger_joints_gt = gt_joints[joint_indices]
            finger_error = np.mean(np.linalg.norm(finger_joints_pred - finger_joints_gt, axis=1)) * 1000
            finger_errors.append(finger_error)
            finger_names.append(finger_name)

        # Add wrist error
        wrist_error = np.linalg.norm(pred_joints[0] - gt_joints[0]) * 1000
        finger_errors.append(wrist_error)
        finger_names.append('Wrist')

        # Create bar plot
        colors = ['#C73E1D', '#2E86AB', '#A23B72', '#3A86FF', '#06FFA5', '#F18F01']
        bars = ax.bar(finger_names, finger_errors, color=colors, alpha=0.7)

        ax.set_ylabel('Mean Error (mm)')
        ax.set_title('Finger-wise Error Analysis')

        # Add value labels on bars
        for bar, error in zip(bars, finger_errors):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{error:.1f}', ha='center', va='bottom')

    def _plot_metrics_summary(self, ax, metrics: Dict[str, float], pred_joints: np.ndarray, gt_joints: np.ndarray):
        """Plot validation metrics summary"""

        # Compute additional metrics for this sample
        joint_errors = np.linalg.norm(pred_joints - gt_joints, axis=1) * 1000
        sample_mpjpe = np.mean(joint_errors)
        sample_max_error = np.max(joint_errors)
        sample_min_error = np.min(joint_errors)

        # Create metrics text
        metrics_text = f"""
Validation Metrics:

MPJPE: {sample_mpjpe:.2f} mm
Max Error: {sample_max_error:.2f} mm
Min Error: {sample_min_error:.2f} mm

Batch Metrics:
Train Loss: {metrics.get('train_loss', 0):.0f}
Val Loss: {metrics.get('val_loss', 0):.0f}
Learning Rate: {metrics.get('lr', 0):.2e}

Joint Statistics:
Wrist Error: {joint_errors[0]:.1f} mm
Fingertip Errors:
  Thumb: {joint_errors[4]:.1f} mm
  Index: {joint_errors[8]:.1f} mm
  Middle: {joint_errors[12]:.1f} mm
  Ring: {joint_errors[16]:.1f} mm
  Pinky: {joint_errors[20]:.1f} mm
        """

        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

    def create_3d_space_visualization(self,
                                     pred_joints: torch.Tensor,
                                     gt_joints: torch.Tensor,
                                     batch_idx: int,
                                     epoch: int,
                                     sample_idx: int = 0,
                                     frame_idx: int = 0) -> None:
        """
        Create 3D space visualization showing predicted and GT hands in 3D space

        Args:
            pred_joints: Predicted 3D joints [B, T, 21, 3]
            gt_joints: Ground truth 3D joints [B, T, 21, 3]
            batch_idx: Batch index
            epoch: Current epoch
            sample_idx: Sample index in batch
            frame_idx: Frame index in sequence
        """

        # Extract data for visualization
        pred_joints_frame = pred_joints[sample_idx, frame_idx].cpu().numpy()  # [21, 3]
        gt_joints_frame = gt_joints[sample_idx, frame_idx].cpu().numpy()      # [21, 3]

        # Create figure with multiple 3D views
        fig = plt.figure(figsize=(20, 15))

        # Main 3D view - side by side hands
        ax1 = plt.subplot(2, 3, (1, 2), projection='3d')
        self._plot_3d_hands_side_by_side(ax1, pred_joints_frame, gt_joints_frame)
        ax1.set_title(f'3D Hand Comparison - Epoch {epoch}, Batch {batch_idx}\nBlue: Ground Truth, Purple: Predicted', fontsize=14)

        # Front view
        ax2 = plt.subplot(2, 3, 3, projection='3d')
        self._plot_3d_hands_overlaid(ax2, pred_joints_frame, gt_joints_frame, view='front')
        ax2.set_title('Front View (Overlaid)', fontsize=12)

        # Side view
        ax3 = plt.subplot(2, 3, 4, projection='3d')
        self._plot_3d_hands_overlaid(ax3, pred_joints_frame, gt_joints_frame, view='side')
        ax3.set_title('Side View (Overlaid)', fontsize=12)

        # Top view
        ax4 = plt.subplot(2, 3, 5, projection='3d')
        self._plot_3d_hands_overlaid(ax4, pred_joints_frame, gt_joints_frame, view='top')
        ax4.set_title('Top View (Overlaid)', fontsize=12)

        # Distance visualization
        ax5 = plt.subplot(2, 3, 6)
        self._plot_3d_distance_analysis(ax5, pred_joints_frame, gt_joints_frame)
        ax5.set_title('3D Distance Analysis', fontsize=12)

        plt.tight_layout()

        # Save the 3D visualization
        filename = f"3d_space_epoch{epoch:02d}_batch{batch_idx:02d}_sample{sample_idx}_frame{frame_idx}.png"
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  🎯 Saved 3D space visualization: {filename}")

    def _plot_3d_hands_side_by_side(self, ax, pred_joints: np.ndarray, gt_joints: np.ndarray):
        """Plot predicted and ground truth hands side by side in 3D space"""

        # Offset hands for side-by-side display
        offset_distance = 0.15  # 15cm separation
        gt_joints_offset = gt_joints.copy()
        pred_joints_offset = pred_joints.copy()
        pred_joints_offset[:, 0] += offset_distance  # Move predicted hand to the right

        # Plot ground truth hand (left side)
        ax.scatter(gt_joints_offset[:, 0], gt_joints_offset[:, 1], gt_joints_offset[:, 2],
                  c=self.colors['gt'], s=100, alpha=0.9, label='Ground Truth', marker='o')

        # Plot predicted hand (right side)
        ax.scatter(pred_joints_offset[:, 0], pred_joints_offset[:, 1], pred_joints_offset[:, 2],
                  c=self.colors['pred'], s=100, alpha=0.9, label='Predicted', marker='^')

        # Draw skeleton for ground truth
        for connection in self.connections:
            if len(gt_joints_offset) > max(connection):
                ax.plot([gt_joints_offset[connection[0], 0], gt_joints_offset[connection[1], 0]],
                       [gt_joints_offset[connection[0], 1], gt_joints_offset[connection[1], 1]],
                       [gt_joints_offset[connection[0], 2], gt_joints_offset[connection[1], 2]],
                       color=self.colors['gt'], alpha=0.8, linewidth=3)

        # Draw skeleton for predicted
        for connection in self.connections:
            if len(pred_joints_offset) > max(connection):
                ax.plot([pred_joints_offset[connection[0], 0], pred_joints_offset[connection[1], 0]],
                       [pred_joints_offset[connection[0], 1], pred_joints_offset[connection[1], 1]],
                       [pred_joints_offset[connection[0], 2], pred_joints_offset[connection[1], 2]],
                       color=self.colors['pred'], alpha=0.8, linewidth=3, linestyle='--')

        # Add connection lines between corresponding joints
        for i in range(len(gt_joints)):
            ax.plot([gt_joints_offset[i, 0], pred_joints_offset[i, 0]],
                   [gt_joints_offset[i, 1], pred_joints_offset[i, 1]],
                   [gt_joints_offset[i, 2], pred_joints_offset[i, 2]],
                   color='gray', alpha=0.3, linewidth=1, linestyle=':')

        # Set labels and legend
        ax.set_xlabel('X (meters)', fontsize=12)
        ax.set_ylabel('Y (meters)', fontsize=12)
        ax.set_zlabel('Z (meters)', fontsize=12)
        ax.legend(fontsize=12)

        # Set equal aspect ratio and good viewing angle
        all_points = np.vstack([gt_joints_offset, pred_joints_offset])
        max_range = np.array([all_points.max()-all_points.min()]).max() / 2.0
        mid_x = (all_points[:, 0].max()+all_points[:, 0].min()) * 0.5
        mid_y = (all_points[:, 1].max()+all_points[:, 1].min()) * 0.5
        mid_z = (all_points[:, 2].max()+all_points[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # Set viewing angle for best perspective
        ax.view_init(elev=20, azim=45)

        # Add grid
        ax.grid(True, alpha=0.3)

    def _plot_3d_hands_overlaid(self, ax, pred_joints: np.ndarray, gt_joints: np.ndarray, view: str = 'front'):
        """Plot overlaid hands from specific viewpoint"""

        # Plot both hands in the same space
        ax.scatter(gt_joints[:, 0], gt_joints[:, 1], gt_joints[:, 2],
                  c=self.colors['gt'], s=80, alpha=0.8, label='GT', marker='o')
        ax.scatter(pred_joints[:, 0], pred_joints[:, 1], pred_joints[:, 2],
                  c=self.colors['pred'], s=80, alpha=0.8, label='Pred', marker='^')

        # Draw skeletons
        for connection in self.connections:
            if len(gt_joints) > max(connection):
                ax.plot([gt_joints[connection[0], 0], gt_joints[connection[1], 0]],
                       [gt_joints[connection[0], 1], gt_joints[connection[1], 1]],
                       [gt_joints[connection[0], 2], gt_joints[connection[1], 2]],
                       color=self.colors['gt'], alpha=0.7, linewidth=2)
                ax.plot([pred_joints[connection[0], 0], pred_joints[connection[1], 0]],
                       [pred_joints[connection[0], 1], pred_joints[connection[1], 1]],
                       [pred_joints[connection[0], 2], pred_joints[connection[1], 2]],
                       color=self.colors['pred'], alpha=0.7, linewidth=2, linestyle='--')

        # Set viewpoint
        if view == 'front':
            ax.view_init(elev=0, azim=0)
        elif view == 'side':
            ax.view_init(elev=0, azim=90)
        elif view == 'top':
            ax.view_init(elev=90, azim=0)

        # Set equal aspect ratio
        all_points = np.vstack([gt_joints, pred_joints])
        max_range = np.array([all_points.max()-all_points.min()]).max() / 2.0
        mid_x = (all_points[:, 0].max()+all_points[:, 0].min()) * 0.5
        mid_y = (all_points[:, 1].max()+all_points[:, 1].min()) * 0.5
        mid_z = (all_points[:, 2].max()+all_points[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_3d_distance_analysis(self, ax, pred_joints: np.ndarray, gt_joints: np.ndarray):
        """Plot 3D distance analysis between predicted and ground truth"""

        # Compute 3D distances for each joint
        distances = np.linalg.norm(pred_joints - gt_joints, axis=1) * 1000  # Convert to mm

        # Create distance visualization
        joint_indices = np.arange(len(self.joint_names))
        colors = plt.cm.viridis(distances / distances.max())

        bars = ax.bar(joint_indices, distances, color=colors, alpha=0.8)

        # Add colorbar to show distance scale
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=distances.max()))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
        cbar.set_label('Distance (mm)', fontsize=10)

        ax.set_xlabel('Joint Index')
        ax.set_ylabel('3D Distance (mm)')
        ax.set_xticks(joint_indices[::2])  # Show every other joint name
        ax.set_xticklabels([self.joint_names[i].replace('_', '\n') for i in joint_indices[::2]],
                          rotation=45, ha='right', fontsize=8)

        # Add statistics
        mean_dist = np.mean(distances)
        max_dist = np.max(distances)
        ax.axhline(y=mean_dist, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_dist:.1f}mm')
        ax.legend()

        # Add text with statistics
        stats_text = f'3D Distance Stats:\nMean: {mean_dist:.1f}mm\nMax: {max_dist:.1f}mm\nMin: {np.min(distances):.1f}mm'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=9)

    def create_training_progress_visualization(self, epoch_metrics: List[Dict], epoch: int):
        """Create training progress visualization"""

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Training and validation loss
        epochs = list(range(1, len(epoch_metrics) + 1))
        train_losses = [m.get('train_loss', 0) for m in epoch_metrics]
        val_losses = [m.get('val_loss', 0) for m in epoch_metrics]

        axes[0, 0].plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        axes[0, 0].plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Progress')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Learning rate
        learning_rates = [m.get('lr', 0) for m in epoch_metrics]
        axes[0, 1].plot(epochs, learning_rates, 'g-', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].set_title('Learning Rate Schedule')
        axes[0, 1].grid(True, alpha=0.3)

        # Loss improvement
        if len(train_losses) > 1:
            train_improvement = [(train_losses[0] - loss) / train_losses[0] * 100 for loss in train_losses]
            val_improvement = [(val_losses[0] - loss) / val_losses[0] * 100 for loss in val_losses]

            axes[1, 0].plot(epochs, train_improvement, 'b-', label='Training', linewidth=2)
            axes[1, 0].plot(epochs, val_improvement, 'r-', label='Validation', linewidth=2)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Improvement (%)')
            axes[1, 0].set_title('Loss Improvement')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # Summary statistics
        axes[1, 1].text(0.1, 0.9, f'Training Summary (Epoch {epoch}):', fontsize=14, fontweight='bold',
                       transform=axes[1, 1].transAxes)

        summary_text = f"""
Final Training Loss: {train_losses[-1]:.0f}
Final Validation Loss: {val_losses[-1]:.0f}
Best Validation Loss: {min(val_losses):.0f}
Training Improvement: {train_improvement[-1]:.1f}%
Validation Improvement: {val_improvement[-1]:.1f}%

Current Learning Rate: {learning_rates[-1]:.2e}
Epochs Completed: {len(epoch_metrics)}
        """

        axes[1, 1].text(0.1, 0.8, summary_text, fontsize=10, fontfamily='monospace',
                       transform=axes[1, 1].transAxes,
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')

        plt.tight_layout()

        # Save training progress
        progress_path = self.output_dir / f'training_progress_epoch_{epoch:02d}.png'
        plt.savefig(progress_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"📈 Saved training progress visualization: {progress_path}")