#!/usr/bin/env python3
"""
HaWoR Evaluation Framework
Implements comprehensive evaluation metrics for hand pose estimation and camera tracking
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm
import cv2

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.hawor_model import HaWoRModel
from src.datasets.arctic_dataset_real import ARCTICDataset
from torch.utils.data import DataLoader


class HaWoREvaluator:
    """
    Comprehensive evaluation framework for HaWoR models
    """

    def __init__(self,
                 model: HaWoRModel,
                 device: torch.device,
                 output_dir: str = "evaluation_results"):

        self.model = model
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Evaluation metrics storage
        self.metrics = {
            'hand_pose_metrics': {},
            'keypoint_metrics': {},
            'camera_metrics': {},
            'temporal_metrics': {},
            'per_sequence_metrics': {}
        }

        print(f"🔍 HaWoR Evaluator initialized")
        print(f"📁 Output directory: {self.output_dir}")

    def evaluate_dataset(self, data_loader: DataLoader,
                         hand_type: str = 'left',
                         save_visualizations: bool = True) -> Dict[str, float]:
        """
        Evaluate model on entire dataset

        Args:
            data_loader: DataLoader for evaluation dataset
            hand_type: 'left', 'right', or 'both'
            save_visualizations: Whether to save visualization results

        Returns:
            Dictionary of evaluation metrics
        """

        print(f"🧪 Evaluating on {len(data_loader.dataset)} samples...")

        self.model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(data_loader, desc="Evaluating")):
                try:
                    # Move batch to device
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                            for k, v in batch.items()}

                    # Forward pass
                    images = batch['images']  # [B, T, C, H, W]
                    outputs = self.model(images)

                    # Extract predictions and targets
                    predictions, targets = self._extract_predictions_and_targets(
                        outputs, batch, hand_type
                    )

                    all_predictions.append(predictions)
                    all_targets.append(targets)

                    # Save sample visualizations
                    if save_visualizations and batch_idx < 5:
                        self._save_sample_visualization(
                            batch, outputs, batch_idx, hand_type
                        )

                except Exception as e:
                    print(f"❌ Error in evaluation batch {batch_idx}: {e}")
                    continue

        # Compute comprehensive metrics
        metrics = self._compute_comprehensive_metrics(all_predictions, all_targets)

        # Save results
        self._save_evaluation_results(metrics)

        print(f"✅ Evaluation completed!")
        return metrics

    def _extract_predictions_and_targets(self,
                                       outputs: Dict[str, torch.Tensor],
                                       batch: Dict[str, torch.Tensor],
                                       hand_type: str) -> Tuple[Dict, Dict]:
        """Extract predictions and targets for specified hand type"""

        if hand_type == 'left':
            predictions = {
                'hand_pose': outputs['left_hand_pose'],      # [B, T, 45]
                'hand_shape': outputs['left_hand_shape'],    # [B, T, 10]
                'hand_trans': outputs['left_hand_trans'],    # [B, T, 3]
                'hand_rot': outputs['left_hand_rot'],        # [B, T, 3]
                'hand_joints': outputs['left_hand_joints'],  # [B, T, 21, 3]
                'confidence': outputs['left_confidence']     # [B, T]
            }
        elif hand_type == 'right':
            predictions = {
                'hand_pose': outputs['right_hand_pose'],
                'hand_shape': outputs['right_hand_shape'],
                'hand_trans': outputs['right_hand_trans'],
                'hand_rot': outputs['right_hand_rot'],
                'hand_joints': outputs['right_hand_joints'],
                'confidence': outputs['right_confidence']
            }
        else:  # both hands
            predictions = {
                'left_hand_pose': outputs['left_hand_pose'],
                'left_hand_joints': outputs['left_hand_joints'],
                'right_hand_pose': outputs['right_hand_pose'],
                'right_hand_joints': outputs['right_hand_joints']
            }

        # Camera predictions
        predictions['camera_pose'] = outputs['camera_pose']  # [B, T, 6]

        # Extract corresponding targets
        targets = {
            'hand_pose': batch['hand_pose'],        # [B, T, 45]
            'hand_shape': batch['hand_shape'],      # [B, T, 10]
            'hand_trans': batch['hand_trans'],      # [B, T, 3]
            'hand_rot': batch['hand_rot'],          # [B, T, 3]
            'hand_joints': batch['hand_joints'],    # [B, T, 21, 3]
            'keypoints_2d': batch['keypoints_2d'],  # [B, T, 21, 2]
            'camera_pose': batch['camera_pose'],    # [B, T, 6]
            'hand_valid': batch['hand_valid']       # [B, T]
        }

        return predictions, targets

    def _compute_comprehensive_metrics(self,
                                     all_predictions: List[Dict],
                                     all_targets: List[Dict]) -> Dict[str, float]:
        """Compute comprehensive evaluation metrics"""

        print("📊 Computing evaluation metrics...")

        metrics = {}

        # Concatenate all predictions and targets
        pred_joints = torch.cat([p['hand_joints'] for p in all_predictions], dim=0)
        pred_poses = torch.cat([p['hand_pose'] for p in all_predictions], dim=0)
        pred_camera = torch.cat([p['camera_pose'] for p in all_predictions], dim=0)

        gt_joints = torch.cat([t['hand_joints'] for t in all_targets], dim=0)
        gt_poses = torch.cat([t['hand_pose'] for t in all_targets], dim=0)
        gt_camera = torch.cat([t['camera_pose'] for t in all_targets], dim=0)
        hand_valid = torch.cat([t['hand_valid'] for t in all_targets], dim=0)

        # 1. Hand Pose Metrics
        metrics.update(self._compute_hand_pose_metrics(pred_poses, gt_poses, hand_valid))

        # 2. 3D Keypoint Metrics
        metrics.update(self._compute_keypoint_metrics(pred_joints, gt_joints, hand_valid))

        # 3. Camera Tracking Metrics
        metrics.update(self._compute_camera_metrics(pred_camera, gt_camera))

        # 4. Temporal Consistency Metrics
        metrics.update(self._compute_temporal_metrics(pred_joints, gt_joints))

        # 5. Per-joint Analysis
        metrics.update(self._compute_per_joint_metrics(pred_joints, gt_joints, hand_valid))

        return metrics

    def _compute_hand_pose_metrics(self,
                                 pred_poses: torch.Tensor,
                                 gt_poses: torch.Tensor,
                                 valid_mask: torch.Tensor) -> Dict[str, float]:
        """Compute MANO pose parameter metrics"""

        metrics = {}

        # Apply valid mask
        valid_frames = valid_mask > 0.5
        if valid_frames.sum() == 0:
            return {'pose_mse': 0.0, 'pose_mae': 0.0}

        pred_valid = pred_poses[valid_frames]
        gt_valid = gt_poses[valid_frames]

        # Mean Squared Error
        pose_mse = torch.mean((pred_valid - gt_valid) ** 2).item()

        # Mean Absolute Error
        pose_mae = torch.mean(torch.abs(pred_valid - gt_valid)).item()

        # Per-parameter analysis
        param_errors = torch.mean(torch.abs(pred_valid - gt_valid), dim=0)

        metrics.update({
            'pose_mse': pose_mse,
            'pose_mae': pose_mae,
            'pose_std': torch.std(pred_valid - gt_valid).item(),
            'global_pose_error': torch.mean(param_errors[:3]).item(),  # Global rotation
            'finger_pose_error': torch.mean(param_errors[3:]).item()   # Finger poses
        })

        return metrics

    def _compute_keypoint_metrics(self,
                                pred_joints: torch.Tensor,
                                gt_joints: torch.Tensor,
                                valid_mask: torch.Tensor) -> Dict[str, float]:
        """Compute 3D keypoint metrics"""

        metrics = {}

        # Apply valid mask
        valid_frames = valid_mask > 0.5
        if valid_frames.sum() == 0:
            return {'mpjpe': 0.0, 'pa_mpjpe': 0.0}

        pred_valid = pred_joints[valid_frames]  # [N, 21, 3]
        gt_valid = gt_joints[valid_frames]      # [N, 21, 3]

        # Mean Per Joint Position Error (MPJPE) in mm
        joint_errors = torch.norm(pred_valid - gt_valid, dim=-1)  # [N, 21]
        mpjpe = torch.mean(joint_errors).item() * 1000  # Convert to mm

        # Procrustes-aligned MPJPE (PA-MPJPE)
        pa_mpjpe = self._compute_pa_mpjpe(pred_valid, gt_valid) * 1000

        # Per-joint errors
        per_joint_errors = torch.mean(joint_errors, dim=0) * 1000  # [21]

        # Joint-specific metrics
        wrist_error = per_joint_errors[0].item()  # Wrist joint
        fingertip_errors = per_joint_errors[[4, 8, 12, 16, 20]]  # Fingertips
        fingertip_error = torch.mean(fingertip_errors).item()

        metrics.update({
            'mpjpe': mpjpe,
            'pa_mpjpe': pa_mpjpe,
            'wrist_error': wrist_error,
            'fingertip_error': fingertip_error,
            'per_joint_errors': per_joint_errors.tolist()
        })

        return metrics

    def _compute_pa_mpjpe(self, pred: torch.Tensor, gt: torch.Tensor) -> float:
        """Compute Procrustes-aligned MPJPE"""

        # Center the predictions and ground truth
        pred_centered = pred - torch.mean(pred, dim=1, keepdim=True)
        gt_centered = gt - torch.mean(gt, dim=1, keepdim=True)

        # Compute optimal rotation using Procrustes analysis
        errors = []
        for i in range(pred.shape[0]):
            try:
                # Compute cross-covariance matrix
                H = torch.matmul(pred_centered[i].T, gt_centered[i])

                # SVD
                U, S, V = torch.svd(H)
                R = torch.matmul(V, U.T)

                # Handle reflection case
                if torch.det(R) < 0:
                    V[:, -1] *= -1
                    R = torch.matmul(V, U.T)

                # Apply rotation
                pred_aligned = torch.matmul(pred_centered[i], R)

                # Compute error
                error = torch.norm(pred_aligned - gt_centered[i], dim=-1)
                errors.append(torch.mean(error))

            except:
                # Fallback to unaligned error
                error = torch.norm(pred_centered[i] - gt_centered[i], dim=-1)
                errors.append(torch.mean(error))

        return torch.mean(torch.stack(errors)).item()

    def _compute_camera_metrics(self,
                              pred_camera: torch.Tensor,
                              gt_camera: torch.Tensor) -> Dict[str, float]:
        """Compute camera tracking metrics"""

        metrics = {}

        if pred_camera.shape[0] == 0:
            return {'camera_trans_error': 0.0, 'camera_rot_error': 0.0}

        # Separate translation and rotation
        pred_trans = pred_camera[..., :3]  # [B, T, 3]
        pred_rot = pred_camera[..., 3:]    # [B, T, 3]
        gt_trans = gt_camera[..., :3]
        gt_rot = gt_camera[..., 3:]

        # Translation error (in meters)
        trans_error = torch.mean(torch.norm(pred_trans - gt_trans, dim=-1)).item()

        # Rotation error (in radians)
        rot_error = torch.mean(torch.norm(pred_rot - gt_rot, dim=-1)).item()

        # Trajectory smoothness
        if pred_trans.shape[1] > 1:
            pred_vel = pred_trans[:, 1:] - pred_trans[:, :-1]
            gt_vel = gt_trans[:, 1:] - gt_trans[:, :-1]
            velocity_error = torch.mean(torch.norm(pred_vel - gt_vel, dim=-1)).item()
        else:
            velocity_error = 0.0

        metrics.update({
            'camera_trans_error': trans_error,
            'camera_rot_error': rot_error,
            'camera_velocity_error': velocity_error
        })

        return metrics

    def _compute_temporal_metrics(self,
                                pred_joints: torch.Tensor,
                                gt_joints: torch.Tensor) -> Dict[str, float]:
        """Compute temporal consistency metrics"""

        metrics = {}

        if pred_joints.shape[1] < 2:
            return {'temporal_consistency': 0.0}

        # Compute velocity differences
        pred_vel = pred_joints[:, 1:] - pred_joints[:, :-1]  # [B, T-1, 21, 3]
        gt_vel = gt_joints[:, 1:] - gt_joints[:, :-1]

        # Velocity error
        vel_error = torch.mean(torch.norm(pred_vel - gt_vel, dim=-1)).item()

        # Acceleration smoothness
        if pred_joints.shape[1] > 2:
            pred_acc = pred_vel[:, 1:] - pred_vel[:, :-1]
            gt_acc = gt_vel[:, 1:] - gt_vel[:, :-1]
            acc_error = torch.mean(torch.norm(pred_acc - gt_acc, dim=-1)).item()
        else:
            acc_error = 0.0

        # Motion smoothness (penalize jittery motion)
        pred_jerk = torch.mean(torch.norm(pred_vel[:, 1:] - pred_vel[:, :-1], dim=-1))
        smoothness_score = 1.0 / (1.0 + pred_jerk.item())

        metrics.update({
            'velocity_error': vel_error * 1000,  # Convert to mm
            'acceleration_error': acc_error * 1000,
            'motion_smoothness': smoothness_score
        })

        return metrics

    def _compute_per_joint_metrics(self,
                                 pred_joints: torch.Tensor,
                                 gt_joints: torch.Tensor,
                                 valid_mask: torch.Tensor) -> Dict[str, float]:
        """Compute detailed per-joint analysis"""

        metrics = {}

        valid_frames = valid_mask > 0.5
        if valid_frames.sum() == 0:
            return {}

        pred_valid = pred_joints[valid_frames]
        gt_valid = gt_joints[valid_frames]

        # Joint errors
        joint_errors = torch.norm(pred_valid - gt_valid, dim=-1) * 1000  # [N, 21]

        # MANO joint names
        joint_names = [
            'wrist', 'thumb_mcp', 'thumb_pip', 'thumb_dip', 'thumb_tip',
            'index_mcp', 'index_pip', 'index_dip', 'index_tip',
            'middle_mcp', 'middle_pip', 'middle_dip', 'middle_tip',
            'ring_mcp', 'ring_pip', 'ring_dip', 'ring_tip',
            'pinky_mcp', 'pinky_pip', 'pinky_dip', 'pinky_tip'
        ]

        # Per-joint statistics
        for i, joint_name in enumerate(joint_names):
            metrics[f'{joint_name}_error_mean'] = torch.mean(joint_errors[:, i]).item()
            metrics[f'{joint_name}_error_std'] = torch.std(joint_errors[:, i]).item()

        # Finger group analysis
        finger_groups = {
            'thumb': [1, 2, 3, 4],
            'index': [5, 6, 7, 8],
            'middle': [9, 10, 11, 12],
            'ring': [13, 14, 15, 16],
            'pinky': [17, 18, 19, 20]
        }

        for finger, joints in finger_groups.items():
            finger_errors = joint_errors[:, joints]
            metrics[f'{finger}_error'] = torch.mean(finger_errors).item()

        return metrics

    def _save_sample_visualization(self,
                                 batch: Dict,
                                 outputs: Dict,
                                 batch_idx: int,
                                 hand_type: str):
        """Save visualization of sample predictions"""

        try:
            vis_dir = self.output_dir / "visualizations"
            vis_dir.mkdir(exist_ok=True)

            # Extract first sample from batch
            images = batch['images'][0].cpu()  # [T, C, H, W]
            gt_joints = batch['hand_joints'][0].cpu()  # [T, 21, 3]

            if hand_type == 'left':
                pred_joints = outputs['left_hand_joints'][0].cpu()
            else:
                pred_joints = outputs['right_hand_joints'][0].cpu()

            # Create visualization for each frame
            for t in range(min(4, images.shape[0])):  # Visualize first 4 frames
                fig, ax = plt.subplots(1, 1, figsize=(8, 8))

                # Show image
                img = images[t].permute(1, 2, 0).numpy()
                img = np.clip(img, 0, 1)
                ax.imshow(img)

                # Project 3D joints to 2D (simplified projection)
                if 'keypoints_2d' in batch:
                    gt_2d = batch['keypoints_2d'][0, t].cpu().numpy()
                    pred_2d = self._project_joints_to_2d(pred_joints[t])

                    # Plot ground truth (blue) and predictions (red)
                    ax.scatter(gt_2d[:, 0], gt_2d[:, 1], c='blue', s=30, alpha=0.7, label='GT')
                    ax.scatter(pred_2d[:, 0], pred_2d[:, 1], c='red', s=30, alpha=0.7, label='Pred')

                    # Draw hand skeleton
                    self._draw_hand_skeleton(ax, gt_2d, color='blue', alpha=0.5)
                    self._draw_hand_skeleton(ax, pred_2d, color='red', alpha=0.5)

                ax.set_title(f'Sample {batch_idx}, Frame {t}')
                ax.legend()
                ax.axis('off')

                plt.tight_layout()
                plt.savefig(vis_dir / f"sample_{batch_idx}_frame_{t}.png", dpi=150, bbox_inches='tight')
                plt.close()

        except Exception as e:
            print(f"Warning: Could not save visualization: {e}")

    def _project_joints_to_2d(self, joints_3d: torch.Tensor) -> np.ndarray:
        """Simple projection of 3D joints to 2D"""
        # Simplified perspective projection
        fx, fy = 500.0, 500.0
        cx, cy = 128.0, 128.0

        x = joints_3d[:, 0]
        y = joints_3d[:, 1]
        z = torch.clamp(joints_3d[:, 2], min=0.1)

        u = (fx * x / z) + cx
        v = (fy * y / z) + cy

        return torch.stack([u, v], dim=1).numpy()

    def _draw_hand_skeleton(self, ax, joints_2d, color='blue', alpha=0.7):
        """Draw hand skeleton connections"""
        # MANO hand connections
        connections = [
            # Thumb
            [0, 1], [1, 2], [2, 3], [3, 4],
            # Index
            [0, 5], [5, 6], [6, 7], [7, 8],
            # Middle
            [0, 9], [9, 10], [10, 11], [11, 12],
            # Ring
            [0, 13], [13, 14], [14, 15], [15, 16],
            # Pinky
            [0, 17], [17, 18], [18, 19], [19, 20]
        ]

        for connection in connections:
            if len(joints_2d) > max(connection):
                x_coords = [joints_2d[connection[0], 0], joints_2d[connection[1], 0]]
                y_coords = [joints_2d[connection[0], 1], joints_2d[connection[1], 1]]
                ax.plot(x_coords, y_coords, color=color, alpha=alpha, linewidth=1)

    def _save_evaluation_results(self, metrics: Dict[str, float]):
        """Save evaluation results to files"""

        # Save detailed metrics
        results_file = self.output_dir / "evaluation_metrics.json"
        with open(results_file, 'w') as f:
            json.dump(metrics, f, indent=2)

        # Create summary report
        summary_file = self.output_dir / "evaluation_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("HaWoR Model Evaluation Summary\n")
            f.write("=" * 40 + "\n\n")

            # Key metrics
            f.write("Key Performance Metrics:\n")
            f.write(f"  MPJPE: {metrics.get('mpjpe', 0):.2f} mm\n")
            f.write(f"  PA-MPJPE: {metrics.get('pa_mpjpe', 0):.2f} mm\n")
            f.write(f"  Wrist Error: {metrics.get('wrist_error', 0):.2f} mm\n")
            f.write(f"  Fingertip Error: {metrics.get('fingertip_error', 0):.2f} mm\n")
            f.write(f"  Camera Trans Error: {metrics.get('camera_trans_error', 0):.4f} m\n")
            f.write(f"  Camera Rot Error: {metrics.get('camera_rot_error', 0):.4f} rad\n")
            f.write(f"  Motion Smoothness: {metrics.get('motion_smoothness', 0):.3f}\n\n")

            # Per-finger analysis
            f.write("Per-Finger Analysis:\n")
            fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
            for finger in fingers:
                error = metrics.get(f'{finger}_error', 0)
                f.write(f"  {finger.capitalize()}: {error:.2f} mm\n")

        print(f"📊 Evaluation results saved to {self.output_dir}")


def evaluate_hawor_model(model_path: str,
                        config_path: str,
                        data_root: str,
                        output_dir: str = "evaluation_results") -> Dict[str, float]:
    """
    Evaluate a trained HaWoR model

    Args:
        model_path: Path to trained model checkpoint
        config_path: Path to training configuration
        data_root: Path to ARCTIC dataset
        output_dir: Output directory for evaluation results

    Returns:
        Dictionary of evaluation metrics
    """

    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    # Load model
    model = create_hawor_model(config)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    # Create evaluation dataset
    eval_dataset = ARCTICDataset(
        data_root=data_root,
        subjects=['s01'],
        split='val',
        sequence_length=config.get('arctic', {}).get('sequence_length', 16),
        use_augmentation=False,
        max_sequences=2,  # Limit for evaluation
        images_per_sequence=20
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    # Run evaluation
    evaluator = HaWoREvaluator(model, device, output_dir)
    metrics = evaluator.evaluate_dataset(eval_loader)

    return metrics


if __name__ == "__main__":
    # Example usage
    model_path = "outputs/real_hawor_training/best_checkpoint.pth"
    config_path = "arctic_training_config.yaml"
    data_root = "thirdparty/arctic"

    if os.path.exists(model_path) and os.path.exists(config_path):
        metrics = evaluate_hawor_model(model_path, config_path, data_root)
        print("Evaluation completed!")
        print(f"MPJPE: {metrics['mpjpe']:.2f} mm")
        print(f"PA-MPJPE: {metrics['pa_mpjpe']:.2f} mm")
    else:
        print("Model or config file not found!")