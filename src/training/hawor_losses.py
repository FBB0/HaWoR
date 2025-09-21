#!/usr/bin/env python3
"""
Real Loss Functions for HaWoR Training
Implements actual loss functions for hand pose estimation and camera trajectory learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
import math

class HaWoRLossFunction(nn.Module):
    """
    Complete loss function for HaWoR training
    Combines multiple loss components for comprehensive hand motion reconstruction
    """

    def __init__(self,
                 lambda_keypoint: float = 10.0,
                 lambda_mano_pose: float = 1.0,
                 lambda_mano_shape: float = 0.1,
                 lambda_temporal: float = 5.0,
                 lambda_camera: float = 1.0,
                 lambda_reprojection: float = 100.0,
                 lambda_consistency: float = 2.0):
        super().__init__()

        # Loss weights
        self.lambda_keypoint = lambda_keypoint
        self.lambda_mano_pose = lambda_mano_pose
        self.lambda_mano_shape = lambda_mano_shape
        self.lambda_temporal = lambda_temporal
        self.lambda_camera = lambda_camera
        self.lambda_reprojection = lambda_reprojection
        self.lambda_consistency = lambda_consistency

        # Loss components
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.smooth_l1 = nn.SmoothL1Loss()

    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor],
                camera_params: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Compute complete HaWoR loss

        Args:
            predictions: Model predictions containing:
                - hand_pose: MANO hand pose parameters [B, T, 45]
                - hand_shape: MANO shape parameters [B, 10]
                - hand_trans: Hand translation [B, T, 3]
                - hand_rot: Hand global rotation [B, T, 3]
                - hand_verts: Hand vertices [B, T, 778, 3]
                - hand_joints: Hand joints [B, T, 21, 3]
                - camera_pose: Camera poses [B, T, 6] (translation + rotation)

            targets: Ground truth targets with same structure
            camera_params: Camera intrinsics

        Returns:
            Dictionary of loss components and total loss
        """

        losses = {}
        total_loss = 0.0

        # 1. MANO Parameter Losses
        if 'hand_pose' in predictions and 'hand_pose' in targets:
            mano_pose_loss = self.mano_pose_loss(predictions['hand_pose'], targets['hand_pose'])
            losses['mano_pose'] = mano_pose_loss
            total_loss += self.lambda_mano_pose * mano_pose_loss

        if 'hand_shape' in predictions and 'hand_shape' in targets:
            mano_shape_loss = self.mano_shape_loss(predictions['hand_shape'], targets['hand_shape'])
            losses['mano_shape'] = mano_shape_loss
            total_loss += self.lambda_mano_shape * mano_shape_loss

        # 2. 3D Keypoint Loss
        if 'hand_joints' in predictions and 'hand_joints' in targets:
            keypoint_loss = self.keypoint_3d_loss(predictions['hand_joints'], targets['hand_joints'])
            losses['keypoint_3d'] = keypoint_loss
            total_loss += self.lambda_keypoint * keypoint_loss

        # 3. Reprojection Loss
        if camera_params is not None and 'hand_joints' in predictions:
            if 'keypoints_2d' in targets:
                reproj_loss = self.reprojection_loss(
                    predictions['hand_joints'], targets['keypoints_2d'], camera_params
                )
                losses['reprojection'] = reproj_loss
                total_loss += self.lambda_reprojection * reproj_loss

        # 4. Temporal Consistency Loss
        temporal_loss = self.temporal_consistency_loss(predictions)
        if temporal_loss is not None:
            losses['temporal'] = temporal_loss
            total_loss += self.lambda_temporal * temporal_loss

        # 5. Camera Trajectory Loss
        if 'camera_pose' in predictions and 'camera_pose' in targets:
            camera_loss = self.camera_trajectory_loss(predictions['camera_pose'], targets['camera_pose'])
            losses['camera'] = camera_loss
            total_loss += self.lambda_camera * camera_loss

        # 6. Hand-World Consistency Loss
        consistency_loss = self.hand_world_consistency_loss(predictions)
        if consistency_loss is not None:
            losses['consistency'] = consistency_loss
            total_loss += self.lambda_consistency * consistency_loss

        losses['total'] = total_loss
        return losses

    def mano_pose_loss(self, pred_pose: torch.Tensor, gt_pose: torch.Tensor) -> torch.Tensor:
        """
        MANO pose parameter loss with angle-aware weighting

        Args:
            pred_pose: Predicted MANO pose [B, T, 45]
            gt_pose: Ground truth MANO pose [B, T, 45]
        """
        # Reshape to handle batch and time dimensions
        pred_flat = pred_pose.view(-1, 45)
        gt_flat = gt_pose.view(-1, 45)

        # L2 loss on pose parameters
        pose_loss = self.mse_loss(pred_flat, gt_flat)

        # Add regularization to prevent extreme poses
        pose_reg = torch.mean(torch.sum(pred_flat**2, dim=-1))

        return pose_loss + 0.01 * pose_reg

    def mano_shape_loss(self, pred_shape: torch.Tensor, gt_shape: torch.Tensor) -> torch.Tensor:
        """
        MANO shape parameter loss

        Args:
            pred_shape: Predicted MANO shape [B, 10]
            gt_shape: Ground truth MANO shape [B, 10]
        """
        # Shape parameters are typically stable across time
        shape_loss = self.mse_loss(pred_shape, gt_shape)

        # Regularization to keep shapes reasonable
        shape_reg = torch.mean(torch.sum(pred_shape**2, dim=-1))

        return shape_loss + 0.001 * shape_reg

    def keypoint_3d_loss(self, pred_joints: torch.Tensor, gt_joints: torch.Tensor,
                        joint_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        3D keypoint loss with optional joint weighting

        Args:
            pred_joints: Predicted 3D joints [B, T, 21, 3]
            gt_joints: Ground truth 3D joints [B, T, 21, 3]
            joint_weights: Optional weights for different joints [21]
        """
        # Flatten batch and time dimensions
        pred_flat = pred_joints.view(-1, 21, 3)  # [B*T, 21, 3]
        gt_flat = gt_joints.view(-1, 21, 3)      # [B*T, 21, 3]

        # L2 distance for each joint
        joint_distances = torch.norm(pred_flat - gt_flat, dim=-1)  # [B*T, 21]

        # Apply joint weights if provided
        if joint_weights is not None:
            joint_weights = joint_weights.to(pred_joints.device)
            joint_distances = joint_distances * joint_weights.unsqueeze(0)

        # Mean over joints and samples
        keypoint_loss = torch.mean(joint_distances)

        return keypoint_loss

    def reprojection_loss(self, joints_3d: torch.Tensor, keypoints_2d: torch.Tensor,
                         camera_params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        2D reprojection loss using camera parameters

        Args:
            joints_3d: 3D joints in camera frame [B, T, 21, 3]
            keypoints_2d: 2D keypoints [B, T, 21, 2]
            camera_params: Camera intrinsics
        """
        # Get camera intrinsics
        fx = camera_params.get('fx', 500.0)
        fy = camera_params.get('fy', 500.0)
        cx = camera_params.get('cx', 320.0)
        cy = camera_params.get('cy', 240.0)

        # Project 3D joints to 2D
        x_3d = joints_3d[..., 0]  # [B, T, 21]
        y_3d = joints_3d[..., 1]  # [B, T, 21]
        z_3d = joints_3d[..., 2]  # [B, T, 21]

        # Avoid division by zero
        z_3d = torch.clamp(z_3d, min=0.1)

        # Project to image coordinates
        x_proj = (fx * x_3d / z_3d) + cx
        y_proj = (fy * y_3d / z_3d) + cy

        projected_2d = torch.stack([x_proj, y_proj], dim=-1)  # [B, T, 21, 2]

        # Compute reprojection error
        reproj_error = self.mse_loss(projected_2d, keypoints_2d)

        return reproj_error

    def temporal_consistency_loss(self, predictions: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Temporal consistency loss for smooth motion

        Args:
            predictions: Model predictions with temporal dimension
        """
        temporal_losses = []

        # Hand pose temporal consistency
        if 'hand_pose' in predictions:
            hand_pose = predictions['hand_pose']  # [B, T, 45]
            if hand_pose.shape[1] > 1:  # Need at least 2 frames
                pose_diff = hand_pose[:, 1:] - hand_pose[:, :-1]  # [B, T-1, 45]
                pose_smooth = torch.mean(torch.sum(pose_diff**2, dim=-1))
                temporal_losses.append(pose_smooth)

        # Hand translation temporal consistency
        if 'hand_trans' in predictions:
            hand_trans = predictions['hand_trans']  # [B, T, 3]
            if hand_trans.shape[1] > 1:
                trans_diff = hand_trans[:, 1:] - hand_trans[:, :-1]  # [B, T-1, 3]
                trans_smooth = torch.mean(torch.sum(trans_diff**2, dim=-1))
                temporal_losses.append(trans_smooth)

        # Camera pose temporal consistency
        if 'camera_pose' in predictions:
            camera_pose = predictions['camera_pose']  # [B, T, 6]
            if camera_pose.shape[1] > 1:
                cam_diff = camera_pose[:, 1:] - camera_pose[:, :-1]  # [B, T-1, 6]
                cam_smooth = torch.mean(torch.sum(cam_diff**2, dim=-1))
                temporal_losses.append(cam_smooth)

        if temporal_losses:
            return sum(temporal_losses) / len(temporal_losses)
        return None

    def camera_trajectory_loss(self, pred_camera: torch.Tensor, gt_camera: torch.Tensor) -> torch.Tensor:
        """
        Camera trajectory loss for SLAM component

        Args:
            pred_camera: Predicted camera poses [B, T, 6] (translation + rotation)
            gt_camera: Ground truth camera poses [B, T, 6]
        """
        # Separate translation and rotation components
        pred_trans = pred_camera[..., :3]  # [B, T, 3]
        pred_rot = pred_camera[..., 3:]    # [B, T, 3]
        gt_trans = gt_camera[..., :3]      # [B, T, 3]
        gt_rot = gt_camera[..., 3:]        # [B, T, 3]

        # Translation loss
        trans_loss = self.mse_loss(pred_trans, gt_trans)

        # Rotation loss (using angle representation)
        rot_loss = self.mse_loss(pred_rot, gt_rot)

        return trans_loss + rot_loss

    def hand_world_consistency_loss(self, predictions: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Consistency loss between hand poses in world coordinates

        Args:
            predictions: Model predictions
        """
        if 'hand_joints' not in predictions or 'camera_pose' not in predictions:
            return None

        # This loss ensures that hand poses are consistent when transformed to world coordinates
        # It's particularly important for egocentric video where both hands and camera move

        hand_joints = predictions['hand_joints']    # [B, T, 21, 3] in camera frame
        camera_pose = predictions['camera_pose']    # [B, T, 6]

        if hand_joints.shape[1] < 2:  # Need at least 2 frames
            return None

        # Transform hands to world coordinates using camera pose
        # This is a simplified version - full implementation would use proper SE(3) transforms
        camera_trans = camera_pose[..., :3]  # [B, T, 3]

        # Add camera translation to hand positions (simplified world transform)
        hand_world = hand_joints + camera_trans.unsqueeze(-2)  # [B, T, 21, 3]

        # Consistency loss: hands should move smoothly in world coordinates
        if hand_world.shape[1] > 1:
            world_diff = hand_world[:, 1:] - hand_world[:, :-1]  # [B, T-1, 21, 3]
            consistency_loss = torch.mean(torch.sum(world_diff**2, dim=-1))
            return consistency_loss

        return None


class AdaptiveLossWeighting(nn.Module):
    """
    Adaptive loss weighting that adjusts loss component weights during training
    """

    def __init__(self, initial_weights: Dict[str, float], adaptation_rate: float = 0.1):
        super().__init__()
        self.adaptation_rate = adaptation_rate

        # Initialize learnable weights
        self.log_weights = nn.ParameterDict({
            name: nn.Parameter(torch.log(torch.tensor(weight)))
            for name, weight in initial_weights.items()
        })

    def forward(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute weighted total loss with adaptive weights

        Args:
            losses: Dictionary of loss components

        Returns:
            Weighted total loss
        """
        total_loss = 0.0

        for name, loss_value in losses.items():
            if name in self.log_weights and name != 'total':
                weight = torch.exp(self.log_weights[name])
                total_loss += weight * loss_value

        return total_loss

    def get_current_weights(self) -> Dict[str, float]:
        """Get current loss weights"""
        return {name: torch.exp(param).item() for name, param in self.log_weights.items()}


def create_joint_weights() -> torch.Tensor:
    """
    Create joint weights for hand keypoints
    Higher weights for more important joints (fingertips, etc.)
    """
    # MANO hand joint weights (21 joints)
    # Indices: 0=wrist, 1-4=thumb, 5-8=index, 9-12=middle, 13-16=ring, 17-20=pinky
    weights = torch.ones(21)

    # Higher weight for wrist (most important for hand position)
    weights[0] = 2.0

    # Higher weights for fingertips
    fingertip_indices = [4, 8, 12, 16, 20]  # tip of each finger
    weights[fingertip_indices] = 1.5

    # Higher weights for finger joints (more visible and important)
    finger_joint_indices = [2, 3, 6, 7, 10, 11, 14, 15, 18, 19]
    weights[finger_joint_indices] = 1.2

    return weights