#!/usr/bin/env python3
"""
Real HaWoR Model Architecture
Implements the actual HaWoR model for hand motion reconstruction from egocentric videos
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
import math

try:
    # Try to import vision transformer components
    from timm import create_model
    from timm.models.vision_transformer import VisionTransformer
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Warning: timm not available, using custom ViT implementation")

class PositionalEncoding(nn.Module):
    """Positional encoding for temporal sequences"""

    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]


class SimpleViT(nn.Module):
    """Simplified Vision Transformer for when timm is not available"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Patch embedding
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        # Class token and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=embed_dim*4, dropout=0.1)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Patch embedding
        x = self.patch_embed(x)  # [B, embed_dim, H//patch_size, W//patch_size]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]

        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add position embedding
        x = x + self.pos_embed

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x[:, 0]  # Return class token


class HandPoseRegressor(nn.Module):
    """Hand pose regression head"""

    def __init__(self, input_dim: int, num_joints: int = 21, mano_dim: int = 45):
        super().__init__()

        self.num_joints = num_joints
        self.mano_dim = mano_dim

        # Shared feature processing
        self.shared_fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # MANO pose parameters (45D)
        self.pose_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, mano_dim)
        )

        # MANO shape parameters (10D)
        self.shape_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

        # Hand translation (3D)
        self.trans_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

        # Hand global rotation (3D)
        self.rot_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

        # 3D joint positions (21 x 3)
        self.joints_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_joints * 3)
        )

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        shared_feat = self.shared_fc(features)

        outputs = {
            'hand_pose': self.pose_head(shared_feat),      # [B, 45]
            'hand_shape': self.shape_head(shared_feat),    # [B, 10]
            'hand_trans': self.trans_head(shared_feat),    # [B, 3]
            'hand_rot': self.rot_head(shared_feat),        # [B, 3]
            'hand_joints': self.joints_head(shared_feat).view(-1, self.num_joints, 3)  # [B, 21, 3]
        }

        return outputs


class CameraPoseRegressor(nn.Module):
    """Camera pose regression for SLAM component"""

    def __init__(self, input_dim: int):
        super().__init__()

        self.camera_head = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 6)  # 3D translation + 3D rotation
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.camera_head(features)


class MotionInfillerNetwork(nn.Module):
    """Motion infiller network for handling missing frames"""

    def __init__(self, pose_dim: int = 45, hidden_dim: int = 256, num_layers: int = 2):
        super().__init__()

        self.pose_dim = pose_dim
        self.hidden_dim = hidden_dim

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=pose_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # bidirectional
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, pose_dim)
        )

        # Confidence prediction
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, pose_sequence: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Fill missing poses in a sequence

        Args:
            pose_sequence: [B, T, pose_dim] - input pose sequence (may have missing frames)
            mask: [B, T] - binary mask indicating valid frames

        Returns:
            Dictionary with filled poses and confidence scores
        """
        B, T, _ = pose_sequence.shape

        # Pass through LSTM
        lstm_out, _ = self.lstm(pose_sequence)  # [B, T, hidden_dim*2]

        # Generate filled poses
        filled_poses = self.output_proj(lstm_out)  # [B, T, pose_dim]

        # Generate confidence scores
        confidence = self.confidence_head(lstm_out).squeeze(-1)  # [B, T]

        # If mask is provided, only fill missing frames
        if mask is not None:
            filled_poses = torch.where(mask.unsqueeze(-1), pose_sequence, filled_poses)

        return {
            'filled_poses': filled_poses,
            'confidence': confidence
        }


class HaWoRModel(nn.Module):
    """
    Complete HaWoR Model for World-Space Hand Motion Reconstruction

    Integrates:
    1. Vision Transformer backbone for feature extraction
    2. Hand pose regression
    3. Camera pose estimation (SLAM component)
    4. Motion infiller for temporal consistency
    """

    def __init__(self,
                 img_size: int = 256,
                 patch_size: int = 16,
                 backbone_type: str = 'vit_small',
                 pretrained: bool = True,
                 num_joints: int = 21,
                 sequence_length: int = 16):
        super().__init__()

        self.img_size = img_size
        self.sequence_length = sequence_length
        self.num_joints = num_joints

        # Vision Transformer Backbone
        if TIMM_AVAILABLE and backbone_type.startswith('vit'):
            try:
                self.backbone = create_model(
                    backbone_type,
                    pretrained=pretrained,
                    img_size=img_size,
                    num_classes=0  # Remove classification head
                )
                embed_dim = self.backbone.embed_dim
            except:
                print(f"Warning: Could not load {backbone_type}, using simple ViT")
                self.backbone = SimpleViT(img_size=img_size, patch_size=patch_size)
                embed_dim = 768
        else:
            self.backbone = SimpleViT(img_size=img_size, patch_size=patch_size)
            embed_dim = 768

        # Feature dimension
        self.feature_dim = embed_dim

        # Temporal encoding for video sequences
        self.temporal_encoder = PositionalEncoding(embed_dim, max_len=sequence_length)

        # Temporal transformer for sequence modeling
        self.temporal_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=8,
                dim_feedforward=embed_dim * 2,
                dropout=0.1
            ),
            num_layers=3
        )

        # Hand pose regression heads
        self.left_hand_regressor = HandPoseRegressor(embed_dim, num_joints)
        self.right_hand_regressor = HandPoseRegressor(embed_dim, num_joints)

        # Camera pose regressor (SLAM component)
        self.camera_regressor = CameraPoseRegressor(embed_dim)

        # Motion infiller network
        self.motion_infiller = MotionInfillerNetwork(pose_dim=45)

        # Feature fusion for multi-hand scenarios
        self.hand_fusion = nn.MultiheadAttention(embed_dim, num_heads=8, dropout=0.1)

        print(f"HaWoR Model initialized:")
        print(f"  - Backbone: {backbone_type}")
        print(f"  - Feature dim: {embed_dim}")
        print(f"  - Image size: {img_size}")
        print(f"  - Sequence length: {sequence_length}")

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input images

        Args:
            images: [B, T, C, H, W] - batch of image sequences

        Returns:
            features: [B, T, feature_dim] - extracted features
        """
        B, T, C, H, W = images.shape

        # Reshape for batch processing
        images_flat = images.view(B * T, C, H, W)

        # Extract features using backbone
        features_flat = self.backbone(images_flat)  # [B*T, feature_dim]

        # Reshape back to sequence format
        features = features_flat.view(B, T, -1)  # [B, T, feature_dim]

        return features

    def process_temporal_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Process features with temporal modeling

        Args:
            features: [B, T, feature_dim]

        Returns:
            temporal_features: [B, T, feature_dim]
        """
        # Add temporal positional encoding
        features = features.transpose(0, 1)  # [T, B, feature_dim]
        features = self.temporal_encoder(features)

        # Apply temporal transformer
        temporal_features = self.temporal_transformer(features)

        # Reshape back
        temporal_features = temporal_features.transpose(0, 1)  # [B, T, feature_dim]

        return temporal_features

    def forward(self, images: torch.Tensor,
                hand_masks: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of HaWoR model

        Args:
            images: [B, T, C, H, W] - input image sequences
            hand_masks: Optional masks indicating hand presence

        Returns:
            Dictionary containing all predictions
        """
        B, T = images.shape[:2]

        # Extract visual features
        features = self.extract_features(images)  # [B, T, feature_dim]

        # Process temporal dependencies
        temporal_features = self.process_temporal_features(features)  # [B, T, feature_dim]

        # Predict hand poses for each frame
        left_hand_preds = []
        right_hand_preds = []
        camera_preds = []

        for t in range(T):
            frame_features = temporal_features[:, t]  # [B, feature_dim]

            # Predict left hand
            left_pred = self.left_hand_regressor(frame_features)
            left_hand_preds.append(left_pred)

            # Predict right hand
            right_pred = self.right_hand_regressor(frame_features)
            right_hand_preds.append(right_pred)

            # Predict camera pose
            camera_pred = self.camera_regressor(frame_features)  # [B, 6]
            camera_preds.append(camera_pred)

        # Stack temporal predictions
        left_hand_outputs = self._stack_hand_predictions(left_hand_preds)
        right_hand_outputs = self._stack_hand_predictions(right_hand_preds)
        camera_poses = torch.stack(camera_preds, dim=1)  # [B, T, 6]

        # Apply motion infiller to smooth poses
        left_filled = self.motion_infiller(left_hand_outputs['hand_pose'])
        right_filled = self.motion_infiller(right_hand_outputs['hand_pose'])

        # Combine outputs
        outputs = {
            # Left hand
            'left_hand_pose': left_filled['filled_poses'],
            'left_hand_shape': left_hand_outputs['hand_shape'],
            'left_hand_trans': left_hand_outputs['hand_trans'],
            'left_hand_rot': left_hand_outputs['hand_rot'],
            'left_hand_joints': left_hand_outputs['hand_joints'],
            'left_confidence': left_filled['confidence'],

            # Right hand
            'right_hand_pose': right_filled['filled_poses'],
            'right_hand_shape': right_hand_outputs['hand_shape'],
            'right_hand_trans': right_hand_outputs['hand_trans'],
            'right_hand_rot': right_hand_outputs['hand_rot'],
            'right_hand_joints': right_hand_outputs['hand_joints'],
            'right_confidence': right_filled['confidence'],

            # Camera
            'camera_pose': camera_poses,

            # Features for analysis
            'features': temporal_features
        }

        return outputs

    def _stack_hand_predictions(self, hand_preds: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Stack hand predictions across time"""
        stacked = {}
        for key in hand_preds[0].keys():
            stacked[key] = torch.stack([pred[key] for pred in hand_preds], dim=1)
        return stacked

    def get_loss_predictions(self, outputs: Dict[str, torch.Tensor],
                           hand_type: str = 'both') -> Dict[str, torch.Tensor]:
        """
        Format predictions for loss computation

        Args:
            outputs: Model outputs
            hand_type: 'left', 'right', or 'both'

        Returns:
            Formatted predictions for loss function
        """
        loss_preds = {}

        if hand_type in ['left', 'both']:
            loss_preds.update({
                'hand_pose': outputs['left_hand_pose'],
                'hand_shape': outputs['left_hand_shape'],
                'hand_trans': outputs['left_hand_trans'],
                'hand_rot': outputs['left_hand_rot'],
                'hand_joints': outputs['left_hand_joints']
            })

        if hand_type in ['right', 'both'] and hand_type != 'left':
            # For 'right' only, replace with right hand
            # For 'both', we might need to handle differently
            loss_preds.update({
                'hand_pose': outputs['right_hand_pose'],
                'hand_shape': outputs['right_hand_shape'],
                'hand_trans': outputs['right_hand_trans'],
                'hand_rot': outputs['right_hand_rot'],
                'hand_joints': outputs['right_hand_joints']
            })

        # Always include camera
        loss_preds['camera_pose'] = outputs['camera_pose']

        return loss_preds


def create_hawor_model(config: Dict) -> HaWoRModel:
    """
    Factory function to create HaWoR model from configuration

    Args:
        config: Model configuration dictionary

    Returns:
        Initialized HaWoR model
    """
    model_config = config.get('model', {})

    model = HaWoRModel(
        img_size=model_config.get('img_size', 256),
        patch_size=model_config.get('patch_size', 16),
        backbone_type=model_config.get('backbone_type', 'vit_small'),
        pretrained=model_config.get('pretrained', True),
        num_joints=model_config.get('num_joints', 21),
        sequence_length=config.get('arctic', {}).get('sequence_length', 16)
    )

    return model