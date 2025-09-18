#!/usr/bin/env python3
"""
ARCTIC Evaluation Framework for HaWoR
Compare HaWoR predictions with ARCTIC ground truth data
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import json
import cv2
from typing import Dict, List, Tuple, Optional
import argparse
from dataclasses import dataclass
import logging

# Add HaWoR to path
sys.path.append(str(Path(__file__).parent))

from hawor_interface import HaWoRInterface
from lib.models.hawor import HAWOR
from lib.utils.geometry import rot6d_to_rotmat, angle_axis_to_rotation_matrix
from lib.utils.rotation import rotation_matrix_to_angle_axis

@dataclass
class ArcticEvaluationMetrics:
    """Container for ARCTIC evaluation metrics"""
    # 3D Keypoint Metrics
    mpjpe_3d: float = 0.0  # Mean Per Joint Position Error (3D)
    pck_3d_5mm: float = 0.0  # Percentage of Correct Keypoints (3D, 5mm threshold)
    pck_3d_10mm: float = 0.0  # Percentage of Correct Keypoints (3D, 10mm threshold)
    pck_3d_15mm: float = 0.0  # Percentage of Correct Keypoints (3D, 15mm threshold)
    
    # 2D Keypoint Metrics
    mpjpe_2d: float = 0.0  # Mean Per Joint Position Error (2D)
    pck_2d_5px: float = 0.0  # Percentage of Correct Keypoints (2D, 5px threshold)
    pck_2d_10px: float = 0.0  # Percentage of Correct Keypoints (2D, 10px threshold)
    
    # MANO Parameter Metrics
    mano_pose_error: float = 0.0  # MANO pose parameter error
    mano_shape_error: float = 0.0  # MANO shape parameter error
    mano_global_orient_error: float = 0.0  # MANO global orientation error
    
    # Mesh Metrics
    mesh_vertices_error: float = 0.0  # Mesh vertices error
    mesh_faces_error: float = 0.0  # Mesh faces error
    
    # Detection Metrics
    hand_detection_rate: float = 0.0  # Percentage of frames with detected hands
    left_hand_detection_rate: float = 0.0
    right_hand_detection_rate: float = 0.0
    
    # Temporal Metrics
    temporal_consistency: float = 0.0  # Temporal smoothness of predictions
    
    def to_dict(self) -> Dict:
        """Convert metrics to dictionary"""
        return {
            'mpjpe_3d': self.mpjpe_3d,
            'pck_3d_5mm': self.pck_3d_5mm,
            'pck_3d_10mm': self.pck_3d_10mm,
            'pck_3d_15mm': self.pck_3d_15mm,
            'mpjpe_2d': self.mpjpe_2d,
            'pck_2d_5px': self.pck_2d_5px,
            'pck_2d_10px': self.pck_2d_10px,
            'mano_pose_error': self.mano_pose_error,
            'mano_shape_error': self.mano_shape_error,
            'mano_global_orient_error': self.mano_global_orient_error,
            'mesh_vertices_error': self.mesh_vertices_error,
            'mesh_faces_error': self.mesh_faces_error,
            'hand_detection_rate': self.hand_detection_rate,
            'left_hand_detection_rate': self.left_hand_detection_rate,
            'right_hand_detection_rate': self.right_hand_detection_rate,
            'temporal_consistency': self.temporal_consistency
        }

class ArcticLossFunction:
    """Loss function for comparing HaWoR predictions with ARCTIC ground truth"""
    
    def __init__(self, loss_weights: Optional[Dict] = None):
        """
        Initialize ARCTIC loss function
        
        Args:
            loss_weights: Dictionary of loss weights for different components
        """
        self.loss_weights = loss_weights or {
            'KEYPOINTS_3D': 0.05,
            'KEYPOINTS_2D': 0.01,
            'GLOBAL_ORIENT': 0.001,
            'HAND_POSE': 0.001,
            'BETAS': 0.0005,
            'MESH_VERTICES': 0.01,
            'TEMPORAL_CONSISTENCY': 0.001
        }
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.smooth_l1_loss = nn.SmoothL1Loss()
        
    def compute_keypoint_3d_loss(self, pred_keypoints_3d: torch.Tensor, 
                                gt_keypoints_3d: torch.Tensor,
                                pelvis_id: int = 0) -> torch.Tensor:
        """
        Compute 3D keypoint loss with pelvis alignment
        
        Args:
            pred_keypoints_3d: Predicted 3D keypoints [B, N, 3]
            gt_keypoints_3d: Ground truth 3D keypoints [B, N, 3]
            pelvis_id: Index of pelvis joint for alignment
            
        Returns:
            3D keypoint loss
        """
        # Align predictions to ground truth using pelvis
        pred_aligned = pred_keypoints_3d - pred_keypoints_3d[:, pelvis_id:pelvis_id+1, :]
        gt_aligned = gt_keypoints_3d - gt_keypoints_3d[:, pelvis_id:pelvis_id+1, :]
        
        # Compute MPJPE
        mpjpe = torch.norm(pred_aligned - gt_aligned, dim=-1).mean()
        
        return mpjpe
    
    def compute_keypoint_2d_loss(self, pred_keypoints_2d: torch.Tensor,
                                gt_keypoints_2d: torch.Tensor) -> torch.Tensor:
        """
        Compute 2D keypoint loss
        
        Args:
            pred_keypoints_2d: Predicted 2D keypoints [B, N, 2]
            gt_keypoints_2d: Ground truth 2D keypoints [B, N, 2]
            
        Returns:
            2D keypoint loss
        """
        # Compute MPJPE for 2D
        mpjpe = torch.norm(pred_keypoints_2d - gt_keypoints_2d, dim=-1).mean()
        
        return mpjpe
    
    def compute_mano_parameter_loss(self, pred_params: torch.Tensor,
                                   gt_params: torch.Tensor) -> torch.Tensor:
        """
        Compute MANO parameter loss
        
        Args:
            pred_params: Predicted MANO parameters [B, N]
            gt_params: Ground truth MANO parameters [B, N]
            
        Returns:
            MANO parameter loss
        """
        return self.smooth_l1_loss(pred_params, gt_params)
    
    def compute_mesh_vertices_loss(self, pred_vertices: torch.Tensor,
                                  gt_vertices: torch.Tensor) -> torch.Tensor:
        """
        Compute mesh vertices loss
        
        Args:
            pred_vertices: Predicted mesh vertices [B, V, 3]
            gt_vertices: Ground truth mesh vertices [B, V, 3]
            
        Returns:
            Mesh vertices loss
        """
        return self.l1_loss(pred_vertices, gt_vertices)
    
    def compute_temporal_consistency_loss(self, pred_sequence: torch.Tensor) -> torch.Tensor:
        """
        Compute temporal consistency loss
        
        Args:
            pred_sequence: Predicted sequence [T, B, N]
            
        Returns:
            Temporal consistency loss
        """
        if pred_sequence.shape[0] < 2:
            return torch.tensor(0.0, device=pred_sequence.device)
        
        # Compute difference between consecutive frames
        temporal_diff = pred_sequence[1:] - pred_sequence[:-1]
        temporal_loss = torch.norm(temporal_diff, dim=-1).mean()
        
        return temporal_loss
    
    def compute_total_loss(self, pred_output: Dict, gt_data: Dict) -> Tuple[torch.Tensor, Dict]:
        """
        Compute total loss for ARCTIC evaluation
        
        Args:
            pred_output: HaWoR prediction output
            gt_data: ARCTIC ground truth data
            
        Returns:
            Total loss and individual loss components
        """
        losses = {}
        
        # 3D Keypoint Loss
        if 'pred_keypoints_3d' in pred_output and 'gt_keypoints_3d' in gt_data:
            losses['keypoints_3d'] = self.compute_keypoint_3d_loss(
                pred_output['pred_keypoints_3d'],
                gt_data['gt_keypoints_3d']
            )
        
        # 2D Keypoint Loss
        if 'pred_keypoints_2d' in pred_output and 'gt_keypoints_2d' in gt_data:
            losses['keypoints_2d'] = self.compute_keypoint_2d_loss(
                pred_output['pred_keypoints_2d'],
                gt_data['gt_keypoints_2d']
            )
        
        # MANO Parameter Losses
        if 'pred_mano_params' in pred_output and 'gt_mano_params' in gt_data:
            pred_mano = pred_output['pred_mano_params']
            gt_mano = gt_data['gt_mano_params']
            
            if 'global_orient' in pred_mano and 'global_orient' in gt_mano:
                losses['global_orient'] = self.compute_mano_parameter_loss(
                    pred_mano['global_orient'].reshape(pred_mano['global_orient'].shape[0], -1),
                    gt_mano['global_orient'].reshape(gt_mano['global_orient'].shape[0], -1)
                )
            
            if 'hand_pose' in pred_mano and 'hand_pose' in gt_mano:
                losses['hand_pose'] = self.compute_mano_parameter_loss(
                    pred_mano['hand_pose'].reshape(pred_mano['hand_pose'].shape[0], -1),
                    gt_mano['hand_pose'].reshape(gt_mano['hand_pose'].shape[0], -1)
                )
            
            if 'betas' in pred_mano and 'betas' in gt_mano:
                losses['betas'] = self.compute_mano_parameter_loss(
                    pred_mano['betas'],
                    gt_mano['betas']
                )
        
        # Mesh Vertices Loss
        if 'pred_vertices' in pred_output and 'gt_vertices' in gt_data:
            losses['mesh_vertices'] = self.compute_mesh_vertices_loss(
                pred_output['pred_vertices'],
                gt_data['gt_vertices']
            )
        
        # Temporal Consistency Loss
        if 'pred_sequence' in pred_output:
            losses['temporal_consistency'] = self.compute_temporal_consistency_loss(
                pred_output['pred_sequence']
            )
        
        # Compute weighted total loss
        total_loss = torch.tensor(0.0, device=next(iter(losses.values())).device)
        for loss_name, loss_value in losses.items():
            weight = self.loss_weights.get(loss_name.upper(), 1.0)
            total_loss += weight * loss_value
            losses[f'weighted_{loss_name}'] = weight * loss_value
        
        return total_loss, losses

class ArcticEvaluator:
    """Evaluator for comparing HaWoR with ARCTIC ground truth"""
    
    def __init__(self, hawor_interface: HaWoRInterface, 
                 arctic_data_root: str = "./thirdparty/arctic/unpack/arctic_data/data"):
        """
        Initialize ARCTIC evaluator
        
        Args:
            hawor_interface: HaWoR interface for predictions
            arctic_data_root: Root directory of ARCTIC data
        """
        self.hawor_interface = hawor_interface
        self.arctic_data_root = Path(arctic_data_root)
        self.loss_function = ArcticLossFunction()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def load_arctic_sequence(self, subject: str, sequence: str) -> Dict:
        """
        Load ARCTIC sequence data
        
        Args:
            subject: Subject ID (e.g., 's01')
            sequence: Sequence name (e.g., 'box_grab_01')
            
        Returns:
            Dictionary containing ARCTIC ground truth data
        """
        seq_path = self.arctic_data_root / "raw_seqs" / subject
        
        # Load MANO parameters
        mano_file = seq_path / f"{sequence}.mano.npy"
        if mano_file.exists():
            mano_data = np.load(mano_file, allow_pickle=True).item()
        else:
            raise FileNotFoundError(f"MANO file not found: {mano_file}")
        
        # Load egocentric camera data
        egocam_file = seq_path / f"{sequence}.egocam.dist.npy"
        if egocam_file.exists():
            egocam_data = np.load(egocam_file, allow_pickle=True).item()
        else:
            raise FileNotFoundError(f"Egocentric camera file not found: {egocam_file}")
        
        # Load object data
        object_file = seq_path / f"{sequence}.object.npy"
        if object_file.exists():
            object_data = np.load(object_file, allow_pickle=True).item()
        else:
            object_data = None
        
        return {
            'mano': mano_data,
            'egocam': egocam_data,
            'object': object_data,
            'subject': subject,
            'sequence': sequence
        }
    
    def convert_arctic_to_hawor_format(self, arctic_data: Dict) -> Dict:
        """
        Convert ARCTIC data to HaWoR format
        
        Args:
            arctic_data: ARCTIC ground truth data
            
        Returns:
            Data in HaWoR format
        """
        mano_data = arctic_data['mano']
        egocam_data = arctic_data['egocam']
        
        # Convert MANO parameters to HaWoR format
        # ARCTIC MANO format: rot, pose, trans, shape, fitting_err
        # HaWoR format: global_orient, hand_pose, betas
        
        # Global orientation (rot)
        global_orient = torch.tensor(mano_data['rot'], dtype=torch.float32)
        
        # Hand pose (pose)
        hand_pose = torch.tensor(mano_data['pose'], dtype=torch.float32)
        
        # Shape parameters (betas)
        betas = torch.tensor(mano_data['shape'], dtype=torch.float32)
        
        # Translation
        trans = torch.tensor(mano_data['trans'], dtype=torch.float32)
        
        # Camera intrinsics
        intrinsics = torch.tensor(egocam_data['intrinsics'], dtype=torch.float32)
        
        return {
            'gt_mano_params': {
                'global_orient': global_orient,
                'hand_pose': hand_pose,
                'betas': betas
            },
            'gt_trans': trans,
            'gt_intrinsics': intrinsics,
            'gt_camera_poses': {
                'R': torch.tensor(egocam_data['R_k_cam_np'], dtype=torch.float32),
                'T': torch.tensor(egocam_data['T_k_cam_np'], dtype=torch.float32)
            }
        }
    
    def evaluate_sequence(self, subject: str, sequence: str, 
                         image_path: Optional[str] = None) -> ArcticEvaluationMetrics:
        """
        Evaluate HaWoR on a single ARCTIC sequence
        
        Args:
            subject: Subject ID (e.g., 's01')
            sequence: Sequence name (e.g., 'box_grab_01')
            image_path: Path to input image (if None, will use ARCTIC images)
            
        Returns:
            Evaluation metrics
        """
        self.logger.info(f"Evaluating sequence: {subject}/{sequence}")
        
        # Load ARCTIC ground truth
        arctic_data = self.load_arctic_sequence(subject, sequence)
        gt_data = self.convert_arctic_to_hawor_format(arctic_data)
        
        # Get HaWoR prediction
        if image_path is None:
            # Use ARCTIC images
            image_path = self.arctic_data_root / "cropped_images" / subject / sequence / "0" / "00023.jpg"
        
        if not Path(image_path).exists():
            self.logger.warning(f"Image not found: {image_path}")
            return ArcticEvaluationMetrics()
        
        # Run HaWoR prediction
        try:
            pred_output = self.hawor_interface.process_video(str(image_path))
        except Exception as e:
            self.logger.error(f"HaWoR prediction failed: {e}")
            return ArcticEvaluationMetrics()
        
        # Compute losses
        total_loss, losses = self.loss_function.compute_total_loss(pred_output, gt_data)
        
        # Compute evaluation metrics
        metrics = self.compute_evaluation_metrics(pred_output, gt_data)
        
        self.logger.info(f"Total loss: {total_loss.item():.4f}")
        self.logger.info(f"MPJPE 3D: {metrics.mpjpe_3d:.4f}")
        self.logger.info(f"MPJPE 2D: {metrics.mpjpe_2d:.4f}")
        
        return metrics
    
    def compute_evaluation_metrics(self, pred_output: Dict, gt_data: Dict) -> ArcticEvaluationMetrics:
        """
        Compute evaluation metrics
        
        Args:
            pred_output: HaWoR prediction output
            gt_data: ARCTIC ground truth data
            
        Returns:
            Evaluation metrics
        """
        metrics = ArcticEvaluationMetrics()
        
        # 3D Keypoint Metrics
        if 'pred_keypoints_3d' in pred_output and 'gt_keypoints_3d' in gt_data:
            pred_kp_3d = pred_output['pred_keypoints_3d']
            gt_kp_3d = gt_data['gt_keypoints_3d']
            
            # MPJPE
            metrics.mpjpe_3d = torch.norm(pred_kp_3d - gt_kp_3d, dim=-1).mean().item()
            
            # PCK
            errors_3d = torch.norm(pred_kp_3d - gt_kp_3d, dim=-1)
            metrics.pck_3d_5mm = (errors_3d < 0.005).float().mean().item()
            metrics.pck_3d_10mm = (errors_3d < 0.010).float().mean().item()
            metrics.pck_3d_15mm = (errors_3d < 0.015).float().mean().item()
        
        # 2D Keypoint Metrics
        if 'pred_keypoints_2d' in pred_output and 'gt_keypoints_2d' in gt_data:
            pred_kp_2d = pred_output['pred_keypoints_2d']
            gt_kp_2d = gt_data['gt_keypoints_2d']
            
            # MPJPE
            metrics.mpjpe_2d = torch.norm(pred_kp_2d - gt_kp_2d, dim=-1).mean().item()
            
            # PCK
            errors_2d = torch.norm(pred_kp_2d - gt_kp_2d, dim=-1)
            metrics.pck_2d_5px = (errors_2d < 5.0).float().mean().item()
            metrics.pck_2d_10px = (errors_2d < 10.0).float().mean().item()
        
        # MANO Parameter Metrics
        if 'pred_mano_params' in pred_output and 'gt_mano_params' in gt_data:
            pred_mano = pred_output['pred_mano_params']
            gt_mano = gt_data['gt_mano_params']
            
            if 'hand_pose' in pred_mano and 'hand_pose' in gt_mano:
                metrics.mano_pose_error = torch.norm(
                    pred_mano['hand_pose'] - gt_mano['hand_pose'], dim=-1
                ).mean().item()
            
            if 'betas' in pred_mano and 'betas' in gt_mano:
                metrics.mano_shape_error = torch.norm(
                    pred_mano['betas'] - gt_mano['betas'], dim=-1
                ).mean().item()
            
            if 'global_orient' in pred_mano and 'global_orient' in gt_mano:
                metrics.mano_global_orient_error = torch.norm(
                    pred_mano['global_orient'] - gt_mano['global_orient'], dim=-1
                ).mean().item()
        
        # Detection Metrics
        if 'pred_keypoints_3d' in pred_output:
            # Simple detection based on keypoint validity
            pred_kp_3d = pred_output['pred_keypoints_3d']
            valid_frames = torch.any(pred_kp_3d.abs() > 1e-6, dim=-1).any(dim=-1)
            metrics.hand_detection_rate = valid_frames.float().mean().item()
        
        return metrics
    
    def evaluate_dataset(self, sequences: List[Tuple[str, str]], 
                        output_file: Optional[str] = None) -> Dict:
        """
        Evaluate HaWoR on multiple ARCTIC sequences
        
        Args:
            sequences: List of (subject, sequence) tuples
            output_file: Optional output file for results
            
        Returns:
            Aggregated evaluation results
        """
        self.logger.info(f"Evaluating {len(sequences)} sequences")
        
        all_metrics = []
        for subject, sequence in sequences:
            try:
                metrics = self.evaluate_sequence(subject, sequence)
                all_metrics.append(metrics)
            except Exception as e:
                self.logger.error(f"Failed to evaluate {subject}/{sequence}: {e}")
                continue
        
        # Aggregate metrics
        aggregated = self.aggregate_metrics(all_metrics)
        
        # Save results
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(aggregated, f, indent=2)
            self.logger.info(f"Results saved to {output_file}")
        
        return aggregated
    
    def aggregate_metrics(self, metrics_list: List[ArcticEvaluationMetrics]) -> Dict:
        """
        Aggregate metrics across multiple sequences
        
        Args:
            metrics_list: List of metrics from individual sequences
            
        Returns:
            Aggregated metrics
        """
        if not metrics_list:
            return {}
        
        # Convert to dictionaries and aggregate
        metrics_dicts = [m.to_dict() for m in metrics_list]
        
        aggregated = {}
        for key in metrics_dicts[0].keys():
            values = [d[key] for d in metrics_dicts if key in d]
            if values:
                aggregated[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values)
                }
        
        return aggregated

def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='ARCTIC Evaluation Framework for HaWoR')
    parser.add_argument('--arctic-root', type=str, 
                       default='./thirdparty/arctic/unpack/arctic_data/data',
                       help='ARCTIC data root directory')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda, mps)')
    parser.add_argument('--sequences', type=str, nargs='+',
                       help='Specific sequences to evaluate (format: s01/box_grab_01)')
    parser.add_argument('--output', type=str, default='arctic_evaluation_results.json',
                       help='Output file for results')
    parser.add_argument('--max-sequences', type=int, default=10,
                       help='Maximum number of sequences to evaluate')
    
    args = parser.parse_args()
    
    # Initialize HaWoR interface
    hawor_interface = HaWoRInterface(device=args.device)
    hawor_interface.initialize_pipeline()
    
    # Initialize evaluator
    evaluator = ArcticEvaluator(hawor_interface, args.arctic_root)
    
    # Get sequences to evaluate
    if args.sequences:
        sequences = [seq.split('/') for seq in args.sequences]
    else:
        # Default sequences for testing
        sequences = [
            ('s01', 'box_grab_01'),
            ('s01', 'phone_use_01'),
            ('s02', 'laptop_use_01'),
        ][:args.max_sequences]
    
    # Run evaluation
    results = evaluator.evaluate_dataset(sequences, args.output)
    
    # Print summary
    print("\n" + "="*60)
    print("ARCTIC EVALUATION RESULTS")
    print("="*60)
    
    for metric, stats in results.items():
        print(f"{metric}:")
        print(f"  Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
        print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        print(f"  Count: {stats['count']}")
        print()

if __name__ == "__main__":
    main()
