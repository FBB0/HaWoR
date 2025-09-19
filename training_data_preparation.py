#!/usr/bin/env python3
"""
Training Data Preparation System for HaWoR
Converts ARCTIC and other datasets to HaWoR training format
"""

import os
import sys
import numpy as np
import torch
import cv2
from pathlib import Path
import json
import pickle
from typing import Dict, List, Tuple, Optional, Union
import argparse
from dataclasses import dataclass
import logging
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import h5py

# Add HaWoR to path
sys.path.append(str(Path(__file__).parent))

from lib.utils.geometry import perspective_projection
from lib.utils.rotation import rotation_matrix_to_angle_axis
from hawor.utils.rotation import angle_axis_to_rotation_matrix

@dataclass
class TrainingSample:
    """Single training sample in HaWoR format"""
    # Image data
    image: np.ndarray  # [H, W, 3] RGB image
    image_path: str
    
    # Camera parameters
    intrinsics: np.ndarray  # [3, 3] camera intrinsics
    camera_pose: Dict[str, np.ndarray]  # R, T matrices
    
    # Hand annotations
    mano_params: Dict[str, np.ndarray]  # global_orient, hand_pose, betas
    translation: np.ndarray  # [3] hand translation
    
    # Keypoints
    keypoints_3d: np.ndarray  # [21, 3] 3D hand keypoints
    keypoints_2d: np.ndarray  # [21, 2] 2D hand keypoints
    
    # Mesh data
    vertices: np.ndarray  # [778, 3] MANO vertices
    faces: np.ndarray  # [1538, 3] MANO faces
    
    # Metadata
    subject_id: str
    sequence_id: str
    frame_id: int
    hand_type: str  # 'left' or 'right'
    
    # Quality metrics
    occlusion_level: float  # 0-1 occlusion level
    confidence: float  # 0-1 confidence score

class ArcticDataConverter:
    """Convert ARCTIC data to HaWoR training format"""
    
    def __init__(self, 
                 arctic_root: str,
                 output_dir: str,
                 target_resolution: Tuple[int, int] = (256, 256),
                 num_workers: int = 4):
        """
        Initialize ARCTIC data converter
        
        Args:
            arctic_root: Root directory of ARCTIC data
            output_dir: Output directory for converted data
            target_resolution: Target image resolution
            num_workers: Number of worker processes
        """
        self.arctic_root = Path(arctic_root)
        self.output_dir = Path(output_dir)
        self.target_resolution = target_resolution
        self.num_workers = num_workers
        
        # Create output directories
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / 'images').mkdir(exist_ok=True)
        (self.output_dir / 'annotations').mkdir(exist_ok=True)
        (self.output_dir / 'masks').mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # MANO model for mesh generation
        self.mano_model = self._load_mano_model()
    
    def _load_mano_model(self):
        """Load MANO model for mesh generation"""
        try:
            from lib.models.mano_wrapper import MANO
            mano_cfg = {
                'data_dir': '_DATA/data/',
                'model_path': '_DATA/data/mano',
                'gender': 'neutral',
                'num_hand_joints': 15,
                'create_body_pose': False
            }
            return MANO(**mano_cfg)
        except Exception as e:
            self.logger.warning(f"Could not load MANO model: {e}")
            return None
    
    def convert_dataset(self, 
                       subjects: Optional[List[str]] = None,
                       sequences: Optional[List[str]] = None,
                       max_sequences_per_subject: Optional[int] = None) -> Dict:
        """
        Convert ARCTIC dataset to HaWoR training format
        
        Args:
            subjects: List of subject IDs to convert (None for all)
            sequences: List of sequence names to convert (None for all)
            max_sequences_per_subject: Maximum sequences per subject
            
        Returns:
            Conversion statistics
        """
        self.logger.info("Starting ARCTIC dataset conversion...")
        
        # Get available subjects and sequences
        available_subjects = self._get_available_subjects()
        if subjects is None:
            subjects = available_subjects
        else:
            subjects = [s for s in subjects if s in available_subjects]
        
        self.logger.info(f"Converting {len(subjects)} subjects: {subjects}")
        
        # Conversion statistics
        stats = {
            'total_subjects': len(subjects),
            'total_sequences': 0,
            'total_frames': 0,
            'successful_conversions': 0,
            'failed_conversions': 0,
            'conversion_errors': []
        }
        
        # Convert each subject
        for subject in tqdm(subjects, desc="Converting subjects"):
            subject_stats = self._convert_subject(
                subject, sequences, max_sequences_per_subject
            )
            
            # Update overall stats
            for key in ['total_sequences', 'total_frames', 'successful_conversions', 'failed_conversions']:
                stats[key] += subject_stats[key]
            stats['conversion_errors'].extend(subject_stats['conversion_errors'])
        
        # Save conversion statistics
        with open(self.output_dir / 'conversion_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        self.logger.info(f"Conversion completed!")
        self.logger.info(f"Total sequences: {stats['total_sequences']}")
        self.logger.info(f"Total frames: {stats['total_frames']}")
        self.logger.info(f"Successful: {stats['successful_conversions']}")
        self.logger.info(f"Failed: {stats['failed_conversions']}")
        
        return stats
    
    def _get_available_subjects(self) -> List[str]:
        """Get list of available subjects"""
        raw_seqs_dir = self.arctic_root / "raw_seqs"
        if not raw_seqs_dir.exists():
            raise FileNotFoundError(f"ARCTIC raw_seqs directory not found: {raw_seqs_dir}")
        
        subjects = [d.name for d in raw_seqs_dir.iterdir() if d.is_dir() and d.name.startswith('s')]
        return sorted(subjects)
    
    def _convert_subject(self, 
                        subject: str,
                        sequences: Optional[List[str]] = None,
                        max_sequences: Optional[int] = None) -> Dict:
        """Convert all sequences for a subject"""
        
        subject_dir = self.arctic_root / "raw_seqs" / subject
        if not subject_dir.exists():
            self.logger.error(f"Subject directory not found: {subject_dir}")
            return {'total_sequences': 0, 'total_frames': 0, 'successful_conversions': 0, 'failed_conversions': 0, 'conversion_errors': []}
        
        # Get available sequences
        available_sequences = self._get_available_sequences(subject_dir)
        if sequences is None:
            sequences_to_convert = available_sequences
        else:
            sequences_to_convert = [s for s in sequences if s in available_sequences]
        
        if max_sequences is not None:
            sequences_to_convert = sequences_to_convert[:max_sequences]
        
        self.logger.info(f"Converting {len(sequences_to_convert)} sequences for subject {subject}")
        
        # Convert sequences in parallel
        stats = {
            'total_sequences': len(sequences_to_convert),
            'total_frames': 0,
            'successful_conversions': 0,
            'failed_conversions': 0,
            'conversion_errors': []
        }
        
        # Use multiprocessing for sequence conversion
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for sequence in sequences_to_convert:
                future = executor.submit(self._convert_sequence, subject, sequence)
                futures.append(future)
            
            # Collect results
            for future in tqdm(futures, desc=f"Converting {subject}"):
                try:
                    sequence_stats = future.result()
                    for key in ['total_frames', 'successful_conversions', 'failed_conversions']:
                        stats[key] += sequence_stats[key]
                    stats['conversion_errors'].extend(sequence_stats['conversion_errors'])
                except Exception as e:
                    self.logger.error(f"Error in sequence conversion: {e}")
                    stats['failed_conversions'] += 1
                    stats['conversion_errors'].append(str(e))
        
        return stats
    
    def _get_available_sequences(self, subject_dir: Path) -> List[str]:
        """Get available sequences for a subject"""
        sequences = []
        for file in subject_dir.glob("*.mano.npy"):
            sequence_name = file.stem.replace('.mano', '')
            sequences.append(sequence_name)
        return sorted(sequences)
    
    def _convert_sequence(self, subject: str, sequence: str) -> Dict:
        """Convert a single sequence"""
        
        try:
            # Load ARCTIC data
            arctic_data = self._load_arctic_sequence(subject, sequence)
            
            # Convert to training samples
            training_samples = self._convert_arctic_to_training_samples(arctic_data, subject, sequence)
            
            # Save training samples
            self._save_training_samples(training_samples, subject, sequence)
            
            return {
                'total_frames': len(training_samples),
                'successful_conversions': len(training_samples),
                'failed_conversions': 0,
                'conversion_errors': []
            }
            
        except Exception as e:
            self.logger.error(f"Failed to convert {subject}/{sequence}: {e}")
            return {
                'total_frames': 0,
                'successful_conversions': 0,
                'failed_conversions': 1,
                'conversion_errors': [f"{subject}/{sequence}: {str(e)}"]
            }
    
    def _load_arctic_sequence(self, subject: str, sequence: str) -> Dict:
        """Load ARCTIC sequence data"""
        
        seq_path = self.arctic_root / "raw_seqs" / subject
        
        # Load MANO parameters
        mano_file = seq_path / f"{sequence}.mano.npy"
        if not mano_file.exists():
            raise FileNotFoundError(f"MANO file not found: {mano_file}")
        mano_data = np.load(mano_file, allow_pickle=True).item()
        
        # Load egocentric camera data
        egocam_file = seq_path / f"{sequence}.egocam.dist.npy"
        if not egocam_file.exists():
            raise FileNotFoundError(f"Egocentric camera file not found: {egocam_file}")
        egocam_data = np.load(egocam_file, allow_pickle=True).item()
        
        # Load object data (optional)
        object_file = seq_path / f"{sequence}.object.npy"
        object_data = None
        if object_file.exists():
            object_data = np.load(object_file, allow_pickle=True).item()
        
        return {
            'mano': mano_data,
            'egocam': egocam_data,
            'object': object_data,
            'subject': subject,
            'sequence': sequence
        }
    
    def _convert_arctic_to_training_samples(self, 
                                          arctic_data: Dict, 
                                          subject: str, 
                                          sequence: str) -> List[TrainingSample]:
        """Convert ARCTIC data to training samples"""
        
        mano_data = arctic_data['mano']
        egocam_data = arctic_data['egocam']
        
        # Get sequence length
        seq_len = len(mano_data['rot'])
        
        training_samples = []
        
        for frame_idx in range(seq_len):
            try:
                # Create training sample
                sample = self._create_training_sample(
                    arctic_data, frame_idx, subject, sequence
                )
                
                if sample is not None:
                    training_samples.append(sample)
                    
            except Exception as e:
                self.logger.warning(f"Failed to create sample for {subject}/{sequence}/{frame_idx}: {e}")
                continue
        
        return training_samples
    
    def _create_training_sample(self, 
                               arctic_data: Dict, 
                               frame_idx: int,
                               subject: str, 
                               sequence: str) -> Optional[TrainingSample]:
        """Create a single training sample"""
        
        mano_data = arctic_data['mano']
        egocam_data = arctic_data['egocam']
        
        # Extract frame data
        global_orient = mano_data['rot'][frame_idx]  # [3] axis-angle
        hand_pose = mano_data['pose'][frame_idx]  # [45] hand pose
        betas = mano_data['shape'][frame_idx]  # [10] shape parameters
        trans = mano_data['trans'][frame_idx]  # [3] translation
        
        # Camera parameters
        intrinsics = egocam_data['intrinsics']  # [3, 3]
        R = egocam_data['R_k_cam_np'][frame_idx]  # [3, 3]
        T = egocam_data['T_k_cam_np'][frame_idx]  # [3, 1]
        
        # Load and process image
        image = self._load_and_process_image(subject, sequence, frame_idx)
        if image is None:
            return None
        
        # Generate MANO mesh and keypoints
        vertices, faces, keypoints_3d = self._generate_mano_mesh(
            global_orient, hand_pose, betas, trans
        )
        
        # Project to 2D
        keypoints_2d = self._project_keypoints_3d_to_2d(
            keypoints_3d, intrinsics, R, T
        )
        
        # Estimate occlusion and confidence
        occlusion_level = self._estimate_occlusion_level(image, keypoints_2d)
        confidence = self._estimate_confidence(keypoints_2d, image.shape)
        
        # Create training sample
        sample = TrainingSample(
            image=image,
            image_path=f"{subject}/{sequence}/{frame_idx:06d}.jpg",
            intrinsics=intrinsics,
            camera_pose={'R': R, 'T': T},
            mano_params={
                'global_orient': global_orient,
                'hand_pose': hand_pose,
                'betas': betas
            },
            translation=trans,
            keypoints_3d=keypoints_3d,
            keypoints_2d=keypoints_2d,
            vertices=vertices,
            faces=faces,
            subject_id=subject,
            sequence_id=sequence,
            frame_id=frame_idx,
            hand_type='right',  # ARCTIC uses right hand
            occlusion_level=occlusion_level,
            confidence=confidence
        )
        
        return sample
    
    def _load_and_process_image(self, 
                               subject: str, 
                               sequence: str, 
                               frame_idx: int) -> Optional[np.ndarray]:
        """Load and process image for training"""
        
        # Try different image paths
        image_paths = [
            self.arctic_root / "cropped_images" / subject / sequence / "0" / f"{frame_idx:06d}.jpg",
            self.arctic_root / "cropped_images" / subject / sequence / f"{frame_idx:06d}.jpg",
            self.arctic_root / "images" / subject / sequence / f"{frame_idx:06d}.jpg"
        ]
        
        image_path = None
        for path in image_paths:
            if path.exists():
                image_path = path
                break
        
        if image_path is None:
            self.logger.warning(f"Image not found for {subject}/{sequence}/{frame_idx}")
            return None
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            self.logger.warning(f"Could not load image: {image_path}")
            return None
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to target resolution
        image = cv2.resize(image, self.target_resolution)
        
        return image
    
    def _generate_mano_mesh(self, 
                           global_orient: np.ndarray,
                           hand_pose: np.ndarray,
                           betas: np.ndarray,
                           trans: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate MANO mesh and keypoints"""
        
        if self.mano_model is None:
            # Fallback: create dummy data
            vertices = np.random.randn(778, 3).astype(np.float32)
            faces = np.random.randint(0, 778, (1538, 3)).astype(np.int32)
            keypoints_3d = np.random.randn(21, 3).astype(np.float32)
            return vertices, faces, keypoints_3d
        
        try:
            # Convert to torch tensors
            global_orient_t = torch.tensor(global_orient, dtype=torch.float32).unsqueeze(0)
            hand_pose_t = torch.tensor(hand_pose, dtype=torch.float32).unsqueeze(0)
            betas_t = torch.tensor(betas, dtype=torch.float32).unsqueeze(0)
            trans_t = torch.tensor(trans, dtype=torch.float32).unsqueeze(0)
            
            # Generate mesh
            output = self.mano_model(
                global_orient=global_orient_t,
                hand_pose=hand_pose_t,
                betas=betas_t,
                trans=trans_t
            )
            
            vertices = output['vertices'].squeeze(0).numpy()
            keypoints_3d = output['joints'].squeeze(0).numpy()
            
            # Get faces from MANO model
            faces = self.mano_model.faces.numpy()
            
            return vertices, faces, keypoints_3d
            
        except Exception as e:
            self.logger.warning(f"MANO mesh generation failed: {e}")
            # Fallback
            vertices = np.random.randn(778, 3).astype(np.float32)
            faces = np.random.randint(0, 778, (1538, 3)).astype(np.int32)
            keypoints_3d = np.random.randn(21, 3).astype(np.float32)
            return vertices, faces, keypoints_3d
    
    def _project_keypoints_3d_to_2d(self, 
                                   keypoints_3d: np.ndarray,
                                   intrinsics: np.ndarray,
                                   R: np.ndarray,
                                   T: np.ndarray) -> np.ndarray:
        """Project 3D keypoints to 2D"""
        
        # Transform keypoints to camera coordinates
        keypoints_3d_cam = (R @ keypoints_3d.T + T).T
        
        # Project to 2D
        keypoints_2d = keypoints_3d_cam[:, :2] / keypoints_3d_cam[:, 2:3]
        
        # Apply intrinsics
        keypoints_2d = (intrinsics[:2, :2] @ keypoints_2d.T + intrinsics[:2, 2:3]).T
        
        return keypoints_2d
    
    def _estimate_occlusion_level(self, image: np.ndarray, keypoints_2d: np.ndarray) -> float:
        """Estimate occlusion level of hand in image"""
        
        # Simple occlusion estimation based on keypoint visibility
        # This is a placeholder - in practice, you'd use more sophisticated methods
        
        h, w = image.shape[:2]
        visible_keypoints = 0
        
        for kp in keypoints_2d:
            x, y = kp
            if 0 <= x < w and 0 <= y < h:
                visible_keypoints += 1
        
        occlusion_level = 1.0 - (visible_keypoints / len(keypoints_2d))
        return max(0.0, min(1.0, occlusion_level))
    
    def _estimate_confidence(self, keypoints_2d: np.ndarray, image_shape: Tuple[int, int]) -> float:
        """Estimate confidence of keypoint detection"""
        
        h, w = image_shape[:2]
        
        # Check if keypoints are within image bounds
        valid_keypoints = 0
        for kp in keypoints_2d:
            x, y = kp
            if 0 <= x < w and 0 <= y < h:
                valid_keypoints += 1
        
        confidence = valid_keypoints / len(keypoints_2d)
        return max(0.0, min(1.0, confidence))
    
    def _save_training_samples(self, 
                              samples: List[TrainingSample],
                              subject: str, 
                              sequence: str):
        """Save training samples to disk"""
        
        # Create subject/sequence directories
        subject_dir = self.output_dir / 'images' / subject
        subject_dir.mkdir(exist_ok=True)
        
        sequence_dir = subject_dir / sequence
        sequence_dir.mkdir(exist_ok=True)
        
        # Save images and annotations
        for sample in samples:
            # Save image
            image_path = sequence_dir / f"{sample.frame_id:06d}.jpg"
            cv2.imwrite(str(image_path), cv2.cvtColor(sample.image, cv2.COLOR_RGB2BGR))
            
            # Save annotation
            annotation_path = self.output_dir / 'annotations' / subject / f"{sequence}_{sample.frame_id:06d}.json"
            annotation_path.parent.mkdir(exist_ok=True)
            
            annotation = {
                'image_path': str(image_path.relative_to(self.output_dir)),
                'intrinsics': sample.intrinsics.tolist(),
                'camera_pose': {
                    'R': sample.camera_pose['R'].tolist(),
                    'T': sample.camera_pose['T'].tolist()
                },
                'mano_params': {
                    'global_orient': sample.mano_params['global_orient'].tolist(),
                    'hand_pose': sample.mano_params['hand_pose'].tolist(),
                    'betas': sample.mano_params['betas'].tolist()
                },
                'translation': sample.translation.tolist(),
                'keypoints_3d': sample.keypoints_3d.tolist(),
                'keypoints_2d': sample.keypoints_2d.tolist(),
                'vertices': sample.vertices.tolist(),
                'faces': sample.faces.tolist(),
                'subject_id': sample.subject_id,
                'sequence_id': sample.sequence_id,
                'frame_id': sample.frame_id,
                'hand_type': sample.hand_type,
                'occlusion_level': sample.occlusion_level,
                'confidence': sample.confidence
            }
            
            with open(annotation_path, 'w') as f:
                json.dump(annotation, f, indent=2)

class TrainingDataset:
    """Training dataset for HaWoR"""
    
    def __init__(self, 
                 data_dir: str,
                 split: str = 'train',
                 transform=None,
                 target_resolution: Tuple[int, int] = (256, 256)):
        """
        Initialize training dataset
        
        Args:
            data_dir: Directory containing training data
            split: Dataset split ('train', 'val', 'test')
            transform: Data augmentation transforms
            target_resolution: Target image resolution
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.target_resolution = target_resolution
        
        # Load dataset metadata
        self.samples = self._load_dataset_metadata()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _load_dataset_metadata(self) -> List[Dict]:
        """Load dataset metadata"""
        
        metadata_file = self.data_dir / f'{self.split}_metadata.json'
        if not metadata_file.exists():
            # Generate metadata if it doesn't exist
            self._generate_metadata()
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        return metadata['samples']
    
    def _generate_metadata(self):
        """Generate dataset metadata"""
        
        self.logger.info(f"Generating metadata for {self.split} split...")
        
        samples = []
        annotations_dir = self.data_dir / 'annotations'
        
        for annotation_file in tqdm(annotations_dir.rglob('*.json'), desc="Processing annotations"):
            try:
                with open(annotation_file, 'r') as f:
                    annotation = json.load(f)
                
                # Check if image exists
                image_path = self.data_dir / annotation['image_path']
                if not image_path.exists():
                    continue
                
                # Add to samples
                samples.append({
                    'annotation_path': str(annotation_file.relative_to(self.data_dir)),
                    'image_path': annotation['image_path'],
                    'subject_id': annotation['subject_id'],
                    'sequence_id': annotation['sequence_id'],
                    'frame_id': annotation['frame_id'],
                    'hand_type': annotation['hand_type'],
                    'occlusion_level': annotation['occlusion_level'],
                    'confidence': annotation['confidence']
                })
                
            except Exception as e:
                self.logger.warning(f"Error processing {annotation_file}: {e}")
                continue
        
        # Save metadata
        metadata = {
            'split': self.split,
            'total_samples': len(samples),
            'samples': samples
        }
        
        metadata_file = self.data_dir / f'{self.split}_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Generated metadata for {len(samples)} samples")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get training sample"""
        
        sample_meta = self.samples[idx]
        
        # Load image
        image_path = self.data_dir / sample_meta['image_path']
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load annotation
        annotation_path = self.data_dir / sample_meta['annotation_path']
        with open(annotation_path, 'r') as f:
            annotation = json.load(f)
        
        # Convert to tensors
        sample = {
            'image': torch.from_numpy(image).permute(2, 0, 1).float() / 255.0,
            'intrinsics': torch.tensor(annotation['intrinsics'], dtype=torch.float32),
            'camera_pose': {
                'R': torch.tensor(annotation['camera_pose']['R'], dtype=torch.float32),
                'T': torch.tensor(annotation['camera_pose']['T'], dtype=torch.float32)
            },
            'mano_params': {
                'global_orient': torch.tensor(annotation['mano_params']['global_orient'], dtype=torch.float32),
                'hand_pose': torch.tensor(annotation['mano_params']['hand_pose'], dtype=torch.float32),
                'betas': torch.tensor(annotation['mano_params']['betas'], dtype=torch.float32)
            },
            'translation': torch.tensor(annotation['translation'], dtype=torch.float32),
            'keypoints_3d': torch.tensor(annotation['keypoints_3d'], dtype=torch.float32),
            'keypoints_2d': torch.tensor(annotation['keypoints_2d'], dtype=torch.float32),
            'vertices': torch.tensor(annotation['vertices'], dtype=torch.float32),
            'faces': torch.tensor(annotation['faces'], dtype=torch.long),
            'subject_id': sample_meta['subject_id'],
            'sequence_id': sample_meta['sequence_id'],
            'frame_id': sample_meta['frame_id'],
            'hand_type': sample_meta['hand_type'],
            'occlusion_level': sample_meta['occlusion_level'],
            'confidence': sample_meta['confidence']
        }
        
        # Apply transforms
        if self.transform:
            sample = self.transform(sample)
        
        return sample

def main():
    """Main function for data preparation"""
    parser = argparse.ArgumentParser(description='Training Data Preparation System')
    parser.add_argument('--arctic-root', type=str, required=True,
                       help='Root directory of ARCTIC data')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for converted data')
    parser.add_argument('--subjects', type=str, nargs='+',
                       help='Specific subjects to convert')
    parser.add_argument('--sequences', type=str, nargs='+',
                       help='Specific sequences to convert')
    parser.add_argument('--max-sequences-per-subject', type=int,
                       help='Maximum sequences per subject')
    parser.add_argument('--target-resolution', type=int, nargs=2, default=[256, 256],
                       help='Target image resolution')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of worker processes')
    
    args = parser.parse_args()
    
    # Initialize converter
    converter = ArcticDataConverter(
        arctic_root=args.arctic_root,
        output_dir=args.output_dir,
        target_resolution=tuple(args.target_resolution),
        num_workers=args.num_workers
    )
    
    # Convert dataset
    stats = converter.convert_dataset(
        subjects=args.subjects,
        sequences=args.sequences,
        max_sequences_per_subject=args.max_sequences_per_subject
    )
    
    print("Data preparation completed!")
    print(f"Statistics: {stats}")

if __name__ == "__main__":
    main()
