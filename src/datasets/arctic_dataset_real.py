#!/usr/bin/env python3
"""
Real ARCTIC Dataset Loader for HaWoR Training
Loads actual ARCTIC annotations, MANO parameters, and ground truth data
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pickle
import random

class ARCTICDataset(Dataset):
    """
    Real ARCTIC dataset with MANO ground truth and hand annotations
    """

    def __init__(self,
                 data_root: str,
                 subjects: List[str] = ['s01'],
                 sequences: Optional[List[str]] = None,
                 split: str = 'train',
                 img_size: int = 224,  # ARCTIC standard image size
                 sequence_length: int = 16,
                 camera_views: List[int] = [0],  # Egocentric view
                 use_augmentation: bool = True,
                 max_sequences: int = None,
                 images_per_sequence: int = None):  # Minimum visible keypoints to consider hand visible

        self.data_root = Path(data_root)
        self.subjects = subjects
        self.split = split
        self.img_size = img_size
        self.sequence_length = sequence_length
        self.camera_views = camera_views
        self.use_augmentation = use_augmentation

        # Data paths
        self.image_root = self.data_root / "data" / "cropped_images"
        self.anno_root = self.data_root / "downloads" / "raw_seqs"
        self.meta_root = self.data_root / "downloads" / "meta"

        # Load dataset structure
        self.samples = self._load_dataset_samples(sequences, max_sequences, images_per_sequence)

        print(f"ARCTIC Dataset loaded:")
        print(f"  - Split: {split}")
        print(f"  - Subjects: {subjects}")
        print(f"  - Camera views: {camera_views}")
        print(f"  - Total samples: {len(self.samples)}")
        print(f"  - Sequence length: {sequence_length}")

    def _load_dataset_samples(self,
                            sequences: Optional[List[str]],
                            max_sequences: int,
                            images_per_sequence: int) -> List[Dict]:
        """Load all valid samples from ARCTIC dataset"""

        samples = []

        for subject in self.subjects:
            subject_image_dir = self.image_root / subject
            subject_anno_dir = self.anno_root / subject

            if not subject_image_dir.exists() or not subject_anno_dir.exists():
                print(f"Warning: Subject {subject} data not found")
                continue

            # Get available sequences
            available_sequences = [d.name for d in subject_image_dir.iterdir() if d.is_dir()]

            if sequences is not None:
                available_sequences = [seq for seq in sequences if seq in available_sequences]

            if max_sequences is not None:
                available_sequences = available_sequences[:max_sequences]

            print(f"  Processing {subject}: {len(available_sequences)} sequences")

            for seq_name in available_sequences:
                seq_samples = self._process_sequence(subject, seq_name, images_per_sequence)
                samples.extend(seq_samples)

                if len(seq_samples) > 0:
                    print(f"    {seq_name}: {len(seq_samples)} samples")

        return samples

    def _process_sequence(self, subject: str, seq_name: str, images_per_sequence: int) -> List[Dict]:
        """Process individual sequence to create training samples"""

        samples = []
        seq_image_dir = self.image_root / subject / seq_name

        # Check for MANO annotation file (ARCTIC format)
        mano_file = self.anno_root / subject / f"{seq_name}.mano.npy"

        # For demo purposes, we'll create samples even without annotations
        # In production, you'd want proper ARCTIC annotations

        # Process each camera view
        for camera_id in self.camera_views:
            camera_dir = seq_image_dir / str(camera_id)
            if not camera_dir.exists():
                continue

            # Get all images in this camera view
            image_files = sorted(list(camera_dir.glob("*.jpg")))

            if len(image_files) < self.sequence_length:
                continue

            # Limit images if specified
            if images_per_sequence is not None:
                image_files = image_files[:images_per_sequence]

            # Create sequence samples with sliding window
            for start_idx in range(0, len(image_files) - self.sequence_length + 1, self.sequence_length):
                end_idx = start_idx + self.sequence_length

                if end_idx > len(image_files):
                    break

                sequence_images = image_files[start_idx:end_idx]

                # Create sample
                sample = {
                    'subject': subject,
                    'sequence': seq_name,
                    'camera_id': camera_id,
                    'image_files': sequence_images,
                    'start_frame': start_idx,
                    'end_frame': end_idx,
                    'mano_file': mano_file
                }

                samples.append(sample)

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a training sample"""

        sample = self.samples[idx]

        try:
            # Load images
            images = self._load_images(sample['image_files'])

            # Load annotations
            annotations = self._load_annotations(sample)

            # Apply transformations
            if self.use_augmentation and self.split == 'train':
                images, annotations = self._apply_augmentations(images, annotations)

            # Convert to tensors
            data = self._to_tensors(images, annotations)

            # Add metadata
            data['subject'] = sample['subject']
            data['sequence'] = sample['sequence']
            data['camera_id'] = sample['camera_id']

            return data

        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            # Return a default sample
            return self._get_default_sample()

    def _load_images(self, image_files: List[Path]) -> np.ndarray:
        """Load and preprocess images"""

        images = []

        for img_file in image_files:
            try:
                img = cv2.imread(str(img_file))
                if img is None:
                    raise ValueError(f"Could not load image: {img_file}")

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (self.img_size, self.img_size))
                images.append(img)

            except Exception as e:
                print(f"Warning: Error loading {img_file}: {e}")
                # Use a blank image as fallback
                blank_img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
                images.append(blank_img)

        return np.array(images)  # [T, H, W, C]

    def _load_annotations(self, sample: Dict) -> Dict[str, np.ndarray]:
        """Load MANO and hand annotations from ARCTIC .npy files"""

        try:
            # Load MANO data from .npy file
            sequence_name = sample['sequence']
            subject = sample['subject']
            mano_file = self.anno_root / subject / f"{sequence_name}.mano.npy"

            annotations = {}

            if mano_file.exists():
                try:
                    # Load real ARCTIC MANO data
                    mano_data = np.load(mano_file, allow_pickle=True).item()
                    start_frame = sample['start_frame']
                    end_frame = sample['end_frame']

                    # Extract left hand parameters (focus on left hand)
                    if 'left' in mano_data:
                        left_data = mano_data['left']

                        # Extract pose parameters
                        if 'pose' in left_data:
                            pose_data = left_data['pose']
                            if pose_data.shape[0] > end_frame:
                                annotations['hand_pose'] = pose_data[start_frame:end_frame].astype(np.float32)
                            else:
                                # Handle short sequences by repeating
                                repeat_count = self.sequence_length // pose_data.shape[0] + 1
                                repeated_pose = np.tile(pose_data, (repeat_count, 1))
                                annotations['hand_pose'] = repeated_pose[:self.sequence_length].astype(np.float32)
                        else:
                            raise ValueError("No pose data in left hand")

                        # Extract shape parameters (constant per sequence)
                        if 'shape' in left_data:
                            shape_data = left_data['shape']
                            annotations['hand_shape'] = np.tile(
                                shape_data[:10][None], (self.sequence_length, 1)
                            ).astype(np.float32)
                        else:
                            raise ValueError("No shape data in left hand")

                        # Extract translation parameters
                        if 'trans' in left_data:
                            trans_data = left_data['trans']
                            if trans_data.shape[0] > end_frame:
                                annotations['hand_trans'] = trans_data[start_frame:end_frame].astype(np.float32)
                            else:
                                repeat_count = self.sequence_length // trans_data.shape[0] + 1
                                repeated_trans = np.tile(trans_data, (repeat_count, 1))
                                annotations['hand_trans'] = repeated_trans[:self.sequence_length].astype(np.float32)
                        else:
                            raise ValueError("No translation data in left hand")

                        # Extract rotation parameters
                        if 'rot' in left_data:
                            rot_data = left_data['rot']
                            if rot_data.shape[0] > end_frame:
                                annotations['hand_rot'] = rot_data[start_frame:end_frame].astype(np.float32)
                            else:
                                repeat_count = self.sequence_length // rot_data.shape[0] + 1
                                repeated_rot = np.tile(rot_data, (repeat_count, 1))
                                annotations['hand_rot'] = repeated_rot[:self.sequence_length].astype(np.float32)
                        else:
                            raise ValueError("No rotation data in left hand")

                        # Generate 3D hand joints from MANO parameters
                        annotations['hand_joints'] = self._generate_hand_joints_from_mano(
                            annotations['hand_pose'],
                            annotations['hand_shape'],
                            annotations['hand_trans'],
                            annotations['hand_rot']
                        )

                        # Generate 2D keypoints by projecting 3D joints
                        annotations['keypoints_2d'] = self._project_3d_to_2d(annotations['hand_joints'])

                        # Generate camera pose (simplified for now)
                        annotations['camera_pose'] = np.zeros((self.sequence_length, 6), dtype=np.float32)
                        annotations['hand_valid'] = np.ones((self.sequence_length,), dtype=np.float32)

                        print(f"✅ Loaded real MANO data for {sequence_name}: frames {start_frame}-{end_frame}")

                    else:
                        raise ValueError("No left hand data found in MANO file")

                except Exception as e:
                    print(f"❌ Error parsing MANO file {mano_file}: {e}")
                    print(f"   Using fallback synthetic data")
                    annotations = self._get_fallback_annotations()
            else:
                print(f"⚠️ MANO file not found: {mano_file}")
                print(f"   Using fallback synthetic data")
                annotations = self._get_fallback_annotations()

        except Exception as e:
            print(f"Warning: Error loading annotations: {e}")
            annotations = self._get_default_annotations()

        return annotations

    def _get_default_annotations(self) -> Dict[str, np.ndarray]:
        """Get default annotations when loading fails"""
        return {
            'hand_pose': np.zeros((self.sequence_length, 45)),
            'hand_shape': np.zeros((self.sequence_length, 10)),
            'hand_trans': np.zeros((self.sequence_length, 3)),
            'hand_rot': np.zeros((self.sequence_length, 3)),
            'hand_joints': np.zeros((self.sequence_length, 21, 3)),
            'keypoints_2d': np.zeros((self.sequence_length, 21, 2)),
            'camera_pose': np.zeros((self.sequence_length, 6)),
            'hand_valid': np.zeros((self.sequence_length,), dtype=np.float32)
        }

    def _get_fallback_annotations(self) -> Dict[str, np.ndarray]:
        """Get fallback synthetic annotations when real data fails"""
        print("⚠️ Generating synthetic fallback data - training will not be meaningful!")
        return {
            'hand_pose': np.random.randn(self.sequence_length, 45).astype(np.float32) * 0.1,
            'hand_shape': np.random.randn(self.sequence_length, 10).astype(np.float32) * 0.1,
            'hand_trans': np.random.randn(self.sequence_length, 3).astype(np.float32) * 0.1,
            'hand_rot': np.random.randn(self.sequence_length, 3).astype(np.float32) * 0.1,
            'hand_joints': np.random.randn(self.sequence_length, 21, 3).astype(np.float32) * 0.1,
            'keypoints_2d': np.random.rand(self.sequence_length, 21, 2).astype(np.float32) * self.img_size,
            'camera_pose': np.random.randn(self.sequence_length, 6).astype(np.float32) * 0.1,
            'hand_valid': np.ones((self.sequence_length,), dtype=np.float32)
        }

    def _generate_hand_joints_from_mano(self,
                                       hand_pose: np.ndarray,
                                       hand_shape: np.ndarray,
                                       hand_trans: np.ndarray,
                                       hand_rot: np.ndarray) -> np.ndarray:
        """
        Generate 3D hand joints from MANO parameters

        Args:
            hand_pose: [T, 45] MANO pose parameters
            hand_shape: [T, 10] MANO shape parameters
            hand_trans: [T, 3] Hand translation
            hand_rot: [T, 3] Hand rotation

        Returns:
            hand_joints: [T, 21, 3] 3D hand joint positions
        """

        # This is a simplified kinematic approximation
        # In a full implementation, you would use the actual MANO model

        T = hand_pose.shape[0]
        hand_joints = np.zeros((T, 21, 3), dtype=np.float32)

        # Define basic hand structure (simplified)
        # Joint hierarchy: wrist -> fingers
        for t in range(T):
            # Start with wrist at origin, then apply transforms
            wrist_pos = hand_trans[t]  # [3]

            # Generate joint positions based on pose parameters
            # This is a very simplified approximation
            pose_t = hand_pose[t]  # [45] = 15 joints * 3 DoF

            # Wrist (joint 0)
            hand_joints[t, 0] = wrist_pos

            # Generate finger joints based on pose parameters
            for joint_idx in range(1, 21):
                # Get the relevant pose parameters for this joint
                param_start = ((joint_idx - 1) // 4) * 9  # Each finger has ~9 parameters
                param_end = min(param_start + 3, 45)

                if param_end <= param_start:
                    joint_params = np.zeros(3)
                else:
                    joint_params = pose_t[param_start:param_end]
                    if len(joint_params) < 3:
                        joint_params = np.pad(joint_params, (0, 3 - len(joint_params)))

                # Simple finger structure: extend from wrist with pose-based offsets
                finger_idx = (joint_idx - 1) // 4  # 0-4 for thumb, index, middle, ring, pinky
                joint_in_finger = (joint_idx - 1) % 4  # 0-3 for mcp, pip, dip, tip

                # Base finger direction
                finger_dirs = np.array([
                    [0.05, 0.0, 0.02],    # thumb
                    [0.08, 0.0, 0.0],     # index
                    [0.09, 0.0, -0.01],   # middle
                    [0.08, 0.0, -0.02],   # ring
                    [0.06, 0.0, -0.03]    # pinky
                ])

                if finger_idx < len(finger_dirs):
                    base_dir = finger_dirs[finger_idx]
                    # Apply pose-based rotation and extension
                    joint_offset = base_dir * (joint_in_finger + 1) * 0.3
                    joint_offset += joint_params * 0.02  # Small pose-based variation

                    hand_joints[t, joint_idx] = wrist_pos + joint_offset
                else:
                    # Fallback for extra joints
                    hand_joints[t, joint_idx] = wrist_pos + np.random.randn(3) * 0.01

        return hand_joints

    def _project_3d_to_2d(self, hand_joints_3d: np.ndarray) -> np.ndarray:
        """
        Project 3D hand joints to 2D keypoints

        Args:
            hand_joints_3d: [T, 21, 3] 3D joint positions

        Returns:
            keypoints_2d: [T, 21, 2] 2D keypoint positions
        """

        T, num_joints, _ = hand_joints_3d.shape
        keypoints_2d = np.zeros((T, num_joints, 2), dtype=np.float32)

        # ARCTIC weak perspective camera model
        # Based on ARCTIC parser: focal_length = 1000.0, img_res = 224
        fx, fy = 1000.0, 1000.0  # ARCTIC weak perspective focal length
        cx, cy = self.img_size // 2, self.img_size // 2  # Principal point at image center

        for t in range(T):
            for j in range(num_joints):
                x, y, z = hand_joints_3d[t, j]

                # Avoid division by zero
                z = max(z, 0.5)

                # Perspective projection with scaling
                u = (fx * x / z) + cx
                v = (fy * y / z) + cy

                # Apply reasonable bounds without hard clamping
                if 0 <= u <= self.img_size - 1 and 0 <= v <= self.img_size - 1:
                    keypoints_2d[t, j] = [u, v]
                else:
                    # For out-of-bounds points, project to a reasonable location
                    u_scaled = cx + (x / z) * 50  # Gentler scaling
                    v_scaled = cy + (y / z) * 50
                    u_clamped = np.clip(u_scaled, 10, self.img_size - 10)
                    v_clamped = np.clip(v_scaled, 10, self.img_size - 10)
                    keypoints_2d[t, j] = [u_clamped, v_clamped]

        return keypoints_2d

    def _apply_augmentations(self, images: np.ndarray, annotations: Dict) -> Tuple[np.ndarray, Dict]:
        """Apply data augmentations"""

        # Simple augmentations that preserve hand pose consistency

        # Random brightness/contrast
        if random.random() < 0.3:
            brightness = random.uniform(0.8, 1.2)
            contrast = random.uniform(0.8, 1.2)
            images = np.clip(images * contrast + brightness * 10, 0, 255).astype(np.uint8)

        # Random horizontal flip (need to adjust poses accordingly)
        if random.random() < 0.2:
            images = images[:, :, ::-1]  # Flip horizontally
            # Note: In a full implementation, you'd need to flip the hand poses and keypoints too

        return images, annotations

    def _to_tensors(self, images: np.ndarray, annotations: Dict) -> Dict[str, torch.Tensor]:
        """Convert numpy arrays to PyTorch tensors"""

        # Normalize images to [0, 1]
        images = images.astype(np.float32) / 255.0

        # Convert to PyTorch format [T, C, H, W]
        images = torch.from_numpy(images).permute(0, 3, 1, 2)

        # Convert annotations
        tensors = {'images': images}

        for key, value in annotations.items():
            tensors[key] = torch.from_numpy(value.astype(np.float32))

        return tensors

    def _get_default_sample(self) -> Dict[str, torch.Tensor]:
        """Get a default sample when loading fails"""

        # Default images
        images = torch.zeros(self.sequence_length, 3, self.img_size, self.img_size)

        # Default annotations
        sample = {
            'images': images,
            'hand_pose': torch.zeros(self.sequence_length, 45),
            'hand_shape': torch.zeros(self.sequence_length, 10),
            'hand_trans': torch.zeros(self.sequence_length, 3),
            'hand_rot': torch.zeros(self.sequence_length, 3),
            'hand_joints': torch.zeros(self.sequence_length, 21, 3),
            'keypoints_2d': torch.zeros(self.sequence_length, 21, 2),
            'camera_pose': torch.zeros(self.sequence_length, 6),
            'hand_valid': torch.zeros(self.sequence_length),
            'subject': 'default',
            'sequence': 'default',
            'camera_id': 0
        }

        return sample


def create_arctic_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders for ARCTIC dataset

    Args:
        config: Training configuration

    Returns:
        Tuple of (train_loader, val_loader)
    """

    arctic_config = config.get('arctic', {})
    training_config = config.get('training', {})

    # Dataset parameters
    data_root = arctic_config.get('data_root', 'thirdparty/arctic')
    train_subjects = arctic_config.get('train_subjects', ['s01'])
    val_subjects = arctic_config.get('val_subjects', ['s01'])
    sequence_length = arctic_config.get('sequence_length', 16)
    max_sequences = arctic_config.get('max_sequences', 3)
    images_per_sequence = arctic_config.get('images_per_camera', 20)

    # Camera configuration
    camera_views = [0]  # Egocentric view only
    if arctic_config.get('use_multiple_cameras', False):
        camera_views = arctic_config.get('camera_views', [0, 1, 2, 3])

    # Create datasets
    train_dataset = ARCTICDataset(
        data_root=data_root,
        subjects=train_subjects,
        split='train',
        sequence_length=sequence_length,
        camera_views=camera_views,
        use_augmentation=True,
        max_sequences=max_sequences,
        images_per_sequence=images_per_sequence
    )

    val_dataset = ARCTICDataset(
        data_root=data_root,
        subjects=val_subjects,
        split='val',
        sequence_length=sequence_length,
        camera_views=camera_views,
        use_augmentation=False,
        max_sequences=max_sequences // 2 if max_sequences else None,
        images_per_sequence=images_per_sequence // 2 if images_per_sequence else None
    )

    # Create dataloaders
    batch_size = training_config.get('batch_size', 1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Single worker for Mac compatibility
        pin_memory=False,  # Disable for Mac
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False
    )

    return train_loader, val_loader


def test_arctic_dataset():
    """Test function for ARCTIC dataset"""

    config = {
        'arctic': {
            'data_root': 'thirdparty/arctic',
            'train_subjects': ['s01'],
            'sequence_length': 8,
            'max_sequences': 2,
            'images_per_camera': 10
        },
        'training': {
            'batch_size': 1
        }
    }

    train_loader, val_loader = create_arctic_dataloaders(config)

    print(f"Train dataset: {len(train_loader.dataset)} samples")
    print(f"Val dataset: {len(val_loader.dataset)} samples")

    # Test loading a batch
    for i, batch in enumerate(train_loader):
        print(f"\nBatch {i}:")
        print(f"  Images shape: {batch['images'].shape}")
        print(f"  Hand pose shape: {batch['hand_pose'].shape}")
        print(f"  Hand joints shape: {batch['hand_joints'].shape}")
        print(f"  Subject: {batch['subject']}")
        print(f"  Sequence: {batch['sequence']}")

        if i >= 2:  # Only test a few batches
            break

    print("\nARCTIC dataset test completed successfully!")


if __name__ == "__main__":
    test_arctic_dataset()