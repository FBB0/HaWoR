#!/usr/bin/env python3
"""
ARCTIC Data Preparation System for HaWoR
Simplified data preparation adapted for ARCTIC dataset
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Optional
import logging
from dataclasses import dataclass

# Add HaWoR to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

@dataclass
class ArcticTrainingSample:
    """Single training sample in ARCTIC format"""

    # Image data
    images: List[str]  # List of image file paths
    sequence_name: str

    # Hand annotations (MANO)
    mano_params: Optional[Dict] = None  # Left and right hand parameters
    keypoints_3d: Optional[np.ndarray] = None  # 3D keypoints
    keypoints_2d: Optional[np.ndarray] = None  # 2D keypoints

    # Camera parameters
    camera_intrinsics: Optional[np.ndarray] = None
    camera_extrinsics: Optional[Dict] = None

    # Metadata
    subject_id: str = "s01"
    frame_indices: Optional[List[int]] = None

class ArcticDataPreparer:
    """Prepare ARCTIC data for HaWoR training"""

    def __init__(self, data_root: str = "thirdparty/arctic", config: Optional[Dict] = None):
        """Initialize data preparer"""
        self.data_root = Path(data_root)
        self.arctic_root = self.data_root
        self._config = config or {}
        print(f"  📁 ARCTIC Data Preparer initialized with root: {self.data_root}")
        if self._config.get('use_egocentric_only'):
            print(f"    📹 Configured for egocentric-only training")
        elif self._config.get('camera_views'):
            print(f"    📹 Configured for camera views: {self._config['camera_views']}")

    def check_data_availability(self) -> Dict:
        """Check what ARCTIC data is available"""
        print("  🔍 Checking ARCTIC data availability...")

        availability = {
            "images": False,
            "mano": False,
            "meta": False,
            "sequences": []
        }

        # Check images
        image_dir = self.arctic_root / "data/cropped_images/s01/"
        if image_dir.exists():
            sequences = [d.name for d in image_dir.iterdir() if d.is_dir()]
            availability["sequences"] = sequences
            availability["images"] = len(sequences) > 0
            print(f"    📷 Images: {len(sequences)} sequences found")

        # Check MANO data
        mano_dir = self.arctic_root / "downloads/raw_seqs/s01/"
        if mano_dir.exists():
            mano_files = list(mano_dir.glob("*.npy"))
            availability["mano"] = len(mano_files) > 0
            print(f"    🧴 MANO: {len(mano_files)} files found")

        # Check meta data
        meta_dir = self.arctic_root / "downloads/meta/"
        if meta_dir.exists():
            availability["meta"] = True
            print("    📊 Meta: Available")

        return availability

    def load_sequence_data(self, sequence_name: str, max_images: int = 20) -> List[ArcticTrainingSample]:
        """Load data for a specific sequence"""
        print(f"    📊 Loading sequence: {sequence_name}")

        try:
            samples = []

            # Get image directories for this sequence
            seq_dir = self.arctic_root / f"data/cropped_images/s01/{sequence_name}/"
            if not seq_dir.exists():
                print(f"      ❌ Sequence directory not found: {seq_dir}")
                return samples

            camera_dirs = [d for d in seq_dir.iterdir() if d.is_dir() and d.name.isdigit()]
            if not camera_dirs:
                print(f"      ❌ No camera directories found for {sequence_name}")
                return samples

            # Load images from selected camera views
            all_images = []
            # Get view configuration from the config
            use_egocentric_only = getattr(self, '_config', {}).get('use_egocentric_only', True)
            camera_views = getattr(self, '_config', {}).get('camera_views', [0])

            if use_egocentric_only:
                # Use only egocentric view (camera 0)
                camera_views = [0]
                print(f"      📹 Using egocentric view only (camera 0)")

            print(f"      📹 Using camera views: {camera_views}")

            # Load images from specified camera views
            for view_id in camera_views:
                cam_dir = seq_dir / str(view_id)
                if cam_dir.exists():
                    images = list(cam_dir.glob("*.jpg"))
                    if images:
                        # Take subset of images
                        selected_images = images[:max_images//len(camera_views)]
                        all_images.extend(selected_images)
                        print(f"        ✅ Camera {view_id}: {len(selected_images)} images")
                    else:
                        print(f"        ❌ Camera {view_id}: No images found")
                else:
                    print(f"        ❌ Camera {view_id}: Directory not found")

            if not all_images:
                print(f"      ❌ No images found for {sequence_name}")
                return samples

            # Create training sample
            sample = ArcticTrainingSample(
                images=[str(img) for img in all_images],
                sequence_name=sequence_name,
                subject_id="s01",
                frame_indices=list(range(len(all_images)))
            )

            # Try to load MANO data
            mano_file = self.arctic_root / f"downloads/raw_seqs/s01/{sequence_name}.mano.npy"
            if mano_file.exists():
                try:
                    mano_data = np.load(mano_file, allow_pickle=True).item()
                    sample.mano_params = mano_data
                    print(f"      ✅ Loaded MANO data: {len(mano_data)} hands")
                except Exception as e:
                    print(f"      ⚠️  Could not load MANO data: {e}")

            samples.append(sample)
            print(f"      ✅ Created {len(samples)} sample(s) with {len(all_images)} images")

            return samples

        except Exception as e:
            print(f"      ❌ Error loading sequence {sequence_name}: {e}")
            return []

    def prepare_training_data(self, config: Dict) -> List[ArcticTrainingSample]:
        """Prepare all training data"""
        print("  📊 Preparing ARCTIC training data...")

        # Check data availability
        availability = self.check_data_availability()
        if not (availability["images"] and availability["mano"]):
            print("    ❌ Required ARCTIC data not available")
            return []

        # Get sequences to use
        sequences = availability["sequences"]
        max_sequences = config.get('arctic', {}).get('max_sequences', 3)
        train_sequences = sequences[:max_sequences]

        print(f"    🎯 Using {len(train_sequences)} sequences for training")

        # Load data for each sequence
        all_samples = []
        max_images_per_seq = config.get('arctic', {}).get('images_per_camera', 20)

        for seq in train_sequences:
            samples = self.load_sequence_data(seq, max_images_per_seq)
            all_samples.extend(samples)

        print(f"    ✅ Prepared {len(all_samples)} training samples")
        return all_samples

    def get_data_statistics(self, samples: List[ArcticTrainingSample]) -> Dict:
        """Get statistics about the training data"""
        if not samples:
            return {"total_samples": 0}

        total_images = sum(len(sample.images) for sample in samples)
        sequences = list(set(sample.sequence_name for sample in samples))

        mano_available = sum(1 for sample in samples if sample.mano_params is not None)

        return {
            "total_samples": len(samples),
            "total_images": total_images,
            "sequences": sequences,
            "mano_available": mano_available,
            "images_per_sample": total_images / len(samples) if samples else 0
        }

def create_arctic_data_loader(config: Dict) -> ArcticDataPreparer:
    """Create ARCTIC data preparer"""
    data_root = config.get('data', {}).get('data_root', 'thirdparty/arctic')
    return ArcticDataPreparer(data_root=data_root)
