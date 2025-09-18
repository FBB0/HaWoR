#!/usr/bin/env python3
"""
Advanced HaWoR Pipeline that integrates with the actual models
"""

import os
import sys
import torch
import numpy as np
import cv2
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import joblib
import yaml

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

try:
    # Try to import actual HaWoR components
    from scripts.scripts_test_video.detect_track_video import detect_track_video
    from scripts.scripts_test_video.hawor_video import hawor_motion_estimation, hawor_infiller
    HAWOR_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import HaWoR modules: {e}")
    HAWOR_AVAILABLE = False

try:
    # Try to import MANO utilities
    from hawor.utils.process import get_mano_faces, run_mano, run_mano_left
    MANO_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import MANO utilities: {e}")
    MANO_AVAILABLE = False

class AdvancedHaWoR:
    """
    Advanced HaWoR pipeline that uses actual models when available
    Falls back to simplified implementations when models are not available
    """

    def __init__(self,
                 checkpoint_path: str = './weights/hawor/checkpoints/hawor.ckpt',
                 infiller_path: str = './weights/hawor/checkpoints/infiller.pt',
                 detector_path: str = './weights/external/detector.pt',
                 config_path: str = './weights/hawor/model_config.yaml',
                 device: str = 'auto'):

        self.device = self._setup_device(device)
        self.checkpoint_path = checkpoint_path
        self.infiller_path = infiller_path
        self.detector_path = detector_path
        self.config_path = config_path

        # Load configuration
        self.config = self._load_config()

        # Check model availability
        self.models_available = self._check_model_availability()

        print(f"AdvancedHaWoR initialized on device: {self.device}")
        print(f"HaWoR models available: {HAWOR_AVAILABLE}")
        print(f"MANO utilities available: {MANO_AVAILABLE}")

    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device"""
        if device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        else:
            return torch.device(device)

    def _load_config(self) -> Dict[str, Any]:
        """Load model configuration"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            print(f"Warning: Could not load config from {self.config_path}: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'MODEL': {
                'IMAGE_SIZE': 256,
                'IMAGE_MEAN': [0.485, 0.456, 0.406],
                'IMAGE_STD': [0.229, 0.224, 0.225]
            },
            'MANO': {
                'DATA_DIR': '_DATA/data/',
                'MODEL_PATH': '_DATA/data//mano'
            }
        }

    def _check_model_availability(self) -> Dict[str, bool]:
        """Check which models are available"""
        return {
            'checkpoint': os.path.exists(self.checkpoint_path),
            'infiller': os.path.exists(self.infiller_path),
            'detector': os.path.exists(self.detector_path),
            'hawor_modules': HAWOR_AVAILABLE,
            'mano_utils': MANO_AVAILABLE
        }

    def process_video_with_hawor(self,
                               video_path: str,
                               output_dir: Optional[str] = None,
                               img_focal: Optional[float] = None) -> Dict[str, Any]:
        """
        Process video using actual HaWoR pipeline when available
        """

        if not HAWOR_AVAILABLE:
            raise RuntimeError("HaWoR modules not available. Use process_video_simplified instead.")

        if output_dir is None:
            output_dir = f"hawor_output_{Path(video_path).stem}"
        os.makedirs(output_dir, exist_ok=True)

        print(f"Processing video with HaWoR pipeline: {video_path}")

        try:
            # Create args object for HaWoR functions
            args = type('Args', (), {})()
            args.video_path = video_path
            args.img_focal = img_focal
            args.checkpoint = self.checkpoint_path
            args.infiller_weight = self.infiller_path

            # Step 1: Detect and track hands
            print("Step 1: Detecting and tracking hands...")
            start_idx, end_idx, seq_folder, imgfiles = detect_track_video(args)

            # Step 2: Hand motion estimation
            print("Step 2: Estimating hand motion...")
            frame_chunks_all, img_focal = hawor_motion_estimation(args, start_idx, end_idx, seq_folder)

            # Step 3: Motion infilling
            print("Step 3: Filling missing frames...")
            pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = hawor_infiller(
                args, start_idx, end_idx, frame_chunks_all
            )

            # Step 4: Process MANO meshes if available
            results = {
                'video_path': video_path,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'seq_folder': seq_folder,
                'pred_trans': pred_trans,
                'pred_rot': pred_rot,
                'pred_hand_pose': pred_hand_pose,
                'pred_betas': pred_betas,
                'pred_valid': pred_valid,
                'img_focal': img_focal,
                'imgfiles': imgfiles
            }

            if MANO_AVAILABLE:
                print("Step 4: Generating MANO meshes...")
                results.update(self._generate_mano_meshes(results))

            # Save results
            results_path = os.path.join(output_dir, 'hawor_results.npz')
            # Convert tensors to numpy for saving
            save_dict = {}
            for key, value in results.items():
                if isinstance(value, torch.Tensor):
                    save_dict[key] = value.cpu().numpy()
                elif isinstance(value, (list, tuple, str, int, float)):
                    save_dict[key] = value
                elif isinstance(value, dict):
                    # Handle nested dictionaries
                    save_dict[key] = {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v
                                    for k, v in value.items()}

            np.savez(results_path, **save_dict)
            print(f"Results saved to: {results_path}")

            return results

        except Exception as e:
            print(f"Error in HaWoR pipeline: {e}")
            print("Falling back to simplified pipeline...")
            return self.process_video_simplified(video_path, output_dir)

    def _generate_mano_meshes(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate MANO hand meshes from pose parameters"""

        pred_trans = results['pred_trans']
        pred_rot = results['pred_rot']
        pred_hand_pose = results['pred_hand_pose']
        pred_betas = results['pred_betas']

        hand2idx = {"right": 1, "left": 0}
        vis_start = 0
        vis_end = pred_trans.shape[1] - 1

        mano_results = {}

        try:
            # Get MANO faces
            faces = get_mano_faces()
            faces_new = np.array([[92, 38, 234], [234, 38, 239], [38, 122, 239], [239, 122, 279],
                                [122, 118, 279], [279, 118, 215], [118, 117, 215], [215, 117, 214],
                                [117, 119, 214], [214, 119, 121], [119, 120, 121], [121, 120, 78],
                                [120, 108, 78], [78, 108, 79]])
            faces_right = np.concatenate([faces, faces_new], axis=0)

            # Generate right hand vertices
            hand_idx = hand2idx['right']
            pred_glob_r = run_mano(
                pred_trans[hand_idx:hand_idx+1, vis_start:vis_end],
                pred_rot[hand_idx:hand_idx+1, vis_start:vis_end],
                pred_hand_pose[hand_idx:hand_idx+1, vis_start:vis_end],
                betas=pred_betas[hand_idx:hand_idx+1, vis_start:vis_end]
            )
            right_verts = pred_glob_r['vertices'][0]

            mano_results['right_hand'] = {
                'vertices': right_verts,
                'faces': faces_right
            }

            # Generate left hand vertices
            faces_left = faces_right[:,[0,2,1]]
            hand_idx = hand2idx['left']
            pred_glob_l = run_mano_left(
                pred_trans[hand_idx:hand_idx+1, vis_start:vis_end],
                pred_rot[hand_idx:hand_idx+1, vis_start:vis_end],
                pred_hand_pose[hand_idx:hand_idx+1, vis_start:vis_end],
                betas=pred_betas[hand_idx:hand_idx+1, vis_start:vis_end]
            )
            left_verts = pred_glob_l['vertices'][0]

            mano_results['left_hand'] = {
                'vertices': left_verts,
                'faces': faces_left
            }

            print("MANO meshes generated successfully")

        except Exception as e:
            print(f"Error generating MANO meshes: {e}")

        return mano_results

    def process_video_simplified(self,
                               video_path: str,
                               output_dir: Optional[str] = None,
                               visualize: bool = True) -> Dict[str, Any]:
        """
        Simplified processing pipeline for when full HaWoR is not available
        """

        if output_dir is None:
            output_dir = f"simplified_output_{Path(video_path).stem}"
        os.makedirs(output_dir, exist_ok=True)

        print(f"Processing video with simplified pipeline: {video_path}")

        # Load video frames
        frames = self._load_video_frames(video_path)
        print(f"Loaded {len(frames)} frames")

        # Simple hand detection using OpenCV or MediaPipe
        print("Detecting hands...")
        hand_detections = self._detect_hands_simple(frames)

        # Generate mock hand poses
        print("Generating hand poses...")
        hand_poses = self._generate_mock_poses(len(frames), hand_detections)

        results = {
            'video_path': video_path,
            'num_frames': len(frames),
            'hand_poses': hand_poses,
            'hand_detections': hand_detections,
            'method': 'simplified'
        }

        # Save results
        results_path = os.path.join(output_dir, 'simplified_results.npz')
        np.savez(results_path, **results)
        print(f"Results saved to: {results_path}")

        # Create visualizations
        if visualize:
            self._create_simple_visualizations(frames, hand_detections, output_dir)

        return results

    def _load_video_frames(self, video_path: str) -> List[np.ndarray]:
        """Load all frames from video"""
        cap = cv2.VideoCapture(video_path)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

        cap.release()
        return frames

    def _detect_hands_simple(self, frames: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Simple hand detection using basic computer vision"""

        detections = []

        for i, frame in enumerate(frames):
            # Simple skin color detection as a proxy for hands
            detection = self._detect_skin_regions(frame)
            detections.append(detection)

            if i % 20 == 0:
                print(f"Processed {i}/{len(frames)} frames for detection")

        return detections

    def _detect_skin_regions(self, frame: np.ndarray) -> Dict[str, Any]:
        """Detect skin-colored regions as potential hands"""

        # Convert to HSV for skin detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

        # Define skin color range in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)

        # Create mask for skin regions
        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        # Apply morphological operations to clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours by area and find potential hands
        hand_candidates = []
        min_area = 500  # Minimum area for a hand region

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                hand_candidates.append({
                    'bbox': [x, y, x+w, y+h],
                    'area': area,
                    'confidence': min(area / 5000.0, 1.0)  # Normalize confidence
                })

        # Sort by area and take top 2 (for left and right hands)
        hand_candidates.sort(key=lambda x: x['area'], reverse=True)

        result = {'left': None, 'right': None}

        if len(hand_candidates) >= 1:
            # Assign larger hand to right, smaller to left (rough heuristic)
            bbox = hand_candidates[0]['bbox']
            center_x = (bbox[0] + bbox[2]) / 2

            if center_x > frame.shape[1] / 2:
                result['right'] = hand_candidates[0]
            else:
                result['left'] = hand_candidates[0]

        if len(hand_candidates) >= 2:
            bbox = hand_candidates[1]['bbox']
            center_x = (bbox[0] + bbox[2]) / 2

            if center_x > frame.shape[1] / 2 and result['right'] is None:
                result['right'] = hand_candidates[1]
            elif center_x <= frame.shape[1] / 2 and result['left'] is None:
                result['left'] = hand_candidates[1]

        return result

    def _generate_mock_poses(self, num_frames: int, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate mock 3D hand poses based on detections"""

        poses = {
            'translations': {'left': [], 'right': []},
            'rotations': {'left': [], 'right': []},
            'hand_poses': {'left': [], 'right': []},
            'confidences': {'left': [], 'right': []}
        }

        for detection in detections:
            for hand_type in ['left', 'right']:
                if detection[hand_type] is not None:
                    bbox = detection[hand_type]['bbox']
                    confidence = detection[hand_type]['confidence']

                    # Convert 2D bbox to rough 3D pose
                    center_x = (bbox[0] + bbox[2]) / 2
                    center_y = (bbox[1] + bbox[3]) / 2
                    size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])

                    # Mock 3D translation (camera frame)
                    translation = [
                        (center_x - 320) / 320.0,  # Normalize x
                        (center_y - 240) / 240.0,  # Normalize y
                        -0.5 - size / 1000.0       # Mock depth based on size
                    ]

                    # Mock rotation
                    rotation = [0.0, 0.0, 0.0]  # Neutral rotation

                    # Mock hand pose (MANO has 45 parameters for finger poses)
                    hand_pose = np.random.normal(0, 0.1, 45).tolist()  # Small random poses

                else:
                    # No detection - use None or previous pose
                    translation = None
                    rotation = None
                    hand_pose = None
                    confidence = 0.0

                poses['translations'][hand_type].append(translation)
                poses['rotations'][hand_type].append(rotation)
                poses['hand_poses'][hand_type].append(hand_pose)
                poses['confidences'][hand_type].append(confidence)

        return poses

    def _create_simple_visualizations(self,
                                    frames: List[np.ndarray],
                                    detections: List[Dict[str, Any]],
                                    output_dir: str):
        """Create simple visualizations showing detected hands"""

        vis_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)

        for i, (frame, detection) in enumerate(zip(frames[:50], detections[:50])):
            vis_frame = frame.copy()

            # Draw hand bounding boxes
            for hand_type in ['left', 'right']:
                if detection[hand_type] is not None:
                    bbox = detection[hand_type]['bbox']
                    confidence = detection[hand_type]['confidence']

                    x1, y1, x2, y2 = bbox
                    color = (255, 0, 0) if hand_type == 'left' else (0, 255, 0)

                    # Draw bounding box
                    cv2.rectangle(vis_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                    # Draw label with confidence
                    label = f"{hand_type} ({confidence:.2f})"
                    cv2.putText(vis_frame, label, (int(x1), int(y1-10)),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Save visualization frame
            vis_path = os.path.join(vis_dir, f'frame_{i:04d}.jpg')
            cv2.imwrite(vis_path, cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR))

        print(f"Visualizations saved to: {vis_dir}")

    def create_summary_report(self, results: Dict[str, Any], output_dir: str):
        """Create a summary report of the processing results"""

        report_path = os.path.join(output_dir, 'summary_report.txt')

        with open(report_path, 'w') as f:
            f.write("HaWoR Processing Summary Report\n")
            f.write("=" * 40 + "\n\n")

            f.write(f"Video Path: {results['video_path']}\n")
            f.write(f"Processing Method: {results.get('method', 'hawor')}\n")
            f.write(f"Device Used: {self.device}\n")
            f.write(f"Models Available: {self.models_available}\n\n")

            if 'num_frames' in results:
                f.write(f"Total Frames: {results['num_frames']}\n")

            if 'hand_poses' in results:
                poses = results['hand_poses']
                if 'confidences' in poses:
                    left_detections = sum(1 for c in poses['confidences']['left'] if c > 0.5)
                    right_detections = sum(1 for c in poses['confidences']['right'] if c > 0.5)
                    f.write(f"Left Hand Detections: {left_detections}\n")
                    f.write(f"Right Hand Detections: {right_detections}\n")

            f.write(f"\nOutput Directory: {output_dir}\n")
            f.write(f"Report Generated: {Path(__file__).name}\n")

        print(f"Summary report saved to: {report_path}")


def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='Advanced HaWoR for hand pose estimation')
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--output', type=str, help='Output directory')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cuda/mps/cpu/auto)')
    parser.add_argument('--mode', type=str, default='auto', choices=['auto', 'hawor', 'simplified'],
                       help='Processing mode')
    parser.add_argument('--focal', type=float, help='Camera focal length')
    parser.add_argument('--no-vis', action='store_true', help='Skip visualization')

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = AdvancedHaWoR(device=args.device)

    # Determine processing mode
    if args.mode == 'auto':
        if HAWOR_AVAILABLE and pipeline.models_available['checkpoint']:
            mode = 'hawor'
        else:
            mode = 'simplified'
    else:
        mode = args.mode

    print(f"Using processing mode: {mode}")

    # Process video
    if mode == 'hawor':
        results = pipeline.process_video_with_hawor(
            video_path=args.video,
            output_dir=args.output,
            img_focal=args.focal
        )
    else:
        results = pipeline.process_video_simplified(
            video_path=args.video,
            output_dir=args.output,
            visualize=not args.no_vis
        )

    # Create summary report
    output_dir = args.output or f"output_{Path(args.video).stem}"
    pipeline.create_summary_report(results, output_dir)

    print("Processing complete!")
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()