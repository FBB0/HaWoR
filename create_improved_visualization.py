#!/usr/bin/env python3
"""
Create improved hand visualization with better rendering and debugging info
"""

import os
import sys
import numpy as np
import cv2
import torch
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

def project_3d_to_2d(vertices_3d, focal_length, img_center):
    """
    Project 3D vertices to 2D image coordinates with better handling
    """
    # Filter out vertices behind the camera
    valid_mask = vertices_3d[:, 2] > 0.1  # Only vertices in front of camera
    
    if not np.any(valid_mask):
        return None, valid_mask
    
    # Simple perspective projection
    x_2d = (vertices_3d[:, 0] * focal_length / vertices_3d[:, 2]) + img_center[0]
    y_2d = (vertices_3d[:, 1] * focal_length / vertices_3d[:, 2]) + img_center[1]
    
    vertices_2d = np.column_stack([x_2d, y_2d])
    return vertices_2d, valid_mask

def draw_hand_mesh(frame, vertices_2d, faces, valid_mask, color, thickness=1):
    """
    Draw hand mesh on frame with proper filtering
    """
    if vertices_2d is None:
        return frame
    
    # Draw mesh edges
    for face in faces:
        # Check if all vertices in face are valid
        if all(valid_mask[face[i]] for i in range(3)):
            # Get triangle vertices
            v1 = vertices_2d[face[0]].astype(int)
            v2 = vertices_2d[face[1]].astype(int)
            v3 = vertices_2d[face[2]].astype(int)
            
            # Draw triangle edges
            cv2.line(frame, tuple(v1), tuple(v2), color, thickness)
            cv2.line(frame, tuple(v2), tuple(v3), color, thickness)
            cv2.line(frame, tuple(v3), tuple(v1), color, thickness)
    
    return frame

def draw_hand_keypoints(frame, vertices_2d, valid_mask, color, radius=3):
    """
    Draw hand keypoints (joints) on frame
    """
    if vertices_2d is None:
        return frame
    
    # Draw keypoints for valid vertices
    for i, (vertex, is_valid) in enumerate(zip(vertices_2d, valid_mask)):
        if is_valid:
            cv2.circle(frame, tuple(vertex.astype(int)), radius, color, -1)
    
    return frame

def create_improved_visualization(results_path, output_dir, video_path=None):
    """
    Create improved hand visualization with better rendering
    """
    
    # Load results
    print(f"Loading results from: {results_path}")
    data = np.load(results_path, allow_pickle=True)
    
    # Extract data
    pred_trans = data['pred_trans']  # (2, 121, 3)
    pred_rot = data['pred_rot']      # (2, 121, 3)
    pred_hand_pose = data['pred_hand_pose']  # (2, 121, 45)
    pred_betas = data['pred_betas']  # (2, 121, 10)
    img_focal = data['img_focal']    # focal length
    imgfiles = data['imgfiles']      # list of image file paths
    
    # Get hand meshes
    right_hand_data = data['right_hand'].item()
    left_hand_data = data['left_hand'].item()
    
    right_vertices = right_hand_data['vertices']  # (120, 778, 3)
    left_vertices = left_hand_data['vertices']    # (120, 778, 3)
    right_faces = right_hand_data['faces']        # (1538, 3)
    left_faces = left_hand_data['faces']          # (1538, 3)
    
    print(f"Right hand vertices shape: {right_vertices.shape}")
    print(f"Left hand vertices shape: {left_vertices.shape}")
    print(f"Number of frames: {len(imgfiles)}")
    print(f"Focal length: {img_focal}")
    
    # Create output directory
    vis_dir = os.path.join(output_dir, 'improved_visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Load first image to get dimensions
    if len(imgfiles) > 0:
        first_img = cv2.imread(imgfiles[0])
        if first_img is not None:
            height, width = first_img.shape[:2]
            img_center = (width // 2, height // 2)
        else:
            width, height = 640, 480
            img_center = (320, 240)
    else:
        width, height = 640, 480
        img_center = (320, 240)
    
    print(f"Image dimensions: {width}x{height}")
    print(f"Image center: {img_center}")
    
    # Process each frame
    num_frames = min(len(imgfiles), right_vertices.shape[0])
    print(f"Processing {num_frames} frames...")
    
    # Statistics
    right_hand_visible = 0
    left_hand_visible = 0
    
    for frame_idx in range(num_frames):
        if frame_idx % 10 == 0:
            print(f"Processing frame {frame_idx}/{num_frames}")
        
        # Load frame
        if frame_idx < len(imgfiles):
            frame = cv2.imread(imgfiles[frame_idx])
            if frame is None:
                frame = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Get hand vertices for this frame
        right_verts = right_vertices[frame_idx]  # (778, 3)
        left_verts = left_vertices[frame_idx]    # (778, 3)
        
        # Project 3D vertices to 2D
        right_2d, right_valid = project_3d_to_2d(right_verts, img_focal, img_center)
        left_2d, left_valid = project_3d_to_2d(left_verts, img_focal, img_center)
        
        # Count visible hands
        if right_2d is not None and np.any(right_valid):
            right_hand_visible += 1
        if left_2d is not None and np.any(left_valid):
            left_hand_visible += 1
        
        # Draw right hand (green)
        if right_2d is not None:
            frame = draw_hand_mesh(frame, right_2d, right_faces, right_valid, (0, 255, 0), 2)
            frame = draw_hand_keypoints(frame, right_2d, right_valid, (0, 255, 0), 2)
        
        # Draw left hand (red)
        if left_2d is not None:
            frame = draw_hand_mesh(frame, left_2d, left_faces, left_valid, (0, 0, 255), 2)
            frame = draw_hand_keypoints(frame, left_2d, left_valid, (0, 0, 255), 2)
        
        # Add frame info
        cv2.putText(frame, f"Frame {frame_idx}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add hand visibility info
        right_vis = "R" if (right_2d is not None and np.any(right_valid)) else " "
        left_vis = "L" if (left_2d is not None and np.any(left_valid)) else " "
        cv2.putText(frame, f"Hands: {right_vis}{left_vis}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add depth info
        if right_2d is not None and np.any(right_valid):
            avg_depth = np.mean(right_verts[right_valid, 2])
            cv2.putText(frame, f"R Depth: {avg_depth:.2f}", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if left_2d is not None and np.any(left_valid):
            avg_depth = np.mean(left_verts[left_valid, 2])
            cv2.putText(frame, f"L Depth: {avg_depth:.2f}", (10, 140), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Save visualization frame
        vis_path = os.path.join(vis_dir, f'frame_{frame_idx:04d}.jpg')
        cv2.imwrite(vis_path, frame)
    
    print(f"Improved visualizations saved to: {vis_dir}")
    print(f"Right hand visible in {right_hand_visible}/{num_frames} frames")
    print(f"Left hand visible in {left_hand_visible}/{num_frames} frames")
    
    # Create a video from the frames
    if len(os.listdir(vis_dir)) > 0:
        create_video_from_frames(vis_dir, output_dir, "improved_hand_visualization.mp4")
    
    return vis_dir

def create_video_from_frames(vis_dir, output_dir, video_name):
    """Create a video from the visualization frames"""
    
    # Get all frame files
    frame_files = sorted([f for f in os.listdir(vis_dir) if f.endswith('.jpg')])
    
    if len(frame_files) == 0:
        print("No frame files found for video creation")
        return
    
    # Load first frame to get dimensions
    first_frame = cv2.imread(os.path.join(vis_dir, frame_files[0]))
    height, width, _ = first_frame.shape
    
    # Create video writer
    video_path = os.path.join(output_dir, video_name)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    print(f"Creating video: {video_path}")
    
    # Write frames to video
    for frame_file in frame_files:
        frame_path = os.path.join(vis_dir, frame_file)
        frame = cv2.imread(frame_path)
        if frame is not None:
            out.write(frame)
    
    out.release()
    print(f"Video saved to: {video_path}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create improved hand visualization from HaWoR results')
    parser.add_argument('--results', type=str, required=True, help='Path to hawor_results.npz file')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--video', type=str, help='Original video path (optional)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results):
        print(f"Error: Results file not found: {args.results}")
        return
    
    create_improved_visualization(args.results, args.output, args.video)

if __name__ == '__main__':
    main()
