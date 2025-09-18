#!/usr/bin/env python3
"""
Create hand visualization by overlaying 3D hand meshes on video frames
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
    Project 3D vertices to 2D image coordinates
    
    Args:
        vertices_3d: (N, 3) array of 3D vertices
        focal_length: camera focal length
        img_center: (cx, cy) image center
    
    Returns:
        vertices_2d: (N, 2) array of 2D coordinates
    """
    # Simple perspective projection
    x_2d = (vertices_3d[:, 0] * focal_length / vertices_3d[:, 2]) + img_center[0]
    y_2d = (vertices_3d[:, 1] * focal_length / vertices_3d[:, 2]) + img_center[1]
    
    return np.column_stack([x_2d, y_2d])

def create_hand_visualization(results_path, output_dir, video_path=None):
    """
    Create hand visualization by overlaying 3D hand meshes on video frames
    """
    
    # Load results
    print(f"Loading results from: {results_path}")
    data = np.load(results_path, allow_pickle=True)
    
    # Extract data
    pred_trans = data['pred_trans']  # (2, 121, 3) - left and right hand translations
    pred_rot = data['pred_rot']      # (2, 121, 3) - left and right hand rotations
    pred_hand_pose = data['pred_hand_pose']  # (2, 121, 45) - hand pose parameters
    pred_betas = data['pred_betas']  # (2, 121, 10) - hand shape parameters
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
    vis_dir = os.path.join(output_dir, 'hand_visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Load first image to get dimensions
    if len(imgfiles) > 0:
        first_img = cv2.imread(imgfiles[0])
        if first_img is not None:
            height, width = first_img.shape[:2]
            img_center = (width // 2, height // 2)
        else:
            # Default values if image can't be loaded
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
    
    for frame_idx in range(num_frames):
        if frame_idx % 10 == 0:
            print(f"Processing frame {frame_idx}/{num_frames}")
        
        # Load frame
        if frame_idx < len(imgfiles):
            frame = cv2.imread(imgfiles[frame_idx])
            if frame is None:
                # Create a black frame if image can't be loaded
                frame = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Get hand vertices for this frame
        right_verts = right_vertices[frame_idx]  # (778, 3)
        left_verts = left_vertices[frame_idx]    # (778, 3)
        
        # Project 3D vertices to 2D
        right_2d = project_3d_to_2d(right_verts, img_focal, img_center)
        left_2d = project_3d_to_2d(left_verts, img_focal, img_center)
        
        # Draw right hand (green)
        for face in right_faces:
            # Get triangle vertices
            v1 = right_2d[face[0]].astype(int)
            v2 = right_2d[face[1]].astype(int)
            v3 = right_2d[face[2]].astype(int)
            
            # Check if vertices are within image bounds and have positive depth
            if (0 <= v1[0] < width and 0 <= v1[1] < height and right_verts[face[0], 2] > 0 and
                0 <= v2[0] < width and 0 <= v2[1] < height and right_verts[face[1], 2] > 0 and
                0 <= v3[0] < width and 0 <= v3[1] < height and right_verts[face[2], 2] > 0):
                
                # Draw triangle with green color
                cv2.polylines(frame, [np.array([v1, v2, v3])], True, (0, 255, 0), 1)
        
        # Draw left hand (red)
        for face in left_faces:
            # Get triangle vertices
            v1 = left_2d[face[0]].astype(int)
            v2 = left_2d[face[1]].astype(int)
            v3 = left_2d[face[2]].astype(int)
            
            # Check if vertices are within image bounds and have positive depth
            if (0 <= v1[0] < width and 0 <= v1[1] < height and left_verts[face[0], 2] > 0 and
                0 <= v2[0] < width and 0 <= v2[1] < height and left_verts[face[1], 2] > 0 and
                0 <= v3[0] < width and 0 <= v3[1] < height and left_verts[face[2], 2] > 0):
                
                # Draw triangle with red color
                cv2.polylines(frame, [np.array([v1, v2, v3])], True, (0, 0, 255), 1)
        
        # Add frame number
        cv2.putText(frame, f"Frame {frame_idx}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Save visualization frame
        vis_path = os.path.join(vis_dir, f'frame_{frame_idx:04d}.jpg')
        cv2.imwrite(vis_path, frame)
    
    print(f"Hand visualizations saved to: {vis_dir}")
    
    # Create a simple video from the frames
    if len(os.listdir(vis_dir)) > 0:
        create_video_from_frames(vis_dir, output_dir)
    
    return vis_dir

def create_video_from_frames(vis_dir, output_dir):
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
    video_path = os.path.join(output_dir, 'hand_visualization.mp4')
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
    
    parser = argparse.ArgumentParser(description='Create hand visualization from HaWoR results')
    parser.add_argument('--results', type=str, required=True, help='Path to hawor_results.npz file')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--video', type=str, help='Original video path (optional)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results):
        print(f"Error: Results file not found: {args.results}")
        return
    
    create_hand_visualization(args.results, args.output, args.video)

if __name__ == '__main__':
    main()
