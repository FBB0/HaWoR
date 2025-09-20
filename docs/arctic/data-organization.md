# ARCTIC Data Organization

## Current Data Structure

Based on the official ARCTIC documentation and our downloaded data, here's how the data is organized:

### 📁 Data Directory Structure

```
thirdparty/arctic/unpack/arctic_data/data/
├── raw_seqs/                    # Raw ground truth sequences
│   ├── s01/                     # Subject 1
│   │   ├── box_grab_01.mano.npy      # MANO hand parameters
│   │   ├── box_grab_01.egocam.dist.npy # Egocentric camera poses
│   │   ├── box_grab_01.object.npy    # Object parameters
│   │   └── box_grab_01.smplx.npy     # SMPLX full-body parameters
│   ├── s02/                     # Subject 2
│   └── ...                      # s03-s10
├── cropped_images/              # Cropped images (116GB) 🔄 DOWNLOADING
│   ├── s01/
│   │   ├── box_grab_01/         # Image sequence folder
│   │   │   ├── 0/               # Camera 0 images
│   │   │   │   ├── 00023.jpg
│   │   │   │   ├── 00063.jpg
│   │   │   │   └── ...
│   │   │   ├── 1/               # Camera 1 images
│   │   │   ├── 2/               # Camera 2 images
│   │   │   └── ...
│   │   └── ...
│   └── ...
├── meta/                        # Metadata and templates
│   ├── misc.json                # Camera parameters, sync offsets
│   ├── object_vtemplates/       # Object 3D templates
│   └── subject_vtemplates/      # Subject-specific templates
├── splits_json/                 # Train/val/test splits
│   ├── protocol_p1.json         # Protocol 1 (allocentric)
│   └── protocol_p2.json         # Protocol 2 (egocentric)
└── body_models/                 # MANO and SMPLX models
    ├── mano/                    # MANO hand models
    └── smplx/                   # SMPLX full-body models
```

## 🖼️ Images Organization

### Cropped Images Structure:
- **Location**: `cropped_images/s{subject}/{sequence_name}/{camera_id}/`
- **Format**: `.jpg` files
- **Naming**: Frame numbers (e.g., `00023.jpg`, `00063.jpg`)
- **Cameras**: Multiple camera views (0, 1, 2, etc.)
- **Cropping**: Loosely cropped around object center for fast loading

### Image-Correspondence:
- Each image corresponds to a frame in the MANO parameter files
- Frame numbers in image filenames match frame indices in `.mano.npy` files
- Multiple camera views available for each sequence

## 🎭 Masks and Segmentation

### **Important Finding**: 
**ARCTIC does NOT provide pre-computed masks or segmentation files.**

### Why No Masks?
1. **Raw Dataset**: ARCTIC provides raw MoCap data and images
2. **Research Focus**: The dataset is designed for hand-object reconstruction, not segmentation
3. **Custom Processing**: Masks need to be generated from MANO parameters + object models

### How to Get Masks:
1. **From MANO Parameters**: Use MANO model to generate hand meshes
2. **From Object Models**: Use object templates to generate object meshes  
3. **Rendering**: Render meshes to create segmentation masks
4. **Pre-processed Splits**: Download `splits.zip` (18GB) which may contain processed data

## 📊 Data Correspondence

### Frame Synchronization:
```
Image: cropped_images/s01/box_grab_01/0/00023.jpg
↓ (same frame)
MANO: raw_seqs/s01/box_grab_01.mano.npy[frame_23]
Camera: raw_seqs/s01/box_grab_01.egocam.dist.npy[frame_23]
Object: raw_seqs/s01/box_grab_01.object.npy[frame_23]
```

### File Formats:
- **Images**: `.jpg` (cropped, multiple cameras)
- **MANO**: `.npy` (hand parameters, 45 pose + 10 shape + 3 trans + 3 rot)
- **Camera**: `.npy` (egocentric camera poses, intrinsics, distortion)
- **Objects**: `.npy` (articulated object poses, 7 parameters)
- **SMPLX**: `.npy` (full-body parameters)

## 🚀 For HaWoR Training

### What We Have:
✅ **Images**: Cropped egocentric images (116GB)  
✅ **Labels**: MANO hand parameters (ground truth)  
✅ **Camera**: Egocentric camera poses and intrinsics  
✅ **Objects**: Object interaction data  
✅ **Splits**: Train/val/test splits  

### What We Need to Generate:
🔄 **Masks**: Hand and object segmentation masks  
🔄 **Keypoints**: 2D/3D hand keypoints from MANO  
🔄 **HaWoR Format**: Convert to HaWoR training format  

### Next Steps:
1. **Download splits.zip** (18GB) - may contain processed data with masks
2. **Generate masks** from MANO parameters + object models
3. **Create HaWoR dataset** with images + masks + keypoints
4. **Set up training pipeline** with proper data loading

## 📈 Data Statistics

- **Subjects**: 10 (s01-s10)
- **Sequences**: 200+ interaction sequences
- **Images**: ~200+ image sequences (116GB cropped)
- **Cameras**: Multiple views per sequence
- **Frames**: Variable length sequences
- **Objects**: 10+ different objects (box, phone, laptop, etc.)

## 🎯 Key Insights

1. **No Pre-computed Masks**: Need to generate from MANO + object models
2. **Multiple Camera Views**: Rich multi-view data available
3. **Egocentric Focus**: Perfect for HaWoR's first-person perspective
4. **High Quality**: MoCap ground truth with precise hand tracking
5. **Large Scale**: 200+ sequences across 10 subjects

The data is well-organized and ready for HaWoR integration once we generate the necessary masks and convert to HaWoR format!
