# Supervised Dataset Download Instructions

## Overview
This directory is set up for HaWoR supervised training and evaluation. The following datasets are commonly used for hand pose estimation:

## Required Datasets

### 1. HOT3D Dataset
- **URL**: https://hot3d-dataset.org/
- **Description**: Hand Object Tracking 3D Dataset
- **Size**: ~50GB
- **Format**: RGB images + 3D annotations
- **Download**: Register and download from official website
- **Place in**: `supervised_data/HOT3D/raw_data/`

### 2. ARCTIC Dataset  
- **URL**: https://arctic.is.tue.mpg.de/
- **Description**: ARCTIC Dataset for Hand-Object Interaction
- **Size**: ~100GB
- **Format**: RGB-D + 3D poses
- **Download**: Register and download from official website
- **Place in**: `supervised_data/ARCTIC/raw_data/`

### 3. DEXYCB Dataset
- **URL**: https://dex-ycb.github.io/
- **Description**: DEX-YCB Dataset for Hand-Object Interaction  
- **Size**: ~200GB
- **Format**: RGB-D + annotations
- **Download**: Register and download from official website
- **Place in**: `supervised_data/DEXYCB/raw_data/`

### 4. HO3D Dataset
- **URL**: https://www.tugraz.at/index.php?id=40231
- **Description**: Hand-Object 3D Dataset
- **Size**: ~30GB
- **Format**: RGB + depth + annotations
- **Download**: Register and download from official website
- **Place in**: `supervised_data/HO3D/raw_data/`

## Setup Steps

1. **Download datasets** from official sources (registration required)
2. **Extract datasets** to their respective `raw_data/` directories
3. **Run preparation script**: `python prepare_for_training.py`
4. **Run evaluation**: `python evaluate_current_model.py`

## Current Status

- ✅ Directory structure created
- ✅ Evaluation scripts prepared
- ✅ Training preparation scripts ready
- ⏳ Waiting for dataset downloads
- ⏳ Waiting for HaWoR training code release

## Notes

- Training code is not yet released by HaWoR authors
- This setup prepares for future training and evaluation
- Current model can be evaluated on downloaded datasets
- All scripts are ready for when training code becomes available
