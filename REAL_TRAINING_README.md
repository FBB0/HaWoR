# Real HaWoR Training Pipeline

This implementation provides **actual neural network training** for HaWoR models, replacing the previous simulation-based approach with real model training, loss computation, and optimization.

## 🎯 What's New

### Previously (Simulation)
- Mock training loops with fake metrics
- No actual model forward/backward passes
- Simulated loss values and progress
- No real gradient computation

### Now (Real Training)
- **Actual neural network training** with real gradients
- **Real loss functions** based on MANO reconstruction
- **Vision Transformer backbone** with temporal modeling
- **Comprehensive evaluation framework**
- **Real optimization** with AdamW and learning rate scheduling

## 🏗️ Architecture Overview

### Model Components
1. **Vision Transformer Backbone**: Feature extraction from egocentric images
2. **Temporal Transformer**: Sequence modeling for video inputs
3. **Hand Pose Regressor**: MANO parameter prediction (pose, shape, translation, rotation)
4. **Camera Pose Regressor**: SLAM component for camera trajectory estimation
5. **Motion Infiller Network**: LSTM-based temporal consistency and missing frame completion

### Loss Functions
1. **MANO Parameter Loss**: MSE on pose and shape parameters
2. **3D Keypoint Loss**: L2 distance on hand joint positions
3. **Reprojection Loss**: 2D keypoint reprojection error
4. **Temporal Consistency Loss**: Smooth motion across frames
5. **Camera Trajectory Loss**: SLAM-based camera pose estimation
6. **Hand-World Consistency Loss**: Consistent hand poses in world coordinates

## 📂 File Structure

```
src/
├── models/
│   ├── hawor_model.py           # Real HaWoR model architecture
│   ├── advanced_hawor.py        # Enhanced pipeline (existing)
│   └── simplified_hawor.py      # Fallback implementation
├── training/
│   ├── real_hawor_trainer.py    # Real training pipeline
│   ├── hawor_losses.py          # Comprehensive loss functions
│   └── arctic_training_pipeline.py  # Previous simulation (deprecated)
├── datasets/
│   ├── arctic_dataset_real.py   # Real ARCTIC data loading
│   └── arctic_dataset.py        # Original dataset interface
└── evaluation/
    ├── hawor_evaluation.py      # Comprehensive evaluation framework
    └── arctic_evaluation_framework.py  # Previous evaluation

train_real_hawor.py              # Main training script
test_real_training.py            # Test pipeline validation
arctic_training_config.yaml     # Updated training configuration
```

## 🚀 Usage

### 1. Test the Pipeline
```bash
python train_real_hawor.py --test-only
```

### 2. Train the Model
```bash
python train_real_hawor.py --config arctic_training_config.yaml
```

### 3. Resume Training
```bash
python train_real_hawor.py --resume outputs/real_hawor_training/latest_checkpoint.pth
```

### 4. Evaluate Trained Model
```bash
python train_real_hawor.py --eval-only
```

### 5. Debug Mode
```bash
python train_real_hawor.py --debug
```

## 📊 Training Configuration

### Key Parameters
- **Learning Rate**: 5e-5 with cosine annealing
- **Batch Size**: 1 (memory efficient for sequences)
- **Sequence Length**: 8 frames
- **Image Size**: 256x256
- **Optimizer**: AdamW with weight decay
- **Loss Weights**: Balanced for hand pose and camera tracking

### Hardware Support
- **CUDA**: Full GPU training support
- **Apple Silicon (MPS)**: Optimized for Mac M1/M2
- **CPU**: Fallback support

## 🔍 Evaluation Metrics

### Hand Pose Metrics
- **MPJPE**: Mean Per Joint Position Error (mm)
- **PA-MPJPE**: Procrustes-Aligned MPJPE (mm)
- **Per-joint Analysis**: Individual finger and wrist errors
- **Temporal Consistency**: Motion smoothness evaluation

### Camera Tracking Metrics
- **Translation Error**: Camera position accuracy (meters)
- **Rotation Error**: Camera orientation accuracy (radians)
- **Trajectory Smoothness**: SLAM consistency evaluation

## 🎯 Key Improvements

### 1. Real Neural Network Training
- Actual forward/backward passes through Vision Transformer
- Real gradient computation and optimization
- Proper model parameter updates

### 2. Comprehensive Loss Functions
- MANO-based reconstruction loss
- 3D keypoint supervision
- 2D reprojection constraints
- Temporal consistency regularization
- Camera pose supervision

### 3. Advanced Model Architecture
- Vision Transformer backbone (timm integration)
- Temporal sequence modeling
- Multi-head attention for hand fusion
- LSTM-based motion infiller

### 4. Production-Ready Features
- Checkpoint saving/loading
- Training visualization
- Comprehensive evaluation
- Error handling and debugging

## 📈 Expected Performance

### Training Progress
- **Initial Loss**: ~0.4-0.5
- **Converged Loss**: ~0.1-0.2
- **Keypoint Error**: <10mm MPJPE target
- **Training Time**: ~1-2 hours per epoch (depending on hardware)

### Evaluation Targets
- **MPJPE**: <15mm on ARCTIC dataset
- **PA-MPJPE**: <10mm after Procrustes alignment
- **Camera Error**: <5cm translation, <0.1 rad rotation

## 🛠️ Technical Details

### Model Architecture
```python
HaWoRModel(
    backbone: VisionTransformer,      # Feature extraction
    temporal_encoder: TransformerEncoder,  # Sequence modeling
    hand_regressors: {left, right},   # Hand pose prediction
    camera_regressor: CameraPose,     # SLAM component
    motion_infiller: LSTM             # Temporal consistency
)
```

### Loss Computation
```python
total_loss = (
    λ_keypoint * keypoint_loss +
    λ_mano * mano_loss +
    λ_temporal * temporal_loss +
    λ_camera * camera_loss +
    λ_reprojection * reprojection_loss
)
```

## 🔧 Requirements

### Core Dependencies
- PyTorch ≥ 1.12
- torchvision
- numpy
- opencv-python
- matplotlib
- tqdm
- PyYAML

### Optional Dependencies
- timm (for advanced Vision Transformer models)
- wandb (for experiment tracking)
- tensorboard (for visualization)

## 🎉 Success Indicators

### Training Success
- ✅ Loss decreases consistently over epochs
- ✅ Validation metrics improve
- ✅ Checkpoints save successfully
- ✅ Visualizations show learning progress

### Model Quality
- ✅ Hand poses look anatomically correct
- ✅ Temporal motion is smooth
- ✅ Camera tracking follows scene motion
- ✅ Evaluation metrics meet targets

## 🐛 Troubleshooting

### Common Issues
1. **Out of Memory**: Reduce batch size or sequence length
2. **Slow Training**: Check device utilization (CUDA/MPS)
3. **High Loss**: Verify data loading and loss weights
4. **NaN Loss**: Enable gradient clipping, check learning rate

### Debug Mode
```bash
python train_real_hawor.py --debug
```
Enables anomaly detection and verbose error reporting.

---

**This implementation transforms HaWoR from a simulation-based system to a real neural network training pipeline capable of learning hand pose estimation from egocentric videos.**