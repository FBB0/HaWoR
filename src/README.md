# HaWoR Source Code Structure

This directory contains all the source code for the HaWoR project, organized by functionality for better maintainability.

## 📁 Directory Structure

```
src/
├── README.md           # 📋 This file
├── core/              # 🏗️ Core HaWoR implementation
├── models/            # 🤖 Model implementations
├── utils/             # 🛠️ Utility functions and tools
├── scripts/           # 📜 Scripts and tools
├── evaluation/        # 📊 Evaluation frameworks
├── training/          # 🏋️ Training pipelines and tools
└── integration/       # 🔗 Integration and setup scripts
```

## 🏗️ Core (`core/`)
Contains the core HaWoR implementation:
- `hawor/` - Main HaWoR modules and utilities
- `lib/` - Core libraries and dependencies
- `infiller/` - Hand mesh infilling components
- Core processing logic and algorithms

## 🤖 Models (`models/`)
Contains model implementations:
- `simplified_hawor.py` - Simplified HaWoR pipeline
- `advanced_hawor.py` - Advanced HaWoR implementation
- `hawor_torch_fix.py` - PyTorch compatibility fixes
- Model architectures and implementations

## 🛠️ Utils (`utils/`)
Contains utility functions and tools:
- `create_hand_visualization.py` - Hand visualization tools
- `create_improved_visualization.py` - Improved visualization tools
- `view_meshes.py` - Mesh viewing utilities
- `validate_framework.py` - Framework validation tools
- `validate_metrics.py` - Metrics validation tools
- `verify_dataset_setup.py` - Dataset verification tools
- `memory_optimization.py` - Memory optimization utilities

## 📜 Scripts (`scripts/`)
Contains scripts and automation tools:
- `comprehensive_test_suite.py` - Comprehensive testing suite
- `test_arctic_evaluation.py` - ARCTIC evaluation tests
- `simple_eval.py` - Simple evaluation scripts
- `quick_inference.py` - Quick inference tools
- `run_hawor_with_visualization.py` - Visualization scripts
- `evaluate_and_visualize.py` - Evaluation and visualization tools
- `scripts/` - Additional script collections

## 📊 Evaluation (`evaluation/`)
Contains evaluation frameworks and tools:
- `arctic_evaluation_framework.py` - ARCTIC evaluation framework
- `arctic_comparison_script.py` - ARCTIC comparison tools
- `evaluation_framework/` - Evaluation framework components
- Evaluation metrics and comparison tools

## 🏋️ Training (`training/`)
Contains training pipelines and tools:
- `enhanced_training_evaluation.py` - Enhanced training evaluation
- `enhanced_training_pipeline.py` - Training pipeline implementation
- `training_data_preparation.py` - Data preparation utilities
- `optimize_training_config.py` - Training configuration optimization
- `launch_training.py` - Training launcher
- `monitor_training.py` - Training monitoring tools
- `prepare_training_data.py` - Training data preparation
- `test_training_config.yaml` - Test training configuration

## 🔗 Integration (`integration/`)
Contains integration and setup scripts:
- `setup_arctic_integration.py` - ARCTIC dataset integration
- `setup_training_pipeline.py` - Training pipeline setup
- Integration tools and setup scripts

## 🚀 Usage

The organized structure makes it easy to:
- Find specific functionality by category
- Maintain and extend the codebase
- Add new features in appropriate locations
- Navigate the project structure efficiently

## 📝 Development Guidelines

When adding new code:
1. Place files in the appropriate category directory
2. Follow the existing naming conventions
3. Update this README if adding new categories
4. Maintain the organized structure for future development
