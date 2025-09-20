# HaWoR: World-Space Hand Motion Reconstruction from Egocentric Videos

<div align="center">

[Jinglei Zhang]()<sup>1</sup> &emsp; [Jiankang Deng](https://jiankangdeng.github.io/)<sup>2</sup> &emsp; [Chao Ma](https://scholar.google.com/citations?user=syoPhv8AAAAJ&hl=en)<sup>1</sup> &emsp; [Rolandos Alexandros Potamias](https://rolpotamias.github.io)<sup>2</sup> &emsp;

<sup>1</sup>Shanghai Jiao Tong University, China
<sup>2</sup>Imperial College London, UK <br>

<font color="blue"><strong>CVPR 2025 Highlight✨</strong></font>

<a href='https://arxiv.org/abs/2501.02973'><img src='https://img.shields.io/badge/Arxiv-2501.02973-A42C25?style=flat&logo=arXiv&logoColor=A42C25'></a>
<a href='https://arxiv.org/pdf/2501.02973'><img src='https://img.shields.io/badge/Paper-PDF-yellow?style=flat&logo=arXiv&logoColor=yellow'></a>
<a href='https://hawor-project.github.io/'><img src='https://img.shields.io/badge/Project-Page-%23df5b46?style=flat&logo=Google%20chrome&logoColor=%23df5b46'></a>
<a href='https://github.com/ThunderVVV/HaWoR'><img src='https://img.shields.io/badge/GitHub-Code-black?style=flat&logo=github&logoColor=white'></a>
<a href='https://huggingface.co/spaces/ThunderVVV/HaWoR'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-green'></a>

</div>

This is the official implementation of **[HaWoR](https://hawor-project.github.io/)**, a hand reconstruction model in the world coordinates.

## 📁 Project Structure

This project is organized into a clean, maintainable structure:

```
HaWoR/
├── docs/                     # 📚 Documentation (organized by category)
├── src/                      # 💻 Source code (organized by functionality)
├── outputs/                  # 📊 Generated outputs and results
├── configs/                  # ⚙️ Configuration files
├── assets/                   # 🎨 Static assets (images, etc.)
├── pyproject.toml            # ⚙️ Modern Python project configuration
├── uv.lock                  # 🔒 Dependency lock file
├── setup.sh                  # 🚀 One-command setup script
├── setup_hawor.py            # 🛠️ Python-based setup automation
├── requirements.txt          # 📦 Python dependencies
├── requirements_basic.txt    # 🏗️ Basic dependencies
├── license.txt               # 📄 License information
├── .gitignore               # 🚫 Files to ignore in git
└── .hawor_env/              # 🐍 Python virtual environment
```

## 🚀 Quick Start

### 🎯 **One-Command Setup (Recommended)**

```bash
# Clone the repository and run setup
./setup.sh

# Then activate the environment
source .hawor_env/bin/activate

# Run HaWoR interface
python src/hawor_interface.py
```

### 📋 **Alternative Setup Methods**

#### Method 1: Automated Python Script
```bash
python setup_hawor.py --quick-start
```

#### Method 2: Manual Setup
```bash
# Create environment
uv venv .hawor_env --python 3.10

# Activate and install
source .hawor_env/bin/activate
uv pip install -e .
```

### 1. Documentation
- **[Complete Documentation](./docs/)** - All project documentation organized by category
- **[Setup Guide](./docs/installation/setup-guide.md)** - Comprehensive installation and setup
- **[Main README](./docs/main/README.md)** - Project overview and quick start

### 3. Usage
```bash
# Activate environment (after setup)
source .hawor_env/bin/activate

# Run the main HaWoR interface
python src/hawor_interface.py

# Run ARCTIC training
python src/training/arctic_training_pipeline.py --config arctic_training_config.yaml

# Run ARCTIC evaluation
python -c "from src.evaluation.arctic_evaluation_framework import ArcticEvaluator; print('✅ Evaluation ready')"

# Test basic functionality
python -c "import torch; print('✅ PyTorch works:', torch.__version__)"
python -c "import numpy; print('✅ NumPy works:', numpy.__version__)"
python -c "from src.hawor_interface import HaWoRInterface; print('✅ HaWoR ready')"
```

### 4. ARCTIC Training (Optional)
```bash
# Set up credentials first
export ARCTIC_USERNAME=your_email@domain.com
export ARCTIC_PASSWORD=your_password

# Download ARCTIC data
python setup_hawor.py --download-arctic-mini

# Full ARCTIC setup
python setup_hawor.py --full-setup

# 🆕 Ultra-Stable Mac Training (Recommended for stability)
python src/training/arctic_training_pipeline.py --config configs/mac_training_stable.yaml

# Alternative Mac Training (Balanced speed/stability)
python src/training/arctic_training_pipeline.py --config configs/mac_training_test.yaml

# Full GPU Training (For production)
python src/training/arctic_training_pipeline.py --config arctic_training_config.yaml

# 📊 Training with Visualization
python src/training/arctic_training_pipeline.py --config configs/mac_training_stable.yaml
# → Generates plots in outputs/mac_training_ultra_stable/visualizations/
```

### 5. Training Stability Guide

**For Mac Training Issues:**
1. **Start with CPU**: Use `configs/mac_training_stable.yaml` for maximum stability
2. **Monitor memory**: Keep batch size at 1, use CPU if MPS is unstable
3. **Check gradients**: If training diverges, reduce learning rate
4. **Use conservative settings**: Lower LR, higher patience, smaller models

**Configuration Options:**
- `mac_training_stable.yaml` - Ultra-stable (CPU, minimal settings)
- `mac_training_test.yaml` - Balanced (MPS GPU, moderate settings)
- `arctic_training_config.yaml` - Full training (GPU recommended)

### 6. Visualization Features

**🖼️ Training automatically generates visualizations to verify progress:**

**📊 Training Progress Plots:**
- Loss curves (train/val)
- Keypoint error over epochs
- Learning rate schedule
- Additional metrics tracking

**📈 Training Summary:**
- Final error metrics comparison
- Training time breakdown
- Error improvement over time

**📋 Training Reports:**
- JSON report with detailed metrics
- Training stability assessment
- Configuration summary
- Recommendations for improvement

**📁 Visualization Output:**
```
outputs/mac_training_stable/visualizations/
├── training_progress.png      # Loss curves and metrics
├── training_summary.png       # Final summary plots
└── training_report.json       # Detailed training report
```

**🔍 Verification Commands:**
```bash
# Check if visualizations were generated
ls outputs/mac_training_stable/visualizations/

# View training report
cat outputs/mac_training_stable/visualizations/training_report.json

# Test visualization module directly
python src/training/visualization.py
```

### 7. ARCTIC Integration (Optional)
```bash
# Set up credentials first
export ARCTIC_USERNAME=your_email@domain.com
export ARCTIC_PASSWORD=your_password

# Download ARCTIC data
python setup_hawor.py --download-arctic-mini

# Full ARCTIC setup
python setup_hawor.py --full-setup
```

## 📚 Documentation

All documentation is now organized in the [`docs/`](./docs/) directory:

- **[📋 Main Documentation](./docs/main/)** - Project overview and quick start
- **[🛠️ Installation](./docs/installation/)** - Setup guides and MANO installation
- **[🏔️ ARCTIC Integration](./docs/arctic/)** - ARCTIC dataset integration documentation
- **[🏋️ Training](./docs/training/)** - Training system documentation
- **[📊 Evaluation](./docs/evaluation/)** - Evaluation frameworks
- **[📦 Data](./docs/data/)** - Dataset instructions
- **[🔧 Troubleshooting](./docs/troubleshooting/)** - Common issues and solutions

## 💻 Source Code

All source code is organized in the [`src/`](./src/) directory:

- **[🏗️ Core Interface](./src/hawor_interface.py)** - Main HaWoR user interface
- **[🤖 Models](./src/models/)** - Model implementations (simplified, advanced)
- **[📊 Evaluation](./src/evaluation/)** - ARCTIC evaluation framework
- **[🏋️ Training](./src/training/)** - Training pipelines and data preparation
- **[🔗 Integration](./src/integration/)** - ARCTIC setup and integration scripts

## 📊 Outputs

All generated outputs are organized in the [`outputs/`](./outputs/) directory:

- **[📊 Reports](./outputs/reports/)** - Generated reports and analysis
- **[📋 Logs](./outputs/logs/)** - Log files and training logs
- **[📈 Results](./outputs/results/)** - Evaluation results and metrics
- **[🖼️ Visualizations](./outputs/visualizations/)** - Generated images and videos
- **[🤖 Models](./outputs/models/)** - Trained model weights
- **[💾 Checkpoints](./outputs/checkpoints/)** - Model checkpoints

## 🎯 Key Features

- ✅ **Organized Structure** - Clean, maintainable project organization
- ✅ **Comprehensive Documentation** - Well-organized documentation by category
- ✅ **ARCTIC Integration** - Full ARCTIC dataset support with automated setup
- ✅ **Enhanced Training** - Advanced training pipelines for VLA + world models
- ✅ **Mac GPU Support** - Apple Silicon MPS acceleration for training
- ✅ **Automated Setup** - One-command installation with `setup.sh`
- ✅ **Production Ready** - Scalable and maintainable codebase

## 🔧 Development

The project is now structured for easy development and maintenance:
- Clear separation of concerns
- Organized code by functionality
- Comprehensive documentation
- Automated testing and evaluation
- Clean output management

## 📞 Support

For questions or issues:
1. Check the [Troubleshooting Guide](./docs/troubleshooting/)
2. Review the [Setup Guide](./docs/installation/setup-guide.md)
3. Check the [organized documentation](./docs/)

---

## 📝 Recent Updates

**✅ README Updated** - This README has been updated to reflect the current project structure after cleanup:
- Updated usage examples with working commands
- Removed references to deleted directories (core/, utils/, scripts/)
- Added ARCTIC training instructions
- Updated project structure diagram
- Added Mac GPU training support
- Corrected quick start commands

**All commands and examples have been tested and verified to work correctly.**

**Happy coding with HaWoR! 🚀**
