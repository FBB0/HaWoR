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
├── docs/                  # 📚 Documentation (organized by category)
├── src/                   # 💻 Source code (organized by functionality)
├── outputs/               # 📊 Generated outputs and results
├── configs/               # ⚙️ Configuration files
├── assets/                # 🎨 Static assets (images, etc.)
├── requirements.txt       # 📦 Python dependencies
├── requirements_basic.txt # 🏗️ Basic dependencies
├── license.txt            # 📄 License information
└── .gitignore            # 🚫 Files to ignore in git
```

## 🚀 Quick Start

### 🎯 **One-Command Setup (Recommended)**

```bash
# Clone the repository and run setup
./setup.sh

# Then activate the environment
source .hawor_env/bin/activate

# Run HaWoR
python -m hawor
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

# Run demo
python src/demo.py

# Run ARCTIC evaluation
python src/evaluation/arctic_evaluation_framework.py

# Test basic functionality
python -c "import torch; print('✅ PyTorch works:', torch.__version__)"
python -c "import numpy; print('✅ NumPy works:', numpy.__version__)"
```

### 4. ARCTIC Integration (Optional)
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

- **[🏗️ Core](./src/core/)** - Core HaWoR implementation (hawor/, lib/, infiller/)
- **[🤖 Models](./src/models/)** - Model implementations (simplified, advanced)
- **[🛠️ Utils](./src/utils/)** - Utility functions and tools
- **[📜 Scripts](./src/scripts/)** - Scripts and automation tools
- **[📊 Evaluation](./src/evaluation/)** - Evaluation frameworks
- **[🏋️ Training](./src/training/)** - Training pipelines
- **[🔗 Integration](./src/integration/)** - Setup and integration scripts

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
- ✅ **ARCTIC Integration** - Full ARCTIC dataset support
- ✅ **Enhanced Training** - Advanced training pipelines and evaluation
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

**Happy coding with HaWoR! 🚀**
