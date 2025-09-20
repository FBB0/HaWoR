#!/bin/bash
set -e

echo "🚀 HaWoR Setup Script"
echo "===================="

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

echo "✅ uv is available"

# Check Python version
PYTHON_VERSION=$(python3 --version | awk '{print $2}')
echo "📦 Python version: $PYTHON_VERSION"

if [[ "$(echo $PYTHON_VERSION | cut -d. -f1)" -lt 3 ]] || [[ "$(echo $PYTHON_VERSION | cut -d. -f2)" -lt 10 ]]; then
    echo "❌ Python 3.10+ is required. Current version: $PYTHON_VERSION"
    echo "Please install Python 3.10 or higher."
    exit 1
fi

echo "✅ Python 3.10+ detected"

# Create virtual environment
echo "🔧 Creating virtual environment..."
uv venv .hawor_env --python 3.10

# Activate and install
echo "📦 Installing HaWoR..."
source .hawor_env/bin/activate
uv pip install -e .

echo ""
echo "✅ HaWoR setup complete!"
echo ""
echo "📋 Available commands:"
echo "  source .hawor_env/bin/activate    # Activate environment"
echo "  python -m hawor                  # Run HaWoR interface"
echo "  python -m hawor-demo             # Run demo"
echo "  python setup_hawor.py --help     # See all options"
echo ""
echo "🎯 Next steps:"
echo "  1. Activate environment: source .hawor_env/bin/activate"
echo "  2. Set up ARCTIC credentials (optional):"
echo "     export ARCTIC_USERNAME=your_email@domain.com"
echo "     export ARCTIC_PASSWORD=your_password"
echo "  3. Download ARCTIC data: python setup_hawor.py --download-arctic-mini"
echo ""
echo "📖 For more information, see README.md"
