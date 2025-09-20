#!/usr/bin/env python3
"""
HaWoR Setup Automation Script

This script automates the entire HaWoR setup process including:
1. Environment setup
2. Package installation
3. ARCTIC integration
4. Credential management
5. Data download preparation

Usage:
    python setup_hawor.py --help
    python setup_hawor.py --quick-start
    python setup_hawor.py --full-setup
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import argparse
from typing import List, Optional

class HaWoRSetup:
    """Automated HaWoR setup manager"""

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.env_name = ".hawor_env"

    def check_requirements(self) -> bool:
        """Check if uv is available"""
        try:
            subprocess.run(["uv", "--version"], capture_output=True, check=True)
            print("✅ uv is available")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ uv is not installed. Please install uv first:")
            print("   curl -LsSf https://astral.sh/uv/install.sh | sh")
            return False

    def create_environment(self) -> bool:
        """Create Python 3.10 virtual environment"""
        print("🚀 Creating Python 3.10 environment...")

        try:
            subprocess.run([
                "uv", "venv", self.env_name, "--python", "3.10"
            ], check=True, cwd=self.project_root)

            print(f"✅ Environment created: {self.env_name}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create environment: {e}")
            return False

    def install_dependencies(self, with_arctic: bool = False) -> bool:
        """Install dependencies using uv"""
        print("📦 Installing dependencies...")

        try:
            # Activate environment and install
            env = os.environ.copy()
            env["PATH"] = f"{self.project_root / self.env_name / 'bin'}:{env['PATH']}"

            # Install basic dependencies
            subprocess.run([
                "uv", "pip", "install", "-e", "."
            ], check=True, cwd=self.project_root, env=env)

            if with_arctic:
                print("📦 Installing ARCTIC dependencies...")
                subprocess.run([
                    "uv", "pip", "install", "-e", ".[arctic]"
                ], check=True, cwd=self.project_root, env=env)

            print("✅ Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False

    def setup_arctic(self) -> bool:
        """Set up ARCTIC integration"""
        print("🏔️ Setting up ARCTIC integration...")

        try:
            # Clone ARCTIC repository
            arctic_path = self.project_root / "thirdparty" / "arctic"
            if not arctic_path.exists():
                subprocess.run([
                    "git", "clone", "https://github.com/zc-alexfan/arctic.git",
                    str(arctic_path)
                ], check=True, cwd=self.project_root)

            # Fix Python commands in ARCTIC scripts
            self.fix_arctic_scripts()

            print("✅ ARCTIC integration ready")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to setup ARCTIC: {e}")
            return False

    def fix_arctic_scripts(self):
        """Fix Python commands in ARCTIC scripts"""
        arctic_path = self.project_root / "thirdparty" / "arctic" / "bash"

        if arctic_path.exists():
            # Fix python -> python3 in all scripts
            for script in arctic_path.glob("*.sh"):
                with open(script, 'r') as f:
                    content = f.read()

                if 'python ' in content and 'python3' not in content:
                    content = content.replace('python ', 'python3 ')

                    with open(script, 'w') as f:
                        f.write(content)

    def create_credentials_guide(self):
        """Create credentials setup guide"""
        guide_path = self.project_root / "ARCTIC_CREDENTIALS_SETUP.md"

        guide_content = '''# ARCTIC Credentials Setup Guide

## Required Accounts

You need to register accounts on the following websites:

1. **ARCTIC Dataset**: https://arctic.is.tue.mpg.de/register.php
2. **SMPL-X**: https://smpl-x.is.tue.mpg.de/
3. **MANO**: https://mano.is.tue.mpg.de/

## Setting Up Credentials

After registering, set your credentials:

```bash
# Set credentials (replace with your actual credentials)
export ARCTIC_USERNAME=your_email@domain.com
export ARCTIC_PASSWORD=your_password
export SMPLX_USERNAME=your_email@domain.com
export SMPLX_PASSWORD=your_password
export MANO_USERNAME=your_email@domain.com
export MANO_PASSWORD=your_password
```

## Verify Credentials

Test if your credentials are set correctly:

```bash
echo "ARCTIC: $ARCTIC_USERNAME"
echo "SMPLX: $SMPLX_USERNAME"
echo "MANO: $MANO_USERNAME"
```

## Next Steps

Once credentials are set, run:

```bash
# Test download
python setup_hawor.py --download-arctic-mini

# Full setup
python setup_hawor.py --full-setup
```
'''

        with open(guide_path, 'w') as f:
            f.write(guide_content)

        print(f"📖 Created credentials guide: {guide_path}")

    def download_arctic_mini(self) -> bool:
        """Download mini ARCTIC dataset"""
        print("📥 Downloading mini ARCTIC dataset...")

        try:
            # Check if credentials are set
            if not all([
                os.environ.get('ARCTIC_USERNAME'),
                os.environ.get('ARCTIC_PASSWORD'),
                os.environ.get('SMPLX_USERNAME'),
                os.environ.get('SMPLX_PASSWORD'),
                os.environ.get('MANO_USERNAME'),
                os.environ.get('MANO_PASSWORD')
            ]):
                print("❌ Credentials not set. Please set up credentials first.")
                self.create_credentials_guide()
                return False

            # Run ARCTIC integration script
            subprocess.run([
                "python", "src/integration/setup_arctic_integration.py",
                "--download-mini"
            ], check=True, cwd=self.project_root)

            print("✅ Mini ARCTIC dataset downloaded successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to download ARCTIC data: {e}")
            return False

    def quick_start(self) -> bool:
        """Run quick start setup"""
        print("🚀 HaWoR Quick Start Setup")
        print("=" * 50)

        # Check requirements
        if not self.check_requirements():
            return False

        # Create environment
        if not self.create_environment():
            return False

        # Install dependencies
        if not self.install_dependencies():
            return False

        print("\n" + "=" * 50)
        print("✅ Quick Start Complete!")
        print("=" * 50)
        print("📋 Next steps:")
        print("  1. Set up ARCTIC credentials (see ARCTIC_CREDENTIALS_SETUP.md)")
        print("  2. Download ARCTIC data: python setup_hawor.py --download-arctic-mini")
        print("  3. Run demo: python -m hawor")
        print("\n🎯 HaWoR is ready for basic use!")

        return True

    def full_setup(self) -> bool:
        """Run full setup including ARCTIC"""
        print("🚀 HaWoR Full Setup")
        print("=" * 50)

        # Quick start first
        if not self.quick_start():
            return False

        # Setup ARCTIC
        if not self.setup_arctic():
            return False

        # Download ARCTIC data
        if not self.download_arctic_mini():
            return False

        print("\n" + "=" * 50)
        print("✅ Full Setup Complete!")
        print("=" * 50)
        print("🎯 HaWoR is fully set up with ARCTIC integration!")

        return True

def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description='HaWoR Setup Automation')
    parser.add_argument('--quick-start', action='store_true',
                       help='Quick setup (environment + basic packages)')
    parser.add_argument('--full-setup', action='store_true',
                       help='Full setup (everything including ARCTIC)')
    parser.add_argument('--download-arctic-mini', action='store_true',
                       help='Download mini ARCTIC dataset')
    parser.add_argument('--check-requirements', action='store_true',
                       help='Check if uv is available')

    args = parser.parse_args()

    setup = HaWoRSetup()

    if args.check_requirements:
        return 0 if setup.check_requirements() else 1

    if args.quick_start:
        return 0 if setup.quick_start() else 1

    if args.full_setup:
        return 0 if setup.full_setup() else 1

    if args.download_arctic_mini:
        return 0 if setup.download_arctic_mini() else 1

    # Default: show help
    parser.print_help()
    return 0

if __name__ == "__main__":
    sys.exit(main())
