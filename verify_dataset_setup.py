#!/usr/bin/env python3
"""
Quick Dataset Setup Verification Script
Checks if datasets are positioned correctly for HaWoR training
"""

import os
from pathlib import Path
import sys

def check_path(path, description, required=True):
    """Check if a path exists and report status"""
    path = Path(path)
    exists = path.exists()
    status = "✅" if exists else ("❌" if required else "⚠️")
    print(f"{status} {description}: {path}")

    if exists and path.is_dir():
        try:
            items = list(path.iterdir())
            print(f"   📁 Contains {len(items)} items")
            if len(items) <= 10:
                for item in items[:5]:
                    print(f"      - {item.name}")
                if len(items) > 5:
                    print(f"      ... and {len(items) - 5} more")
        except PermissionError:
            print("   🔒 Permission denied")

    return exists

def main():
    """Verify dataset positioning"""
    print("🔍 HaWoR Dataset Setup Verification")
    print("=" * 50)

    print("\n📂 Required Directories:")

    # Check MANO data
    mano_ok = check_path("_DATA/data", "MANO data directory")
    if mano_ok:
        check_path("_DATA/data/mano", "MANO models")
        check_path("_DATA/data/mano_mean_params.npz", "MANO mean parameters")

    print("\n📂 ARCTIC Dataset:")
    arctic_root = check_path("thirdparty/arctic/unpack/arctic_data/data", "ARCTIC data root")
    if arctic_root:
        check_path("thirdparty/arctic/unpack/arctic_data/data/raw_seqs", "ARCTIC sequences")
        check_path("thirdparty/arctic/unpack/arctic_data/data/meta", "ARCTIC metadata")

        # Check specific subjects
        subjects = ["s01", "s02", "s04", "s05", "s06", "s07", "s08", "s09", "s10"]
        available_subjects = []
        for subject in subjects:
            subject_path = f"thirdparty/arctic/unpack/arctic_data/data/raw_seqs/{subject}"
            if Path(subject_path).exists():
                available_subjects.append(subject)

        print(f"   📊 Available subjects: {len(available_subjects)}/9")
        print(f"      {', '.join(available_subjects)}")

    print("\n📂 Training Data Directories:")
    training_ok = check_path("training_data", "Training data directory")
    if training_ok:
        check_path("training_data/images", "Training images")
        check_path("training_data/annotations", "Training annotations")
        check_path("training_data/masks", "Training masks")

        # Count files
        try:
            img_count = len(list(Path("training_data/images").glob("*.jpg")))
            ann_count = len(list(Path("training_data/annotations").glob("*.json")))
            print(f"   📊 Training files: {img_count} images, {ann_count} annotations")
        except:
            print("   📊 Unable to count training files")

    validation_ok = check_path("validation_data", "Validation data directory", required=False)
    if validation_ok:
        try:
            val_img_count = len(list(Path("validation_data/images").glob("*.jpg")))
            val_ann_count = len(list(Path("validation_data/annotations").glob("*.json")))
            print(f"   📊 Validation files: {val_img_count} images, {val_ann_count} annotations")
        except:
            print("   📊 Unable to count validation files")

    print("\n📂 Configuration Files:")
    check_path("configs/macbook_training_config.json", "MacBook training config")
    check_path("enhanced_training_pipeline.py", "Enhanced training pipeline")
    check_path("prepare_training_data.py", "Data preparation script")

    print("\n📋 Summary:")

    # Determine current status
    if not mano_ok:
        print("❌ CRITICAL: MANO models missing - cannot train")
        status = "❌ SETUP INCOMPLETE"
    elif not arctic_root:
        print("❌ CRITICAL: ARCTIC dataset missing - cannot convert data")
        status = "❌ SETUP INCOMPLETE"
    elif not training_ok or not Path("training_data/images").exists():
        print("⚠️  ARCTIC data available, but need to convert to training format")
        print("   Run: python prepare_training_data.py")
        status = "⚠️  NEEDS DATA CONVERSION"
    else:
        try:
            img_count = len(list(Path("training_data/images").glob("*.jpg")))
            if img_count == 0:
                print("⚠️  Training directories exist but are empty")
                print("   Run: python prepare_training_data.py")
                status = "⚠️  NEEDS DATA CONVERSION"
            else:
                print(f"✅ Ready to train! Found {img_count} training images")
                status = "✅ READY TO TRAIN"
        except:
            print("⚠️  Cannot verify training data")
            status = "⚠️  NEEDS VERIFICATION"

    print(f"\n🎯 Status: {status}")

    if "READY TO TRAIN" in status:
        print("\n🚀 Next steps:")
        print("   python launch_training.py --config configs/macbook_training_config.json --dry-run")
        print("   python launch_training.py --config configs/macbook_training_config.json")
    elif "NEEDS DATA CONVERSION" in status:
        print("\n🔄 Next steps:")
        print("   source .venv/bin/activate")
        print("   python prepare_training_data.py")
    elif "SETUP INCOMPLETE" in status:
        print("\n📥 Next steps:")
        print("   1. Download ARCTIC dataset to thirdparty/arctic/unpack/arctic_data/data/")
        print("   2. Download MANO models to _DATA/data/mano/")
        print("   3. Run this verification script again")

if __name__ == "__main__":
    main()