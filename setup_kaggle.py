#!/usr/bin/env python3
"""
Setup script for Kaggle/Colab environment
Run this in a Kaggle/Colab notebook cell
"""

import sys
import subprocess

def run_command(cmd, description):
    """Run a shell command and print status"""
    print(f"\n{'='*60}")
    print(f"▶ {description}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✓ Success")
        if result.stdout:
            print(result.stdout)
    else:
        print(f"✗ Failed (exit code {result.returncode})")
        if result.stderr:
            print(result.stderr)
    return result.returncode == 0

def main():
    print("="*60)
    print("iOrthoPredictor Setup for Kaggle/Colab")
    print("="*60)

    # Check Python version
    print(f"\nPython version: {sys.version}")

    # Check TensorFlow
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
        gpus = tf.config.list_physical_devices('GPU')
        print(f"GPU available: {len(gpus) > 0}")
        if gpus:
            for gpu in gpus:
                print(f"  - {gpu}")
    except ImportError:
        print("⚠ TensorFlow not found - will install")

    print("\n" + "="*60)
    print("Installing dependencies...")
    print("="*60)

    # Install missing dependencies
    packages = [
        "googledrivedownloader",
        "tf-slim"
    ]

    for package in packages:
        run_command(
            f"pip install {package} -q",
            f"Installing {package}"
        )

    # Download model and data
    print("\n" + "="*60)
    print("Downloading pre-trained model and data...")
    print("="*60)

    run_command(
        "python scripts/download_model.py",
        "Downloading pre-trained model (checkpoints)"
    )

    run_command(
        "python scripts/download_dataset.py",
        "Downloading example dataset"
    )

    print("\n" + "="*60)
    print("✓ Setup Complete!")
    print("="*60)
    print("\nYou can now run:")
    print("  python test.py --test_data_dir=examples/cases_for_testing --use_gan --use_style_cont --use_skip")
    print("\nResults will be saved in: examples/cases_for_testing/")

if __name__ == "__main__":
    main()
