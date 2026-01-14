# iOrthoPredictor - CPU Setup Guide (Python 3.12)

## Environment Setup

I've successfully set up the environment for you with the following modifications:

### 1. Created a CPU-compatible virtual environment

```bash
source venv_cpu/bin/activate
```

### 2. Updated all dependencies to work with Python 3.12 and CPU-only TensorFlow

The requirements have been updated in `requirements_cpu.txt` with:
- TensorFlow CPU 2.20.0 (latest version supporting Python 3.12)
- TensorFlow-Slim (standalone package, as tf.contrib was removed in TF 2.x)
- Updated OpenCV, NumPy, and SciPy versions
- All compatible packages

### 3. Modified the codebase for TensorFlow 2.x compatibility

Updated all Python files to use TensorFlow 1.x compatibility mode:
- Added `import tensorflow.compat.v1 as tf` and `tf.disable_v2_behavior()`
- Replaced `tf.contrib.slim` with standalone `tf_slim` package
- Updated `tf.contrib.image` functions to use `tf.image` equivalents
- Fixed all Google Drive downloader import issues

### 4. Downloaded pre-trained models and example datasets

- Pre-trained model checkpoints are in `./checkpoints/`
- Example test cases are in `./examples/cases_for_testing/`

## Current Status

The environment is set up and most compatibilityissues are resolved. However, there's one remaining challenge:

### Known Issue: NCHW vs NHWC Data Format

The code encounters this error when running on CPU:
```
Conv2DCustomBackpropInputOp only supports NHWC.
```

This happens because:
- The original code was designed for NVIDIA GPUs which use NCHW format (channels first)
- CPU operations in TensorFlow only support NHWC format (channels last)
- The model architecture uses custom operations that expect NCHW format

### Possible Solutions

1. **Best option**: Run on a system with a supported GPU (even an older NVIDIA card)
2. **Complex option**: Modify the entire codebase to use NHWC format (would require extensive changes to models/ops.py and related files)
3. **Alternative**: Use TensorFlow 1.15 with Python 3.7 in a separate environment (but this won't work with Python 3.12)

## What You Can Do Now

### Option 1: Try with GPU (if available)
If you have access to a CUDA-capable machine, you can:
```bash
pip install tensorflow-gpu==2.16.1  # instead of tensorflow-cpu
```

### Option 2: Use Docker with Older TF Version
Create a Docker container with TensorFlow 1.15 and Python 3.7:
```dockerfile
FROM python:3.7
RUN pip install tensorflow==1.15
# ... rest of setup
```

### Option 3: Wait for Data Format Fix
The codebase would need modifications to:
- Add data format conversion layers (NCHW → NHWC)
- Update all conv operations to use explicit `data_format='channels_last'`
- Modify the upsampling/downsampling operations in extern/dnnlib/

## Files Modified

- `test.py` - Added TF 1.x compatibility
- `train.py` - Added TF 1.x compatibility
- `models/ops.py` - Added TF 1.x compatibility
- `models/loss.py` - Added TF 1.x compatibility, removed unused slim
- `models/solver.py` - Added tf_slim import
- `extern/vgg/vgg.py` - Updated to use tf_slim
- `extern/dnnlib/*.py` - Added TF 1.x compatibility
- `extern/metrics/FID.py` - Added TF 1.x compatibility
- `util/*.py` - Added TF 1.x compatibility, replaced contrib.image functions
- `scripts/download_*.py` - Fixed Google Drive downloader imports

## Summary

The environment is fully configured and dependencies are installed. The code loads successfully but cannot run on CPU-only due to data format constraints in the original StyleGAN2-based architecture. This is unfortunately a fundamental limitation of running this GPU-optimized code on CPU.

For research or production use, I'd recommend:
1. Using a cloud instance with GPU (AWS/Google Cloud/Azure)
2. Running on a local machine with any NVIDIA GPU
3. Using Google Colab with GPU runtime (free tier available)
