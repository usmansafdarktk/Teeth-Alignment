# Kaggle Notebook Instructions

## Step-by-Step Guide to Run on Kaggle

### Step 1: Create a New Kaggle Notebook

1. Go to https://www.kaggle.com
2. Click "Code" → "New Notebook"
3. **IMPORTANT**: Enable GPU
   - Click on the three dots `⋮` in the top right
   - Select "Accelerator" → "GPU T4 x2" (or any GPU option)
   - Click "Save"

### Step 2: Upload Your Code

**Option A: Upload as ZIP**
```python
# Cell 1: Upload the code
from google.colab import files
uploaded = files.upload()  # Select iOrthopredictor.zip

# Extract
!unzip -q iOrthopredictor.zip
%cd iOrthopredictor
!ls
```

**Option B: Clone from GitHub** (if you uploaded to GitHub)
```python
# Cell 1: Clone repository
!git clone https://github.com/YOUR_USERNAME/iOrthopredictor.git
%cd iOrthopredictor
!ls
```

### Step 3: Check Environment

```python
# Cell 2: Verify TensorFlow and GPU
import tensorflow as tf
import sys

print(f"Python version: {sys.version}")
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
```

Expected output should show TensorFlow 2.x and at least one GPU.

### Step 4: Install Dependencies

```python
# Cell 3: Install required packages
!pip install -qq googledrivedownloader tf-slim

# Verify installation
import googledrivedownloader
import tf_slim
print("✓ All dependencies installed")
```

### Step 5: Setup and Download Models

```python
# Cell 4: Run setup script
!python setup_kaggle.py
```

This will:
- Download the pre-trained model (~1GB)
- Download example test datasets
- Verify everything is ready

### Step 6: Run Test

```python
# Cell 5: Run inference on example cases
!python test.py \
  --test_data_dir=examples/cases_for_testing \
  --use_gan \
  --use_style_cont \
  --use_skip
```

### Step 7: View Results

```python
# Cell 6: List output files
!ls -lh examples/cases_for_testing/*/

# Display result images
from IPython.display import Image, display
import os

# Show results for first case
case_dir = "examples/cases_for_testing/C0000"
for img_file in os.listdir(case_dir):
    if img_file.endswith(('.png', '.jpg')):
        img_path = os.path.join(case_dir, img_file)
        print(f"\n{img_file}:")
        display(Image(filename=img_path, width=400))
```

### Step 8: Process Your Own Images

To process your own images, you need to structure them like the examples:

```python
# Cell 7: Upload and process custom image
# First, upload your image files
from google.colab import files
uploaded = files.upload()

# Structure them properly (you'll need the proper masks)
# See examples/cases_for_testing/C0000/ for the required structure:
# - Img.jpg (input image)
# - TeethEdgeUpNew.png (upper teeth mask)
# - TeethEdgeDownNew.png (lower teeth mask)
# - MouthMask.png (mouth mask)
```

## Troubleshooting

### Error: "No GPU available"
- Make sure you enabled GPU in notebook settings
- Go to Settings → Accelerator → Select GPU

### Error: "Module not found"
- Re-run the dependency installation cell
- Make sure tf-slim is installed: `!pip install tf-slim`

### Error: "Out of memory"
- The model is large - this is expected on free tier
- Try processing one image at a time
- Reduce batch size in test.py (it's already set to 1)

### Download is slow
- Kaggle has good bandwidth - downloads should be fast
- Google Drive downloads might be rate-limited
- Wait for the download to complete

## Complete Notebook (Copy-Paste Ready)

```python
# ========== CELL 1: Clone/Upload Code ==========
# Option A: Upload ZIP
from google.colab import files
uploaded = files.upload()
!unzip -q iOrthopredictor.zip
%cd iOrthopredictor

# OR Option B: Clone from GitHub
# !git clone https://github.com/YOUR_USERNAME/iOrthopredictor.git
# %cd iOrthopredictor


# ========== CELL 2: Check Environment ==========
import tensorflow as tf
import sys
print(f"Python: {sys.version}")
print(f"TensorFlow: {tf.__version__}")
print(f"GPUs: {tf.config.list_physical_devices('GPU')}")


# ========== CELL 3: Install Dependencies ==========
!pip install -qq googledrivedownloader tf-slim
print("✓ Dependencies installed")


# ========== CELL 4: Setup ==========
!python setup_kaggle.py


# ========== CELL 5: Run Test ==========
!python test.py --test_data_dir=examples/cases_for_testing --use_gan --use_style_cont --use_skip


# ========== CELL 6: View Results ==========
from IPython.display import Image, display
import os

case_dir = "examples/cases_for_testing/C0000"
for img in os.listdir(case_dir):
    if img.endswith(('.png', '.jpg')):
        print(f"\n{img}:")
        display(Image(filename=os.path.join(case_dir, img), width=400))
```

## Notes

- Processing takes about 2-5 minutes per case on GPU
- Results are saved in the same directory as input images
- Free Kaggle GPU quota: 30 hours/week
- Notebook sessions timeout after 9-12 hours of inactivity
