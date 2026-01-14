# TGeoNet - Missing Mask Generation Network

## Critical Discovery

**Based on the paper description you provided, this repository is INCOMPLETE.**

The paper describes a **two-stage system**:
1. **TGeoNet** - U-Net based segmentation network (generates masks)
2. **TSynNet** - Synthesis network (generates aligned teeth images)

**This repository only contains TSynNet** (the synthesis part). The TGeoNet (mask generation) code is NOT included.

## Evidence from Repository Analysis

### What's Included (TSynNet):
✅ **RenderingNet** - Main synthesis network (models/module.py:4-121)
✅ **Discriminator** - GAN discriminator (models/module.py:123-151)
✅ **Checkpoint** - Pre-trained model: `TSynNet_gan_skip_scont_2`
✅ **Test script** - Inference code (test.py)
✅ **Training script** - Training code (train.py)

### What's Missing (TGeoNet):
❌ **No TGeoNet class** - Searched entire codebase, no U-Net implementation
❌ **No segmentation network** - No encoder-decoder architecture for masks
❌ **No mask generation code** - Only mask LOADING code exists (data/data_loader.py)
❌ **No TGeoNet checkpoint** - Only TSynNet checkpoint available
❌ **No preprocessing scripts** - Scripts folder only has download utilities

## Repository Structure Analysis

```
iOrthopredictor/
├── models/
│   ├── module.py          ✅ Contains RenderingNet (TSynNet)
│   ├── solver.py          ✅ Contains TSynNetSolver (training/inference)
│   ├── ops.py             ✅ Neural network operations
│   └── loss.py            ✅ Loss functions
├── data/
│   └── data_loader.py     ⚠️  Only LOADS masks, doesn't generate them
├── checkpoints/
│   └── TSynNet_*/         ✅ TSynNet checkpoint (synthesis network)
│       └── NO TGeoNet checkpoint
└── scripts/
    └── download_*.py      ❌ Only download scripts, no preprocessing
```

## What the Code Expects

Looking at `data/data_loader.py`, the code expects **pre-existing masks**:

```python
# data/data_loader.py:11-29
def get_gt_labels(path, names, load_size=256, thresh=50, mdilate=True):
    out = []
    for name in names:
        b = cv2.imread(os.path.join(path, name))  # ← Reads existing mask
        b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
        _, b = cv2.threshold(b, thresh, 1, type=0)
        # ... processes but doesn't generate
```

The code:
1. **Reads** mask files from disk
2. **Processes** them (resize, threshold, dilate)
3. **Does NOT generate** them from input images

## Paper Description vs Repository

### According to You (from the paper):
- **TGeoNet** uses U-Net architecture
- **Input**: 256×256 mouth image
- **Output**: Three binary maps:
  - Upper teeth silhouette (ḡu)
  - Lower teeth silhouette (ḡl)
  - Mouth cavity mask (ḡm)

### In This Repository:
- **TSynNet** uses StyleGAN2-based architecture
- **Input**: Image + pre-existing masks (TeethEdgeUpNew.png, TeethEdgeDownNew.png, MouthMask.png)
- **Output**: Aligned teeth image

**The repository assumes you already have the masks that TGeoNet would generate.**

## Why TGeoNet Might Be Missing

Possible reasons:

1. **Separate Repository** - The authors may have released TGeoNet separately
2. **Proprietary Code** - TGeoNet might use proprietary/licensed components
3. **Simplified Release** - Authors only released the "interesting" synthesis part
4. **Pre-computed Masks** - Authors provided pre-computed masks for the test dataset
5. **Incomplete Release** - This is a partial code release for research purposes

## Implications for Your Use Case

### For the 5 Provided Examples:
✅ **Works perfectly** - Masks are pre-generated, just run inference

### For Your Own Images:
❌ **Cannot process directly** - You need to:
1. Manually create the three masks, OR
2. Find/build a U-Net segmentation model yourself

## What You Need to Do

### Option 1: Contact Original Authors
- Repository: https://github.com/usmansafdarktk/Teeth-Alignment (this is a fork)
- Original author: LCYan1997 (from LICENSE)
- Ask for:
  - TGeoNet code
  - TGeoNet pre-trained checkpoint
  - Instructions for mask generation

### Option 2: Find the Original Paper
Search for the paper to check if:
- There's supplementary code
- There's a different repository link
- The paper explains the mask generation process in detail

Search terms:
- "TGeoNet teeth alignment"
- "TSynNet orthodontic prediction"
- "LCYan1997 teeth synthesis"
- Look for papers on arXiv, IEEE, or CVPR/ICCV proceedings

### Option 3: Build Your Own TGeoNet

Since the paper describes it as "U-Net architecture", you could:

1. **Use a Standard U-Net** for semantic segmentation
   - TensorFlow/Keras U-Net implementation
   - Train on dental images with labeled masks

2. **Use Existing Teeth Segmentation Models**
   - Face-parsing models with mouth segmentation
   - Dental image segmentation papers/code

3. **Use Approximate Methods**
   - MediaPipe for mouth detection
   - Edge detection for teeth boundaries
   - Manual refinement

### Option 4: Use Pre-computed Examples Only
- Stick to the 5 provided test cases
- Use them for demonstration/learning
- Accept the limitation

## Recommended Next Steps

1. **Search for the original paper**
   ```bash
   # Search for:
   - "TGeoNet TSynNet teeth alignment"
   - Author: "LCYan1997"
   - Conference: CVPR, ICCV, ECCV, MICCAI
   ```

2. **Check for related repositories**
   - Original author's GitHub
   - Paper's official code link
   - Supplementary materials

3. **Contact the authors directly**
   - Ask for TGeoNet code release
   - Request pre-trained TGeoNet checkpoint
   - Ask about mask generation methodology

## Summary

**Direct Answer:**
- ❌ **NO**, this repository does NOT contain TGeoNet
- ❌ **NO**, there is NO automatic mask generation
- ✅ **YES**, you must provide pre-made masks or create them yourself
- ⚠️ **This is only half of the system described in the paper**

The repository is **incomplete** - it contains only the synthesis network (TSynNet) but not the mask generation network (TGeoNet) that should precede it.

For production use, you would need:
1. TGeoNet code + checkpoint (missing from this repo)
2. TSynNet code + checkpoint (✅ included in this repo)

Without TGeoNet, you're limited to using the 5 example cases or manually creating masks.
