# Mask Creation Guide for iOrthoPredictor

## The Problem

**To process your own images, you need 4 files:**
1. `Img.jpg` - Your input photo
2. `MouthMask.png` - Mouth region mask (256x256, grayscale)
3. `TeethEdgeUpNew.png` - Upper teeth edge mask (256x256, grayscale)
4. `TeethEdgeDownNew.png` - Lower teeth edge mask (256x256, grayscale)

**The repository does NOT provide automated mask generation code.** You must create these masks manually or use external tools.

## What the Masks Look Like

Based on analysis of the example data:

### MouthMask.png
- **Format**: 256x256 grayscale PNG
- **Content**: White (255) pixels mark the mouth/teeth region, black (0) for background
- **Purpose**: Defines the region where teeth alignment will be applied
- **Processing**: The code dilates this mask by 7x7 pixels (data/data_loader.py:25)

### TeethEdgeUpNew.png & TeethEdgeDownNew.png
- **Format**: 256x256 grayscale PNG
- **Content**: White (255) pixels mark the teeth edges, black (0) for background
- **Purpose**: Defines the contours of upper and lower teeth separately
- **Note**: These are actual edge/boundary maps, not filled regions

## How to Create Masks

### Option 1: Manual Annotation (RECOMMENDED FOR LEARNING)

**Tools Needed:**
- GIMP (free, open-source) or Photoshop
- Good mouse or drawing tablet

**Steps:**
1. Open your image in GIMP
2. Create a new layer for each mask
3. Use brush/pencil tool with white color (255)
4. Trace around the teeth edges carefully
5. Export each layer as PNG (256x256)

**Pros:** Full control, works for any image
**Cons:** Time-consuming (10-30 minutes per image)

### Option 2: Use Face Segmentation Models

**Recommended Tools:**

#### A. MediaPipe Face Mesh (FREE)
```python
# Install
pip install mediapipe opencv-python

# Example code to detect face landmarks
import mediapipe as mp
import cv2
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

# Load image
image = cv2.imread('your_image.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process
results = face_mesh.process(image_rgb)

# Extract mouth landmarks (indices 61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95)
# Create mask from these landmarks
```

**Pros:** Free, automatic, real-time capable
**Cons:** Doesn't separate upper/lower teeth, requires manual processing

#### B. Face-parsing.PyTorch (BETTER FOR TEETH)
Repository: https://github.com/zllrunning/face-parsing.PyTorch

```python
# This model can segment mouth/lips region
# You'll need to:
# 1. Run face parsing to get mouth region
# 2. Apply edge detection (Canny) to get teeth edges
# 3. Manually separate upper/lower teeth
```

**Pros:** More accurate mouth/lip segmentation
**Cons:** Doesn't distinguish upper/lower teeth automatically

### Option 3: Use OpenCV Edge Detection

**Semi-automated approach:**

```python
import cv2
import numpy as np

# Load image
img = cv2.imread('Img.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Crop to mouth region (you need to find coordinates)
mouth_region = gray[100:200, 80:180]  # ADJUST THESE VALUES

# Apply edge detection
edges = cv2.Canny(mouth_region, threshold1=50, threshold2=150)

# Refine and split into upper/lower
# This requires manual adjustment
```

**Pros:** Fast for batch processing
**Cons:** Still requires manual coordinate selection and refinement

### Option 4: Use Existing Dental Segmentation Tools

**DentalSegmentator**: https://github.com/IvisionLab/dental-image-segmentation
- Specifically designed for teeth segmentation
- May require training on your data

**Pros:** Purpose-built for teeth
**Cons:** Complex setup, may need retraining

## Realistic Workflow for Your Own Images

### For Single Image Testing:
1. **Quick & dirty**: Copy an existing example's masks and test (won't be accurate but tests the pipeline)
2. **Manual annotation**: Spend 15-30 minutes with GIMP to create proper masks
3. **Use the result**: This is a research project, not production-ready

### For Multiple Images:
1. Use MediaPipe to detect mouth region automatically
2. Apply edge detection within mouth region
3. Manually refine and separate upper/lower teeth in GIMP
4. Create a template workflow for similar images

## Why the Repository Doesn't Include Mask Generation

This is a **research code release** from an academic paper. The authors likely:
- Created masks manually for their dataset
- Used proprietary annotation tools
- Assumed users would have their own annotation pipeline
- Focused on the teeth alignment algorithm, not data preprocessing

## Practical Recommendation

**For your use case (testing with internet images):**

1. **Start with provided examples** - These already have proper masks
2. **If you must use your own image:**
   - Use MediaPipe to get mouth landmarks
   - Use GIMP to manually refine edges
   - Accept that this is time-consuming
3. **Contact the original authors** - They might have mask generation code not included in the repo

## Example: Quick MediaPipe Script

```python
# save as create_mouth_mask.py
import mediapipe as mp
import cv2
import numpy as np

def create_mouth_mask(image_path, output_path):
    # Initialize MediaPipe
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

    # Load image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]

    # Process
    results = face_mesh.process(image_rgb)

    if not results.multi_face_landmarks:
        print("No face detected!")
        return False

    # Create mask
    mask = np.zeros((h, w), dtype=np.uint8)

    # Mouth landmarks (approximate)
    mouth_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324]

    landmarks = results.multi_face_landmarks[0]
    points = []
    for idx in mouth_indices:
        landmark = landmarks.landmark[idx]
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        points.append([x, y])

    # Draw filled polygon
    points = np.array(points, dtype=np.int32)
    cv2.fillPoly(mask, [points], 255)

    # Resize to 256x256
    mask = cv2.resize(mask, (256, 256))

    # Save
    cv2.imwrite(output_path, mask)
    print(f"Mask saved to {output_path}")
    return True

# Usage
create_mouth_mask('your_image.jpg', 'MouthMask.png')
```

**Note:** This creates a basic mouth mask but you'll still need to manually create the teeth edge masks.

## Summary

**Direct Answer to Your Question:**
- **NO**, the repository does NOT include code to generate masks
- **YES**, you need to create them manually or use external tools
- **RECOMMENDATION**: Use the provided examples for testing, or be prepared to spend time on manual annotation

The teeth edge masks (`TeethEdgeUpNew.png`, `TeethEdgeDownNew.png`) are particularly challenging because they require precise edge detection and manual separation of upper/lower teeth. This is likely why the authors only provided 5 example cases with masks already created.
