#!/usr/bin/env python3
"""
Automatic Mask Generation using SAM2 (Segment Anything Model)
This replaces the missing TGeoNet functionality
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path

try:
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    print("ERROR: SAM2 not installed!")
    print("\nTo install SAM2, run:")
    print("  pip install git+https://github.com/facebookresearch/segment-anything-2.git")
    sys.exit(1)


class TeethMaskGenerator:
    """
    Generates the three required masks for iOrthoPredictor using SAM2:
    1. MouthMask.png - Full mouth/teeth region
    2. TeethEdgeUpNew.png - Upper teeth edges
    3. TeethEdgeDownNew.png - Lower teeth edges
    """

    def __init__(self, sam_checkpoint="sam2_hiera_large.pt", model_cfg="sam2_hiera_l.yaml", device="cuda"):
        """
        Initialize SAM2 model

        Args:
            sam_checkpoint: Path to SAM2 checkpoint
            model_cfg: SAM2 model configuration
            device: 'cuda' or 'cpu'
        """
        print(f"Loading SAM2 model on {device}...")
        self.device = device if torch.cuda.is_available() else "cpu"

        # Build SAM2 predictor
        sam2_model = build_sam2(model_cfg, sam_checkpoint, device=self.device)
        self.predictor = SAM2ImagePredictor(sam2_model)
        print("✓ SAM2 loaded successfully")

    def segment_mouth_region(self, image, mouth_center=None):
        """
        Segment the full mouth region using SAM2

        Args:
            image: Input image (H, W, 3) in RGB
            mouth_center: Optional (x, y) point for mouth, auto-detected if None

        Returns:
            mouth_mask: Binary mask of mouth region (H, W)
        """
        h, w = image.shape[:2]

        # Auto-detect mouth center if not provided
        if mouth_center is None:
            # Assume mouth is in lower-center of face
            mouth_center = (w // 2, int(h * 0.65))

        # Set image for SAM2
        self.predictor.set_image(image)

        # Prompt: point in mouth center
        input_point = np.array([mouth_center])
        input_label = np.array([1])  # 1 = foreground

        # Predict
        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True
        )

        # Choose best mask (highest score)
        best_mask = masks[scores.argmax()]

        return best_mask.astype(np.uint8)

    def segment_teeth(self, image, mouth_mask):
        """
        Segment teeth within the mouth region

        Args:
            image: Input image (H, W, 3) in RGB
            mouth_mask: Binary mask of mouth region

        Returns:
            teeth_mask: Binary mask of teeth region
        """
        # Find bounding box of mouth
        coords = np.column_stack(np.where(mouth_mask > 0))
        if len(coords) == 0:
            return np.zeros_like(mouth_mask)

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        # Center of mouth for teeth prompt
        teeth_center_x = (x_min + x_max) // 2
        teeth_center_y = (y_min + y_max) // 2

        # Set image
        self.predictor.set_image(image)

        # Prompt for teeth (center of mouth)
        input_point = np.array([[teeth_center_x, teeth_center_y]])
        input_label = np.array([1])

        # Use bounding box to constrain
        input_box = np.array([x_min, y_min, x_max, y_max])

        # Predict
        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=input_box[None, :],
            multimask_output=True
        )

        # Get best mask and constrain to mouth region
        teeth_mask = masks[scores.argmax()].astype(np.uint8)
        teeth_mask = teeth_mask * mouth_mask  # Keep only within mouth

        return teeth_mask

    def split_upper_lower_teeth(self, teeth_mask):
        """
        Split teeth mask into upper and lower teeth based on geometry

        Args:
            teeth_mask: Binary mask of all teeth

        Returns:
            upper_teeth: Mask of upper teeth
            lower_teeth: Mask of lower teeth
        """
        # Find vertical center of teeth region
        coords = np.column_stack(np.where(teeth_mask > 0))
        if len(coords) == 0:
            return np.zeros_like(teeth_mask), np.zeros_like(teeth_mask)

        y_coords = coords[:, 0]
        y_center = (y_coords.min() + y_coords.max()) // 2

        # Split at center
        upper_teeth = teeth_mask.copy()
        lower_teeth = teeth_mask.copy()

        upper_teeth[y_center:, :] = 0  # Keep only upper half
        lower_teeth[:y_center, :] = 0  # Keep only lower half

        return upper_teeth, lower_teeth

    def extract_edges(self, mask, thickness=2):
        """
        Extract edge contours from a binary mask

        Args:
            mask: Binary mask
            thickness: Edge thickness in pixels

        Returns:
            edges: Binary edge map
        """
        # Find contours
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )

        # Draw contours as edges
        edges = np.zeros_like(mask, dtype=np.uint8)
        cv2.drawContours(edges, contours, -1, 255, thickness)

        return edges

    def generate_all_masks(self, image_path, output_dir, mouth_prompt=None, visualize=True):
        """
        Complete pipeline: Generate all three required masks

        Args:
            image_path: Path to input image
            output_dir: Directory to save masks
            mouth_prompt: Optional (x, y) point for mouth center
            visualize: Whether to save visualization

        Returns:
            dict with paths to generated masks
        """
        print(f"\n{'='*60}")
        print(f"Processing: {image_path}")
        print(f"{'='*60}")

        # Load image
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise ValueError(f"Could not load image: {image_path}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]

        # Step 1: Segment mouth region
        print("Step 1/5: Segmenting mouth region...")
        mouth_mask = self.segment_mouth_region(image_rgb, mouth_prompt)

        # Step 2: Segment teeth within mouth
        print("Step 2/5: Segmenting teeth...")
        teeth_mask = self.segment_teeth(image_rgb, mouth_mask)

        # Step 3: Split into upper and lower teeth
        print("Step 3/5: Splitting upper/lower teeth...")
        upper_teeth, lower_teeth = self.split_upper_lower_teeth(teeth_mask)

        # Step 4: Extract edges
        print("Step 4/5: Extracting tooth edges...")
        upper_edges = self.extract_edges(upper_teeth, thickness=3)
        lower_edges = self.extract_edges(lower_teeth, thickness=3)

        # Step 5: Resize to 256x256 and save
        print("Step 5/5: Resizing and saving masks...")
        os.makedirs(output_dir, exist_ok=True)

        mouth_mask_resized = cv2.resize(
            (mouth_mask * 255).astype(np.uint8),
            (256, 256),
            interpolation=cv2.INTER_NEAREST
        )
        upper_edges_resized = cv2.resize(
            upper_edges,
            (256, 256),
            interpolation=cv2.INTER_NEAREST
        )
        lower_edges_resized = cv2.resize(
            lower_edges,
            (256, 256),
            interpolation=cv2.INTER_NEAREST
        )

        # Save masks
        mouth_mask_path = os.path.join(output_dir, "MouthMask.png")
        upper_edges_path = os.path.join(output_dir, "TeethEdgeUpNew.png")
        lower_edges_path = os.path.join(output_dir, "TeethEdgeDownNew.png")

        cv2.imwrite(mouth_mask_path, mouth_mask_resized)
        cv2.imwrite(upper_edges_path, upper_edges_resized)
        cv2.imwrite(lower_edges_path, lower_edges_resized)

        print(f"✓ Saved: {mouth_mask_path}")
        print(f"✓ Saved: {upper_edges_path}")
        print(f"✓ Saved: {lower_edges_path}")

        # Visualization
        if visualize:
            self._save_visualization(
                image_bgr, mouth_mask, upper_edges, lower_edges, output_dir
            )

        return {
            'mouth_mask': mouth_mask_path,
            'upper_edges': upper_edges_path,
            'lower_edges': lower_edges_path
        }

    def _save_visualization(self, image, mouth_mask, upper_edges, lower_edges, output_dir):
        """Save a visualization of the masks overlaid on the image"""
        # Resize masks to original image size
        h, w = image.shape[:2]
        mouth_viz = cv2.resize((mouth_mask * 255).astype(np.uint8), (w, h))
        upper_viz = cv2.resize(upper_edges, (w, h))
        lower_viz = cv2.resize(lower_edges, (w, h))

        # Create overlay
        overlay = image.copy()
        overlay[mouth_viz > 0] = overlay[mouth_viz > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
        overlay[upper_viz > 0] = [255, 0, 0]  # Red for upper
        overlay[lower_viz > 0] = [0, 0, 255]  # Blue for lower

        # Save
        viz_path = os.path.join(output_dir, "visualization.png")
        cv2.imwrite(viz_path, overlay)
        print(f"✓ Saved visualization: {viz_path}")


def main():
    """
    Example usage
    """
    import argparse

    parser = argparse.ArgumentParser(description="Generate teeth masks using SAM2")
    parser.add_argument('--image', type=str, required=True, help='Input image path')
    parser.add_argument('--output', type=str, required=True, help='Output directory for masks')
    parser.add_argument('--sam-checkpoint', type=str, default='checkpoints/sam2_hiera_large.pt',
                        help='Path to SAM2 checkpoint')
    parser.add_argument('--sam-config', type=str, default='sam2_hiera_l.yaml',
                        help='SAM2 model config')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device to run on')
    parser.add_argument('--mouth-x', type=int, default=None, help='Mouth center X coordinate (optional)')
    parser.add_argument('--mouth-y', type=int, default=None, help='Mouth center Y coordinate (optional)')

    args = parser.parse_args()

    # Initialize generator
    generator = TeethMaskGenerator(
        sam_checkpoint=args.sam_checkpoint,
        model_cfg=args.sam_config,
        device=args.device
    )

    # Optional mouth prompt
    mouth_prompt = None
    if args.mouth_x is not None and args.mouth_y is not None:
        mouth_prompt = (args.mouth_x, args.mouth_y)

    # Generate masks
    result = generator.generate_all_masks(
        image_path=args.image,
        output_dir=args.output,
        mouth_prompt=mouth_prompt,
        visualize=True
    )

    print(f"\n{'='*60}")
    print("SUCCESS! Masks generated:")
    print(f"  - Mouth mask: {result['mouth_mask']}")
    print(f"  - Upper teeth edges: {result['upper_edges']}")
    print(f"  - Lower teeth edges: {result['lower_edges']}")
    print(f"{'='*60}")
    print("\nYou can now run inference:")
    print(f"  python test.py --test_data_dir={args.output} --use_gan --use_style_cont --use_skip")


if __name__ == "__main__":
    main()
