"""
Image processing service for extracting segmented objects from panorama.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from ..core.config import settings

logger = logging.getLogger(__name__)

def crop_assets_with_transparency(
    panorama_path: str,
    segmentation_output_dir: str,
    scene_name: str = None,
) -> Dict[str, List[str]]:
    """
    Extract segmented assets from panorama with transparent background.

    Args:
        panorama_path: Path to original panorama image
        segmentation_output_dir: Directory containing segmentation results
        scene_name: Scene name for folder organization

    Returns:
        Dictionary mapping asset names to lists of image paths
    """

    if not os.path.exists(panorama_path):
        raise FileNotFoundError(f"Panorama not found: {panorama_path}")

    # Load panorama
    panorama = cv2.imread(panorama_path, cv2.IMREAD_UNCHANGED)
    if panorama is None:
        raise ValueError(f"Failed to load panorama: {panorama_path}")

    # Convert BGR to BGRA
    if panorama.shape[2] == 3:
        panorama = cv2.cvtColor(panorama, cv2.COLOR_BGR2BGRA)

    logger.info(f"📸 Loaded panorama: {panorama.shape}")

    # Setup output directory
    if scene_name:
        output_dir = settings.get_task_paths(scene_name)["assets"]
    else:
        output_dir = settings.output_dir / "default" / "assets"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Masks directory
    masks_dir = Path(segmentation_output_dir) / "masks"
    if not masks_dir.exists():
        raise FileNotFoundError(f"Masks directory not found: {masks_dir}")

    cropped_files = {}

    # Process mask files
    mask_files = sorted(masks_dir.glob("*.png"))

    for mask_path in mask_files:
        asset_name = mask_path.stem

        logger.info(f"🔍 Processing {asset_name} with transparency...")

        # Load mask
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            logger.warning(f"⚠️ Failed to load mask: {mask_path}")
            continue

        # Resize mask if needed
        if mask.shape != panorama.shape[:2]:
            mask = cv2.resize(mask, (panorama.shape[1], panorama.shape[0]))

        # Copy panorama
        result = panorama.copy()

        # Apply mask to alpha channel
        result[:, :, 3] = mask

        # Find bounding box
        coords = cv2.findNonZero(mask)
        if coords is None:
            logger.warning(f"⚠️ Empty mask for {asset_name}")
            continue

        x, y, w, h = cv2.boundingRect(coords)

        # Crop
        cropped = result[y:y+h, x:x+w]

        # Save
        output_path = output_dir / f"{asset_name}.png"
        cv2.imwrite(str(output_path), cropped)

        # Record result
        if asset_name not in cropped_files:
            cropped_files[asset_name] = []
        cropped_files[asset_name].append(str(output_path))

        logger.info(f"✅ Saved with transparency: {output_path} (size: {w}x{h})")

    logger.info(f"\n🎉 Cropped {len(cropped_files)} asset types with transparency")
    logger.info(f"📁 Output directory: {output_dir}")

    return cropped_files
