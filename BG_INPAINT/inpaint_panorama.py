#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
panorama_inpaint_auto.py

An advanced panorama background inpainting script using Diffusers,
aware of equirectangular projection properties.

Key Features:
 - Full support for all original CLI parameters.
 - 'auto' mode: Automatically calculates optimal wrap_pad, dilate, and feather
   values based on image and mask dimensions.
 - Batch processing for directories of images (`--input_dir`).
 - Flexible mask directory mapping (`--mask_dir`), supporting both a single
   mask folder and per-image subfolders.
 - Option to save intermediate processing steps for debugging.
 - Memory-saving optimizations (VAE tiling, attention slicing) and robust
   exception handling.
"""

import os
import sys
import argparse
import math
import shutil
from pathlib import Path
from typing import Tuple, Optional, List

import cv2
import numpy as np
from PIL import Image
import torch

try:
    from diffusers import AutoPipelineForInpainting
except ImportError:
    print("⚠️ Warning: diffusers library not found. Please ensure it is installed.")
    AutoPipelineForInpainting = None

# ---------------------------
# Helper Functions
# ---------------------------
def load_union_binary(mask_dir: str, H: int, W: int) -> np.ndarray:
    """
    Loads all mask images from a directory and computes their union,
    returning a single binary (0/255) numpy array of size (H, W).
    """
    union_mask = np.zeros((H, W), dtype=np.uint8)
    if not os.path.isdir(mask_dir):
        return union_mask

    for filename in sorted(os.listdir(mask_dir)):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        
        mask_path = os.path.join(mask_dir, filename)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            continue
            
        if mask.shape[:2] != (H, W):
            # Use nearest-neighbor interpolation for binary masks to avoid artifacts.
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
            
        # Combine with the main mask.
        union_mask = np.maximum(union_mask, (mask > 127).astype(np.uint8) * 255)
        
    return union_mask


def build_edge_feather_mask(binary_255: np.ndarray, dilate_px: int, feather_px: int) -> Image.Image:
    """
    Creates a feathered mask primarily at the edges.

    This method creates a mask that is solid white inside the dilated area
    and feathered only on the outer edge. This is crucial for avoiding the
    soft, gray blob artifacts that simple blurring can cause in inpainting.
    """
    if binary_255.max() == 0:
        return Image.fromarray(binary_255, mode="L")

    # 1. Dilate the mask to expand the inpainting region.
    k_dilate = max(1, int(dilate_px))
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k_dilate + 1, 2 * k_dilate + 1))
    dilated_mask = cv2.dilate(binary_255, kernel_dilate, iterations=1)

    if feather_px <= 0:
        return Image.fromarray(dilated_mask, mode="L")

    # 2. Erode the dilated mask to find the solid inner core.
    k_feather = max(1, int(feather_px))
    kernel_feather = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k_feather + 1, 2 * k_feather + 1))
    eroded_core = cv2.erode(dilated_mask, kernel_feather, iterations=1)

    # 3. Subtract the core from the dilated mask to get the edge "ring".
    feather_ring = cv2.subtract(dilated_mask, eroded_core)

    # 4. Blur only the ring.
    blurred_ring = cv2.GaussianBlur(feather_ring, (0, 0), sigmaX=float(k_feather), sigmaY=float(k_feather))
    
    # 5. Combine the solid core with the blurred ring.
    final_mask = np.where(eroded_core > 0, 255, blurred_ring).astype(np.uint8)
    
    return Image.fromarray(final_mask, mode="L")


def erase_masked_region(img: Image.Image, mask_L: Image.Image, mode: str = "gray") -> Image.Image:
    """Overwrites the masked region in the input image to remove visual hints for the model."""
    if mode == "none":
        return img
        
    arr = np.array(img).copy()
    mask_bool = np.array(mask_L) > 0
    
    if not mask_bool.any():
        return img
        
    fill_value = 127 if mode == "gray" else 0
    
    if arr.ndim == 3:
        arr[mask_bool] = [fill_value, fill_value, fill_value]
    else:
        arr[mask_bool] = fill_value
        
    return Image.fromarray(arr, mode="RGB")


def wrap_horizontally(img: Image.Image, mask_L: Image.Image, pad: int) -> Tuple[Image.Image, Image.Image]:
    """
    Pads an image and its mask horizontally to handle panorama seams.
    Resulting image has a width of (W + 2*pad).
    """
    W, H = img.size
    pad = max(1, min(pad, W // 2))
    extended_width = W + pad * 2

    # Wrap the main image
    extended_img = Image.new(img.mode, (extended_width, H))
    extended_img.paste(img.crop((W - pad, 0, W, H)), (0, 0))
    extended_img.paste(img, (pad, 0))
    extended_img.paste(img.crop((0, 0, pad, H)), (pad + W, 0))

    # Wrap the mask
    extended_mask = Image.new("L", (extended_width, H))
    extended_mask.paste(mask_L.crop((W - pad, 0, W, H)), (0, 0))
    extended_mask.paste(mask_L, (pad, 0))
    extended_mask.paste(mask_L.crop((0, 0, pad, H)), (pad + W, 0))
    
    return extended_img, extended_mask


def crop_center(img: Image.Image, W: int) -> Image.Image:
    """Crops the central W region from a horizontally wrapped image."""
    pad = (img.size[0] - W) // 2
    return img.crop((pad, 0, pad + W, img.size[1]))


# ---------------------------
# Auto Heuristics
# ---------------------------
def parse_auto_int(value: str, fallback: int) -> Optional[int]:
    """Parses a string that can be an integer or 'auto'."""
    if value is None:
        return fallback
    if isinstance(value, int):
        return value
        
    s = str(value).lower().strip()
    if s in ("auto", "a", "none", ""):
        return None # Sentinel for auto-computation
        
    try:
        return int(s)
    except (ValueError, TypeError):
        return fallback


def compute_auto_params(W: int, H: int, mask_nonzero_px: int) -> dict:
    """
    Computes heuristic-based parameters for inpainting if 'auto' is selected.
    - wrap_pad: 4% to 8% of image width (min 64px, max W/4).
    - dilate: ~2-5% of image diagonal, adjusted by mask size.
    - feather: ~45% of the dilation value.
    """
    # Heuristic for wrap padding
    wrap_pad = int(np.clip(W * 0.06, 64, max(64, W // 4)))
    
    # Heuristic for dilation
    image_diag = math.hypot(W, H)
    base_dilate = max(4, int(image_diag * 0.02))
    
    # Adjust dilation based on the relative size of the mask
    if mask_nonzero_px > 0:
        mask_area_ratio = mask_nonzero_px / float(W * H)
        if mask_area_ratio < 0.001:   # Very small mask
            base_dilate = max(2, int(base_dilate * 0.5))
        elif mask_area_ratio < 0.01:  # Small mask
            base_dilate = max(3, int(base_dilate * 0.8))
        elif mask_area_ratio > 0.2:   # Very large mask
            base_dilate = int(base_dilate * 1.5)
            
    dilate = int(np.clip(base_dilate, 2, min(W // 8, 200)))
    
    # Heuristic for feathering
    feather = max(0, int(np.clip(dilate * 0.45, 1, 128)))
    
    return {"wrap_pad": wrap_pad, "dilate": dilate, "feather": feather}


# ---------------------------
# Diffusers Pipeline Wrapper
# ---------------------------
class PanoramaInpainter:
    """A wrapper for the Diffusers inpainting pipeline."""
    def __init__(self, model_id: str, device: str = None, dtype=torch.float16, 
                 enable_vae_tiling=True, enable_attention_slicing=True):
        if AutoPipelineForInpainting is None:
            raise RuntimeError("Diffusers AutoPipelineForInpainting failed to import. Please check your installation.")
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        
        print(f"[INFO] Loading pipeline '{model_id}' to device={self.device}, dtype={self.dtype}...")
        self.pipe = AutoPipelineForInpainting.from_pretrained(model_id, torch_dtype=self.dtype)
        self.pipe = self.pipe.to(self.device)
        
        # Apply memory-saving optimizations if available
        if enable_vae_tiling and hasattr(self.pipe, "enable_vae_tiling"):
            try:
                self.pipe.enable_vae_tiling()
                print("[INFO] VAE tiling enabled for memory efficiency.")
            except Exception:
                pass
                
        if enable_attention_slicing and hasattr(self.pipe, "enable_attention_slicing"):
            try:
                self.pipe.enable_attention_slicing()
                print("[INFO] Attention slicing enabled for memory efficiency.")
            except Exception:
                pass

    def inpaint(self, prompt: str, negative_prompt: str, image_ext: Image.Image, mask_ext: Image.Image,
                strength: float, guidance: float, steps: int, generator: Optional[torch.Generator] = None) -> Image.Image:
        
        with torch.autocast(self.device) if self.device == "cuda" else torch.no_grad():
            result_image = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image_ext,
                mask_image=mask_ext,
                strength=strength,
                guidance_scale=guidance,
                num_inference_steps=steps,
                generator=generator,
                height=image_ext.size[1],
                width=image_ext.size[0]
            ).images[0]
            
        return result_image


# ---------------------------
# Main Processing Logic
# ---------------------------
def process_single_image(
    inpainter: PanoramaInpainter,
    img_path: str,
    masks_root: str,
    output_path: str,
    args: argparse.Namespace
):
    """Handles the full inpainting process for a single image."""
    out_dir = os.path.dirname(output_path)
    os.makedirs(out_dir, exist_ok=True)

    img = Image.open(img_path).convert("RGB")
    W, H = img.size

    # --- 1. Resolve mask directory ---
    img_basename = Path(img_path).stem
    per_image_mask_dir = os.path.join(masks_root, img_basename)
    if os.path.isdir(per_image_mask_dir):
        mask_dir = per_image_mask_dir
    else:
        mask_dir = masks_root

    # --- 2. Load and check masks ---
    binary_mask = load_union_binary(mask_dir, H, W)
    if binary_mask.max() == 0:
        print(f"[WARN] No valid masks found for '{img_path}' in '{mask_dir}'. Copying original image.")
        shutil.copy(img_path, output_path)
        return {"status": "empty_mask", "path": output_path}

    # --- 3. Compute parameters (auto or manual) ---
    auto_params = compute_auto_params(W, H, int(np.sum(binary_mask > 0)))
    wrap_pad = args.wrap_pad if isinstance(args.wrap_pad, int) else auto_params["wrap_pad"]
    dilate = args.dilate if isinstance(args.dilate, int) else auto_params["dilate"]
    feather = args.feather if isinstance(args.feather, int) else auto_params["feather"]

    # --- 4. Prepare image and mask ---
    feathered_mask = build_edge_feather_mask(binary_mask, dilate_px=dilate, feather_px=feather)
    erased_src = erase_masked_region(img, feathered_mask, mode=args.erase)
    extended_src, extended_mask = wrap_horizontally(erased_src, feathered_mask, pad=wrap_pad)

    # --- 5. Save intermediate steps if requested ---
    if args.save_intermediate:
        inter_dir = os.path.join(out_dir, "intermediate", img_basename)
        os.makedirs(inter_dir, exist_ok=True)
        feathered_mask.save(os.path.join(inter_dir, "step1_feathered_mask.png"))
        erased_src.save(os.path.join(inter_dir, "step2_erased_source.png"))
        extended_src.save(os.path.join(inter_dir, "step3_wrapped_source.png"))
        extended_mask.save(os.path.join(inter_dir, "step4_wrapped_mask.png"))

    # --- 6. Run Inpainting ---
    print(f"[INFO] Inpainting '{img_path}' with params: pad={wrap_pad}, dilate={dilate}, feather={feather}...")
    generator = torch.Generator(device=inpainter.device).manual_seed(args.seed) if args.seed != 0 else None
    
    result_extended = inpainter.inpaint(
        prompt=args.prompt, negative_prompt=args.neg_prompt,
        image_ext=extended_src, mask_ext=extended_mask,
        strength=args.strength, guidance=args.guidance, steps=args.steps, generator=generator
    )

    # --- 7. Finalize and Save ---
    final_result = crop_center(result_extended, W)
    final_result.save(output_path)
    print(f"[SUCCESS] Saved inpainted image to '{output_path}'")
    
    return {"status": "success", "path": output_path, "params": {"pad": wrap_pad, "dilate": dilate, "feather": feather}}


# ---------------------------
# CLI and Batch Orchestration
# ---------------------------
def gather_input_images(input_file: Optional[str], input_dir: Optional[str]) -> List[str]:
    """Collects all image paths from file or directory inputs."""
    imgs = []
    if input_file:
        imgs.append(input_file)
    if input_dir:
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.tif", "*.tiff"):
            imgs.extend(sorted([str(p) for p in Path(input_dir).glob(ext)]))
    return sorted(list(set(imgs)))


def main():
    """Main entrypoint for the script."""
    parser = argparse.ArgumentParser(description="Panorama inpainting with auto-parameters and batch processing.")
    
    # Input/Output arguments
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="Path to a single input panorama (equirectangular).")
    group.add_argument("--input_dir", help="Path to a folder of panoramas to process in batch.")
    parser.add_argument("--mask_dir", required=True, help="Root folder with asset masks (white=remove). For batch mode, place per-image masks under mask_dir/<image_basename>/.")
    parser.add_argument("--output", required=True, help="Path to the output file (for single image) or output directory (for batch mode).")

    # Inpainting model and prompts
    parser.add_argument("--model_id", default="diffusers/stable-diffusion-xl-1.0-inpainting-0.1", help="Hugging Face model ID for the inpainting pipeline.")
    parser.add_argument("--prompt", default="clean empty interior background, seamless walls and floor, photorealistic, matching lighting, no new objects", help="Positive prompt for inpainting.")
    parser.add_argument("--neg_prompt", default="sofa, couch, armchair, chair, bench, text, watermark, logo, artifacts, distortion, blurry, people, signature", help="Negative prompt for inpainting.")
    
    # Core inpainting parameters
    parser.add_argument("--strength", type=float, default=0.94, help="Inpainting strength (0.0 to 1.0).")
    parser.add_argument("--guidance", type=float, default=5.0, help="Guidance scale (CFG).")
    parser.add_argument("--steps", type=int, default=40, help="Number of inference steps.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility (0 for random).")

    # Mask and wrapping parameters (can be 'auto')
    parser.add_argument("--wrap_pad", default="auto", help="Horizontal padding in pixels for panorama wrapping, or 'auto'.")
    parser.add_argument("--dilate", default="auto", help="Mask dilation in pixels, or 'auto'.")
    parser.add_argument("--feather", default=0, help="Mask feathering (blur) in pixels, or 'auto'.")
    
    # Utility arguments
    parser.add_argument("--erase", choices=["none", "gray", "black"], default="gray", help="How to pre-erase the masked region before inpainting.")
    parser.add_argument("--save_intermediate", action="store_true", help="Save intermediate images (e.g., union mask, wrapped input) for debugging.")
    parser.add_argument("--device", default=None, help="Force device ('cuda' or 'cpu'). Defaults to auto-detection.")
    parser.add_argument("--batch_overwrite", action="store_true", help="In batch mode, overwrite output files if they already exist.")

    args = parser.parse_args()

    # --- 1. Resolve input and output paths ---
    input_image_paths = gather_input_images(args.image, args.input_dir)
    if not input_image_paths:
        print("[ERROR] No input images found. Exiting.")
        sys.exit(1)

    is_batch_mode = args.input_dir is not None
    if is_batch_mode:
        os.makedirs(args.output, exist_ok=True)

    # --- 2. Parse 'auto' arguments ---
    # These will be `None` if 'auto', or an `int` if a number was provided.
    args.wrap_pad = parse_auto_int(args.wrap_pad, None)
    args.dilate = parse_auto_int(args.dilate, None)
    args.feather = parse_auto_int(args.feather, None)

    # --- 3. Load pipeline once ---
    try:
        dtype = torch.float16 if (args.device is None and torch.cuda.is_available()) or (args.device == "cuda") else torch.float32
        inpainter = PanoramaInpainter(model_id=args.model_id, device=args.device, dtype=dtype)
    except Exception as e:
        print(f"[ERROR] Failed to load the inpainting pipeline: {e}")
        sys.exit(1)

    # --- 4. Process images ---
    results_summary = []
    for img_path in input_image_paths:
        try:
            if is_batch_mode:
                output_path = os.path.join(args.output, Path(img_path).stem + "_inpainted.png")
                if os.path.exists(output_path) and not args.batch_overwrite:
                    print(f"[SKIP] Output '{output_path}' exists. Use --batch_overwrite to force.")
                    results_summary.append({"image": img_path, "status": "skipped"})
                    continue
            else:
                output_path = args.output

            result = process_single_image(
                inpainter=inpainter,
                img_path=img_path,
                masks_root=args.mask_dir,
                output_path=output_path,
                args=args
            )
            result["image"] = img_path
            results_summary.append(result)

        except KeyboardInterrupt:
            print("\n[INTERRUPT] User interrupted the process.")
            break
        except Exception as e:
            print(f"[ERROR] Failed to process '{img_path}': {e}")
            import traceback
            traceback.print_exc()
            results_summary.append({"image": img_path, "status": "error", "error": str(e)})
            continue

    # --- 5. Print summary ---
    print("\n[DONE] Processing Summary:")
    for r in results_summary:
        print(f"  - Image: {r.get('image')}, Status: {r.get('status')}")
    
    return 0


if __name__ == "__main__":
    main()

'''
#!/usr/bin/env python3
"""
Panorama background inpainting (equirect-aware) using Diffusers.

Why this version:
- Keeps the original panorama resolution and actually runs the model on the
  horizontally-extended canvas (W + 2*pad) to hide the 0/360 seam.
- Builds a UNION mask where the INTERIOR stays 255 (pure white) and only the
  edge is feathered. This prevents "gray blobs" caused by soft masks mixing the
  pre-erased input with the generated content.
- Optional pre-erase of the masked region (gray/black) to remove latent hints.
- Enables VAE tiling and attention slicing for memory efficiency.

Mask convention: white (255) = to inpaint; black (0) = keep.
"""

import os, argparse, cv2, numpy as np
from PIL import Image
import torch
from diffusers import AutoPipelineForInpainting


# ---------------------------
# Helpers
# ---------------------------
def load_union_binary(mask_dir: str, H: int, W: int) -> np.ndarray:
    """Union all masks into a binary 0/255 array at (H, W)."""
    union = np.zeros((H, W), dtype=np.uint8)
    for name in os.listdir(mask_dir):
        if not name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        m = cv2.imread(os.path.join(mask_dir, name), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        if m.shape[:2] != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        union = np.maximum(union, (m > 127).astype(np.uint8) * 255)
    return union


def build_edge_feather_mask(binary_255: np.ndarray, dilate_px: int, feather_px: int) -> Image.Image:
    """
    Create a mask that is PURE WHITE inside the dilated area, and only the outer
    edge is feathered. This avoids soft interiors that cause gray blobs.
    """
    if binary_255.max() == 0:
        return Image.fromarray(binary_255, mode="L")

    k = max(1, int(dilate_px))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    dilated = cv2.dilate(binary_255, kernel, iterations=1)   # interior = 255

    if feather_px <= 0:
        return Image.fromarray(dilated, mode="L")

    # Edge ring = dilated - eroded(dilated)
    kf = max(1, int(feather_px))
    kernel_f = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * kf + 1, 2 * kf + 1))
    eroded = cv2.erode(dilated, kernel_f, iterations=1)
    ring = cv2.subtract(dilated, eroded)                     # ring ∈ [0,255]

    # Blur the ring only, then add back the solid core (eroded area = 255)
    ring_blur = cv2.GaussianBlur(ring, (0, 0), sigmaX=float(kf), sigmaY=float(kf))
    # Normalize ring_blur to 0..255
    ring_blur = np.clip(ring_blur, 0, 255).astype(np.uint8)

    # Core should remain 255, ring is soft, outside is 0
    mask = np.where(eroded > 0, 255, ring_blur).astype(np.uint8)
    return Image.fromarray(mask, mode="L")


def erase_masked_region(img: Image.Image, mask_L: Image.Image, mode: str = "gray") -> Image.Image:
    """Overwrite masked region in the input image to remove hints (optional)."""
    if mode not in ("none", "gray", "black"):
        mode = "gray"
    if mode == "none":
        return img
    arr = np.array(img).copy()
    m = np.array(mask_L) > 0
    if m.sum() == 0:
        return img
    arr[m] = 127 if mode == "gray" else 0
    return Image.fromarray(arr, mode="RGB")


def wrap_horizontally(img: Image.Image, mask_L: Image.Image, pad: int) -> tuple[Image.Image, Image.Image]:
    """Concatenate right-pad + image + left-pad for both RGB image and L mask."""
    W, H = img.size
    pad = max(1, min(pad, W // 2))
    extW = W + pad * 2

    ext = Image.new(img.mode, (extW, H))
    ext.paste(img.crop((W - pad, 0, W, H)), (0, 0))
    ext.paste(img, (pad, 0))
    ext.paste(img.crop((0, 0, pad, H)), (pad + W, 0))

    mext = Image.new("L", (extW, H))
    mext.paste(mask_L.crop((W - pad, 0, W, H)), (0, 0))
    mext.paste(mask_L, (pad, 0))
    mext.paste(mask_L.crop((0, 0, pad, H)), (pad + W, 0))
    return ext, mext


def crop_center(img: Image.Image, W: int) -> Image.Image:
    """Crop the center W region from a horizontally wrapped image."""
    pad = (img.size[0] - W) // 2
    return img.crop((pad, 0, pad + W, img.size[1]))


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser(description="Panorama background inpainting with edge-feather masks and seam wrap.")
    ap.add_argument("--image", required=True, help="Input panorama (equirect).")
    ap.add_argument("--mask_dir", required=True, help="Folder with asset masks (white=remove).")
    ap.add_argument("--output", required=True, help="Path to save inpainted panorama.")
    ap.add_argument("--prompt", default="clean empty interior background, seamless walls and floor, photorealistic, matching lighting, no new objects")
    ap.add_argument("--neg_prompt", default="sofa, couch, armchair, chair, bench, text, watermark, logo, artifacts, distortion, blurry")
    ap.add_argument("--model_id", default="diffusers/stable-diffusion-xl-1.0-inpainting-0.1")
    ap.add_argument("--strength", type=float, default=0.94, help="0..1; use 1.0 to completely overwrite masked regions.")
    ap.add_argument("--guidance", type=float, default=5.0)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--wrap_pad", type=int, default=256, help="Horizontal wrap pad in pixels.")
    ap.add_argument("--dilate", type=int, default=18, help="Mask dilation in pixels.")
    ap.add_argument("--feather", type=int, default=8, help="Edge feather width (px). Interior stays 255.")
    ap.add_argument("--erase", choices=["none", "gray", "black"], default="gray", help="Pre-erase masked region.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Load panorama and sizes
    src = Image.open(args.image).convert("RGB")
    W, H = src.size

    # Build mask: binary union → dilate → edge-only feather (interior=255)
    binary = load_union_binary(args.mask_dir, H, W)
    if binary.max() == 0:
        print("⚠️ Empty mask: nothing to inpaint. Copying input.")
        src.save(args.output); return
    union = build_edge_feather_mask(binary, dilate_px=args.dilate, feather_px=args.feather)

    # Optional pre-erase to remove hints
    src_erased = erase_masked_region(src, union, mode=args.erase)

    # Horizontal wrap (actually run the model at extended width)
    src_ext, mask_ext = wrap_horizontally(src_erased, union, pad=args.wrap_pad)
    extW, extH = src_ext.size

    # Pipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = AutoPipelineForInpainting.from_pretrained(args.model_id, torch_dtype=dtype).to(device)

    if hasattr(pipe, "enable_vae_tiling"): pipe.enable_vae_tiling()
    if hasattr(pipe, "enable_attention_slicing"): pipe.enable_attention_slicing("max")

    generator = torch.Generator(device=device).manual_seed(args.seed) if args.seed != 0 else None

    # IMPORTANT: run at the extended resolution (extW x extH), not W x H.
    with (torch.autocast(device) if device == "cuda" else torch.no_grad()):
        result = pipe(
            prompt=args.prompt,
            negative_prompt=args.neg_prompt,
            image=src_ext,
            mask_image=mask_ext,
            strength=args.strength,
            guidance_scale=args.guidance,
            num_inference_steps=args.steps,
            generator=generator,
            height=extH, width=extW
        ).images[0]

    # Crop center back to original size
    result = crop_center(result, W)
    result.save(args.output)
    print(f"✅ Inpainted panorama saved: {args.output} (size={W}x{H})")

if __name__ == "__main__":
    main()
'''