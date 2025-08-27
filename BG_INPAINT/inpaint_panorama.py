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
