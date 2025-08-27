#!/usr/bin/env python3
"""
Panorama background inpainting (Equirectangular-aware) with Diffusers.

- Keeps original panorama resolution (no unintended downscale).
- Uses SDXL inpaint by default for better large-res quality.
- Does mask dilation to avoid halos.
- Wraps horizontally (3*W) to avoid seams at 0/360 boundary, then crops center.

White pixels in unioned mask = region to remove/inpaint.
"""

import os
import argparse
import cv2
import numpy as np
from PIL import Image

import torch
from diffusers import AutoPipelineForInpainting

# ---------------------------
# Helpers
# ---------------------------
def union_masks(mask_dir: str, H: int, W: int, dilate_px: int = 6) -> Image.Image:
    """Load and union all binary masks; return L-mode 0/255. Apply dilation to avoid halos."""
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
    if union.max() > 0 and dilate_px > 0:
        k = max(1, dilate_px)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        union = cv2.dilate(union, kernel, iterations=1)
    return Image.fromarray(union, mode="L")

def wrap_horizontally(img: Image.Image, mask: Image.Image, pad: int) -> tuple[Image.Image, Image.Image]:
    """Create a 3*W wrap by concatenating right-pad + image + left-pad. Works for both RGB and L."""
    W, H = img.size
    pad = max(1, min(pad, W // 2))  # safe clamp
    left = img.crop((0, 0, pad, H))
    right = img.crop((W - pad, 0, W, H))
    ext = Image.new(img.mode, (W + pad * 2, H))
    ext.paste(right, (0, 0))
    ext.paste(img, (pad, 0))
    ext.paste(left, (pad + W, 0))

    mleft = mask.crop((0, 0, pad, H))
    mright = mask.crop((W - pad, 0, W, H))
    mext = Image.new("L", (W + pad * 2, H))
    mext.paste(mright, (0, 0))
    mext.paste(mask, (pad, 0))
    mext.paste(mleft, (pad + W, 0))
    return ext, mext

def crop_center(img: Image.Image, W: int) -> Image.Image:
    """Crop the center W region from a 3*W image."""
    assert img.size[0] >= W
    pad = (img.size[0] - W) // 2
    return img.crop((pad, 0, pad + W, img.size[1]))

# ---------------------------
# Main
# ---------------------------
def main():
    p = argparse.ArgumentParser(description="Panorama background inpainting with wrap-around.")
    p.add_argument("--image", required=True, help="Input panorama (equirectangular).")
    p.add_argument("--mask_dir", required=True, help="Folder with asset masks (white=remove).")
    p.add_argument("--output", required=True, help="Output path for inpainted panorama.")
    p.add_argument("--prompt", default="clean empty interior background, seamless walls and floor, photorealistic, neat, no objects",
                   help="Positive prompt to guide background.")
    p.add_argument("--neg_prompt", default="text, watermark, logo, artifacts, extra objects, distortion, blurry",
                   help="Negative prompt to avoid undesired content.")
    p.add_argument("--model_id", default="diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                   help="Hugging Face model id. Examples: "
                        "'diffusers/stable-diffusion-xl-1.0-inpainting-0.1', "
                        "'stabilityai/stable-diffusion-2-inpainting', "
                        "'runwayml/stable-diffusion-inpainting'")
    p.add_argument("--strength", type=float, default=0.6, help="Denoising strength (0..1).")
    p.add_argument("--guidance", type=float, default=5.5, help="CFG scale.")
    p.add_argument("--steps", type=int, default=35, help="Diffusion steps.")
    p.add_argument("--wrap_pad", type=int, default=256, help="Horizontal wrap pad in pixels (to hide 0/360 seam).")
    p.add_argument("--dilate", type=int, default=6, help="Mask dilation pixels to avoid halos.")
    p.add_argument("--seed", type=int, default=0, help="Random seed (0 = random).")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Load source image
    src = Image.open(args.image).convert("RGB")
    W, H = src.size

    # Build union mask (white=to inpaint)
    union = union_masks(args.mask_dir, H, W, dilate_px=args.dilate)
    if np.array(union).max() < 1:
        print("⚠️ Empty mask: nothing to inpaint. Copying input.")
        src.save(args.output)
        return

    # Create a horizontal wrap (3*W x H) to avoid seam artifacts
    src_ext, mask_ext = wrap_horizontally(src, union, pad=args.wrap_pad)

    # Pipeline (SDXL inpaint by default)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = AutoPipelineForInpainting.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
    ).to(device)

    # Memory-friendly for large images
    if hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing("max")

    generator = None
    if args.seed != 0:
        generator = torch.Generator(device=device).manual_seed(args.seed)

    # Run inpainting on the extended canvas
    with torch.autocast(device) if device == "cuda" else torch.no_grad():
        result = pipe(
            prompt=args.prompt,
            negative_prompt=args.neg_prompt,
            image=src_ext,           # keep original (extended) resolution
            mask_image=mask_ext,
            strength=args.strength,
            guidance_scale=args.guidance,
            num_inference_steps=args.steps,
            generator=generator,
        ).images[0]

    # Crop center back to original W×H
    result = crop_center(result, W)
    assert result.size == (W, H)
    result.save(args.output)
    print(f"✅ Inpainted panorama saved: {args.output}  (size={W}x{H})")

if __name__ == "__main__":
    main()
