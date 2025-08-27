#!/usr/bin/env python3
"""
Panorama background inpainting with Stable Diffusion Inpainting (Diffusers).

- Reads the panorama (equirectangular) and unions all masks from ASSET_SEG.
- White pixels in mask = region to inpaint (remove assets).
- Writes the inpainted panorama to --output.

Note: SD-inpainting may internally resize; for very large panoramas consider tiling.
"""

import os
import argparse
import cv2
import numpy as np
from PIL import Image

import torch
from diffusers import StableDiffusionInpaintPipeline

def load_and_union_masks(mask_dir: str, H: int, W: int) -> Image.Image:
    """Union all binary masks in a folder; return white=to-inpaint (uint8 0/255)."""
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
    return Image.fromarray(union, mode="L")

def main():
    parser = argparse.ArgumentParser(description="Panorama background inpainting using Diffusers.")
    parser.add_argument("--image", required=True, help="Path to input panorama (equirect).")
    parser.add_argument("--mask_dir", required=True, help="Directory of asset masks (white regions will be removed).")
    parser.add_argument("--output", required=True, help="Path to save inpainted panorama.")
    parser.add_argument("--prompt", default="clean, consistent background, no objects, seamless room interior",
                        help="Text prompt to guide background inpainting.")
    parser.add_argument("--model_id", default="runwayml/stable-diffusion-inpainting", help="HF model id for inpainting.")
    parser.add_argument("--strength", type=float, default=0.75, help="Denoising strength (0..1).")
    parser.add_argument("--guidance", type=float, default=7.5, help="CFG guidance scale.")
    parser.add_argument("--steps", type=int, default=40, help="Number of diffusion steps.")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Load panorama
    src = Image.open(args.image).convert("RGB")
    W, H = src.size

    # Union mask: white = to-inpaint
    union_mask = load_and_union_masks(args.mask_dir, H, W)

    # Early exit if empty mask
    if np.array(union_mask).max() < 1:
        print("⚠️ Empty mask: nothing to inpaint. Copying input to output.")
        src.save(args.output)
        return

    # Create pipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionInpaintPipeline.from_pretrained(args.model_id, torch_dtype=dtype).to(device)

    # Run inpainting
    with (torch.autocast(device) if device == "cuda" else torch.inference_mode()):
        out = pipe(
            prompt=args.prompt,
            image=src,
            mask_image=union_mask,
            strength=args.strength,
            guidance_scale=args.guidance,
            num_inference_steps=args.steps,
        ).images[0]

    out.save(args.output)
    print(f"✅ Inpainted panorama saved: {args.output}")

if __name__ == "__main__":
    main()
