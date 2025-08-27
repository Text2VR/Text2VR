#!/usr/bin/env python3
"""
DreamScene360 simple panorama generator.
Text in → Panorama out.
"""

import os
import sys
import argparse
import shutil

# Make internal modules importable
sys.path.append('stitch_diffusion/kohya_trainer')
sys.path.append('stitch_diffusion/kohya_trainer/library')  # add library module path

# Try to import StitchDiffusion pipeline
try:
    from stitch_diffusion.kohya_trainer.StitchDiffusionPipeline import StitchDiffusion, my_args
    STITCHDIFFUSION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ StitchDiffusion import failed: {e}")
    STITCHDIFFUSION_AVAILABLE = False

# Self-refinement runner using GPT-4o (optional)
from Text2PanoRunner import Text2PanoRunner


def generate_panorama(text_prompt, output_dir="panorama_output", api_key=None, use_self_refinement=False):
    """Generate a panorama image from a text prompt."""
    print(f"🎨 Generating panorama from: '{text_prompt}'")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save prompt to a text file (used by the self-refinement runner)
    text_file = os.path.join(output_dir, "prompt.txt")
    with open(text_file, 'w', encoding='utf-8') as f:
        f.write(text_prompt)

    # Optional: self-refinement using GPT-4o
    if use_self_refinement and api_key:
        print("📝 Using self-refinement with GPT-4o...")
        runner = Text2PanoRunner(
            api_key=api_key,
            testfile=text_file,
            num_prompt=3,
            max_rounds=3,
            foldername="panorama_scene"
        )
        runner.run_command()

        # Check a few candidate result paths created by the refinement pipeline
        possible_paths = [
            "self_refinement/panorama_scene/iter_best/image.png",
            "candidates/panorama_scene/iter_best/image.png",
            "panorama_scene/iter_best/image.png",
            "iter_best/image.png",
        ]

        pano_path = None
        for path in possible_paths:
            if os.path.exists(path):
                pano_path = path
                print(f"📁 Found result at: {path}")
                break

        # Copy the result into the output directory
        if pano_path and os.path.exists(pano_path):
            final_path = os.path.join(output_dir, "panorama.png")
            try:
                shutil.copy2(pano_path, final_path)
            except Exception as e:
                print(f"❌ Failed to copy result: {e}")
                return None
            return final_path
        else:
            print("❌ Self-refinement output not found, falling back to basic generation...")

    # Fallback: basic StitchDiffusion generation
    if not STITCHDIFFUSION_AVAILABLE:
        print("❌ StitchDiffusion is not available due to import errors.")
        return None

    print("🖼️ Using basic StitchDiffusion...")
    try:
        sd = StitchDiffusion(my_args)
        pano_path = os.path.join(output_dir, "panorama.png")
        sd.inference(text_prompt, savename=pano_path)

        if os.path.exists(pano_path):
            return pano_path
        else:
            print("❌ StitchDiffusion did not produce an output image.")
            return None
    except Exception as e:
        print(f"❌ StitchDiffusion failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='DreamScene360 Simple Panorama Generator')
    parser.add_argument('--text', type=str, required=True, help='Text prompt for panorama generation')
    parser.add_argument('--output_dir', type=str, default='panorama_output', help='Output directory')
    parser.add_argument('--api_key', type=str, help='OpenAI API key (required for self-refinement)')
    parser.add_argument('--self_refinement', action='store_true',
                        help='Use self-refinement for better quality (requires --api_key)')

    args = parser.parse_args()

    # Self-refinement requires an API key
    if args.self_refinement and not args.api_key:
        print("❌ Self-refinement requires --api_key")
        return

    print("🚀 DreamScene360 Simple Panorama Generator")
    print("=" * 50)
    print(f"📝 Text prompt: {args.text}")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"🔧 Self-refinement: {'Yes' if args.self_refinement else 'No'}")
    print("=" * 50)

    # Generate panorama
    pano_path = generate_panorama(
        args.text,
        args.output_dir,
        args.api_key,
        args.self_refinement
    )

    if pano_path and os.path.exists(pano_path):
        print("✅ Panorama generation completed!")
        print(f"🖼️ Result saved to: {pano_path}")

        # Print file info
        try:
            file_size_mb = os.path.getsize(pano_path) / (1024 * 1024)
            print(f"📊 File size: {file_size_mb:.2f} MB")
        except Exception:
            pass

        # Print image resolution
        try:
            from PIL import Image
            with Image.open(pano_path) as img:
                print(f"📐 Resolution: {img.width} x {img.height}")
        except Exception:
            pass
    else:
        print("❌ Panorama generation failed!")


if __name__ == "__main__":
    main()
