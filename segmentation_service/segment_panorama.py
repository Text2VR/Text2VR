#!/usr/bin/env python3
import torch
import numpy as np
from PIL import Image
import cv2
import os
import json
import requests
import base64
import argparse
from typing import List, Dict

# --- Dependency Availability Check ---
try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    print("⚠️ Warning: segment_anything library not found.")
    SAM_AVAILABLE = False
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("⚠️ Warning: transformers library not found.")
    TRANSFORMERS_AVAILABLE = False

# --- OpenAI GPT-4V Integration ---
def get_asset_prompts_from_gpt(image_path: str, api_key: str) -> List[str]:
    """
    Analyzes a panorama with GPT-4V and generates a list of asset prompts.
    """
    print("🧠 Contacting GPT-4V to analyze the panorama and identify assets...")
    if not api_key or api_key == "your_openai_api_key_here":
        print("❌ ERROR: OpenAI API key is not provided or is a placeholder.")
        return []

    def encode_image_to_base64(path):
        with open(path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    base64_image = encode_image_to_base64(image_path)
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this panoramic image of a scene. List the primary, distinct, segmentable objects that could be interactive assets. Ignore general background elements like walls, floors, ceilings, sky, and windows. Return ONLY a comma-separated list of simple, lowercase, singular nouns. For example: sofa, chair, table, plant, lamp, car, tree"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                ]
            }
        ],
        "max_tokens": 100
    }
    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        content = response.json()['choices'][0]['message']['content']
        prompts = [p.strip() for p in content.split(',') if p.strip()]
        print(f"✅ GPT-4V identified the following assets: {prompts}")
        return prompts
    except Exception as e:
        print(f"❌ ERROR: Failed to get prompts from GPT-4V: {e}")
        return []

# --- Main GroundedSAM Class ---
class HuggingFaceGroundedSAM:
    def __init__(self, sam_checkpoint: str, output_dir: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sam_predictor = None
        self.grounding_detector = None
        self.output_dir = output_dir

        os.makedirs(os.path.join(self.output_dir, "masks"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "visualizations"), exist_ok=True)

        if SAM_AVAILABLE and os.path.exists(sam_checkpoint):
            self._init_sam(sam_checkpoint)
        if TRANSFORMERS_AVAILABLE:
            self._init_grounding_detector()

    def _init_sam(self, checkpoint_path: str):
        try:
            print("🔧 Initializing SAM model...")
            sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
            sam.to(self.device)
            self.sam_predictor = SamPredictor(sam)
            print(f"✅ SAM initialized successfully (Device: {self.device})")
        except Exception as e:
            print(f"❌ Failed to initialize SAM: {e}")

    def _init_grounding_detector(self):
        try:
            print("🔧 Initializing Hugging Face GroundingDINO model...")
            self.grounding_detector = pipeline(
                "zero-shot-object-detection",
                model="IDEA-Research/grounding-dino-base",
                device=0 if self.device == "cuda" else -1
            )
            print("✅ GroundingDINO initialized successfully.")
        except Exception as e:
            print(f"❌ Failed to initialize GroundingDINO: {e}")
            
    def detect_with_text(self, image: Image.Image, text_prompt: str, threshold: float) -> List[Dict]:
        if not self.grounding_detector:
            print("❌ Grounding detector not initialized.")
            return []
        
        print(f"🧠 Detecting objects for prompt: '{text_prompt}'")
        predictions = self.grounding_detector(image, candidate_labels=[text_prompt])
        filtered = [p for p in predictions if p.get('score', 0) >= threshold]
        print(f"  > Found {len(filtered)} instances with confidence > {threshold}.")
        return filtered

    def segment_with_sam(self, image_rgb: np.ndarray, boxes: List[np.ndarray]) -> List[np.ndarray]:
        if not self.sam_predictor:
            print("❌ SAM not initialized.")
            return []
        
        self.sam_predictor.set_image(image_rgb)
        all_masks = []
        for box in boxes:
            masks, scores, _ = self.sam_predictor.predict(box=box, multimask_output=True)
            best_mask = masks[np.argmax(scores)]
            all_masks.append(best_mask)
        return all_masks

    def save_and_visualize(self, image_path: str, all_results: Dict):
        image_bgr = cv2.imread(image_path)
        vis_image = image_bgr.copy()
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        
        for i, (prompt, results) in enumerate(all_results.items()):
            if not results: continue
            
            color = colors[i % len(colors)]
            combined_mask_for_prompt = np.zeros_like(results[0]['mask'], dtype=np.uint8)
            
            for result in results:
                mask = result['mask'].astype(np.uint8)
                combined_mask_for_prompt = np.maximum(combined_mask_for_prompt, mask)
                
                # Visualization logic
                overlay = np.zeros_like(vis_image)
                overlay[mask > 0] = color
                vis_image = cv2.addWeighted(vis_image, 1.0, overlay, 0.5, 0)
            
            mask_filename = f"{prompt.replace(' ', '_')}.png"
            cv2.imwrite(os.path.join(self.output_dir, "masks", mask_filename), combined_mask_for_prompt * 255)
        
        vis_filename = os.path.basename(image_path).replace('.png', '_visualization.png')
        vis_path = os.path.join(self.output_dir, "visualizations", vis_filename)
        cv2.imwrite(vis_path, vis_image)
        print(f"✅ Visualization saved to: {vis_path}")


def main(args):
    print("🚀 Starting Panorama Segmentation Pipeline")
    print("=" * 60)

    prompts_to_run = get_asset_prompts_from_gpt(args.panorama_path, args.openai_api_key)
    if not prompts_to_run:
        print("❌ No prompts generated. Using default fallback prompts.")
        prompts_to_run = ["sofa", "chair", "table", "plant", "lamp", "car", "tree", "bench"]

    grounded_sam = HuggingFaceGroundedSAM(args.sam_checkpoint, args.output_dir)

    try:
        panorama_pil = Image.open(args.panorama_path).convert("RGB")
    except FileNotFoundError:
        print(f"❌ Panorama image not found at: {args.panorama_path}")
        return

    all_results = {}
    for prompt in prompts_to_run:
        detections = grounded_sam.detect_with_text(panorama_pil, prompt, threshold=0.3)
        if not detections:
            all_results[prompt] = []
            continue
        
        boxes = [np.array([d['box']['xmin'], d['box']['ymin'], d['box']['xmax'], d['box']['ymax']]) for d in detections]
        masks = grounded_sam.segment_with_sam(np.array(panorama_pil), boxes)
        
        results_for_prompt = [{"mask": mask} for mask in masks]
        all_results[prompt] = results_for_prompt

    grounded_sam.save_and_visualize(args.panorama_path, all_results)
    
    print("=" * 60)
    print("🎉 Pipeline Finished Successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment panoramas using GPT-4V and GroundedSAM.")
    parser.add_argument("--sam_checkpoint", type=str, required=True, help="Path to the SAM checkpoint file.")
    parser.add_argument("--panorama_path", type=str, required=True, help="Path to the panorama image to process.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the results.")
    parser.add_argument("--openai_api_key", type=str, default=os.getenv("OPENAI_API_KEY"), help="OpenAI API key. Can also be set as an environment variable.")
    
    args = parser.parse_args()
    main(args)
