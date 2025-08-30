#!/bin/bash

# Stop script on any error
set -e

# --- CONFIGURATION ---
# The OpenAI API key is now automatically loaded from the .env file by Docker Compose.
# Make sure your .env file exists in the root directory and contains:
# OPENAI_API_KEY="sk-..."

# set Text2VR/.env !!
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo "❌ ERROR: .env file not found! Please create one with your OPENAI_API_KEY."
    exit 1
fi
# ------------------------------

# Define the scene name and prompt
SCENE_NAME="indoor_livingroom_compose"
PROMPT="A 360 equirectangular photo of a minimalist and spacious living room. In the center, there is a single modern leather sofa. The room has plain white walls, a smooth light gray concrete floor, and no other furniture or decorations. The scene is brightly lit by soft, natural light from a large window, with no harsh shadows. photorealistic, 8k, sharp focus."

# --- PATH DEFINITIONS (relative to Text2VR root) ---
# Host paths
HOST_SCENE_DIR="./DREAMSCENE360/data/${SCENE_NAME}"
HOST_PANO="${HOST_SCENE_DIR}/panorama.png"
HOST_INPAINT="${HOST_SCENE_DIR}/inpainted_panorama.png"
HOST_MASK_DIR="./output/${SCENE_NAME}_masks/masks"   # <- produced by asset_seg

# Container paths
DS360_SCENE_DIR="/workspace/DREAMSCENE360/data/${SCENE_NAME}"
SEG_PANO="/app/data/${SCENE_NAME}/panorama.png"
SEG_OUT="/app/output/${SCENE_NAME}_masks"
SAM_CKPT="/app/checkpoints/sam_vit_h_4b8939.pth"

# BG_INPAINT uses /workspace because of its compose mount
INPAINT_IN="/workspace/data/${SCENE_NAME}/panorama.png"
INPAINT_MASK_DIR="/workspace/output/${SCENE_NAME}_masks/masks"
INPAINT_OUT="/workspace/data/${SCENE_NAME}/inpainted_panorama.png"

# --- PIPELINE EXECUTION ---
echo "======================================================"
echo "    STARTING END-TO-END TEXT2VR PIPELINE"
echo "======================================================"
echo "Scene: ${SCENE_NAME}"
echo "NOTE: OpenAI API Key will be loaded from your .env file."
echo ""

# --------------------------------------------------------------
### ONLY needs to be done once, or when Dockerfile changes. ### 
echo "================= BUILD ================="
echo "--> Building all services..."
# docker-compose build                                                ### (optional)
# --------------------------------------------------------------

echo "================= STAGE 1: PANORAMA GEN ================="
mkdir -p "${HOST_SCENE_DIR}"
echo "${PROMPT}" > "${HOST_SCENE_DIR}/prompt.txt"

docker-compose run --rm dreamscene360 \
  micromamba run -n dev \
  python pano_generator.py \
    --text "$(cat ${HOST_SCENE_DIR}/prompt.txt)" \
    --output_dir "${DS360_SCENE_DIR}" \
    --api_key "${OPENAI_API_KEY}" \
    --self_refinement \
    --num_prompt 2 \ 
    --max_rounds 2

test -f "${HOST_PANO}" || { echo "❌ Panorama generation failed"; exit 1; }

echo "================= STAGE 2: ASSET SEG ================="
# NOTE: We pass the key explicitly to guarantee GPT usage inside the container.
docker-compose run --rm -e OPENAI_API_KEY \
  asset_seg \
  python segment_panorama.py \
    --panorama_path "${SEG_PANO}" \
    --output_dir "${SEG_OUT}" \
    --sam_checkpoint "${SAM_CKPT}" \
    --openai_api_key "${OPENAI_API_KEY}" \
    --box_threshold 0.30 --text_threshold 0.25 \
    --min_area_ratio 0.01 --max_area_ratio 0.40 \
    --exclusion_use_mask true \
    --exclusion_mask_dilate_px 12 \
    --exclusion_overlap_drop 0.50
    # If your segmenter supports these, keep them; otherwise comment:
    # --max_prompts 5 \
    # --anchor_enable true

# simple heads-up if no masks were produced (doesn't fail the pipeline)
if [ ! -d "${HOST_MASK_DIR}" ] || ! ls "${HOST_MASK_DIR}"/*.png >/dev/null 2>&1 ; then
  echo "⚠️  No masks found at ${HOST_MASK_DIR} (segmentation produced none). Inpainting will effectively be a copy."
fi

echo "================= STAGE 3: BG INPAINT ================="
docker-compose run --rm bg_inpaint \
  python /workspace/inpaint_panorama.py \
    --image "${INPAINT_IN}" \
    --mask_dir "${INPAINT_MASK_DIR}" \
    --output "${INPAINT_OUT}" \
    --prompt "clean empty interior background, seamless walls and floor, photorealistic, matching lighting, no new objects" \
    --neg_prompt "sofa, couch, armchair, chair, bench, flat gray fill, plain color patch, smudged texture, repetitive tiling, text, watermark, logo, artifacts" \
    --model_id "diffusers/stable-diffusion-xl-1.0-inpainting-0.1" \
    --strength 0.88 --guidance 5.0 --steps 40 \
    --dilate 18 --feather 8 --erase none --wrap_pad 256

test -f "${HOST_INPAINT}" || { echo "❌ Inpainting failed"; exit 1; }

echo "================= STAGE 4: DREAMSCENE360 TRAIN ================="
docker-compose run --rm dreamscene360 \
  micromamba run -n dev \
  python train.py \
    -s "${DS360_SCENE_DIR}" \
    -m "/workspace/DREAMSCENE360/output/${SCENE_NAME}_ply" \
    --pano_path "${DS360_SCENE_DIR}/inpainted_panorama.png" \
    --debug

echo -e "\n======================================================"
echo "      PIPELINE FINISHED SUCCESSFULLY!  "
echo "======================================================"
