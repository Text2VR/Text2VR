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
# SCENE_NAME="park" 
# PROMPT="A beautiful park with trees and one benche, one table, and a walking path, photorealistic, 8k, sharp focus"
SCENE_NAME="living_room" 
PROMPT="A modern living room with a sofa, a coffee table, a plant, and a window showing a cityscape, photorealistic, 8k, sharp focus"


# --- PATH DEFINITIONS (relative to Text2VR root) ---
# Host paths
HOST_SCENE_DIR="./DREAMSCENE360/data/${SCENE_NAME}"
HOST_PANO_IMAGE_PATH="${HOST_SCENE_DIR}/panorama.png" # Changed from diffusion_img.png for clarity
HOST_MASK_DIR="./output/${SCENE_NAME}_masks"
HOST_INPAINTED_PANO_PATH="${HOST_SCENE_DIR}/inpainted_panorama.png"

# Define a clean directory for the final training step
HOST_INPAINTED_DIR_FOR_TRAINING="${HOST_SCENE_DIR}/for_training"

# Container paths (do not change)
CONTAINER_DS360_DATA_DIR="/workspace/DREAMSCENE360/data/${SCENE_NAME}"
CONTAINER_DS360_OUTPUT_DIR="/workspace/DREAMSCENE360/output/${SCENE_NAME}_pano_gen"
CONTAINER_SEG_PANO_PATH="/app/data/${SCENE_NAME}/panorama.png"
CONTAINER_SEG_OUTPUT_DIR="/app/output/${SCENE_NAME}_masks"
CONTAINER_SAM_CHECKPOINT="/app/checkpoints/sam_vit_h_4b8939.pth"
CONTAINER_INPAINT_INPUT_PANO="/workspace/data/${SCENE_NAME}/panorama.png"
CONTAINER_INPAINT_MASK_DIR="/workspace/output/${SCENE_NAME}_masks/masks"
CONTAINER_INPAINT_OUTPUT_PANO="/workspace/data/${SCENE_NAME}/inpainted_panorama.png"
CONTAINER_INPAINTED_DIR_FOR_TRAINING="/workspace/DREAMSCENE360/data/${SCENE_NAME}/for_training"


# --- PIPELINE EXECUTION ---
echo "======================================================"
echo "    STARTING END-TO-END TEXT2VR PIPELINE"
echo "======================================================"
echo "Scene: ${SCENE_NAME}"
echo "NOTE: OpenAI API Key will be loaded from your .env file."
echo ""

# --------------------------------------------------------------
### ONLY needs to be done once, or when Dockerfile changes. ### 
echo "================= BUILD (if needed) ================="
echo "--> Building all services..."
# docker-compose build                                                ### (optional)
# --------------------------------------------------------------

echo "================= STAGE 1: PANORAMA GEN (API) ================="
mkdir -p "${HOST_SCENE_DIR}"
echo "${PROMPT}" > "${HOST_SCENE_DIR}/prompt.txt"

echo "================= STAGE 1: PANORAMA GEN ================="
mkdir -p "${HOST_SCENE_DIR}"
echo "${PROMPT}" > "${HOST_SCENE_DIR}/prompt.txt"

docker-compose run --rm dreamscene360 \
  micromamba run -n dev \
  python pano_generator.py \
    --text "$(cat ${HOST_SCENE_DIR}/prompt.txt)" \
    --output_dir "${CONTAINER_DS360_DATA_DIR}" \
    --api_key "${OPENAI_API_KEY}" \
    --self_refinement \
    --num_prompt 1 \
    --max_rounds 2

test -f "${HOST_PANO_IMAGE_PATH}" || { echo "❌ Panorama generation failed"; exit 1; }

# echo "================= STAGE 1: PANORAMA GEN (API) ================="
# mkdir -p "${HOST_SCENE_DIR}"
# echo "${PROMPT}" > "${HOST_SCENE_DIR}/prompt.txt"

# Start panorama API service if not running
# echo "🔄 Ensuring panorama-api service is running..."
# docker compose up -d panorama-api

# Wait for service to be ready
# echo "⏳ Waiting for API service to start..."
# for i in {1..30}; do
#  if curl -s http://localhost:8001/health > /dev/null; then
#     echo "✅ API service is ready"
#     break
#   fi
#   if [ $i -eq 30 ]; then
#     echo "❌ API service failed to start"
#     exit 1
#   fi
#   sleep 2
# done

# Call API to generate panorama
# echo "🎨 Calling panorama generation API..."
# python3 api_client.py "${PROMPT}" "${SCENE_NAME}" true 1 2
#test -f "${HOST_PANO_IMAGE_PATH}" || { echo "❌ Panorama generation failed"; exit 1; }

echo "================= STAGE 2: ASSET SEG ================="
# NOTE: We pass the key explicitly to guarantee GPT usage inside the container.
docker-compose run --rm -e OPENAI_API_KEY \
  asset_seg \
  python segment_panorama.py \
    --panorama_path "${CONTAINER_SEG_PANO_PATH}" \
    --output_dir "${CONTAINER_SEG_OUTPUT_DIR}" \
    --sam_checkpoint "${CONTAINER_SAM_CHECKPOINT}" \
    --openai_api_key "${OPENAI_API_KEY}" \
    # --box_threshold 0.20 --text_threshold 0.15 \
    # --min_area_ratio 0.005 --max_area_ratio 0.40 \
    # --exclusion_use_mask true \
    # --exclusion_mask_dilate_px 4 \
    # --exclusion_overlap_drop 0.0
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
    --image "${CONTAINER_INPAINT_INPUT_PANO}" \
    --mask_dir "${CONTAINER_INPAINT_MASK_DIR}" \
    --output "${CONTAINER_INPAINT_OUTPUT_PANO}" \
    --prompt "a clean empty room background, photorealistic, seamless texture, 8k, sharp focus" \
    --neg_prompt "objects, furniture, sofa, , picture, chair, plant, lamp, table, blurry, hazy, watermark, text, signature, pillow" \
    --model_id "diffusers/stable-diffusion-xl-1.0-inpainting-0.1" \
    --strength 0.95 \
    --guidance 7.5 \
    --steps 40 \
    --seed 42 \
    --wrap_pad "auto" \
    --dilate "auto" \
    --feather 1 \
    --save_intermediate # Optional: for debugging

test -f "${HOST_INPAINTED_PANO_PATH}" || { echo "❌ Inpainting failed"; exit 1; }

echo "================= STAGE 4: Trellis ================="


echo "================= STAGE 5: DREAMSCENE360 TRAIN ================="
# mkdir -p "${HOST_INPAINTED_DIR_FOR_TRAINING}"
# cp "${HOST_INPAINTED_PANO_PATH}" "${HOST_INPAINTED_DIR_FOR_TRAINING}/"
# echo "✅ Copied inpainted panorama to a dedicated training folder."

docker-compose run --rm dreamscene360 \
  micromamba run -n dev \
  python train.py \
    -s "${CONTAINER_DS360_DATA_DIR}" \
    -m "/workspace/DREAMSCENE360/output/${SCENE_NAME}_ply" \
    --pano_path "/workspace/DREAMSCENE360/data/${SCENE_NAME}/inpainted_panorama.png" \
    --no_perturb_loss \
    --iterations 7000 \
    --test_iterations 7000 \
    --save_iterations 5000 7000
#   python train_ds360_only.py \
#     -s "${CONTAINER_INPAINTED_DIR_FOR_TRAINING}" \
#     -m "/workspace/DREAMSCENE360/output/${SCENE_NAME}_ply"

echo "================= STAGE 6: FlashSculptor ================="



echo -e "\n======================================================"
echo "      PIPELINE FINISHED SUCCESSFULLY!  "
echo "======================================================"