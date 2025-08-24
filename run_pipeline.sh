#!/bin/bash

# Stop script on any error
set -e

# --- CONFIGURATION ---
# ★★★ IMPORTANT: Set your OpenAI API key here ★★★
export OPENAI_API_KEY="your_openai_api_key_here"

# Define the scene name and prompt
SCENE_NAME="indoor_livingroom"
PANO_PROMPT="A spacious modern living room with a gray sofa and a potted plant."

# --- PATH DEFINITIONS (relative to Text2VR root) ---
# Host machine paths
HOST_PANO_DATA_DIR="./dreamscene360_service/data/${SCENE_NAME}"
HOST_PANO_IMAGE_PATH="${HOST_PANO_DATA_DIR}/diffusion_img.png"

# Container-internal paths
CONTAINER_DS360_DATA_DIR="/workspace/DreamScene360/data/${SCENE_NAME}"
CONTAINER_DS360_OUTPUT_DIR="/workspace/DreamScene360/output/${SCENE_NAME}_pano_gen"
CONTAINER_SEG_PANO_PATH="/app/data/${SCENE_NAME}/diffusion_img.png"
CONTAINER_SEG_OUTPUT_DIR="/app/output/${SCENE_NAME}_masks"
CONTAINER_SAM_CHECKPOINT="/app/checkpoints/sam_vit_h_4b8939.pth"

# --- PIPELINE EXECUTION ---
echo "======================================================"
echo "    STARTING END-TO-END TEXT2VR PIPELINE"
echo "======================================================"
echo "Scene: ${SCENE_NAME}"
echo ""

# Step 1: Build all services defined in docker-compose.yml
echo "--> Building all services..."
docker-compose build

# Step 2: Generate the panorama using the 'dreamscene360' service
echo -e "\n--> STAGE 1: Generating Panorama via [dreamscene360] service..."
# Create necessary directories and prompt file on the host
mkdir -p ${HOST_PANO_DATA_DIR}
echo "${PANO_PROMPT}" > "${HOST_PANO_DATA_DIR}/${SCENE_NAME}_PROMPT.txt"

# Run the training script for 1 iteration to generate the panorama
docker-compose run --rm dreamscene360 python train.py \
    -s ${CONTAINER_DS360_DATA_DIR} \
    -m ${CONTAINER_DS360_OUTPUT_DIR} \
    --self_refinement \
    --api_key "${OPENAI_API_KEY}" \
    --iterations 1

# Verify that the panorama was created
if [ ! -f "${HOST_PANO_IMAGE_PATH}" ]; then
    echo "❌ ERROR: Panorama generation failed. Exiting."
    exit 1
fi
echo "✅ STAGE 1 Complete: Panorama generated."

# Step 3: Segment the panorama using the 'segmentation' service
echo -e "\n--> STAGE 2: Segmenting Panorama via [segmentation] service..."
docker-compose run --rm segmentation python segment_panorama.py \
    --panorama_path ${CONTAINER_SEG_PANO_PATH} \
    --output_dir ${CONTAINER_SEG_OUTPUT_DIR} \
    --sam_checkpoint ${CONTAINER_SAM_CHECKPOINT}

echo "✅ STAGE 2 Complete: Masks generated in ./output/${SCENE_NAME}_masks"

# Step 4: Placeholder for the final unified training
echo -e "\n--> STAGE 3: Unified Training (Placeholder)..."
echo "    The next step would be to run a new hybrid training script"
echo "    in the 'dreamscene360' service, using the masks from STAGE 2."

echo -e "\n======================================================"
echo "      PIPELINE FINISHED SUCCESSFULLY!  "
echo "======================================================"
