# 🌀 Text2VR: End-to-End Panorama to Interactive Scene Pipeline

## 🚀 Overview

This project provides an end-to-end pipeline to generate a fully interactive 3D VR scene from a single text prompt. It leverages a microservice architecture orchestrated by **Docker Compose** to ensure stability, scalability, and maintainability across different AI model dependencies.

The pipeline is composed of two main services, each running in its own isolated Docker container:

1.  **`dreamscene360_service`**: Utilizes the original DreamScene360 with a stable, legacy environment (`diffusers==0.10.2`, etc.) to generate high-quality 360° panoramas from text.
2.  **`segmentation_service`**: Uses a modern environment with State-of-the-Art models (GPT-4V, GroundingDINO, SAM) to analyze the generated panorama and segment it into interactive assets and a background.

This guide details the setup and execution of the entire automated pipeline for a user who has just cloned the repository.

---

## 🛠️ 1. Environment Setup (for a New User)

### 1.1. Prerequisites

-   NVIDIA GPU with appropriate drivers.
-   [Docker](https://www.docker.com/) installed.
-   [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed.

### 1.2. Directory Structure

After cloning this repository, your project structure should be organized as follows. You only need to create the `output` and `pre_checkpoints` directories.

```bash
# Run from the root of the Text2VR repository
mkdir -p ./output
mkdir -p ./pre_checkpoints
```

### 1.3. Download Pretrained Models
```bash
# Run these commands from the Text2VR/pre_checkpoints/ directory

# SAM Checkpoint (for segmentation_service)
wget [https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)

# GroundingDINO Checkpoint (for segmentation_service)
wget [https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth)

# DreamScene360 DPT-Depth Checkpoint (for dreamscene360_service)
wget "[https://www.dropbox.com/scl/fi/y11c69dd9fjf05s640qj9/omnidata_dpt_depth_v2.ckpt?rlkey=vj7a8n1s2q4q5q5j3q2q2q2q2&dl=1](https://www.dropbox.com/scl/fi/y11c69dd9fjf05s640qj9/omnidata_dpt_depth_v2.ckpt?rlkey=vj7a8n1s2q4q5q5j3q2q2q2q2&dl=1)" -O omnidata_dpt_depth_v2.ckpt
```
or download the omnidata_dpt_depth_v2.ckpt file from the official from this [Dropbox folder](https://www.dropbox.com/scl/fo/348s01x0trt0yxb934cwe/h?rlkey=a96g2incso7g53evzamzo0j0y&dl=0) and place it in `Text2VR/pre_checkpoints/` directory.


The final project structure will be:
```bash
Text2VR/
├── dreamscene360_service/      # Contains the original DreamScene360 source
│   ├── Dockerfile
│   └── ...
├── segmentation_service/       # Contains the new segmentation scripts
│   ├── Dockerfile
│   └── ...
├── docker-compose.yml          # The orchestrator for all services
├── run_pipeline.sh             # The one-click script to run the full pipeline
├── output/                     # Shared output directory for all services (created by you)
└── pre_checkpoints/            # Shared directory for pretrained models (created by you)
    └── big-lama.ckpt                   # <-- Pretrained models will be placed here
    └── omnidata_dpt_depth_v2.ckpt      # <-- Pretrained models will be placed here
    └── monidata_dpt_normal_v2.ckpt     # <-- Pretrained models will be placed here
    └── monidata_dpt_normal_v2.ckpt     # <-- Pretrained models will be placed here
    └── sam_vit_h_4b8939.pth            # <-- Pretrained models will be placed here
    └──
```

---

## ✨ 2. End-to-End Pipeline Execution (The "One-Click" Method)
This is the simplest way to run the entire pipeline, from a text prompt to a fully segmented panorama, with a single command.

### 2.1. Configuration
Before running, you must configure the main pipeline script.
1. Open run_pipeline.sh in your editor.
3. Set your OpenAI API Key. This is required for both `self-refinement` in the panorama generation stage and for asset identification in the segmentation stage.
3. Customize the scene and prompt by changing the `SCENE_NAME` and `PANO_PROMPT` variables. This will determine the output folder name and the content of the generated scene.

```bash
# In run_pipeline.sh
export OPENAI_API_KEY="your openai api key" #  IMPORTANT: Set your key
SCENE_NAME="simple_indoor"
PANO_PROMPT="A 360 equirectangular photo of a minimalist and spacious living room. In the center, there is a single modern leather sofa. The room has plain white walls, a smooth light gray concrete floor, and no other furniture or decorations. The scene is brightly lit by soft, natural light from a large window, with no harsh shadows. photorealistic, 8k, sharp focus."
```

### 2.2. Run the Pipeline
From the root of the `Text2VR` repository, execute the following commands:
```bash
# 1. Make the script executable (only needed once)
chmod +x run_pipeline.sh

# 2. Run the entire pipeline
./run_pipeline.sh
```
The `run_pipeline.sh` script will automatically:

1. Build the Docker images for both services using `docker-compose`.
2. Execute the `dreamscene360_service` to generate a panorama.
3. Execute the `segmentation_service` to segment the panorama.
4. Save all results in the `Text2VR/output/` directory, organized by `SCENE_NAME`.

---

## 🔧 3. Individual Service Development & Debugging
If you need to work on a single service without running the entire pipeline, you can use `docker-compose` to enter its specific container.

### 3.1. Working with the `dreamscene360_service`
This is useful for debugging the original `train.py` or other core functionalities.
```bash
# Build and run the container, then drop into a bash shell
docker-compose run --rm dreamscene360 /bin/bash
```
You will now be inside the container at `/workspace/dreamscene360_code`. See the `dreamscene360_service/README.md` for detailed instructions on manual execution.

### 3.2. Working with the `segmentation_service`
This is useful for testing or modifying the panorama segmentation logic.

```bash
# Build and run the container, then drop into a bash shell
docker-compose run --rm segmentation /bin/bash
```
You will now be inside the container at `/app`. See the `segmentation_service/README.md` for detailed instructions on manual execution.
