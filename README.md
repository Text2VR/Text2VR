# 🥽 Text2VR: End-to-End Panorama to Interactive Scene Pipeline

## 🚀 Overview
Create an interactive VR scene from **a single text prompt**.  
Text2VR orchestrates multiple AI components with **Docker Compose** so you can reproduce builds and keep dependencies isolated and stable.

## ✨ What you get

- **Stage 1 — Panorama generation** (DreamScene360): text → 360° equirect panorama  
- **Stage 2 — Asset segmentation** (GroundingDINO + SAM + optional GPT-4o): masks for interactive objects  
- **Stage 3 — Background inpainting** (SDXL inpaint, wrap-aware): remove masked assets cleanly  
- **Stage 4 — 3D training** (DreamScene360): train a Gaussian scene using the inpainted panorama  
- **(Roadmap)** Asset mesh extraction & alignment → **Unity** integration for VR HMD

> Services run in separate containers to avoid dependency conflicts and to allow independent upgrades.

---

## 🧰 Prerequisites

- NVIDIA GPU + recent driver
- [Docker](https://www.docker.com/) and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- [Unity Gaussian Splatting Plugin](https://github.com/aras-p/UnityGaussianSplatting)
---

## 📁 Project layout

```bash
Text2VR/
├── ASSET_SEG/                  # Asset segmentation service (GroundingDINO + SAM + GPT)
│ ├── Dockerfile
│ ├── requirements.txt
│ ├── segment_panorama.py
│ └── README.md
├── BG_INPAINT/                 # Background inpainting service (SDXL inpaint)
│ ├── Dockerfile
│ ├── requirements.txt
│ ├── inpaint_panorama.py
│ └── README.md
├── DREAMSCENE360/              # Panorama & training (legacy-stable env)
│ ├── ...
│ ├── Dockerfile
│ ├── requirements.txt
│ ├── pano_generator.py
│ ├── train.py
│ └── README.md
├── docker-compose.yml
├── run_pipeline.sh             # One-click pipeline
├── output/                     
├── pre_checkpoints/            # Shared pretrained weights (created by you)
```

## ⬇️ First-time setup

### 1.1. `.env` setup 
```bash
# Create .env (Compose auto-loads it)
cat > .env << 'EOF'
OPENAI_API_KEY=...

# Optional: set HF cache inside the repo for persistence (matched to docker-compose.yml)
HF_HOME=/workspace/cache/hf
EOF
```

### 1.2. Download Pretrained Models
```bash
# From repo root
mkdir -p output pre_checkpoints

# Run these commands from the Text2VR/pre_checkpoints/ directory
# SAM Checkpoint (for ASSET_SEG)
wget [https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)

# DreamScene360 DPT-Depth Checkpoint (for DREAMSCENE360)
wget "[https://www.dropbox.com/scl/fi/y11c69dd9fjf05s640qj9/omnidata_dpt_depth_v2.ckpt?rlkey=vj7a8n1s2q4q5q5j3q2q2q2q2&dl=1](https://www.dropbox.com/scl/fi/y11c69dd9fjf05s640qj9/omnidata_dpt_depth_v2.ckpt?rlkey=vj7a8n1s2q4q5q5j3q2q2q2q2&dl=1)" -O omnidata_dpt_depth_v2.ckpt

# Put these files into pre_checkpoints/ !!
```
DreamScene360 DPT-Depth Checkpoint, 
* download the omnidata_dpt_depth_v2.ckpt file from the official from this [Dropbox folder](https://www.dropbox.com/scl/fo/348s01x0trt0yxb934cwe/h?rlkey=a96g2incso7g53evzamzo0j0y&dl=0) and place it in `Text2VR/pre_checkpoints/` directory.

#### The final project structure will be:
```bash
Text2VR/
├── ASSEGT_GEN/      
│   └── ...
├── BG_INPAINT/       
│   └── ...
├── DREAMSCENE360/              # Panorama & training (legacy-stable env)
│   └── ...
├── docker-compose.yml          # The orchestrator for all services
├── run_pipeline.sh             # The one-click script to run the full pipeline
├── output/                     
└── pre_checkpoints/            # Shared directory for pretrained models (created by you)
    └── big-lama.ckpt                   # <-- Pretrained models will be placed here
    └── omnidata_dpt_depth_v2.ckpt      # <-- Pretrained models will be placed here
    └── monidata_dpt_normal_v2.ckpt     # <-- Pretrained models will be placed here
    └── monidata_dpt_normal_v2.ckpt     # <-- Pretrained models will be placed here
    └── sam_vit_h_4b8939.pth            # <-- Pretrained models will be placed here
    └── ...
```

### 1.3. Build `docker-compose`
```bash
docker-compose build
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
export OPENAI_API_KEY="your openai api key" #  IMPORTANT: Set your key here, or use .env !!
SCENE_NAME="simple_indoor"
PANO_PROMPT="A 360 equirectangular photo of a minimalist and spacious living room. In the center, there is a single modern leather sofa. The room has plain white walls, a smooth light gray concrete floor, and no other furniture or decorations. The scene is brightly lit by soft, natural light from a large window, with no harsh shadows. photorealistic, 8k, sharp focus."
```

### 2.2. Run the Pipeline
From the root of the `Text2VR` repository, execute the following commands:
```bash
# 1. Make the script executable (only needed once)
chmod +x run_pipeline.sh

# 2. Run the entire pipeline
bash ./run_pipeline.sh
```

What it does:
1. Stage 1 – Panorama (DreamScene360), optional GPT self-refinement
2. Stage 2 – Segmentation (GroundingDINO+SAM+GPT) → /output/<SCENE>_masks/
3. Stage 3 – Inpainting (SDXL) → /DREAMSCENE360/data/<SCENE>/inpainted_panorama.png
4. Stage 4 – Training (DreamScene360)
> `run_pipeline.sh` reads `OPENAI_API_KEY` from `.env.` Edit the script to set your `SCENE_NAME` and `prompt`.

---

## 🔧 3. Individual Service Development & Debugging
If you need to work on a single service without running the entire pipeline, you can use `docker-compose` to enter its specific container.

### 3.1. Working with the `Panoramic Image Generation` or `DreamScene360`
This is useful for debugging the original `train.py` or other core functionalities.
```bash
# Build and run the container, then drop into a bash shell
docker-compose run --rm dreamscene360 /bin/bash
```
You will now be inside the container at `/workspace/dreamscene360_code`. See the `dreamscene360_service/README.md` for detailed instructions on manual execution.

### 3.2. Working with the `ASSET_SEG`
This is useful for testing or modifying the panorama segmentation logic.

```bash
# Build and run the container, then drop into a bash shell
docker-compose run --rm segmentation /bin/bash
```
You will now be inside the container at `/app`. See the `ASSET_SEG/README.md` for detailed instructions on manual execution.
