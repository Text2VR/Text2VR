🌀 Text2VR: End-to-End Panorama to Interactive Scene Pipeline🚀 OverviewThis project provides an end-to-end pipeline to generate a fully interactive 3D VR scene from a single text prompt. It leverages a microservice architecture orchestrated by Docker Compose to ensure stability, scalability, and maintainability across different AI model dependencies.The pipeline is composed of two main services, each running in its own isolated Docker container:dreamscene360_service: Utilizes the original DreamScene360 with a stable, legacy environment (diffusers==0.10.2, etc.) to generate high-quality 360° panoramas from text.segmentation_service: Uses a modern environment with State-of-the-Art models (GPT-4V, GroundingDINO, SAM) to analyze the generated panorama and segment it into interactive assets and a background.This guide details the setup and execution of the entire automated pipeline.🛠️ 1. Environment Setup1.1. PrerequisitesNVIDIA GPU with appropriate drivers.Docker installed.NVIDIA Container Toolkit installed.1.2. Directory Structure Setup (One-Time Action)To adopt the new microservice architecture, you must restructure your project directory as follows.Action Steps:Rename your existing DreamScene360 folder to dreamscene360_service.Move your existing Docker/Dockerfile into the new dreamscene360_service folder.Create a new folder named segmentation_service.Create all the new files (docker-compose.yml, run_pipeline.sh, etc.) as provided in the project documentation.Your final project structure should look like this:Text2VR/
├── dreamscene360_service/      # Contains the original DreamScene360 source
│   ├── Dockerfile
│   └── ...
├── segmentation_service/       # Contains the new segmentation scripts
│   ├── Dockerfile
│   ├── requirements.txt
│   └── segment_panorama.py
├── docker-compose.yml          # The orchestrator for all services
├── run_pipeline.sh             # The one-click script to run the full pipeline
├── output/                     # Shared output directory for all services
└── pre_checkpoints/            # Shared directory for pretrained models
1.3. Download Pretrained ModelsPlace all required pretrained models in the Text2VR/pre_checkpoints/ directory. This directory is shared across all services.# Run from the Text2VR/pre_checkpoints/ directory

# SAM Checkpoint
wget [https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)

# GroundingDINO Checkpoint
wget [https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth)

# DreamScene360 Checkpoints
wget "[https://www.dropbox.com/scl/fi/y11c69dd9fjf05s640qj9/omnidata_dpt_depth_v2.ckpt?rlkey=vj7a8n1s2q4q5q5j3q2q2q2q2&dl=1](https://www.dropbox.com/scl/fi/y11c69dd9fjf05s640qj9/omnidata_dpt_depth_v2.ckpt?rlkey=vj7a8n1s2q4q5q5j3q2q2q2q2&dl=1)" -O omnidata_dpt_depth_v2.ckpt
# ... add other required checkpoints here ...
✨ 2. End-to-End Pipeline Execution (The "One-Click" Method)This is the simplest way to run the entire pipeline, from a text prompt to a fully segmented panorama, with a single command.2.1. ConfigurationBefore running, you must configure the main pipeline script.Open run_pipeline.sh in your editor.Set your OpenAI API Key and customize the scene and prompt as needed.# In run_pipeline.sh
export OPENAI_API_KEY="sk-..." # ★★★ IMPORTANT: Set your key ★★★
SCENE_NAME="indoor_livingroom_test"
PANO_PROMPT="A spacious modern living room with a gray sofa and a potted plant."
2.2. Run the PipelineFrom the root of the Text2VR repository, execute the following commands:# 1. Make the script executable (only needed once)
chmod +x run_pipeline.sh

# 2. Run the entire pipeline
./run_pipeline.sh
The run_pipeline.sh script will automatically:Build the Docker images for both services using docker-compose.Execute the dreamscene360_service to generate a panorama.Execute the segmentation_service to segment the panorama.Save all results in the Text2VR/output/ directory, organized by SCENE_NAME.🔧 3. Individual Service Development & DebuggingIf you need to work on a single service without running the entire pipeline, you can use docker-compose to enter its specific container.3.1. Working with the dreamscene360_serviceThis is useful for debugging the original train.py or other core functionalities.# Build and run the container, then drop into a bash shell
docker-compose run --rm dreamscene360 /bin/bash

# You are now inside the container at /workspace/DreamScene360
# You can run train.py manually here
# (dev) root@...:/workspace/DreamScene360# python train.py -s data/some_scene ...
3.2. Working with the segmentation_serviceThis is useful for testing or modifying the panorama segmentation logic.# Build and run the container, then drop into a bash shell
docker-compose run --rm segmentation /bin/bash

# You are now inside the container at /app
# You can run segment_panorama.py manually here
# root@...:/app# python segment_panorama.py --panorama_path /app/data/...
