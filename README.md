# Text2VR: Text-to-Interactive VR Scene Pipeline

Transform a **single text prompt** into a fully interactive VR-ready 3D scene.

Text2VR orchestrates multiple AI models through a **LangGraph-based workflow** with **Docker Compose microservices**, delivering end-to-end automation from natural language to immersive 3D environments.

![System Architecture](docs/architecture.png)

---

## Features

- **End-to-End Pipeline**: Text prompt to VR-ready scene in one command
- **LangGraph Orchestration**: Intelligent workflow with automatic query optimization
- **Microservice Architecture**: Isolated, scalable Docker containers for each AI model
- **Real-time Progress**: React frontend with progressive result visualization
- **VRAM Optimization**: Automatic container lifecycle management

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Frontend** | React + TypeScript + Vite |
| **Backend** | FastAPI + LangGraph + LangChain |
| **Panorama** | DreamScene360 (Stable Diffusion + Stitch Diffusion) |
| **Segmentation** | GroundingDINO + SAM + GPT-4o |
| **3D Generation** | TRELLIS |
| **Inpainting** | Stable Diffusion 2 Inpaint |
| **3D Scene** | Gaussian Splatting |
| **Infrastructure** | Docker Compose + NVIDIA Container Toolkit |

- NVIDIA GPU + recent driver
- [Docker](https://www.docker.com/) and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- [Unity Gaussian Splatting Plugin](https://github.com/aras-p/UnityGaussianSplatting)
---

## System Architecture

![Pipeline Flow](docs/pipeline.png)

### Microservices

| Service | Port | Description |
|---------|------|-------------|
| `panorama-api` | 8001 | DreamScene360 - 360° panorama generation |
| `segmentation-api` | 8002 | GroundingDINO + SAM - Asset segmentation |
| `inpainting-api` | 8003 | SD2 Inpaint - Background inpainting |
| `trellis-api` | 8004 | TRELLIS - 2D to 3D asset conversion |

### LangGraph Workflow

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Query Rewrite  │────▶│    Panorama     │────▶│  Segmentation   │
│     (LLM)       │     │   Generation    │     │ (DINO+SAM+GPT)  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
┌─────────────────┐     ┌─────────────────┐     ┌───────▼─────────┐
│  PLY Generation │◀────│   Inpainting    │◀────│ 3D Asset Gen    │
│    (GS Train)   │     │     (SD2)       │     │   (TRELLIS)     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

---

## Pipeline Stages

### Stage 1: Query Rewrite
Transforms user input into optimized prompts for panorama generation using LLM (GPT-4o).

### Stage 2: Panorama Generation
Generates 360° equirectangular panorama using DreamScene360 with optional self-refinement.

### Stage 3: Asset Segmentation
- **GPT-4o**: Identifies interactive objects in the scene
- **GroundingDINO**: Detects object bounding boxes from text prompts
- **SAM**: Generates precise segmentation masks
- **Asset Cropper**: Extracts individual assets with transparency

### Stage 4: 3D Asset Generation
Converts 2D segmented assets to 3D GLB models using TRELLIS.

### Stage 5: Background Inpainting
Removes segmented objects and fills background seamlessly using Stable Diffusion 2 Inpaint with wrap-aware padding.

### Stage 6: Gaussian Splatting
Trains 3D Gaussian Splatting model from the inpainted panorama for immersive VR rendering.

---

## Results Gallery

| | |
|:---:|:---:|
| ![Result 1](docs/results/result_1.png) | ![Result 2](docs/results/result_2.png) |

---

## Prerequisites

- **NVIDIA GPU** with 16GB+ VRAM (recommended: NVIDIA L4 D6 24GB)
- **Docker** and **Docker Compose**
- **NVIDIA Container Toolkit**
- **OpenAI API Key** (for GPT-4o)

---

## Project Structure

```
Text2VR/
├── app/                          # Backend application
│   ├── workflows/                # LangGraph workflow
│   │   ├── workflow.py           # Main workflow assembly
│   │   ├── nodes.py              # Pipeline node implementations
│   │   ├── states.py             # Workflow state definitions
│   │   ├── defaults.py           # API parameter defaults
│   │   ├── pano_client.py        # DreamScene360 API client
│   │   ├── segmentation_client.py
│   │   ├── inpainting_client.py
│   │   └── trellis_client.py
│   ├── api/                      # FastAPI endpoints
│   └── services/                 # Task management
├── src/                          # React frontend
│   ├── components/               # UI components
│   ├── services/                 # API service layer
│   └── types/                    # TypeScript definitions
├── DREAMSCENE360/                # Panorama generation service
├── ASSET_SEG/                    # Segmentation service
├── BG_INPAINT/                   # Inpainting service
├── TRELLIS_API/                  # 3D generation service
├── docker-compose.yml            # Service orchestration
├── orchestrator.py               # FastAPI orchestrator
├── data/                         # Generated panoramas
├── masking_output/               # Segmentation results
├── inpainted_pano/               # Inpainted panoramas
├── output/3d_assets/             # Generated 3D GLB files
├── plyoutput/                    # Gaussian Splatting PLY files
└── pre_checkpoints/              # Pretrained model weights
```

---

## Quick Start

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/your-repo/Text2VR.git
cd Text2VR

# Create .env file
cat > .env << 'EOF'
OPENAI_API_KEY=your_openai_api_key_here
EOF
```

### 2. Download Pretrained Models

```bash
mkdir -p pre_checkpoints

# SAM checkpoint
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth \
  -O pre_checkpoints/sam_vit_h_4b8939.pth

# DreamScene360 DPT-Depth (download from Dropbox)
# https://www.dropbox.com/scl/fo/348s01x0trt0yxb934cwe/h?rlkey=a96g2incso7g53evzamzo0j0y&dl=0
# Place omnidata_dpt_depth_v2.ckpt in pre_checkpoints/
```

### 3. Pull Docker Images from Docker Hub

```bash
# Pull all required images from Docker Hub
docker pull 0in11/text2vr-dreamscene360:v3
docker pull 0in11/text2vr-asset_seg:v2
docker pull 0in11/text2vr-bg_inpaint:v3
docker pull 0in11/trellis:v1
```

> **Note**: These images are pre-built and ready to use. Total download size is approximately 30-40GB.

### 4. Run Services

**Terminal 1: Start AI Services (Docker)**
```bash
# From project root (Text2VR/)
docker-compose up -d
```

**Terminal 2: Start Backend API**
```bash
# From project root (Text2VR/)
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Terminal 3: Start Frontend**
```bash
# From project root (Text2VR/src)
npm install  # first time only
npm run dev
```

### 5. Generate VR Scene

1. Open http://localhost:3000 in your browser
2. Enter your scene description (e.g., "A cozy living room with a fireplace")
3. Click Generate and watch the pipeline progress in real-time

---

## API Reference

### Generate Panorama
```
POST /generate
{
  "text": "Scene description",
  "scene_name": "optional_scene_name"
}
```

### Check Status
```
GET /status/{task_id}
```

### Get Panorama Image
```
GET /panorama/{task_id}
```

### List All Tasks
```
GET /tasks
```

---

## Configuration

### Workflow Parameters

All API parameters are centralized in `app/workflows/defaults.py`:

```python
# Panorama generation
PANORAMA_DEFAULTS.use_self_refinement = True
PANORAMA_DEFAULTS.num_prompt = 2
PANORAMA_DEFAULTS.max_rounds = 2

# Segmentation
SEGMENTATION_DEFAULTS.sam_checkpoint = "/app/checkpoints/sam_vit_h_4b8939.pth"

# Inpainting
INPAINTING_DEFAULTS.strength = 0.95
INPAINTING_DEFAULTS.guidance = 7.5
INPAINTING_DEFAULTS.steps = 40

# Gaussian Splatting
GAUSSIAN_DEFAULTS.iterations = 7000
GAUSSIAN_DEFAULTS.sh_degree = 3
```

---

## Development

### Individual Service Development

```bash
# Enter specific container for debugging
docker-compose run --rm panorama-api /bin/bash
docker-compose run --rm segmentation-api /bin/bash
docker-compose run --rm inpainting-api /bin/bash
docker-compose run --rm trellis-api /bin/bash
```

### Frontend Development

```bash
# Install dependencies
npm install

# Start dev server with hot reload
npm run dev

# Build for production
npm run build
```

---

## Troubleshooting

### VRAM Issues
The pipeline automatically stops containers after each stage to free VRAM. If you encounter OOM errors:
```bash
# Manually stop all containers
docker-compose down

# Check GPU memory
nvidia-smi
```

### Service Health Check
```bash
# Check all service logs
docker-compose logs -f

# Check specific service
docker-compose logs panorama-api
```

---

## Roadmap

- [ ] Unity VR integration for HMD deployment
- [ ] Multi-GPU support for parallel processing
- [ ] Real-time collaborative editing
- [ ] Custom model fine-tuning interface

---

## License

MIT License

---

## Acknowledgments

- [DreamScene360](https://github.com/xxx) - Panorama generation
- [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything)
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
- [TRELLIS](https://github.com/xxx) - 3D generation
- [LangGraph](https://github.com/langchain-ai/langgraph) - Workflow orchestration
