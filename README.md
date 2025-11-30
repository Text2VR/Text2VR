# Text2VR: Text-to-Interactive VR Scene Pipeline

Transform a **single text prompt** into a fully interactive VR-ready 3D scene.

Text2VR orchestrates multiple AI models through a **LangGraph-based workflow** with **Docker Compose microservices**, delivering end-to-end automation from natural language to immersive 3D environments.

![System Architecture](docs/architecture.png)

---

## Features

- **End-to-End Pipeline**: Text prompt to VR-ready scene in one command
- **7-Stage Pipeline Visualization**: Real-time progress tracking with step-by-step feedback
- **LangGraph Orchestration**: Intelligent workflow with automatic query optimization
- **Microservice Architecture**: Isolated, scalable Docker containers for each AI model
- **Interactive 360° Viewer**: A-Frame based panorama viewer with VR support
- **Progressive Results**: Watch each stage complete in real-time
- **Asset Downloads**: Export panoramas, 3D GLB models, and PLY point clouds
- **VRAM Optimization**: Automatic container lifecycle management

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Frontend** | React 19 + TypeScript + Vite + A-Frame |
| **Backend** | FastAPI + LangGraph + LangChain |
| **Panorama** | DreamScene360 (Stable Diffusion + Stitch Diffusion) |
| **Segmentation** | GroundingDINO + SAM + GPT-4o |
| **3D Generation** | TRELLIS |
| **Inpainting** | Stable Diffusion 2 Inpaint |
| **3D Scene** | Gaussian Splatting |
| **Infrastructure** | Docker Compose + NVIDIA Container Toolkit |

---

## System Architecture

### Pipeline Flow

```
┌──────────────────┐
│  Query Rewrite   │ GPT-4o optimizes user prompt
└────────┬─────────┘
         ▼
┌──────────────────┐
│    Panorama      │ DreamScene360 generates 360° image
│   Generation     │
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Segmentation    │ GroundingDINO + SAM + GPT-4o
│                  │ Extracts objects with transparency
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Asset Cropping  │ Creates transparent PNG assets
└────────┬─────────┘
         ▼
┌──────────────────┐
│  3D Generation   │ TRELLIS converts 2D → 3D GLB
└────────┬─────────┘
         ▼
┌──────────────────┐
│   Inpainting     │ SD2 fills removed object areas
└────────┬─────────┘
         ▼
┌──────────────────┐
│  PLY Generation  │ Gaussian Splatting training
└──────────────────┘
```

### Microservices

| Service | Port | Description |
|---------|------|-------------|
| `panorama-api` | 8001 | DreamScene360 - 360° panorama generation |
| `segmentation-api` | 8002 | GroundingDINO + SAM - Asset segmentation |
| `inpainting-api` | 8003 | SD2 Inpaint - Background inpainting |
| `trellis-api` | 8004 | TRELLIS - 2D to 3D asset conversion |

---

## Project Structure

```
Text2VR/
├── app/                          # FastAPI Backend
│   ├── main.py                   # Application entry point
│   ├── api/                      # API routers
│   │   ├── tasks.py              # Task management (/generate, /status)
│   │   ├── assets.py             # Asset serving
│   │   └── unity_assets.py       # Unity export endpoints
│   ├── services/                 # Business logic
│   │   ├── task_manager.py       # Task state management
│   │   ├── panorama_service.py   # Workflow execution
│   │   ├── docker_service.py     # Container management
│   │   └── image_processing.py   # Image utilities
│   ├── clients/                  # Microservice clients
│   │   ├── panorama.py           # DreamScene360 client
│   │   ├── segmentation.py       # Segmentation client
│   │   ├── inpainting.py         # Inpainting client
│   │   └── trellis.py            # TRELLIS client
│   ├── workflows/                # LangGraph workflow
│   │   ├── workflow.py           # Main workflow definition
│   │   └── steps/                # Pipeline stages
│   │       ├── generation.py     # Query rewrite & panorama
│   │       ├── segmentation.py   # Asset segmentation
│   │       ├── three_d.py        # 3D generation & PLY
│   │       └── inpainting.py     # Background inpainting
│   ├── models/                   # Pydantic models
│   └── core/                     # Config & utilities
│
├── src/                          # React Frontend
│   ├── main.tsx                  # React entry point
│   ├── App.tsx                   # Main component
│   ├── App.css                   # Design system & styles
│   ├── components/
│   │   ├── Header.tsx            # App header with logo
│   │   ├── PipelineStepper.tsx   # 7-stage progress indicator
│   │   ├── InputPanel.tsx        # Scene input & advanced options
│   │   ├── ResultPanel.tsx       # Tabbed viewer (Panorama/Seg/Inpainted)
│   │   ├── DownloadHub.tsx       # Asset download buttons
│   │   └── ...
│   ├── services/
│   │   └── apiService.ts         # Backend API abstraction
│   └── types/
│       └── api.ts                # TypeScript definitions
│
├── DREAMSCENE360/                # Panorama generation module
├── ASSET_SEG/                    # Segmentation module
├── BG_INPAINT/                   # Inpainting module
├── output/                       # Generated results
├── docker-compose.yml            # Service orchestration
├── package.json                  # Node.js dependencies
├── vite.config.ts                # Vite configuration
└── tsconfig.json                 # TypeScript configuration
```

---

## Prerequisites

- **NVIDIA GPU** with 12GB+ VRAM (recommended: RTX 4090, L40S)
- **Docker** and **Docker Compose**
- **NVIDIA Container Toolkit**
- **Node.js** 18+ and **npm**
- **Python** 3.10+
- **OpenAI API Key** (for GPT-4o)

---

## Quick Start

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/Text2VR/Text2VR.git
cd Text2VR

# Create .env file
cat > .env << 'EOF'
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o
DREAMSCENE_API_URL=http://localhost:8001
SEGMENTATION_API_URL=http://localhost:8002
INPAINTING_API_URL=http://localhost:8003
TRELLIS_API_URL=http://localhost:8004
EOF
```

### 2. Download Pretrained Models

```bash
mkdir -p pre_checkpoints

# SAM checkpoint
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth \
  -O pre_checkpoints/sam_vit_h_4b8939.pth

# DreamScene360 DPT-Depth
# Download from: https://www.dropbox.com/scl/fo/348s01x0trt0yxb934cwe/h
# Place omnidata_dpt_depth_v2.ckpt in pre_checkpoints/
```

### 3. Install Dependencies

```bash
# Frontend dependencies
npm install

# Backend dependencies (using virtual environment)
python -m venv myenv
source myenv/bin/activate  # Linux/Mac
# or: myenv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 4. Run Services

**Terminal 1: Start AI Microservices**
```bash
docker-compose up -d
```

**Terminal 2: Start Backend API**
```bash
source myenv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 3: Start Frontend**
```bash
npm run dev
```

### 5. Generate VR Scene

1. Open http://localhost:3000 in your browser
2. Enter your scene description (e.g., "A cozy living room with a fireplace")
3. (Optional) Expand "Advanced Options" to configure:
   - Self Refinement
   - Number of Prompts
   - Max Rounds
4. Click **Generate Scene** and watch the 7-stage pipeline progress

---

## Frontend Features

### Pipeline Stepper
Real-time visualization of the 7 pipeline stages:
1. **Query** - Prompt optimization
2. **Pano** - 360° panorama generation
3. **Seg** - Object segmentation
4. **Crop** - Asset extraction
5. **3D** - GLB model generation
6. **Inpaint** - Background restoration
7. **PLY** - Gaussian splatting

### Tabbed Viewer
- **Panorama**: Interactive 360° viewer (drag to rotate)
- **Segmentation**: Grid of extracted assets with labels
- **Inpainted**: Final panorama with objects removed

### Download Hub
- **Panorama**: PNG image download
- **3D Assets**: ZIP of all GLB models
- **Point Cloud**: PLY file for Gaussian Splatting

---

## API Reference

### Generate Scene
```http
POST /generate
Content-Type: application/json

{
  "text": "A modern office with large windows",
  "scene_name": "my-office",
  "use_self_refinement": false,
  "num_prompt": 3,
  "max_rounds": 3
}
```

### Check Status
```http
GET /status/{task_id}
```

Response includes:
- `status`: queued | processing | completed | failed
- `message`: Current stage description
- `panorama_path`: Path when panorama is ready
- `segmentation_visualization_path`: Segmentation result
- `inpainted_panorama_path`: Final inpainted image
- `asset_3d_paths`: Dictionary of GLB file paths
- `ply_path`: Gaussian splatting model path

### Download Assets
```http
GET /panorama/{task_id}          # PNG panorama
GET /inpainted/{task_id}         # Inpainted panorama
GET /segmentation/{task_id}      # Asset list JSON
GET /unity/latest/assets.zip     # All 3D assets
GET /unity/latest/scene.ply      # PLY point cloud
```

---

## Configuration

### Environment Variables (.env)

```bash
# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o

# Microservice URLs
DREAMSCENE_API_URL=http://localhost:8001
SEGMENTATION_API_URL=http://localhost:8002
INPAINTING_API_URL=http://localhost:8003
TRELLIS_API_URL=http://localhost:8004

# Development
MOCK_PIPELINE_MODE=false  # Enable for demo without GPU
```

### Workflow Parameters (app/core/constants.py)

```python
# Panorama generation
use_self_refinement = True
num_prompt = 5
max_rounds = 3

# Inpainting
strength = 0.85
guidance = 7.5
steps = 30

# Gaussian Splatting
iterations = 3000
sh_degree = 3
```

---

## Development

### Frontend Development

```bash
# Development with hot reload
npm run dev

# Production build
npm run build

# Preview production build
npm run preview
```

### Backend Development

```bash
# Activate virtual environment
source myenv/bin/activate

# Run with auto-reload
uvicorn app.main:app --reload --port 8000

# Run tests
pytest app/
```

### Docker Services

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Restart specific service
docker-compose restart panorama-api

# Stop all services
docker-compose down
```

---

## Output Structure

```
output/{scene_name}/
├── stitch/
│   └── im_0.png              # Generated panorama
├── masking/
│   ├── masks/                # Object masks
│   ├── visualizations/       # Segmentation visualization
│   └── results.json          # Segmentation metadata
├── assets/                   # Cropped assets (transparent PNG)
├── 3d/                       # GLB 3D models
├── ply/                      # Gaussian splatting models
└── inpainted.png             # Final inpainted panorama
```

---

## Troubleshooting

### VRAM Issues
```bash
# Stop all containers to free VRAM
docker-compose down

# Check GPU memory
nvidia-smi

# The pipeline automatically manages container lifecycle
```

### Service Health
```bash
# Check all services
docker-compose ps

# View specific service logs
docker-compose logs panorama-api

# Restart failed service
docker-compose restart segmentation-api
```

### Frontend Issues
```bash
# Clear cache and rebuild
rm -rf node_modules web-dist
npm install
npm run build

# Force refresh in browser: Ctrl+Shift+R
```

---

## Roadmap

- [ ] Unity VR integration for HMD deployment
- [ ] Multi-GPU support for parallel processing
- [ ] WebXR direct VR viewing
- [ ] Custom model fine-tuning interface
- [ ] Real-time collaborative editing

---

## License

MIT License

---

## Acknowledgments

- [DreamScene360](https://github.com/dreamscene360) - Panorama generation
- [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything)
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
- [TRELLIS](https://github.com/microsoft/TRELLIS) - 3D generation
- [LangGraph](https://github.com/langchain-ai/langgraph) - Workflow orchestration
- [A-Frame](https://aframe.io/) - WebVR framework
