# Text2VR Container API Parameter Reference

This document summarizes the FastAPI services exposed via Docker containers so that teammates can easily experiment with them.  
Only the fields listed in the tables below are currently used; any missing fields are either ignored or not supported by the backend.

> Default host: `http://localhost:<port>` (based on docker-compose ports)

---

## DreamScene360 Panorama API (`panorama-api`, port 8001)

| Method | Path | Description | Request Parameters (JSON Body) | Default / Notes |
|--------|------|-------------|---------------------------------|-----------------|
| POST | `/generate` | Generate panorama | `text` *(required)* | Scene description text |
|  |  |  | `scene_name` | If omitted, uses `scene_<uuid>` |
|  |  |  | `use_self_refinement` | `false` |
|  |  |  | `num_prompt` | `3` |
|  |  |  | `max_rounds` | `3` |
| POST | `/panorama_to_ply` | Panorama → PLY conversion | `panorama_path` *(required)* | Path inside container (`/workspace/...`) |
|  |  |  | `output_name` | Default `panorama_pointcloud.ply` |
| POST | `/train_gaussian` | Gaussian Splatting training | `panorama_path` *(required)* | Path inside container |
|  |  |  | `scene_name` | If omitted, uses `gaussian_scene_<uuid>` |
|  |  |  | `iterations` | `100` |
|  |  |  | `save_iterations` | `[50, 100]` |
|  |  |  | `gen_res` | `512` *(currently unused)* |
|  |  |  | `white_background` | `false` |
|  |  |  | `sh_degree` | `3` |
| GET | `/health` | Health check | - | JSON health status |
| GET | `/status/{task_id}` | Task status | - | Uses only path parameter |
| GET | `/result/{task_id}` | Download result panorama | - | Uses only path parameter |
| GET | `/tasks` | List all tasks | - |  |

---

## Asset Segmentation API (`segmentation-api`, port 8002)

| Method | Path | Description | Request Parameters (JSON Body) | Default / Notes |
|--------|------|-------------|---------------------------------|-----------------|
| POST | `/segment` | Panorama segmentation | `panorama_path` *(required)* | Path inside container (`/app/host_data/...`) |
|  |  |  | `scene_name` *(required)* | Output directory name |
|  |  |  | `sam_checkpoint` | `/app/checkpoints/sam_vit_h_4b8939.pth` |
|  |  |  | `openai_api_key` | Defaults to env `OPENAI_API_KEY` |
|  |  |  | `box_threshold` | `0.20` |
|  |  |  | `text_threshold` | `0.15` |
| GET | `/health` | Health check | - |  |
| GET | `/status/{task_id}` | Task status | - | Uses only path parameter |
| GET | `/result/{task_id}` | Download segmentation JSON | - | Uses only path parameter |
| GET | `/tasks` | List all tasks | - |  |

---

## Background Inpainting API (`inpainting-api`, port 8003)

| Method | Path | Description | Request Parameters (JSON Body) | Default / Notes |
|--------|------|-------------|---------------------------------|-----------------|
| POST | `/inpaint` | Background inpainting | `panorama_path` *(required)* | Path inside container (`/workspace/data/...`) |
|  |  |  | `mask_dir` *(required)* | Mask directory (`/workspace/masking_output/...`) |
|  |  |  | `scene_name` *(required)* | Output directory name |
|  |  |  | `model_id` | `"diffusers/stable-diffusion-xl-1.0-inpainting-0.1"` |
|  |  |  | `prompt` | `"clean empty interior background, …"` |
|  |  |  | `neg_prompt` | `"sofa, couch, armchair, …"` |
|  |  |  | `strength` | `0.94` |
|  |  |  | `guidance` | `5.0` |
|  |  |  | `steps` | `40` |
|  |  |  | `wrap_pad` | `null` (auto) |
|  |  |  | `dilate` | `null` (auto) |
|  |  |  | `feather` | `0` |
|  |  |  | `erase` | `"gray"` (`"none"` / `"black"` also allowed) |
|  |  |  | `seed` | `0` |
| GET | `/health` | Health check | - | Includes `cuda_available`, `model_loaded` |
| GET | `/status/{task_id}` | Task status | - | Uses only path parameter |
| GET | `/result/{task_id}` | Inpainted result image | - | Uses only path parameter |
| GET | `/tasks` | List all tasks | - |  |

---

## TRELLIS 3D Asset API (`trellis-api`, port 8004)

### File upload mode (`POST /generate-direct`)

- **Form fields** (multipart/form-data):
  - `image` *(required)*: Uploaded image file
  - `asset_name`: Default `"generated_asset"`
  - `seed`: Default `42`
  - `simplify`: Default `0.95` (0.0–1.0)
  - `texture_size`: Default `1024` (512 / 1024 / 2048)
  - `ss_guidance_strength`: Default `7.5`
  - `ss_sampling_steps`: Default `12`
  - `slat_guidance_strength`: Default `3.0`
  - `slat_sampling_steps`: Default `12`

### Path-based mode (`POST /generate`)

- **JSON Body**:
  - `image_path` *(required)*: Path to image inside container (`/app/...`)
  - `asset_name` *(required)*
  - `output_dir` *(required)*: Directory where GLB will be saved
  - `seed`, `simplify`, `texture_size`, `ss_guidance_strength`, `ss_sampling_steps`, `slat_guidance_strength`, `slat_sampling_steps`: Same defaults as in the upload mode

### Other endpoints

| Method | Path | Description | Notes |
|--------|------|-------------|-------|
| GET | `/health` | Pipeline health check | Returns GPU memory and load status |

---

## Usage Examples

```bash
# DreamScene360 panorama generation
curl -X POST http://localhost:8001/generate \
  -H "Content-Type: application/json" \
  -d '{
        "text": "sunny beach boardwalk with cafes",
        "scene_name": "scene_demo",
        "use_self_refinement": true,
        "num_prompt": 4,
        "max_rounds": 2
      }'

# Run segmentation
curl -X POST http://localhost:8002/segment \
  -H "Content-Type: application/json" \
  -d '{
        "panorama_path": "/app/host_data/scene_demo/panorama.png",
        "scene_name": "scene_demo",
        "box_threshold": 0.25,
        "text_threshold": 0.2
      }'

# Run inpainting
curl -X POST http://localhost:8003/inpaint \
  -H "Content-Type: application/json" \
  -d '{
        "panorama_path": "/workspace/data/scene_demo/panorama.png",
        "mask_dir": "/workspace/masking_output/scene_demo/masks",
        "scene_name": "scene_demo",
        "strength": 0.9,
        "guidance": 6.5,
        "steps": 50
      }'

# TRELLIS file upload example
curl -X POST http://localhost:8004/generate-direct \
  -F "image=@./seged_assets/scene_demo/chair.png" \
  -F "asset_name=chair" \
  -F "texture_size=2048" \
  -o chair.glb
````

Feel free to tweak the parameter values above while experimenting.
Any values not defined in the current FastAPI models will be rejected or ignored by the backend, so if you need new parameters, you must first extend the Pydantic models/endpoints.

Common default values used across the pipeline are centralized in `app/workflows/defaults.py`.
Refer to that file when checking or modifying shared defaults so that behavior stays consistent.
