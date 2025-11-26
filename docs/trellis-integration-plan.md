# TRELLIS 3D Asset Generation Integration Plan

**Date**: 2025-10-22  
**Status**: Verified ✅  
**Goal**: Integrate the TRELLIS API into the LangGraph workflow to automatically convert segmented objects into 3D GLB assets.

---

## 📋 Table of Contents

1. [Overview](#overview)  
2. [Current Workflow Analysis](#current-workflow-analysis)  
3. [TRELLIS Integration Goals](#trellis-integration-goals)  
4. [Verified Items](#verified-items)  
5. [Implementation Plan](#implementation-plan)  
6. [Directory Structure](#directory-structure)  
7. [VRAM Management Strategy](#vram-management-strategy)  
8. [Implementation Checklist](#implementation-checklist)  
9. [Expected Issues and Mitigations](#expected-issues-and-mitigations)

---

## Overview

### Purpose

Automatically convert segmented 2D object images into 3D GLB assets via the TRELLIS API, so that VR scenes can be constructed from automatically generated 3D assets.

### Expected Benefits

- **Automation**: Generate 3D assets from 2D images without manual 3D modeling.  
- **Consistency**: Convert objects extracted from panoramas into stylistically consistent 3D assets.  
- **Efficiency**: Handle the 2D→3D conversion as a one-stop process within the workflow.

---

## Current Workflow Analysis

### Existing LangGraph Workflow

```text
query_rewrite → panorama_generation → segmentation → inpainting → ply_generation → END
````

#### Role of Each Node

1. **query_rewrite** (`app/workflows/nodes.py:25-61`)

   * Rewrites user input into a form suitable for panorama generation.
   * Uses an OpenAI LLM.
   * Generates a scene name (UUID-based).

2. **panorama_generation** (`app/workflows/nodes.py:64-221`)

   * Calls the DreamScene360 API.
   * Generates a 360° panorama image.
   * Path: `/home/0in/workspace/Text2VR/data/{scene_name}/panorama.png`

3. **segmentation** (`app/workflows/nodes.py:224-328`)

   * Runs object segmentation using SAM + GroundingDINO.
   * Generates masks:
     `/home/0in/workspace/Text2VR/masking_output/{scene_name}/masks/`
   * **Generates cropped assets**:
     `/home/0in/workspace/Text2VR/seged_assets/{scene_name}/`
   * Saves transparent-background RGBA PNG files.
   * Stops container after completion (to save VRAM).

4. **inpainting** (`app/workflows/nodes.py:331-425`)

   * Performs background inpainting using Stable Diffusion.
   * Produces a panorama with objects removed.
   * Stops container after completion (to save VRAM).

5. **ply_generation** (`app/workflows/nodes.py:428-500`)

   * Converts the inpainted panorama to a PLY point cloud.
   * Based on depth estimation.

### Current State Structure (`app/workflows/states.py`)

```python
class WorkflowState(TypedDict):
    task_id: str
    user_input: str
    rewritten_query: str
    scene_name: str
    panorama_path: str
    segmentation_data: Dict[str, object]
    inpainted_panorama_path: str
    ply_path: str
    messages: Annotated[List[BaseMessage], operator.add]
```

### Cropped Asset Generation Logic (`app/workflows/asset_cropper.py:102-198`)

✅ **Verified**: `crop_assets_with_transparency()` is already implemented.

**Characteristics**:

* Produces transparent-background (RGBA) PNGs.
* Crops based on bounding boxes.
* Output path:
  `/home/0in/workspace/Text2VR/seged_assets/{scene_name}/{asset_name}.png`
* Return value: `Dict[str, List[str]]` (asset_name → list of file paths).

**Verified Behavior**:

```python
# Called at nodes.py:265-269
cropped_assets = crop_assets_with_transparency(
    panorama_path=panorama_path,
    segmentation_output_dir=segmentation_output_dir,
    scene_name=state['scene_name']
)
# Result: {"sofa": ["/path/to/sofa.png"], "plant": ["/path/to/plant.png"], ...}
```

---

## TRELLIS Integration Goals

### Updated Workflow

```text
query_rewrite → panorama_generation → segmentation → asset_3d_generation → inpainting → ply_generation → END
                                                              ↑ newly added
```

### Role of the `asset_3d_generation` Node

1. Read segmented asset images from `cropped_assets`.
2. Send each asset image to the TRELLIS API.
3. Download and store the resulting GLB files.
4. Update `asset_3d_paths` in the state.
5. Stop the TRELLIS container (free VRAM).

### Expected I/O

**Input**:

* `state['cropped_assets']`:
  `{"sofa": ["/path/to/sofa.png"], ...}`
* `state['scene_name']`:
  `"scene_abc123de"`

**Output**:

* `state['asset_3d_paths']`:
  `{"sofa": "/path/to/3d_assets/scene_abc123de/sofa.glb", ...}`

---

## Verified Items

### ✅ 1. Docker Image Availability

```bash
$ docker images | grep trellis
trellis    v1    5745868dd8d4    6 weeks ago    43.2GB
```

**Result**: TRELLIS image is available locally (`trellis:v1`).

### ✅ 2. TRELLIS API Verification

**API to use**: `trellis_api_v2.py`

**Endpoint**: `POST /generate-direct`

* Uses multipart/form-data file upload.
* Directly returns a GLB file.
* No volume mount required for the response itself.

**Code location**:
`/home/0in/workspace/Text2VR/TRELLIS_API/trellis_api_v2.py:98-213`

**Parameters**:

```python
image: UploadFile           # Input image (RGBA PNG)
asset_name: str = "generated_asset"
seed: int = 42
simplify: float = 0.95
texture_size: int = 1024
ss_guidance_strength: float = 7.5
ss_sampling_steps: int = 12
slat_guidance_strength: float = 3.0
slat_sampling_steps: int = 12
```

### ✅ 3. Directory Structure Verification

**Existing directories**:

```text
/home/0in/workspace/Text2VR/
├── data/                    # panoramas
│   └── {scene_name}/
│       └── panorama.png
├── masking_output/          # segmentation results
│   └── {scene_name}/
│       ├── masks/           # mask PNGs
│       └── results.json
├── seged_assets/            # ✅ cropped assets
│   └── {scene_name}/
│       ├── sofa.png
│       ├── plant.png
│       └── ...
└── output/                  # other outputs
    └── {scene_name}/
```

**New directory needed**:

```text
/home/0in/workspace/Text2VR/
└── output/
    └── 3d_assets/           # new
        └── {scene_name}/
            ├── sofa.glb
            ├── plant.glb
            └── ...
```

### ✅ 4. Path Mapping Verification

**Host ↔ Container path mapping**:

| Item           | Host Path                                              | Container Path                  |
| -------------- | ------------------------------------------------------ | ------------------------------- |
| Cropped assets | `/home/0in/workspace/Text2VR/seged_assets/{scene}`     | `/app/seged_assets/{scene}`     |
| 3D assets      | `/home/0in/workspace/Text2VR/output/3d_assets/{scene}` | `/app/output/3d_assets/{scene}` |
| Cache          | `/home/0in/workspace/Text2VR/cache/hf`                 | `/root/.cache/huggingface`      |

### ✅ 5. VRAM Usage Check

| Service       | VRAM (idle) | VRAM (processing) | Notes               |
| ------------- | ----------- | ----------------- | ------------------- |
| DreamScene360 | -           | ~8–10 GB          | Panorama generation |
| Segmentation  | -           | ~6 GB             | SAM + GroundingDINO |
| Inpainting    | -           | ~6 GB             | Stable Diffusion    |
| **TRELLIS**   | **5.3 GB**  | **6–8 GB**        | Image-to-3D         |

**Conclusion**: Running multiple services concurrently can cause VRAM shortage → containers must be stopped after each stage.

---

## Implementation Plan

### Task 1: Add TRELLIS service to `docker-compose.yml`

**File**: `docker-compose.yml`

**Service to add**:

```yaml
services:
  # ... existing services ...

  # TRELLIS 3D Asset Generation API
  trellis-api:
    image: trellis:v1
    container_name: text2vr_trellis_api
    working_dir: /app
    ports:
      - "8004:8000"
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - SPCONV_ALGO=native
    ipc: host
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - ./TRELLIS_API/trellis_api_v2.py:/app/trellis_api_v2.py:ro
      - ./seged_assets:/app/seged_assets:ro             # read cropped assets
      - ./output/3d_assets:/app/output/3d_assets        # save GLBs
      - ./cache/hf:/root/.cache/huggingface
      - ./cache/torch:/root/.cache/torch
    command: ["python", "/app/trellis_api_v2.py"]
    restart: unless-stopped
```

**Port**: 8004 (to avoid conflicts with other services).

---

### Task 2: Extend `WorkflowState` fields

**File**: `app/workflows/states.py`

**Changes**:

```python
class WorkflowState(TypedDict):
    """Represents the shared state that flows through the LangGraph workflow."""

    task_id: str
    user_input: str
    rewritten_query: str
    scene_name: str
    panorama_path: str
    segmentation_data: Dict[str, object]
    inpainted_panorama_path: str
    ply_path: str

    # New: cropped asset paths
    cropped_assets: Dict[str, List[str]]  # {"sofa": ["/path/to/sofa.png"], ...}

    # New: 3D asset paths
    asset_3d_paths: Dict[str, str]        # {"sofa": "/path/to/sofa.glb", ...}

    messages: Annotated[List[BaseMessage], operator.add]
```

**Note**: `cropped_assets` is already used in `nodes.py:309`; we are just formalizing the type.

---

### Task 3: Implement TRELLIS API client

**File**: `app/workflows/trellis_client.py` (new)

**Implementation**:

```python
#!/usr/bin/env python3
"""
TRELLIS API Client for 3D asset generation
"""

import requests
from typing import Optional
from pathlib import Path


class TrellisAPIClient:
    def __init__(self, base_url: str = "http://localhost:8004"):
        self.base_url = base_url

    def health_check(self) -> dict:
        """Check if TRELLIS API is healthy and ready"""
        response = requests.get(f"{self.base_url}/health", timeout=10)
        response.raise_for_status()
        return response.json()

    def generate_3d_asset(
        self,
        image_path: str,
        asset_name: str,
        output_path: str,
        seed: int = 42,
        texture_size: int = 1024,
        simplify: float = 0.95,
        ss_guidance_strength: float = 7.5,
        ss_sampling_steps: int = 12,
        slat_guidance_strength: float = 3.0,
        slat_sampling_steps: int = 12,
        timeout: int = 120
    ) -> str:
        """
        Generate 3D GLB asset from image using TRELLIS API

        Args:
            image_path: Path to input image (RGBA PNG with transparent background)
            asset_name: Name for the asset (used in filename)
            output_path: Full path where GLB file should be saved
            seed: Random seed for reproducibility
            texture_size: Texture resolution (512, 1024, or 2048)
            simplify: Mesh simplification ratio (0.0-1.0)
            ss_guidance_strength: Sparse structure guidance strength
            ss_sampling_steps: Sparse structure sampling steps
            slat_guidance_strength: SLAT guidance strength
            slat_sampling_steps: SLAT sampling steps
            timeout: Request timeout in seconds

        Returns:
            Path to generated GLB file

        Raises:
            FileNotFoundError: If input image doesn't exist
            requests.HTTPError: If API request fails
        """
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Input image not found: {image_path}")

        with open(image_path, 'rb') as f:
            files = {'image': (Path(image_path).name, f, 'image/png')}

            data = {
                'asset_name': asset_name,
                'seed': seed,
                'texture_size': texture_size,
                'simplify': simplify,
                'ss_guidance_strength': ss_guidance_strength,
                'ss_sampling_steps': ss_sampling_steps,
                'slat_guidance_strength': slat_guidance_strength,
                'slat_sampling_steps': slat_sampling_steps
            }

            response = requests.post(
                f"{self.base_url}/generate-direct",
                files=files,
                data=data,
                timeout=timeout,
                stream=True
            )
            response.raise_for_status()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return str(output_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print("Usage: python trellis_client.py <image_path> <asset_name> <output_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    asset_name = sys.argv[2]
    output_path = sys.argv[3]

    client = TrellisAPIClient()

    try:
        health = client.health_check()
        print(f"✅ TRELLIS API Status: {health['status']}")
        print(f"📊 GPU Memory: {health['gpu_memory_used']:.2f}GB")

        print(f"🎯 Generating 3D asset for: {asset_name}")
        result_path = client.generate_3d_asset(
            image_path=image_path,
            asset_name=asset_name,
            output_path=output_path
        )
        print(f"✅ 3D asset saved: {result_path}")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
```

---

### Task 4: Implement `asset_3d_generation_node`

**File**: `app/workflows/nodes.py`

**New node**:

```python
def asset_3d_generation_node(state: WorkflowState) -> WorkflowState:
    """
    Generate 3D GLB assets from cropped segmentation images using TRELLIS API
    """
    from .trellis_client import TrellisAPIClient
    import subprocess
    import time

    cropped_assets = state.get("cropped_assets", {})

    if not cropped_assets:
        print("⚠️ No cropped assets found, skipping 3D generation")
        return {
            **state,
            "asset_3d_paths": {},
            "messages": [
                HumanMessage(content="No assets to convert to 3D")
            ],
        }

    try:
        print(f"🎲 Starting 3D asset generation for {len(cropped_assets)} assets")

        # Start TRELLIS container (ignored if already running)
        try:
            subprocess.run(
                ["docker", "start", "text2vr_trellis_api"],
                capture_output=True,
                timeout=10
            )
            print("🚀 TRELLIS container started")
            # Wait for pipeline loading (~10 seconds)
            time.sleep(10)
        except Exception as e:
            print(f"⚠️ Failed to start TRELLIS container: {e}")

        client = TrellisAPIClient(base_url=settings.TRELLIS_API_URL)

        # Health check
        try:
            health = client.health_check()
            if health['status'] != 'healthy':
                raise Exception(f"TRELLIS API not healthy: {health}")
            print(f"✅ TRELLIS API ready (GPU memory: {health['gpu_memory_used']:.2f}GB)")
        except Exception as e:
            raise Exception(f"TRELLIS API health check failed: {e}")

        asset_3d_paths = {}
        scene_name = state["scene_name"]

        for asset_name, image_paths in cropped_assets.items():
            if not image_paths:
                continue

            image_path = image_paths[0]
            print(f"🎯 Generating 3D for: {asset_name}")

            output_dir = f"/home/0in/workspace/Text2VR/output/3d_assets/{scene_name}"
            output_path = f"{output_dir}/{asset_name}.glb"

            try:
                result_path = client.generate_3d_asset(
                    image_path=image_path,
                    asset_name=asset_name,
                    output_path=output_path,
                    timeout=120
                )

                asset_3d_paths[asset_name] = result_path
                print(f"✅ 3D asset created: {result_path}")

            except Exception as asset_exc:
                print(f"❌ Failed to generate 3D for {asset_name}: {asset_exc}")
                continue

        # Stop TRELLIS container (free VRAM)
        try:
            print("🛑 Stopping TRELLIS container to free VRAM...")
            subprocess.run(
                ["docker", "stop", "text2vr_trellis_api"],
                capture_output=True,
                timeout=10
            )
            print("✅ TRELLIS container stopped")
        except Exception as e:
            print(f"⚠️ Failed to stop TRELLIS container: {e}")

        # Update task manager
        try:
            if state.get("task_id") and asset_3d_paths:
                task_manager.update_task_status(
                    task_id=state["task_id"],
                    status=TaskStatus.PROCESSING,
                    message=f"3D assets generated ({len(asset_3d_paths)} assets), starting inpainting...",
                )
                print(f"✅ Task manager updated for task_id: {state['task_id']}")
        except Exception as tm_exc:
            print(f"⚠️ Failed to update task manager: {tm_exc}")

        print(f"🎉 3D generation completed: {len(asset_3d_paths)}/{len(cropped_assets)} assets")

        return {
            **state,
            "asset_3d_paths": asset_3d_paths,
            "messages": [
                HumanMessage(
                    content=f"3D assets generated: {list(asset_3d_paths.keys())}"
                )
            ],
        }

    except Exception as exc:
        print(f"❌ 3D asset generation failed: {exc}")
        return {
            **state,
            "asset_3d_paths": {},
            "messages": [
                HumanMessage(content=f"3D generation failed: {str(exc)}")
            ],
        }
```

---

### Task 5: Add node to workflow

**File**: `app/workflows/workflow.py`

**Changes**:

```python
from .nodes import (
    panorama_generation_node,
    query_rewrite_node,
    segmentation_node,
    asset_3d_generation_node,  # new
    inpainting_node,
    ply_generation_node,
)

def create_workflow():
    """Compile and return the LangGraph workflow for panorama generation."""

    workflow = StateGraph(WorkflowState)
    workflow.add_node("query_rewrite", query_rewrite_node)
    workflow.add_node("panorama_generation", panorama_generation_node)
    workflow.add_node("segmentation", segmentation_node)
    workflow.add_node("asset_3d_generation", asset_3d_generation_node)  # new
    workflow.add_node("inpainting", inpainting_node)
    workflow.add_node("ply_generation", ply_generation_node)

    workflow.set_entry_point("query_rewrite")
    workflow.add_edge("query_rewrite", "panorama_generation")
    workflow.add_edge("panorama_generation", "segmentation")
    workflow.add_edge("segmentation", "asset_3d_generation")  # new
    workflow.add_edge("asset_3d_generation", "inpainting")    # changed
    workflow.add_edge("inpainting", "ply_generation")
    workflow.add_edge("ply_generation", END)

    return workflow.compile()
```

**`langgraph_workflow.py` changes**:

```python
from .nodes import (
    panorama_generation_node,
    query_rewrite_node,
    segmentation_node,
    asset_3d_generation_node,  # new
)

__all__ = [
    "WorkflowState",
    "create_workflow",
    "query_rewrite_node",
    "panorama_generation_node",
    "segmentation_node",
    "asset_3d_generation_node",  # new
]
```

---

### Task 6: Add TRELLIS API URL to `config.py`

**File**: `app/config.py`

**Changes**:

```python
class Settings(BaseSettings):
    """Application settings"""

    # ... existing fields ...

    # External APIs
    DREAMSCENE_API_URL: str
    SEGMENTATION_API_URL: str = "http://localhost:8002"
    INPAINTING_API_URL: str = "http://localhost:8003"
    TRELLIS_API_URL: str = "http://localhost:8004"  # new

    # ... rest ...
```

---

### Task 7: Create directories

**Command**:

```bash
mkdir -p /home/0in/workspace/Text2VR/output/3d_assets
```

---

## Directory Structure

### Final Directory Layout

```text
/home/0in/workspace/Text2VR/
├── app/
│   └── workflows/
│       ├── states.py                    # state definitions (modified)
│       ├── nodes.py                     # node implementations (added: asset_3d_generation_node)
│       ├── workflow.py                  # workflow assembly (modified)
│       ├── langgraph_workflow.py        # compatibility layer (modified)
│       ├── trellis_client.py            # TRELLIS API client (new)
│       ├── segmentation_client.py       # existing
│       ├── inpainting_client.py         # existing
│       └── asset_cropper.py             # existing
├── config.py                            # settings (modified)
├── docker-compose.yml                   # Docker services (modified)
├── TRELLIS_API/
│   ├── trellis_api.py
│   └── trellis_api_v2.py               # API to use
├── data/
│   └── {scene_name}/
│       └── panorama.png
├── masking_output/
│   └── {scene_name}/
│       ├── masks/
│       │   ├── sofa.png
│       │   └── plant.png
│       └── results.json
├── seged_assets/                        # ✅ cropped assets (transparent)
│   └── {scene_name}/
│       ├── sofa.png
│       └── plant.png
├── output/
│   └── 3d_assets/                       # ✨ new: 3D GLB assets
│       └── {scene_name}/
│           ├── sofa.glb
│           └── plant.glb
└── docs/
    └── trellis-integration-plan.md      # this document
```

---

## VRAM Management Strategy

### VRAM Usage Analysis

| Stage                  | Service       | VRAM (processing) | Container State               |
| ---------------------- | ------------- | ----------------- | ----------------------------- |
| 1. Query Rewrite       | -             | 0 GB              | -                             |
| 2. Panorama Generation | DreamScene360 | 8–10 GB           | running                       |
| 3. Segmentation        | ASSET_SEG     | 6 GB              | running → stop after done ✅   |
| **4. 3D Generation**   | **TRELLIS**   | **6–8 GB**        | **start → stop after done** ✅ |
| 5. Inpainting          | BG_INPAINT    | 6 GB              | start → stop after done ✅     |
| 6. PLY Generation      | DreamScene360 | -                 | reused                        |

### Container Stop Pattern (already implemented)

**After segmentation** (`nodes.py:279-286`):

```python
subprocess.run(["docker", "stop", "text2vr_segmentation_api"],
               capture_output=True, timeout=10)
```

**After inpainting** (`nodes.py:386-393`):

```python
subprocess.run(["docker", "stop", "text2vr_inpainting_api"],
               capture_output=True, timeout=10)
```

**After TRELLIS** (new):

```python
subprocess.run(["docker", "stop", "text2vr_trellis_api"],
               capture_output=True, timeout=10)
```

### Recommended GPU Specs

* **Minimum**: 12 GB VRAM (RTX 3080 Ti, RTX 4070 Ti)
* **Recommended**: 16+ GB VRAM (RTX 4080, A4000, A5000)
* **Ideal**: 24+ GB VRAM (RTX 4090, A6000)

---

## Implementation Checklist

### Phase 1: Environment Preparation

* [x] Confirm TRELLIS Docker image (`trellis:v1`) exists.
* [ ] Create 3D assets directory (`mkdir -p output/3d_assets`).
* [ ] Add TRELLIS service to `docker-compose.yml`.
* [ ] Test starting TRELLIS container via Docker Compose.

### Phase 2: Code Implementation

* [ ] `app/config.py`: add `TRELLIS_API_URL`.
* [ ] `app/workflows/states.py`: extend `WorkflowState`.

  * [ ] Add `cropped_assets: Dict[str, List[str]]`.
  * [ ] Add `asset_3d_paths: Dict[str, str]`.
* [ ] `app/workflows/trellis_client.py`: implement API client.

  * [ ] `health_check()` method.
  * [ ] `generate_3d_asset()` method.
* [ ] `app/workflows/nodes.py`: add `asset_3d_generation_node`.

  * [ ] Container start logic.
  * [ ] Health check.
  * [ ] Loop over assets.
  * [ ] Save GLB files.
  * [ ] Stop container.
* [ ] `app/workflows/workflow.py`: add node to workflow.

  * [ ] `add_node("asset_3d_generation", ...)`.
  * [ ] Update edges (`segmentation` → `asset_3d_generation` → `inpainting`).
* [ ] `app/workflows/langgraph_workflow.py`: update exports.

### Phase 3: Testing

* [ ] Standalone TRELLIS API test (via curl or Python script).
* [ ] Unit tests for `TrellisAPIClient`.
* [ ] Isolated test for `asset_3d_generation_node`.
* [ ] Full workflow integration test.
* [ ] Monitor VRAM usage (`nvidia-smi`).
* [ ] Error case tests:

  * [ ] Missing image file.
  * [ ] API connection failure.
  * [ ] Partial asset generation failures.

### Phase 4: Documentation and Deployment

* [x] Draft integration plan (this document).
* [ ] Update README.
* [ ] Add API usage examples.
* [ ] Write troubleshooting guide.

---

## Expected Issues and Mitigations

### Issue 1: TRELLIS Container Startup Delay

**Problem**: Pipeline loading takes time (~10 seconds).

**Mitigation**:

```python
subprocess.run(["docker", "start", "text2vr_trellis_api"])
time.sleep(10)  # wait for loading

# Additional health check loop
for _ in range(30):  # up to 30 attempts (~30 seconds)
    try:
        health = client.health_check()
        if health['status'] == 'healthy':
            break
    except:
        time.sleep(1)
```

---

### Issue 2: Increased Processing Time

**Problem**: 30–40 seconds per asset; 3 assets → ~2 minutes.

**Mitigation**:

* Consider parallelization when multiple GPUs are available.
* Or keep sequential processing and clearly show progress to the user.

---

### Issue 3: Partial Asset Generation Failures

**Problem**: TRELLIS may fail on some assets.

**Mitigation**:

```python
for asset_name, image_paths in cropped_assets.items():
    try:
        result = client.generate_3d_asset(...)
        asset_3d_paths[asset_name] = result
    except Exception as e:
        print(f"⚠️ Failed for {asset_name}: {e}")
        continue  # proceed with remaining assets
```

---

### Issue 4: Transparent Background Handling

**Problem**: TRELLIS might not handle transparency properly.

**Verified**:

* `crop_assets_with_transparency()` generates RGBA PNGs ✅
* TRELLIS API accepts RGBA input (`trellis_api_v2.py:136`) ✅

---

### Issue 5: VRAM Shortage

**Problem**: Running multiple services concurrently can cause OOM.

**Mitigation**:

* ✅ Pattern already in place: stop containers after each stage.
* Apply the same pattern to TRELLIS.

---

### Issue 6: Volume Permission Issues

**Problem**: Permission errors on mounted volumes.

**Mitigation**:

```bash
chmod -R 755 /home/0in/workspace/Text2VR/output/3d_assets
```

---

## References

### Related Files

* TRELLIS API guide: `/home/0in/workspace/TRELLIS_API_Guide.md`
* TRELLIS API v2: `/home/0in/workspace/Text2VR/TRELLIS_API/trellis_api_v2.py`
* Current workflow: `/home/0in/workspace/Text2VR/app/workflows/`
* Asset cropper: `/home/0in/workspace/Text2VR/app/workflows/asset_cropper.py`

### API Endpoints

| Service       | Port | Endpoints                       |
| ------------- | ---- | ------------------------------- |
| DreamScene360 | 8001 | `/generate`, `/panorama_to_ply` |
| Segmentation  | 8002 | `/segment`, `/status/{task_id}` |
| Inpainting    | 8003 | `/inpaint`, `/status/{task_id}` |
| **TRELLIS**   | 8004 | `/generate-direct`, `/health`   |

---

## Version History

* **v1.0** (2025-10-22): Initial draft and verification completed

  * Verified Docker image.
  * Verified directory structure.
  * Verified API endpoints.
  * Defined VRAM management strategy.

---

## Next Steps

1. **Immediately executable**:

   * `mkdir -p /home/0in/workspace/Text2VR/output/3d_assets`
   * Update `docker-compose.yml`.
   * Update `config.py`.

2. **Code implementation** (1–2 hours):

   * Implement `trellis_client.py`.
   * Implement `asset_3d_generation_node`.
   * Update `workflow.py`.

3. **Testing** (30 minutes):

   * Unit tests.
   * Integration tests.
   * VRAM monitoring.

4. **Deployment**:

   * Git commit.
   * Documentation update.

---

**Author**: Claude (llmops-expert agent)
**Reviewer**: 0in
**Approval Status**: Verified ✅
