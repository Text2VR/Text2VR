#!/usr/bin/env python3
"""
Inpainting API for Background Inpainting
"""

import os
import sys
import uuid
import json
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
import torch

# Add current directory to Python path
sys.path.append('/workspace')

# Import inpainting functions
from inpaint_panorama import PanoramaInpainter, process_single_image
from types import SimpleNamespace

app = FastAPI(title="Text2VR Background Inpainting API", version="1.0.0")

# Task storage
tasks = {}
WORKING_DIR = "/workspace/inpainted_pano"

# Global inpainter instance (loaded once)
inpainter = None


class InpaintingRequest(BaseModel):
    panorama_path: str
    mask_dir: str
    scene_name: str
    model_id: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    prompt: str = "clean empty interior background, seamless walls and floor, photorealistic, matching lighting, no new objects"
    neg_prompt: str = "sofa, couch, armchair, chair, bench, text, watermark, logo, artifacts, distortion, blurry, people, signature"
    strength: float = 0.94
    guidance: float = 5.0
    steps: int = 40
    wrap_pad: Optional[int] = None  # None = auto
    dilate: Optional[int] = None    # None = auto
    feather: int = 0
    erase: str = "gray"
    seed: int = 0


class InpaintingResponse(BaseModel):
    task_id: str
    status: str
    message: str


class StatusResponse(BaseModel):
    task_id: str
    status: str
    message: str
    result_path: Optional[str] = None


def inpaint_task(task_id: str, request: InpaintingRequest):
    """Background task for panorama inpainting"""
    global inpainter

    try:
        tasks[task_id]["status"] = "processing"
        tasks[task_id]["message"] = "Starting panorama inpainting..."

        # Setup paths
        output_dir = os.path.join(WORKING_DIR, request.scene_name)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "inpainted_panorama.png")

        print(f"🎨 Inpainting panorama: {request.panorama_path}")
        print(f"🎭 Mask directory: {request.mask_dir}")
        print(f"📁 Output: {output_path}")

        # Load inpainter if not already loaded
        if inpainter is None:
            print(f"🔄 Loading inpainting model: {request.model_id}...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32
            inpainter = PanoramaInpainter(
                model_id=request.model_id,
                device=device,
                dtype=dtype,
                enable_vae_tiling=True,
                enable_attention_slicing=True
            )
            print(f"✅ Model loaded on {device}")

        # Create args object for process_single_image
        args = SimpleNamespace(
            wrap_pad=request.wrap_pad,
            dilate=request.dilate,
            feather=request.feather,
            prompt=request.prompt,
            neg_prompt=request.neg_prompt,
            strength=request.strength,
            guidance=request.guidance,
            steps=request.steps,
            seed=request.seed,
            erase=request.erase,
            save_intermediate=False
        )

        # Run inpainting
        result = process_single_image(
            inpainter=inpainter,
            img_path=request.panorama_path,
            masks_root=request.mask_dir,
            output_path=output_path,
            args=args
        )

        if result["status"] == "success":
            tasks[task_id]["status"] = "completed"
            tasks[task_id]["message"] = "Inpainting completed successfully"
            tasks[task_id]["result_path"] = output_path
            print(f"✅ Inpainting completed: {output_path}")
        elif result["status"] == "empty_mask":
            tasks[task_id]["status"] = "completed"
            tasks[task_id]["message"] = "No masks found, copied original image"
            tasks[task_id]["result_path"] = output_path
        else:
            raise Exception(f"Unknown status: {result['status']}")

    except Exception as e:
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["message"] = f"Error: {str(e)}"
        print(f"❌ Task {task_id} failed: {e}")
        import traceback
        traceback.print_exc()


@app.on_event("startup")
async def startup_event():
    os.makedirs(WORKING_DIR, exist_ok=True)
    print("🚀 Background Inpainting API started")
    print(f"📁 Working directory: {WORKING_DIR}")
    print(f"🎮 CUDA available: {torch.cuda.is_available()}")


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "background-inpainting",
        "cuda_available": torch.cuda.is_available(),
        "model_loaded": inpainter is not None
    }


@app.post("/inpaint", response_model=InpaintingResponse)
async def inpaint_panorama_endpoint(request: InpaintingRequest, background_tasks: BackgroundTasks):
    """Start panorama inpainting task"""

    # Validate input file exists
    if not os.path.exists(request.panorama_path):
        raise HTTPException(status_code=400, detail=f"Panorama file not found: {request.panorama_path}")

    # Validate mask directory exists
    if not os.path.exists(request.mask_dir):
        raise HTTPException(status_code=400, detail=f"Mask directory not found: {request.mask_dir}")

    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "status": "queued",
        "message": "Task queued",
        "result_path": None
    }

    background_tasks.add_task(inpaint_task, task_id, request)

    return InpaintingResponse(
        task_id=task_id,
        status="queued",
        message="Inpainting started"
    )


@app.get("/status/{task_id}", response_model=StatusResponse)
async def get_status(task_id: str):
    """Get task status"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = tasks[task_id]
    return StatusResponse(
        task_id=task_id,
        status=task["status"],
        message=task["message"],
        result_path=task.get("result_path")
    )


@app.get("/result/{task_id}")
async def get_result(task_id: str):
    """Download inpainting result"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = tasks[task_id]
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Task not completed: {task['status']}")

    result_path = task.get("result_path")
    if not result_path or not os.path.exists(result_path):
        raise HTTPException(status_code=404, detail="Result file not found")

    return FileResponse(
        path=result_path,
        filename=f"inpainted_{task_id[:8]}.png",
        media_type="image/png"
    )


@app.get("/tasks")
async def list_tasks():
    """List all tasks"""
    return [
        {
            "task_id": tid,
            "status": task["status"],
            "message": task["message"]
        }
        for tid, task in tasks.items()
    ]


if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8003,
        log_level="info"
    )
