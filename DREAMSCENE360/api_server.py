#!/usr/bin/env python3
"""
Panorama Generation API for DreamScene360
"""

import os
import uuid
import shutil
from pathlib import Path
from typing import Optional, List
import subprocess
import re
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
import torch
from PIL import Image
import numpy as np
import trimesh
import io

# Import panorama generation functions
from pano_generator import generate_panorama

app = FastAPI(title="DreamScene360 Panorama API", version="1.0.0")

# Task storage
tasks = {}
WORKING_DIR = "/workspace/data"
PROJECT_ROOT = "/workspace/DREAMSCENE360"

class PanoramaRequest(BaseModel):
    text: str
    scene_name: Optional[str] = None
    use_self_refinement: bool = False
    num_prompt: int = 3
    max_rounds: int = 3

class PanoramaResponse(BaseModel):
    task_id: str
    status: str
    message: str

class StatusResponse(BaseModel):
    task_id: str
    status: str
    message: str
    result_path: Optional[str] = None

class PanoramaToPlyRequest(BaseModel):
    panorama_path: str
    output_name: Optional[str] = None

class TrainGaussianRequest(BaseModel):
    panorama_path: str
    scene_name: Optional[str] = None
    iterations: int = 100
    save_iterations: List[int] = [50, 100]
    gen_res: int = 512
    white_background: bool = False
    sh_degree: int = 3

class TrainGaussianResponse(BaseModel):
    status: str
    message: str
    model_path: Optional[str] = None
    initial_ply_path: Optional[str] = None
    trained_ply_paths: Optional[List[str]] = None

def generate_task(task_id: str, request: PanoramaRequest):
    """Background task for panorama generation"""
    try:
        tasks[task_id]["status"] = "processing"
        tasks[task_id]["message"] = "Starting panorama generation..."
        
        # Setup paths - use provided scene_name or fall back to task_id
        scene_name = request.scene_name if request.scene_name else f"scene_{task_id[:8]}"
        print(f"🎬 Using scene name: {scene_name} (requested: {request.scene_name}, task_id: {task_id})")
        output_dir = os.path.join(WORKING_DIR, scene_name)
        
        # Get API key from environment
        api_key = os.getenv("OPENAI_API_KEY") if request.use_self_refinement else None
        
        # Call existing panorama generation function
        result_path = generate_panorama(
            text_prompt=request.text,
            output_dir=output_dir,
            api_key=api_key,
            use_self_refinement=request.use_self_refinement,
            num_prompt=request.num_prompt,
            max_rounds=request.max_rounds
        )
        
        if result_path and os.path.exists(result_path):
            tasks[task_id]["status"] = "completed"
            tasks[task_id]["message"] = "Panorama generation completed"
            tasks[task_id]["result_path"] = result_path
        else:
            raise Exception("Panorama generation failed")
            
    except Exception as e:
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["message"] = f"Error: {str(e)}"
        print(f"Task {task_id} failed: {e}")

@app.on_event("startup")
async def startup_event():
    os.makedirs(WORKING_DIR, exist_ok=True)
    print("🚀 DreamScene360 Panorama API started")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "dreamscene360-panorama"}

@app.post("/generate", response_model=PanoramaResponse)
async def create_panorama(request: PanoramaRequest, background_tasks: BackgroundTasks):
    """Start panorama generation task"""
    
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "status": "queued",
        "message": "Task queued",
        "result_path": None
    }
    
    background_tasks.add_task(generate_task, task_id, request)
    
    return PanoramaResponse(
        task_id=task_id,
        status="queued", 
        message="Panorama generation started"
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
    """Download generated panorama"""
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
        filename=f"panorama_{task_id[:8]}.png",
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

@app.post("/panorama_to_ply")
async def panorama_to_ply(request: PanoramaToPlyRequest):
    """Convert panorama image to PLY point cloud"""
    try:
        # Import required modules (working directory is already /workspace/DREAMSCENE360)
        from geo_predictors.pano_geo_predictor import PanoGeoPredictor
        from utils.camera_utils import img_coord_from_hw, img_coord_to_pano_direction

        # Check if panorama file exists
        if not os.path.exists(request.panorama_path):
            raise HTTPException(status_code=404, detail=f"Panorama file not found: {request.panorama_path}")

        # Load panorama image
        pano_img = Image.open(request.panorama_path).convert('RGB')
        pano_img = pano_img.resize((2048, 1024))
        pano_img_tensor = torch.from_numpy(np.array(pano_img)).float() / 255.0

        height, width = 1024, 2048

        # Initialize depth predictor
        geo_predictor = PanoGeoPredictor(gen_res=512)

        # Predict depth
        with torch.no_grad():
            distances = geo_predictor.predict(pano_img_tensor.cuda().permute(2, 0, 1))

        # Convert to point cloud
        pano_dirs = img_coord_to_pano_direction(img_coord_from_hw(height, width)).cuda()
        scale = distances.max().item() * 0.7
        distances /= scale
        pts = pano_dirs * distances.squeeze()[..., None]
        pts = pts.cpu().numpy().reshape(-1, 3)

        # Create point cloud with colors
        pcd = trimesh.PointCloud(pts, pano_img_tensor.reshape(-1, 3).cpu().numpy())

        # Save PLY file
        output_dir = os.path.dirname(request.panorama_path)
        output_name = request.output_name if request.output_name else "panorama_pointcloud.ply"
        ply_path = os.path.join(output_dir, output_name)
        pcd.export(ply_path)

        return {
            "status": "success",
            "message": "PLY file created successfully",
            "ply_path": ply_path
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error converting to PLY: {str(e)}")

@app.post("/train_gaussian", response_model=TrainGaussianResponse)
async def train_gaussian(request: TrainGaussianRequest):
    """Train Gaussian Splatting model from a panorama"""
    try:
        # Validate panorama path
        if not os.path.exists(request.panorama_path):
            raise HTTPException(status_code=404, detail=f"Panorama file not found: {request.panorama_path}")

        # Prepare source directory
        source_dir = os.path.dirname(request.panorama_path)
        source_name = os.path.splitext(os.path.basename(request.panorama_path))[0]

        # Generate a unique output directory for this training run
        scene_name = request.scene_name or f"gaussian_scene_{str(uuid.uuid4())[:8]}"
        # Use /workspace/plyoutput instead of outputs
        model_path = os.path.join("/workspace/plyoutput", scene_name)
        os.makedirs(model_path, exist_ok=True)

        # Prepare command arguments
        cmd_args = [
            "python", "train.py",
            "--source_path", source_dir,
            "--pano_path", request.panorama_path,
            "--model_path", model_path,
            "--iterations", str(request.iterations),
            "--save_iterations"
        ]
        # Add save_iterations as separate arguments
        cmd_args.extend([str(x) for x in request.save_iterations])
        cmd_args.extend([
            "--sh_degree", str(request.sh_degree)
        ])

        # Add optional white background flag
        if request.white_background:
            cmd_args.append("--white_background")

        # Run training with subprocess
        print(f"🚀 Executing: {' '.join(cmd_args)}")
        process = subprocess.run(
            cmd_args,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=3600
        )

        # Log subprocess output for debugging
        print(f"📊 Return code: {process.returncode}")
        if process.stdout:
            print(f"📤 STDOUT:\n{process.stdout[-2000:]}")  # Last 2000 chars
        if process.stderr:
            print(f"📛 STDERR:\n{process.stderr[-2000:]}")  # Last 2000 chars

        # Check if process failed
        if process.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"Training process failed with code {process.returncode}. Error: {process.stderr[:500]}"
            )

        # Parse output to find model path and PLY files
        model_path_match = re.search(r"Output folder: (.*)", process.stdout)
        initial_ply_path = os.path.join(source_dir, "point_cloud.ply")

        # Find trained PLY paths
        trained_ply_paths = []
        for save_iter in request.save_iterations:
            ply_path = os.path.join(model_path, f"point_cloud_iter_{save_iter}.ply")
            if os.path.exists(ply_path):
                trained_ply_paths.append(ply_path)

        return TrainGaussianResponse(
            status="success",
            message="Gaussian Splatting model training completed",
            model_path=model_path,
            initial_ply_path=initial_ply_path if os.path.exists(initial_ply_path) else None,
            trained_ply_paths=trained_ply_paths
        )

    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Training timed out after 1 hour")
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {e.stderr}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during training: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )