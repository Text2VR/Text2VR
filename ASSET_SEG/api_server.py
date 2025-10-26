#!/usr/bin/env python3
"""
Segmentation API for Asset Segmentation
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

# Add current directory to Python path for imports
sys.path.append('/app')

# Import segmentation functions  
from segment_panorama import main as segment_main
from types import SimpleNamespace

app = FastAPI(title="Text2VR Asset Segmentation API", version="1.0.0")

# Task storage
tasks = {}
WORKING_DIR = "/app/output"

class SegmentationRequest(BaseModel):
    panorama_path: str
    scene_name: str
    sam_checkpoint: str = "/app/checkpoints/sam_vit_h_4b8939.pth"
    openai_api_key: Optional[str] = None
    box_threshold: float = 0.20
    text_threshold: float = 0.15

class SegmentationResponse(BaseModel):
    task_id: str
    status: str
    message: str

class StatusResponse(BaseModel):
    task_id: str
    status: str
    message: str
    result_path: Optional[str] = None
    segmentation_data: Optional[dict] = None

def segment_task(task_id: str, request: SegmentationRequest):
    """Background task for panorama segmentation"""
    try:
        tasks[task_id]["status"] = "processing"
        tasks[task_id]["message"] = "Starting panorama segmentation..."
        
        # Setup paths
        output_dir = os.path.join(WORKING_DIR, request.scene_name)
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"🔍 Segmenting panorama: {request.panorama_path}")
        print(f"📁 Output directory: {output_dir}")
        
        # Create args object for segment_panorama.py main function
        args = SimpleNamespace(
            panorama_path=request.panorama_path,
            output_dir=output_dir,
            sam_checkpoint=request.sam_checkpoint,
            openai_api_key=request.openai_api_key or os.getenv("OPENAI_API_KEY"),
            box_threshold=request.box_threshold,
            text_threshold=request.text_threshold,
            # Add other default values that might be needed
            labels=None,
            max_prompts=12,  # Default from segment_panorama.py
            anchor_enable=True,
            min_area_ratio=0.00025,
            max_area_ratio=0.7,
            # Exclusion controls
            exclusion_use_mask=True,
            exclusion_mask_dilate_px=5,
            exclusion_overlap_drop=0.35,
            exclusion_box_th=0.15,
            exclusion_text_th=0.15,
            exclusion_pad_ratio=0.05,
            # Wrap NMS
            wrap_nms_iou=0.3,
            # Floor filter
            enable_floor_filter=False,
            floor_band_ratio=0.40,
            # Wrap mask merge & cleanup
            min_region_ratio=0.001,
            close_kernel=7
        )
        
        # Call the main segmentation function
        segment_main(args)
        
        # Check if output files were created
        output_json = os.path.join(output_dir, "segmentation_result.json")
        if os.path.exists(output_json):
            tasks[task_id]["status"] = "completed"
            tasks[task_id]["message"] = "Segmentation completed"
            tasks[task_id]["result_path"] = output_json
            
            # Load segmentation data
            with open(output_json, 'r') as f:
                tasks[task_id]["segmentation_data"] = json.load(f)
        else:
            # Look for any JSON files in the output directory
            json_files = [f for f in os.listdir(output_dir) if f.endswith('.json')]
            if json_files:
                result_file = os.path.join(output_dir, json_files[0])
                tasks[task_id]["status"] = "completed"
                tasks[task_id]["message"] = "Segmentation completed"
                tasks[task_id]["result_path"] = result_file
                
                with open(result_file, 'r') as f:
                    tasks[task_id]["segmentation_data"] = json.load(f)
            else:
                raise Exception("No segmentation results found")
            
    except Exception as e:
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["message"] = f"Error: {str(e)}"
        print(f"Task {task_id} failed: {e}")
        import traceback
        traceback.print_exc()

@app.on_event("startup")
async def startup_event():
    os.makedirs(WORKING_DIR, exist_ok=True)
    print("🚀 Asset Segmentation API started")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "asset-segmentation"}

@app.post("/segment", response_model=SegmentationResponse)
async def segment_panorama_endpoint(request: SegmentationRequest, background_tasks: BackgroundTasks):
    """Start panorama segmentation task"""
    
    # Validate input file exists
    if not os.path.exists(request.panorama_path):
        raise HTTPException(status_code=400, detail=f"Panorama file not found: {request.panorama_path}")
    
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "status": "queued",
        "message": "Task queued",
        "result_path": None,
        "segmentation_data": None
    }
    
    background_tasks.add_task(segment_task, task_id, request)
    
    return SegmentationResponse(
        task_id=task_id,
        status="queued", 
        message="Segmentation started"
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
        result_path=task.get("result_path"),
        segmentation_data=task.get("segmentation_data")
    )

@app.get("/result/{task_id}")
async def get_result(task_id: str):
    """Download segmentation result"""
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
        filename=f"segmentation_{task_id[:8]}.json",
        media_type="application/json"
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
        port=8002,
        log_level="info"
    )