#!/usr/bin/env python3
"""
Text2VR Orchestrator with FastAPI and LangGraph
"""

import os
import sys
import subprocess
import uuid
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# Ensure project root is available for package imports
PROJECT_ROOT = os.path.dirname(__file__)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from app.workflows.workflow import create_workflow

app = FastAPI(title="Text2VR Orchestrator", version="1.0.0")

# Task storage
tasks = {}
DATA_DIR = Path("/home/0in/workspace/Text2VR/data")
DATA_DIR.mkdir(exist_ok=True)

class PanoramaRequest(BaseModel):
    text: str
    scene_name: Optional[str] = None

class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: str

class StatusResponse(BaseModel):
    task_id: str
    status: str
    message: str
    panorama_path: Optional[str] = None

def run_panorama_generation(task_id: str, request: PanoramaRequest):
    """Background task for panorama generation using LangGraph"""
    try:
        tasks[task_id]["status"] = "processing"
        tasks[task_id]["message"] = "Starting panorama generation with LangGraph..."
        
        # Start DreamScene360 API server if not running
        try:
            subprocess.run(
                ["curl", "-f", "http://localhost:8001/health"],
                check=True,
                capture_output=True
            )
        except subprocess.CalledProcessError:
            # Start the API server
            tasks[task_id]["message"] = "Starting DreamScene360 API server..."
            dreamscene_process = subprocess.Popen(
                ["python", "/home/0in/workspace/Text2VR/DREAMSCENE360/api_server.py"],
                cwd="/home/0in/workspace/Text2VR/DREAMSCENE360"
            )
            import time
            time.sleep(10)  # Wait for server to start
        
        # Create and run LangGraph workflow
        workflow = create_workflow()
        
        scene_name = request.scene_name or f"scene_{task_id[:8]}"
        initial_state = {
            "user_input": request.text,
            "rewritten_query": "",
            "scene_name": scene_name,
            "panorama_path": "",
            "segmentation_data": {},
            "messages": []
        }
        
        tasks[task_id]["message"] = "Running LangGraph workflow..."
        result = workflow.invoke(initial_state)
        
        if result["panorama_path"] and os.path.exists(result["panorama_path"]):
            tasks[task_id]["status"] = "completed"
            tasks[task_id]["message"] = "Panorama generation completed"
            tasks[task_id]["panorama_path"] = result["panorama_path"]
            tasks[task_id]["scene_name"] = result["scene_name"]
        else:
            raise Exception("Panorama generation failed - no output file")
            
    except Exception as e:
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["message"] = f"Error: {str(e)}"
        print(f"Task {task_id} failed: {e}")

@app.on_event("startup")
async def startup_event():
    print("🚀 Text2VR Orchestrator started")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "text2vr-orchestrator"}

@app.post("/generate", response_model=TaskResponse)
async def generate_panorama(request: PanoramaRequest, background_tasks: BackgroundTasks):
    """Start panorama generation task"""
    
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "status": "queued",
        "message": "Task queued",
        "panorama_path": None,
        "scene_name": None
    }
    
    background_tasks.add_task(run_panorama_generation, task_id, request)
    
    return TaskResponse(
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
        panorama_path=task.get("panorama_path")
    )

@app.get("/panorama/{task_id}")
async def get_panorama(task_id: str):
    """Serve generated panorama image"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Task not completed: {task['status']}")
    
    panorama_path = task.get("panorama_path")
    if not panorama_path or not os.path.exists(panorama_path):
        raise HTTPException(status_code=404, detail="Panorama file not found")
    
    return FileResponse(
        path=panorama_path,
        filename=f"panorama_{task_id[:8]}.png",
        media_type="image/png"
    )

@app.get("/tasks")
async def list_tasks():
    """List all tasks"""
    return [
        {
            "task_id": task_id,
            "status": task["status"],
            "message": task["message"],
            "scene_name": task.get("scene_name")
        }
        for task_id, task in tasks.items()
    ]

if __name__ == "__main__":
    uvicorn.run("orchestrator:app", host="0.0.0.0", port=8000, reload=True)