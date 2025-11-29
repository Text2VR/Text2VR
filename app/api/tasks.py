"""
Task management API routes
"""

import logging
from fastapi import APIRouter, HTTPException, BackgroundTasks

from ..models.panorama import (
    PanoramaRequest, 
    TaskResponse, 
    StatusResponse, 
    TaskListResponse,
    TaskStatus
)
from ..services.task_manager import task_manager
from ..services.panorama_service import panorama_service

logger = logging.getLogger(__name__)
router = APIRouter(tags=["tasks"])


@router.post("/generate", response_model=TaskResponse)
async def generate_panorama(request: PanoramaRequest, background_tasks: BackgroundTasks):
    """Start panorama generation task"""
    try:
        # Create task
        task_id = task_manager.create_task(request)
        
        # Start background generation
        background_tasks.add_task(panorama_service.generate_panorama, task_id, request)
        
        logger.info(f"Started panorama generation task: {task_id}")
        
        return TaskResponse(
            task_id=task_id,
            status=TaskStatus.QUEUED,
            message="Panorama generation task created and queued"
        )
        
    except Exception as e:
        logger.error(f"Failed to create panorama generation task: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to start panorama generation: {str(e)}"
        )


@router.get("/status/{task_id}", response_model=StatusResponse)
async def get_task_status(task_id: str):
    """Get task status and progress"""
    task = task_manager.get_task(task_id)
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return StatusResponse(
        task_id=task.task_id,
        status=task.status,
        message=task.message,
        panorama_path=task.panorama_path,
        segmentation_results_path=task.segmentation_results_path,
        segmentation_visualization_path=task.segmentation_visualization_path,
        inpainted_panorama_path=task.inpainted_panorama_path,
        scene_name=task.scene_name,
        asset_3d_paths=task.asset_3d_paths,
        ply_path=task.ply_path,
        created_at=task.created_at,
        updated_at=task.updated_at
    )


@router.get("/tasks", response_model=TaskListResponse)
async def list_tasks(limit: int = 50):
    """List all panorama generation tasks"""
    try:
        tasks = task_manager.list_tasks(limit=limit)
        
        task_responses = [
            StatusResponse(
                task_id=task.task_id,
                status=task.status,
                message=task.message,
                panorama_path=task.panorama_path,
                scene_name=task.scene_name,
                asset_3d_paths=task.asset_3d_paths,
                ply_path=task.ply_path,
                created_at=task.created_at,
                updated_at=task.updated_at
            )
            for task in tasks
        ]
        
        return TaskListResponse(
            tasks=task_responses,
            total=len(task_responses)
        )
        
    except Exception as e:
        logger.error(f"Failed to list tasks: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve tasks")


@router.delete("/tasks/{task_id}")
async def delete_task(task_id: str):
    """Delete a task"""
    if not task_manager.delete_task(task_id):
        raise HTTPException(status_code=404, detail="Task not found")
    
    logger.info(f"Deleted task: {task_id}")
    return {"message": "Task deleted successfully"}


@router.post("/tasks/cleanup")
async def cleanup_old_tasks(max_age_hours: int = 24):
    """Clean up old completed/failed tasks"""
    try:
        deleted_count = task_manager.cleanup_old_tasks(max_age_hours)
        logger.info(f"Cleaned up {deleted_count} old tasks")
        
        return {
            "message": f"Cleaned up {deleted_count} old tasks",
            "deleted_count": deleted_count
        }
        
    except Exception as e:
        logger.error(f"Failed to cleanup tasks: {e}")
        raise HTTPException(status_code=500, detail="Failed to cleanup tasks")
