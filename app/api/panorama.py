"""
Panorama API routes
"""

import os
import logging
import glob
from typing import List
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse

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
router = APIRouter(tags=["panorama"])


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


@router.get("/panorama/{task_id}")
async def download_panorama(task_id: str):
    """Download generated panorama image"""
    task = task_manager.get_task(task_id)

    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    # Allow access during PROCESSING for realtime updates
    if task.status == TaskStatus.FAILED:
        raise HTTPException(
            status_code=400,
            detail=f"Task failed: {task.message}"
        )

    if not task.panorama_path or not os.path.exists(task.panorama_path):
        # Fallback: try to find the latest panorama from stitch_output
        logger.warning(f"Panorama file not found at {task.panorama_path}, trying stitch_output fallback")
        
        stitch_output_dir = "/home/0in/workspace/Text2VR/stitch_output"
        if os.path.exists(stitch_output_dir):
            png_files = glob.glob(os.path.join(stitch_output_dir, "im_*.png"))
            if png_files:
                # Get the most recent file
                latest_file = max(png_files, key=os.path.getmtime)
                logger.info(f"Using fallback panorama: {latest_file}")
                
                return FileResponse(
                    path=latest_file,
                    filename=f"panorama_{task.scene_name}.png",
                    media_type="image/png"
                )
        
        logger.error(f"No panorama file found for task {task_id}")
        raise HTTPException(status_code=404, detail="Panorama file not found")
    
    return FileResponse(
        path=task.panorama_path,
        filename=f"panorama_{task.scene_name}.png",
        media_type="image/png"
    )


@router.get("/segmentation/{task_id}")
async def get_segmentation_assets(task_id: str):
    """Get list of segmented asset images"""
    from ..config import settings

    task = task_manager.get_task(task_id)

    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    # Check if Mock mode is enabled
    if settings.MOCK_PIPELINE_MODE:
        seged_assets_dir = settings.MOCK_SEGMENTATION_DIR
    else:
        # Extract scene name from panorama_path or segmentation_results_path
        scene_name = None
        if task.panorama_path:
            # Extract from path like /home/.../data/scene_xxx/panorama.png
            import re
            match = re.search(r'scene_[a-f0-9]+', task.panorama_path)
            if match:
                scene_name = match.group(0)

        if not scene_name and task.segmentation_results_path:
            import re
            match = re.search(r'scene_[a-f0-9]+', task.segmentation_results_path)
            if match:
                scene_name = match.group(0)

        if not scene_name:
            raise HTTPException(
                status_code=400,
                detail="Scene name not available"
            )

        # Find segmented assets in seged_assets/{scene_name}/
        seged_assets_dir = f"/home/0in/workspace/Text2VR/seged_assets/{scene_name}"

    logger.info(f"Looking for segmented assets in: {seged_assets_dir}")
    logger.info(f"Directory exists: {os.path.exists(seged_assets_dir)}")

    if not os.path.exists(seged_assets_dir):
        raise HTTPException(
            status_code=404,
            detail=f"Segmented assets not yet available (looking for {scene_name})"
        )

    # Get all PNG files in the directory
    asset_files = glob.glob(os.path.join(seged_assets_dir, "*.png"))

    if not asset_files:
        raise HTTPException(
            status_code=404,
            detail="No segmented assets found"
        )

    # Return list of asset names and URLs
    assets = []
    for file_path in asset_files:
        asset_name = os.path.basename(file_path).replace(".png", "")
        assets.append({
            "name": asset_name,
            "url": f"/segmentation/{task_id}/asset/{asset_name}"
        })

    return {"assets": assets}


@router.get("/segmentation/{task_id}/asset/{asset_name}")
async def get_segmentation_asset(task_id: str, asset_name: str):
    """Download a specific segmented asset image"""
    from ..config import settings

    task = task_manager.get_task(task_id)

    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    # Check if Mock mode is enabled
    if settings.MOCK_PIPELINE_MODE:
        asset_path = f"{settings.MOCK_SEGMENTATION_DIR}/{asset_name}.png"
    else:
        # Extract scene name from panorama_path or segmentation_results_path
        scene_name = None
        if task.panorama_path:
            import re
            match = re.search(r'scene_[a-f0-9]+', task.panorama_path)
            if match:
                scene_name = match.group(0)

        if not scene_name and task.segmentation_results_path:
            import re
            match = re.search(r'scene_[a-f0-9]+', task.segmentation_results_path)
            if match:
                scene_name = match.group(0)

        if not scene_name:
            raise HTTPException(status_code=400, detail="Scene name not available")

        # Construct the asset path
        asset_path = f"/home/0in/workspace/Text2VR/seged_assets/{scene_name}/{asset_name}.png"

    if not os.path.exists(asset_path):
        raise HTTPException(status_code=404, detail="Asset file not found")

    return FileResponse(
        path=asset_path,
        filename=f"{asset_name}.png",
        media_type="image/png"
    )


@router.get("/segmentation/{task_id}/json")
async def get_segmentation_json(task_id: str):
    """Get segmentation metadata (results.json)"""
    task = task_manager.get_task(task_id)

    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if not task.segmentation_results_path:
        raise HTTPException(
            status_code=400,
            detail="Segmentation results not yet available"
        )

    if not os.path.exists(task.segmentation_results_path):
        raise HTTPException(status_code=404, detail="Segmentation results file not found")

    return FileResponse(
        path=task.segmentation_results_path,
        filename=f"segmentation_{task.scene_name}.json",
        media_type="application/json"
    )


@router.get("/inpainted/{task_id}")
async def get_inpainted_panorama(task_id: str):
    """Download inpainted panorama image"""
    task = task_manager.get_task(task_id)

    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if not task.inpainted_panorama_path:
        raise HTTPException(
            status_code=400,
            detail="Inpainted panorama not yet available"
        )

    if not os.path.exists(task.inpainted_panorama_path):
        raise HTTPException(status_code=404, detail="Inpainted panorama file not found")

    return FileResponse(
        path=task.inpainted_panorama_path,
        filename=f"inpainted_{task.scene_name}.png",
        media_type="image/png"
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
