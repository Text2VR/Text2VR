"""
Asset serving API routes
"""

import os
import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ..core.config import settings
from ..models.panorama import TaskStatus
from ..services.task_manager import task_manager

logger = logging.getLogger(__name__)
router = APIRouter(tags=["assets"])


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
        # Fallback: try to find the latest panorama from task-specific stitch directory
        logger.warning(f"Panorama file not found at {task.panorama_path}, trying stitch fallback")

        task_paths = settings.get_task_paths(task.scene_name)
        stitch_output_dir = task_paths["stitch"]
        if stitch_output_dir.exists():
            png_files = list(stitch_output_dir.glob("im_*.png"))
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
    task = task_manager.get_task(task_id)

    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

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

    # Find segmented assets in task-specific assets directory
    task_paths = settings.get_task_paths(scene_name)
    seged_assets_dir = task_paths["assets"]

    logger.info(f"Looking for segmented assets in: {seged_assets_dir}")
    logger.info(f"Directory exists: {seged_assets_dir.exists()}")

    if not seged_assets_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Segmented assets not yet available (looking for {scene_name})"
        )

    # Get all PNG files in the directory
    asset_files = list(seged_assets_dir.glob("*.png"))

    if not asset_files:
        raise HTTPException(
            status_code=404,
            detail="No segmented assets found"
        )

    # Return list of asset names and URLs
    assets = []
    for file_path in asset_files:
        asset_name = file_path.stem
        assets.append({
            "name": asset_name,
            "url": f"/segmentation/{task_id}/asset/{asset_name}"
        })

    return {"assets": assets}


@router.get("/segmentation/{task_id}/asset/{asset_name}")
async def get_segmentation_asset(task_id: str, asset_name: str):
    """Download a specific segmented asset image"""
    task = task_manager.get_task(task_id)

    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

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

    # Construct the asset path from task-specific assets directory
    task_paths = settings.get_task_paths(scene_name)
    asset_path = task_paths["assets"] / f"{asset_name}.png"

    if not asset_path.exists():
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
