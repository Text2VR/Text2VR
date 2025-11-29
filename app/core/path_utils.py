"""
Path utility functions for converting between Host and Container paths.
"""

import os
from pathlib import Path
from .config import settings

def to_container_path(host_path: str) -> str:
    """
    Convert a host path to a container path.
    Assumes the workspace root is mounted to /workspace in the container.
    """
    if not host_path:
        return ""
        
    host_path_obj = Path(host_path).resolve()
    workspace_root = settings.WORKSPACE_ROOT.resolve()
    
    try:
        # Check if the path is relative to the workspace
        relative_path = host_path_obj.relative_to(workspace_root)
        return f"/workspace/{relative_path}"
    except ValueError:
        # Path is not inside the workspace, return as is or handle error
        # For now, we assume all relevant paths are within the workspace
        return host_path

def to_host_path(container_path: str) -> str:
    """
    Convert a container path to a host path.
    Assumes /workspace in the container maps to the workspace root on host.
    """
    if not container_path:
        return ""

    if container_path.startswith("/workspace/"):
        relative_path = container_path.replace("/workspace/", "", 1)
        return str(settings.WORKSPACE_ROOT / relative_path)

    return container_path

def get_container_task_paths(scene_name: str) -> dict:
    """
    Get container paths for a specific task.

    Args:
        scene_name: Name of the scene/task

    Returns:
        Dictionary containing all container paths for the task
    """
    base = f"/workspace/output/{scene_name}"
    return {
        "task_dir": base,
        "panorama": f"{base}/panorama.png",
        "stitch": f"{base}/stitch",
        "masking": f"{base}/masking",
        "inpainted": f"{base}/inpainted.png",
        "assets": f"{base}/assets",
        "ply": f"{base}/ply",
        "3d": f"{base}/3d",
    }

def get_container_mask_dir(scene_name: str) -> str:
    """
    Get container path for mask directory.

    Args:
        scene_name: Name of the scene/task

    Returns:
        Container path to the masks directory
    """
    return f"/workspace/output/{scene_name}/masking/masks"
