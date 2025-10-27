"""
Unity-facing asset download endpoints.

Provides direct access to generated GLB and PLY outputs so that external
clients (e.g., Unity projects) can fetch assets over HTTP.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Optional
from zipfile import ZipFile, ZIP_DEFLATED

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse

from ..models.panorama import TaskStatus
from ..services.task_manager import task_manager

router = APIRouter(prefix="/unity", tags=["unity"])

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GLB_ROOT = PROJECT_ROOT / "output" / "3d_assets"
PLY_ROOT = PROJECT_ROOT / "plyoutput"


def _validate_within_root(file_path: Path, root: Path) -> None:
    """Ensure the resolved path stays under the expected root."""
    try:
        file_path.relative_to(root)
    except ValueError:  # pragma: no cover - safety guard
        raise HTTPException(status_code=400, detail="Invalid path requested")


def _get_latest_completed_task():
    """Return the most recent task that reached COMPLETED status."""
    for task in task_manager.list_tasks():
        if task.status == TaskStatus.COMPLETED:
            return task
    return None


@router.get(
    "/{scene_name}/assets/{asset_name}.glb",
    response_class=FileResponse,
    summary="Download generated GLB asset",
)
def download_glb_asset(scene_name: str, asset_name: str) -> FileResponse:
    """Return a GLB asset produced by the TRELLIS stage."""
    file_path = (GLB_ROOT / scene_name / f"{asset_name}.glb").resolve()

    _validate_within_root(file_path, GLB_ROOT)

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"GLB asset not found for scene '{scene_name}' and asset '{asset_name}'",
        )

    return FileResponse(
        path=file_path,
        filename=file_path.name,
        media_type="application/octet-stream",
    )


@router.get(
    "/{scene_name}/scene.ply",
    response_class=FileResponse,
    summary="Download trained Gaussian PLY",
)
def download_ply_scene(
    scene_name: str,
    iteration: Optional[int] = Query(
        default=None,
        description="Specific iteration to fetch (e.g., 5000). Uses latest if omitted.",
    ),
) -> FileResponse:
    """Return a Gaussian Splatting PLY file for the requested scene."""
    scene_dir = (PLY_ROOT / scene_name).resolve()
    _validate_within_root(scene_dir, PLY_ROOT)

    if not scene_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No PLY output found for scene '{scene_name}'",
        )

    candidates: list[Path] = []
    if iteration is not None:
        candidates.append(scene_dir / f"point_cloud_iter_{iteration}.ply")
    else:
        candidates.extend(sorted(scene_dir.glob("point_cloud_iter_*.ply")))
        candidates.append(scene_dir / "point_cloud.ply")

    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.exists():
            _validate_within_root(candidate, PLY_ROOT)
            return FileResponse(
                path=candidate,
                filename=candidate.name,
                media_type="application/octet-stream",
            )

    raise HTTPException(
        status_code=404,
        detail=f"No PLY file available for scene '{scene_name}'",
    )


@router.get(
    "/latest/assets.zip",
    response_class=StreamingResponse,
    summary="Download GLB assets from the latest completed task",
)
def download_latest_glb_assets() -> StreamingResponse:
    """
    Bundle GLB assets from the most recent completed task into a ZIP archive.
    """
    task = _get_latest_completed_task()
    if not task or not task.asset_3d_paths:
        raise HTTPException(
            status_code=404,
            detail="No completed task with GLB assets found",
        )

    buffer = BytesIO()
    with ZipFile(buffer, mode="w", compression=ZIP_DEFLATED) as zip_file:
        for asset_name, asset_path in task.asset_3d_paths.items():
            path_obj = Path(asset_path)
            if not path_obj.exists():
                continue
            try:
                _validate_within_root(path_obj.resolve(), GLB_ROOT)
            except HTTPException:
                continue
            zip_file.write(path_obj, arcname=path_obj.name)

    if buffer.getbuffer().nbytes == 0:
        raise HTTPException(
            status_code=404,
            detail="Completed task has no available GLB files on disk",
        )

    buffer.seek(0)
    filename = f"{task.scene_name or 'scene'}_assets.zip"
    return StreamingResponse(
        buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get(
    "/latest/scene.ply",
    response_class=FileResponse,
    summary="Download PLY from the latest completed task",
)
def download_latest_ply() -> FileResponse:
    """Serve the PLY file generated by the most recent completed task."""
    task = _get_latest_completed_task()
    if not task or not task.ply_path:
        raise HTTPException(
            status_code=404,
            detail="No completed task with a PLY output found",
        )

    file_path = Path(task.ply_path).resolve()
    _validate_within_root(file_path, PLY_ROOT)

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Latest task PLY file is missing on disk",
        )

    return FileResponse(
        path=file_path,
        filename=file_path.name,
        media_type="application/octet-stream",
    )
