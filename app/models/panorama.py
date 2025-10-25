"""
Pydantic models for panorama generation
"""

from datetime import datetime
from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    """Task status enumeration"""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class PanoramaRequest(BaseModel):
    """Request model for panorama generation"""
    text: str = Field(..., min_length=1, description="Description of the panoramic scene")
    scene_name: Optional[str] = Field(None, description="Optional scene name")
    use_self_refinement: bool = Field(False, description="Enable self-refinement")
    num_prompt: int = Field(3, ge=1, le=10, description="Number of prompts to generate")
    max_rounds: int = Field(3, ge=1, le=5, description="Maximum refinement rounds")


class TaskResponse(BaseModel):
    """Response model for task creation"""
    task_id: str
    status: TaskStatus
    message: str
    created_at: datetime = Field(default_factory=datetime.now)


class StatusResponse(BaseModel):
    """Response model for task status"""
    task_id: str
    status: TaskStatus
    message: str
    panorama_path: Optional[str] = None
    segmentation_results_path: Optional[str] = None
    segmentation_visualization_path: Optional[str] = None
    inpainted_panorama_path: Optional[str] = None
    scene_name: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    progress: Optional[float] = Field(None, ge=0.0, le=1.0, description="Progress percentage")


class TaskInfo(BaseModel):
    """Internal task information"""
    task_id: str
    status: TaskStatus
    message: str
    panorama_path: Optional[str] = None
    segmentation_results_path: Optional[str] = None
    segmentation_visualization_path: Optional[str] = None
    inpainted_panorama_path: Optional[str] = None
    scene_name: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    request: PanoramaRequest
    error_details: Optional[str] = None


class TaskListResponse(BaseModel):
    """Response model for listing tasks"""
    tasks: List[StatusResponse]
    total: int