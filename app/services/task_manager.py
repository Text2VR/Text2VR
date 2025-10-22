"""
Task management service for panorama generation
"""

import uuid
from datetime import datetime
from typing import Dict, Optional, List
from ..models.panorama import TaskInfo, TaskStatus, PanoramaRequest


class TaskManager:
    """Manages panorama generation tasks"""
    
    def __init__(self):
        self._tasks: Dict[str, TaskInfo] = {}
    
    def create_task(self, request: PanoramaRequest) -> str:
        """Create a new task"""
        task_id = str(uuid.uuid4())
        
        task_info = TaskInfo(
            task_id=task_id,
            status=TaskStatus.QUEUED,
            message="Task queued for processing",
            request=request,
            scene_name=request.scene_name or f"scene_{task_id[:8]}"
        )
        
        self._tasks[task_id] = task_info
        return task_id
    
    def get_task(self, task_id: str) -> Optional[TaskInfo]:
        """Get task by ID"""
        return self._tasks.get(task_id)
    
    def update_task_status(
        self,
        task_id: str,
        status: TaskStatus,
        message: str,
        panorama_path: Optional[str] = None,
        segmentation_results_path: Optional[str] = None,
        segmentation_visualization_path: Optional[str] = None,
        inpainted_panorama_path: Optional[str] = None,
        error_details: Optional[str] = None
    ) -> bool:
        """Update task status"""
        if task_id not in self._tasks:
            return False

        task = self._tasks[task_id]
        task.status = status
        task.message = message
        task.updated_at = datetime.now()

        if panorama_path:
            task.panorama_path = panorama_path

        if segmentation_results_path:
            task.segmentation_results_path = segmentation_results_path

        if segmentation_visualization_path:
            task.segmentation_visualization_path = segmentation_visualization_path

        if inpainted_panorama_path:
            task.inpainted_panorama_path = inpainted_panorama_path

        if error_details:
            task.error_details = error_details

        return True
    
    def list_tasks(self, limit: Optional[int] = None) -> List[TaskInfo]:
        """List all tasks, optionally limited"""
        tasks = list(self._tasks.values())
        tasks.sort(key=lambda x: x.created_at, reverse=True)
        
        if limit:
            tasks = tasks[:limit]
        
        return tasks
    
    def delete_task(self, task_id: str) -> bool:
        """Delete a task"""
        if task_id in self._tasks:
            del self._tasks[task_id]
            return True
        return False
    
    def cleanup_old_tasks(self, max_age_hours: int = 24) -> int:
        """Clean up old tasks"""
        cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
        
        old_tasks = [
            task_id for task_id, task in self._tasks.items()
            if task.created_at.timestamp() < cutoff_time
        ]
        
        for task_id in old_tasks:
            del self._tasks[task_id]
        
        return len(old_tasks)


# Global task manager instance
task_manager = TaskManager()