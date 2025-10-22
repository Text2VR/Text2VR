"""
Panorama generation service using LangGraph
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from typing import Optional

from ..config import settings
from ..models.panorama import PanoramaRequest, TaskStatus
from .task_manager import task_manager

# Workflows are now part of the app package

logger = logging.getLogger(__name__)


class PanoramaService:
    """Service for generating panoramas"""
    
    def __init__(self):
        self._dreamscene_process = None
    
    async def generate_panorama(self, task_id: str, request: PanoramaRequest) -> None:
        """
        Generate panorama using LangGraph workflow
        This runs as a background task
        """
        try:
            # Update task status
            task_manager.update_task_status(
                task_id, 
                TaskStatus.PROCESSING, 
                "Starting panorama generation with LangGraph..."
            )
            
            # Ensure DreamScene360 API is running
            await self._ensure_dreamscene_api()
            
            # Import and run LangGraph workflow
            task_manager.update_task_status(
                task_id,
                TaskStatus.PROCESSING,
                "Rewriting query with AI..."
            )
            
            result_path = await self._run_langgraph_workflow(task_id, request)
            
            if result_path and os.path.exists(result_path):
                task_manager.update_task_status(
                    task_id,
                    TaskStatus.COMPLETED,
                    "Panorama generation completed successfully",
                    panorama_path=result_path
                )
                logger.info(f"Task {task_id} completed: {result_path}")
            else:
                raise Exception("Panorama generation failed - no output file created")
                
        except Exception as e:
            error_msg = f"Generation failed: {str(e)}"
            task_manager.update_task_status(
                task_id,
                TaskStatus.FAILED,
                error_msg,
                error_details=str(e)
            )
            logger.error(f"Task {task_id} failed: {e}", exc_info=True)
    
    async def _ensure_dreamscene_api(self) -> None:
        """Ensure DreamScene360 API server is running"""
        try:
            # Check if API is already running
            result = subprocess.run(
                ["curl", "-f", f"{settings.DREAMSCENE_API_URL}/health"],
                capture_output=True,
                timeout=5
            )
            
            if result.returncode == 0:
                logger.info("DreamScene360 API is already running")
                return
                
        except subprocess.TimeoutExpired:
            logger.warning("Health check timed out")
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
        
        # Start the API server if not running
        logger.info("Starting DreamScene360 API server...")
        
        self._dreamscene_process = subprocess.Popen(
            ["python", "api_server.py"],
            cwd=settings.dreamscene_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for server to start
        import asyncio
        await asyncio.sleep(10)
        
        # Verify it started
        try:
            result = subprocess.run(
                ["curl", "-f", f"{settings.DREAMSCENE_API_URL}/health"],
                capture_output=True,
                timeout=10
            )
            
            if result.returncode != 0:
                raise Exception("Failed to start DreamScene360 API server")
            
            logger.info("DreamScene360 API server started successfully")
            
        except Exception as e:
            logger.error(f"Failed to verify API server: {e}")
            raise
    
    async def _run_langgraph_workflow(self, task_id: str, request: PanoramaRequest) -> Optional[str]:
        """Run the LangGraph workflow"""
        try:
            # Import LangGraph workflow
            from ..workflows.workflow import create_workflow
            
            task_manager.update_task_status(
                task_id,
                TaskStatus.PROCESSING,
                "Preparing AI workflow..."
            )
            
            # Create workflow
            workflow = create_workflow()
            
            # Prepare initial state
            task = task_manager.get_task(task_id)
            scene_name = task.scene_name if task else f"scene_{task_id[:8]}"
            
            initial_state = {
                "task_id": task_id,
                "user_input": request.text,
                "rewritten_query": "",
                "scene_name": scene_name,
                "panorama_path": "",
                "segmentation_data": {},
                "messages": []
            }
            
            task_manager.update_task_status(
                task_id,
                TaskStatus.PROCESSING,
                "Generating panorama with AI models..."
            )
            
            # Run workflow
            logger.info(f"Running LangGraph workflow for task {task_id}")
            result = workflow.invoke(initial_state)
            
            task_manager.update_task_status(
                task_id,
                TaskStatus.PROCESSING,
                "Finalizing panorama generation..."
            )
            
            logger.info(f"LangGraph workflow completed for task {task_id}")
            return result.get("panorama_path")
            
        except ImportError as e:
            logger.error(f"Failed to import LangGraph workflow: {e}")
            raise Exception(f"LangGraph workflow import failed: {e}")
        except Exception as e:
            logger.error(f"LangGraph workflow execution failed: {e}")
            raise
    
    def cleanup(self) -> None:
        """Cleanup resources"""
        if self._dreamscene_process:
            try:
                self._dreamscene_process.terminate()
                self._dreamscene_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._dreamscene_process.kill()
            except Exception as e:
                logger.error(f"Error cleaning up DreamScene process: {e}")


# Global panorama service instance
panorama_service = PanoramaService()