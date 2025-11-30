"""
Panorama generation service using LangGraph
"""

import os
import logging
import asyncio
from functools import partial
from typing import Optional

from ..core.config import settings
from ..models.panorama import PanoramaRequest, TaskStatus
from .task_manager import task_manager
from .docker_service import docker_service

logger = logging.getLogger(__name__)


class PanoramaService:
    """Service for generating panoramas"""
    
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

            result_state = await self._run_langgraph_workflow(task_id, request)
            result_path = result_state.get("panorama_path") if result_state else None
            asset_3d_paths = result_state.get("asset_3d_paths") if result_state else None
            ply_path = result_state.get("ply_path") if result_state else None

            if result_path and os.path.exists(result_path):
                task_manager.update_task_status(
                    task_id,
                    TaskStatus.COMPLETED,
                    "Panorama generation completed successfully",
                    panorama_path=result_path,
                    asset_3d_paths=asset_3d_paths,
                    ply_path=ply_path
                )
                logger.info(f"Task {task_id} completed: {result_path}")
            else:
                # Even if result_path is missing, check if we have other artifacts or if it failed gracefully
                if result_state and result_state.get("messages"):
                    last_msg = result_state["messages"][-1].content
                    if "failed" in last_msg.lower():
                        raise Exception(f"Workflow reported failure: {last_msg}")
                
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
        import httpx

        # Try to start the container using DockerService
        # The container name is assumed to be 'text2vr_panorama_api' based on docker-compose
        docker_service.start_container("text2vr_panorama_api")

        try:
            # Check if API is responding
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{settings.DREAMSCENE_API_URL}/health", timeout=5.0)
                if response.status_code == 200:
                    logger.info("DreamScene360 API is running")
                    return

        except (httpx.RequestError, httpx.TimeoutException) as e:
            logger.warning(f"DreamScene360 API not responding immediately: {e}")

        logger.info("Continuing with workflow - API might be starting up")
    
    async def _run_langgraph_workflow(self, task_id: str, request: PanoramaRequest) -> Optional[dict]:
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

            # Run workflow in executor to avoid blocking
            logger.info(f"Running LangGraph workflow for task {task_id}")
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                partial(workflow.invoke, initial_state)
            )

            task_manager.update_task_status(
                task_id,
                TaskStatus.PROCESSING,
                "Finalizing panorama generation..."
            )

            logger.info(f"LangGraph workflow completed for task {task_id}")
            return result
            
        except ImportError as e:
            logger.error(f"Failed to import LangGraph workflow: {e}")
            raise Exception(f"LangGraph workflow import failed: {e}")
        except Exception as e:
            logger.error(f"LangGraph workflow execution failed: {e}")
            raise
    
    def cleanup(self) -> None:
        """Cleanup resources"""
        # No longer managing subprocess directly, so nothing to kill here
        pass


# Global panorama service instance
panorama_service = PanoramaService()
