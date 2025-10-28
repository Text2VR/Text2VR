"""
Panorama generation service using LangGraph
"""

import os
import sys
import subprocess
import logging
import asyncio
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

            # Check if mock mode is enabled
            if settings.MOCK_PIPELINE_MODE:
                logger.info(f"Running in MOCK mode for task {task_id}")
                await self._run_mock_workflow(task_id, request)
                return

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
    
    async def _run_mock_workflow(self, task_id: str, request: PanoramaRequest) -> None:
        """
        Run mock workflow that simulates the pipeline with fixed assets
        """
        try:
            logger.info(f"[MOCK] Starting mock workflow for task {task_id}")

            # Get task to extract scene name
            task = task_manager.get_task(task_id)
            scene_name = task.scene_name if task else f"scene_{task_id[:8]}"

            # Stage 1: Generate Panorama
            task_manager.update_task_status(
                task_id,
                TaskStatus.PROCESSING,
                "Generating panorama from text prompt..."
            )
            await asyncio.sleep(settings.MOCK_PANORAMA_DELAY)

            # Check if mock panorama file exists
            if not os.path.exists(settings.MOCK_PANORAMA_PATH):
                raise Exception(f"Mock panorama file not found at {settings.MOCK_PANORAMA_PATH}")

            task_manager.update_task_status(
                task_id,
                TaskStatus.PROCESSING,
                "Panorama generated successfully!",
                panorama_path=settings.MOCK_PANORAMA_PATH
            )
            logger.info(f"[MOCK] Panorama stage completed: {settings.MOCK_PANORAMA_PATH}")

            # Stage 2: Segmentation
            task_manager.update_task_status(
                task_id,
                TaskStatus.PROCESSING,
                "Segmenting objects in panorama..."
            )
            await asyncio.sleep(settings.MOCK_SEGMENTATION_DELAY)

            # Check if mock segmentation visualization exists
            if not os.path.exists(settings.MOCK_SEGMENTATION_VIZ_PATH):
                raise Exception(f"Mock segmentation visualization not found at {settings.MOCK_SEGMENTATION_VIZ_PATH}")

            task_manager.update_task_status(
                task_id,
                TaskStatus.PROCESSING,
                "Object segmentation completed!",
                panorama_path=settings.MOCK_PANORAMA_PATH,
                segmentation_visualization_path=settings.MOCK_SEGMENTATION_VIZ_PATH
            )
            logger.info(f"[MOCK] Segmentation stage completed: {settings.MOCK_SEGMENTATION_VIZ_PATH}")

            # Stage 3: Inpainting
            task_manager.update_task_status(
                task_id,
                TaskStatus.PROCESSING,
                "Inpainting panorama background..."
            )
            await asyncio.sleep(settings.MOCK_INPAINTING_DELAY)

            # Check if mock inpainted file exists
            if not os.path.exists(settings.MOCK_INPAINTED_PATH):
                raise Exception(f"Mock inpainted file not found at {settings.MOCK_INPAINTED_PATH}")

            # Final completion
            task_manager.update_task_status(
                task_id,
                TaskStatus.COMPLETED,
                "Mock panorama generation completed successfully!",
                panorama_path=settings.MOCK_PANORAMA_PATH,
                segmentation_visualization_path=settings.MOCK_SEGMENTATION_VIZ_PATH,
                inpainted_panorama_path=settings.MOCK_INPAINTED_PATH
            )
            logger.info(f"[MOCK] Mock workflow completed successfully for task {task_id}")

        except Exception as e:
            logger.error(f"[MOCK] Mock workflow failed for task {task_id}: {e}")
            raise

    async def _ensure_dreamscene_api(self) -> None:
        """Ensure DreamScene360 API server is running"""
        import httpx

        try:
            # Check if API is already running using httpx (async)
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{settings.DREAMSCENE_API_URL}/health", timeout=5.0)
                if response.status_code == 200:
                    logger.info("DreamScene360 API is already running")
                    return

        except (httpx.RequestError, httpx.TimeoutException) as e:
            logger.warning(f"DreamScene360 API not responding: {e}")

        # If API is not running, log warning but don't block
        # The API should be started via docker-compose
        logger.warning("DreamScene360 API is not running. Please ensure docker-compose services are started.")
        logger.info("Continuing with workflow - external API calls may fail if services are not available")
    
    async def _run_langgraph_workflow(self, task_id: str, request: PanoramaRequest) -> Optional[dict]:
        """Run the LangGraph workflow"""
        try:
            # Import LangGraph workflow
            from ..workflows.workflow import create_workflow
            import asyncio
            from functools import partial

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
