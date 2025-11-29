"""
Segmentation steps for the workflow
"""

import os
from langchain_core.messages import HumanMessage

from ...core.config import settings
from ...core.path_utils import to_container_path, get_container_task_paths
from ...services.docker_service import docker_service
from ...core.constants import SEGMENTATION_DEFAULTS
from ...models.workflow_state import WorkflowState
from ...services.image_processing import crop_assets_with_transparency
from ...clients.segmentation import SegmentationAPIClient
from ...services.task_manager import task_manager
from ...models.panorama import TaskStatus


def segmentation_node(state: WorkflowState) -> WorkflowState:
    """Run segmentation on the generated panorama image."""

    if not state.get("panorama_path"):
        return {
            **state,
            "segmentation_data": {},
            "messages": [
                HumanMessage(content="No panorama available for segmentation")
            ],
        }

    try:
        print(f"🔍 Starting segmentation for: {state['panorama_path']}")

        panorama_path = state["panorama_path"]
        
        # Convert host path to container path
        container_path = to_container_path(panorama_path)
        print(f"🔄 Container path: {container_path}")

        client = SegmentationAPIClient(base_url=settings.SEGMENTATION_API_URL)

        result = client.segment_panorama(
            panorama_path=container_path,
            scene_name=state["scene_name"],
            sam_checkpoint=state.get("sam_checkpoint", SEGMENTATION_DEFAULTS.sam_checkpoint),
            openai_api_key=state.get("openai_api_key", SEGMENTATION_DEFAULTS.openai_api_key),
        )

        print("✅ Segmentation completed")
        print(
            f"📋 Found objects: {list(result['segmentation_data'].get('prompts', {}).keys())}"
        )

        # Asset cropping with transparency
        try:
            print("🔪 Starting asset cropping with transparency...")

            # masking_output directory path
            # We use the host path for cropping since it runs in the app context
            segmentation_output_dir = settings.get_task_paths(state['scene_name'])["masking"]

            cropped_assets = crop_assets_with_transparency(
                panorama_path=panorama_path,
                segmentation_output_dir=str(segmentation_output_dir),
                scene_name=state['scene_name']
            )

            print(f"✅ Asset cropping completed: {len(cropped_assets)} asset types")
            print(f"📦 Cropped assets: {list(cropped_assets.keys())}")

        except Exception as crop_exc:
            print(f"⚠️ Asset cropping failed (non-critical): {crop_exc}")
            cropped_assets = {}

        # Stop segmentation container to free VRAM
        docker_service.stop_container("text2vr_segmentation_api")

        # Update task status
        try:
            if state.get("task_id"):
                scene_name = state["scene_name"]
                # Construct paths using settings
                masking_output_dir = settings.get_task_paths(scene_name)["masking"]
                segmentation_results_path = str(masking_output_dir / "results.json")
                segmentation_visualization_path = str(masking_output_dir / "visualizations/panorama_visualization.png")

                task_manager.update_task_status(
                    task_id=state["task_id"],
                    status=TaskStatus.PROCESSING,
                    message="Segmentation completed, starting inpainting...",
                    segmentation_results_path=segmentation_results_path,
                    segmentation_visualization_path=segmentation_visualization_path
                )
        except Exception as tm_exc:
            print(f"⚠️ Failed to update task manager: {tm_exc}")

        return {
            **state,
            "segmentation_data": result["segmentation_data"],
            "cropped_assets": cropped_assets,
            "messages": [
                HumanMessage(
                    content=(
                        "Segmentation completed: "
                        f"{len(result['segmentation_data'].get('prompts', {}))} objects found"
                    )
                )
            ],
        }

    except Exception as exc:
        print(f"❌ Segmentation failed: {exc}")
        return {
            **state,
            "segmentation_data": {},
            "messages": [
                HumanMessage(content=f"Segmentation failed: {str(exc)}")
            ],
        }
