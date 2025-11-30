"""
Inpainting steps for the workflow
"""

import os
from langchain_core.messages import HumanMessage

from ...core.config import settings
from ...core.path_utils import to_container_path, to_host_path, get_container_mask_dir
from ...services.docker_service import docker_service
from ...core.constants import INPAINTING_DEFAULTS
from ...models.workflow_state import WorkflowState
from ...clients.inpainting import InpaintingAPIClient
from ...services.task_manager import task_manager
from ...models.panorama import TaskStatus


def inpainting_node(state: WorkflowState) -> WorkflowState:
    """Run background inpainting on the panorama"""

    if not state.get("panorama_path"):
        return {
            **state,
            "inpainted_panorama_path": "",
            "messages": [
                HumanMessage(content="No panorama available for inpainting")
            ],
        }

    # Skip if no segmentation data
    if not state.get("segmentation_data"):
        print("⚠️ No segmentation data, skipping inpainting")
        return {
            **state,
            "inpainted_panorama_path": state["panorama_path"],
            "messages": [
                HumanMessage(content="Skipped inpainting (no segmentation)")
            ],
        }

    try:
        print(f"🎨 Starting background inpainting for: {state['panorama_path']}")

        panorama_path = state["panorama_path"]
        scene_name = state["scene_name"]

        # Convert to container paths
        container_panorama_path = to_container_path(panorama_path)

        # Mask directory in container
        container_mask_dir = get_container_mask_dir(scene_name)

        print(f"🔄 Container panorama path: {container_panorama_path}")
        print(f"🔄 Container mask dir: {container_mask_dir}")

        client = InpaintingAPIClient(base_url=settings.INPAINTING_API_URL)

        result_path = client.inpaint_panorama(
            panorama_path=container_panorama_path,
            mask_dir=container_mask_dir,
            scene_name=scene_name,
            model_id=state.get("inpaint_model_id", INPAINTING_DEFAULTS.model_id),
            prompt=state.get("inpaint_prompt", INPAINTING_DEFAULTS.prompt),
            neg_prompt=state.get("inpaint_neg_prompt", INPAINTING_DEFAULTS.neg_prompt),
            strength=state.get("inpaint_strength", INPAINTING_DEFAULTS.strength),
            guidance=state.get("inpaint_guidance", INPAINTING_DEFAULTS.guidance),
            steps=state.get("inpaint_steps", INPAINTING_DEFAULTS.steps),
            wrap_pad=state.get("inpaint_wrap_pad", INPAINTING_DEFAULTS.wrap_pad),
            dilate=state.get("inpaint_dilate", INPAINTING_DEFAULTS.dilate),
            feather=state.get("inpaint_feather", INPAINTING_DEFAULTS.feather),
            erase=state.get("inpaint_erase", INPAINTING_DEFAULTS.erase),
            seed=state.get("inpaint_seed", INPAINTING_DEFAULTS.seed),
            poll_interval=state.get("inpaint_poll_interval", INPAINTING_DEFAULTS.poll_interval),
            timeout=state.get("inpaint_timeout", INPAINTING_DEFAULTS.timeout),
        )

        print(f"✅ Inpainting completed: {result_path}")

        # Convert result path to host path
        host_result_path = to_host_path(result_path)

        # Stop inpainting container
        docker_service.stop_container("text2vr_inpainting_api")

        # Update task status
        try:
            if state.get("task_id") and host_result_path:
                task_manager.update_task_status(
                    task_id=state["task_id"],
                    status=TaskStatus.PROCESSING,
                    message="Inpainting completed, generating PLY...",
                    inpainted_panorama_path=host_result_path
                )
        except Exception as tm_exc:
            print(f"⚠️ Failed to update task manager: {tm_exc}")

        return {
            **state,
            "inpainted_panorama_path": host_result_path,
            "messages": [
                HumanMessage(content=f"Background inpainting completed: {host_result_path}")
            ],
        }

    except Exception as exc:
        print(f"❌ Inpainting failed: {exc}")
        # Fallback to original panorama
        return {
            **state,
            "inpainted_panorama_path": state["panorama_path"],
            "messages": [
                HumanMessage(content=f"Inpainting failed (using original): {str(exc)}")
            ],
        }
