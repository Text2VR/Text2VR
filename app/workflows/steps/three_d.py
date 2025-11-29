"""
3D generation steps for the workflow (Trellis & Gaussian Splatting)
"""

import os
import time
import requests
from typing import List, Tuple
from langchain_core.messages import HumanMessage

from ...core.config import settings
from ...core.path_utils import to_container_path, to_host_path
from ...core.constants import TRELLIS_DEFAULTS, GAUSSIAN_DEFAULTS
from ...services.docker_service import docker_service
from ...clients.trellis import TrellisAPIClient
from ...models.workflow_state import WorkflowState
from ...services.task_manager import task_manager
from ...models.panorama import TaskStatus


def asset_3d_generation_node(state: WorkflowState) -> WorkflowState:
    """Generate 3D GLB assets from cropped segmentation images using TRELLIS API."""

    cropped_assets = state.get("cropped_assets", {})

    if not cropped_assets:
        print("⚠️ No cropped assets found, skipping 3D generation")
        return {
            **state,
            "asset_3d_paths": {},
            "messages": [
                HumanMessage(content="No assets to convert to 3D")
            ],
        }

    try:
        print(f"🎲 Starting 3D asset generation for {len(cropped_assets)} assets")

        # Start TRELLIS container
        docker_service.start_container("text2vr_trellis_api")
        
        # Wait for pipeline to load
        time.sleep(10)

        client = TrellisAPIClient(base_url=settings.TRELLIS_API_URL)

        # Health check
        try:
            health = client.health_check()
            if health.get('status') != 'healthy':
                raise Exception(f"TRELLIS API not healthy: {health}")
            print(f"✅ TRELLIS API ready (GPU memory: {health.get('gpu_memory_used', 0):.2f}GB)")
        except Exception as e:
            raise Exception(f"TRELLIS API health check failed: {e}")

        asset_3d_paths = {}
        scene_name = state["scene_name"]

        # Generate 3D for each asset
        for asset_name, image_paths in cropped_assets.items():
            if not image_paths:
                continue

            # Use first image
            image_path = image_paths[0]

            print(f"🎯 Generating 3D for: {asset_name}")

            # Output directory
            output_dir = settings.get_task_paths(scene_name)["3d"]
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(output_dir / f"{asset_name}.glb")

            try:
                # TRELLIS API call
                result_path = client.generate_3d_asset(
                    image_path=image_path,
                    asset_name=asset_name,
                    output_path=output_path
                )

                asset_3d_paths[asset_name] = result_path
                print(f"✅ 3D asset created: {result_path}")

            except Exception as asset_exc:
                print(f"❌ Failed to generate 3D for {asset_name}: {asset_exc}")
                continue

        # Stop TRELLIS container
        docker_service.stop_container("text2vr_trellis_api")

        # Update task status
        try:
            if state.get("task_id") and asset_3d_paths:
                task_manager.update_task_status(
                    task_id=state["task_id"],
                    status=TaskStatus.PROCESSING,
                    message=f"3D assets generated ({len(asset_3d_paths)} assets), starting inpainting...",
                )
        except Exception as tm_exc:
            print(f"⚠️ Failed to update task manager: {tm_exc}")

        print(f"🎉 3D generation completed: {len(asset_3d_paths)}/{len(cropped_assets)} assets")

        return {
            **state,
            "asset_3d_paths": asset_3d_paths,
            "messages": [
                HumanMessage(
                    content=f"3D assets generated: {list(asset_3d_paths.keys())}"
                )
            ],
        }

    except Exception as exc:
        print(f"❌ 3D asset generation failed: {exc}")
        return {
            **state,
            "asset_3d_paths": {},
            "messages": [
                HumanMessage(content=f"3D generation failed: {str(exc)}")
            ],
        }


def ply_generation_node(state: WorkflowState) -> WorkflowState:
    """Generate PLY point cloud from inpainted panorama"""

    panorama_path = state.get("inpainted_panorama_path") or state.get("panorama_path")

    if not panorama_path:
        return {
            **state,
            "ply_path": "",
            "messages": [
                HumanMessage(content="No panorama available for PLY generation")
            ],
        }

    try:
        print(f"🎲 Starting PLY generation for: {panorama_path}")

        # Convert to container path
        container_path = to_container_path(panorama_path)
        print(f"🔄 Container path: {container_path}")

        # API Call (Train Gaussian Splatting)
        # Using requests directly as there isn't a dedicated client class for this in the original code
        response = requests.post(
            f"{settings.DREAMSCENE_API_URL}/train_gaussian",
            json={
                "panorama_path": container_path,
                "scene_name": state.get("scene_name", "default_scene"),
                "iterations": state.get("training_iterations", GAUSSIAN_DEFAULTS.iterations),
                "save_iterations": state.get("training_save_iterations", GAUSSIAN_DEFAULTS.save_iterations),
                "test_iterations": state.get("training_test_iterations", GAUSSIAN_DEFAULTS.test_iterations),
                "no_perturb_loss": state.get("training_no_perturb_loss", GAUSSIAN_DEFAULTS.no_perturb_loss),
                "white_background": state.get("training_white_background", GAUSSIAN_DEFAULTS.white_background),
                "sh_degree": state.get("training_sh_degree", GAUSSIAN_DEFAULTS.sh_degree),
                "gen_res": state.get("training_gen_res", GAUSSIAN_DEFAULTS.gen_res),
            },
            timeout=GAUSSIAN_DEFAULTS.request_timeout
        )

        if response.status_code == 200:
            result = response.json()
            print(f"🔍 API Response: {result}")

            # Select the last (most trained) PLY path
            trained_ply_paths = result.get("trained_ply_paths", [])
            container_ply_path = trained_ply_paths[-1] if trained_ply_paths else result.get("initial_ply_path")

            if not container_ply_path:
                error_msg = f"No PLY paths returned from API. Response: {result}"
                print(f"❌ {error_msg}")
                return {
                    **state,
                    "ply_path": "",
                    "messages": [
                        HumanMessage(content=f"PLY generation failed: {error_msg}")
                    ],
                }

            # Convert to host path
            host_ply_path = to_host_path(container_ply_path)
            print(f"✅ Gaussian Splatting training completed: {host_ply_path}")

            return {
                **state,
                "ply_path": host_ply_path,
                "model_path": result.get("model_path", ""),
                "messages": [
                    HumanMessage(content=f"Gaussian Splatting trained. Final PLY: {host_ply_path}")
                ],
            }
        else:
            error_msg = f"API error: {response.status_code} - {response.text}"
            print(f"❌ {error_msg}")
            return {
                **state,
                "ply_path": "",
                "messages": [
                    HumanMessage(content=f"Gaussian Splatting training failed: {error_msg}")
                ],
            }

    except Exception as exc:
        print(f"❌ PLY generation failed: {exc}")
        return {
            **state,
            "ply_path": "",
            "messages": [
                HumanMessage(content=f"PLY generation failed: {str(exc)}")
            ],
        }
