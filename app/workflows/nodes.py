"""LangGraph node implementations for the panorama workflow."""

from __future__ import annotations

import glob
import os
import time
import uuid
import requests
from typing import List, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from ..config import settings
from .api_client import PanoramaAPIClient
from .segmentation_client import SegmentationAPIClient
from .inpainting_client import InpaintingAPIClient
from .states import WorkflowState
from .asset_cropper import crop_assets_with_transparency
from ..services.task_manager import task_manager
from ..models.panorama import TaskStatus


def query_rewrite_node(state: WorkflowState) -> WorkflowState:
    """Rewrite a user's prompt to better suit panorama generation."""

    llm = ChatOpenAI(
        model=settings.OPENAI_MODEL,
        temperature=settings.OPENAI_TEMPERATURE,
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL,
    )

    system_prompt = (
        "You are an expert at rewriting user queries for panorama image generation.\n\n"
        "Transform the user's input into a detailed, vivid description suitable for panorama generation.\n"
        "Focus on:\n"
        "- Visual details and atmosphere\n"
        "- 360-degree scene composition\n"
        "- Lighting and mood\n"
        "- Environmental elements\n\n"
        "Keep it concise but descriptive. Return only the rewritten query."
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"User input: {state['user_input']}")
    ]

    response = llm.invoke(messages)
    rewritten_query = response.content.strip()

    scene_name = f"scene_{str(uuid.uuid4())[:8]}"

    return {
        **state,
        "rewritten_query": rewritten_query,
        "scene_name": scene_name,
        "messages": [HumanMessage(content=f"Query rewritten: {rewritten_query}")],
    }


def panorama_generation_node(state: WorkflowState) -> WorkflowState:
    """Generate a panorama using the DreamScene360 API and post-process the file path."""

    client = PanoramaAPIClient(base_url=settings.DREAMSCENE_API_URL)

    try:
        print(f"🔄 Calling DreamScene360 API for scene: {state['scene_name']}")
        result_path = client.generate_panorama(
            text=state["rewritten_query"],
            scene_name=state["scene_name"],
            use_self_refinement=False,
            num_prompt=3,
            max_rounds=3,
        )

        print(f"📦 DreamScene360 API returned: {result_path}")

        local_result_path = None
        if result_path:
            print(f"🔍 Processing path: {result_path}")
            if result_path.startswith("/workspace/data/"):
                local_result_path = result_path.replace(
                    "/workspace/data/", "/home/0in/workspace/Text2VR/data/"
                )
                print(f"🔄 Converted to: {local_result_path}")
            elif result_path.startswith("/workspace/DREAMSCENE360/output/"):
                local_result_path = result_path.replace(
                    "/workspace/DREAMSCENE360/output/",
                    "/home/0in/workspace/Text2VR/output/",
                )
                print(f"🔄 Converted DREAMSCENE360 path to: {local_result_path}")
            else:
                scene_name = state["scene_name"]
                local_result_path = (
                    f"/home/0in/workspace/Text2VR/data/{scene_name}/panorama.png"
                )
                print(f"🔄 Using fallback path: {local_result_path}")

            print(f"verfying start:{local_result_path}")
            print(
                f"DEBUG: local_result_path type: {type(local_result_path)}, value: {repr(local_result_path)}"
            )
            print(f"DEBUG: bool(local_result_path): {bool(local_result_path)}")
            if local_result_path:
                print("DEBUG: Entered if local_result_path block")
                max_retries = 5
                print(f"DEBUG: Starting retry loop with max_retries={max_retries}")
                for attempt in range(max_retries):
                    print(f"DEBUG: Loop attempt {attempt + 1}")
                    print(
                        f"DEBUG: About to check os.path.exists for: {local_result_path}"
                    )
                    try:
                        exists = os.path.exists(local_result_path)
                        print(f"DEBUG: os.path.exists returned: {exists}")
                    except Exception as exc:
                        print(f"DEBUG: Exception in os.path.exists: {exc}")
                        exists = False
                    if exists:
                        print(f"✅ Path exists: {local_result_path}")
                        break
                    print(
                        f"⚠️ Attempt {attempt + 1}/{max_retries}: Path does not exist yet: {local_result_path}"
                    )
                    if attempt < max_retries - 1:
                        time.sleep(3)

                if not os.path.exists(local_result_path):
                    print(f"🔍 Original path was: {result_path}")

                    scene_name = state["scene_name"]
                    scene_dir = f"/home/0in/workspace/Text2VR/data/{scene_name}"
                    print(f"🔍 Checking scene directory: {scene_dir}")

                    if os.path.exists(scene_dir):
                        pano_files = glob.glob(os.path.join(scene_dir, "*.png"))
                        print(
                            f"🔍 Found PNG files in {scene_dir}: {pano_files}"
                        )
                        if pano_files:
                            local_result_path = pano_files[0]
                            print(
                                f"✅ Found alternative file: {local_result_path}"
                            )
                        else:
                            print("❌ No PNG files found in scene directory")
                            local_result_path = None
                    else:
                        print(
                            f"❌ Scene directory does not exist: {scene_dir}"
                        )
                        print(
                            "🔍 Searching for recent PNG files in data directory..."
                        )
                        data_dir = "/home/0in/workspace/Text2VR/data"
                        all_pngs: List[Tuple[float, str]] = []
                        for root, _, files in os.walk(data_dir):
                            for file in files:
                                if file.endswith(".png") and "panorama" in file:
                                    full_path = os.path.join(root, file)
                                    mtime = os.path.getmtime(full_path)
                                    all_pngs.append((mtime, full_path))

                        if all_pngs:
                            all_pngs.sort(reverse=True)
                            newest_mtime, newest_file = all_pngs[0]
                            current_time = time.time()
                            if current_time - newest_mtime < 300:
                                local_result_path = newest_file
                                print(
                                    f"✅ Found recent panorama file: {local_result_path}"
                                )
                            else:
                                print(
                                    f"⚠️ Latest file is too old ({current_time - newest_mtime:.0f}s ago): {newest_file}"
                                )
                                local_result_path = None
                        else:
                            print("❌ No panorama files found anywhere")
                            local_result_path = None
        else:
            print("❌ DreamScene360 API returned empty/null result_path")

        print(f"DEBUG: About to set final_path, local_result_path = {repr(local_result_path)}")
        final_path = local_result_path or ""
        print(f"🎯 Final panorama path: {final_path}")

        # Update task status after panorama generation
        try:
            if final_path and state.get("task_id"):
                task_manager.update_task_status(
                    task_id=state["task_id"],
                    status=TaskStatus.PROCESSING,
                    message="Panorama generated, processing segmentation...",
                    panorama_path=final_path
                )
                print(f"✅ Task manager updated for task_id: {state['task_id']}")
        except Exception as tm_exc:
            print(f"⚠️ Failed to update task manager: {tm_exc}")

        return {
            **state,
            "panorama_path": final_path,
            "messages": [
                HumanMessage(
                    content=f"Panorama generated: {final_path or result_path}"
                )
            ],
        }

    except Exception as exc:
        return {
            **state,
            "panorama_path": "",
            "messages": [
                HumanMessage(content=f"Generation failed: {str(exc)}")
            ],
        }


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
        container_path = panorama_path.replace(
            "/home/0in/workspace/Text2VR/data/", "/app/host_data/"
        )

        print(f"🔄 Container path: {container_path}")

        client = SegmentationAPIClient(base_url=settings.SEGMENTATION_API_URL)

        result = client.segment_panorama(
            panorama_path=container_path,
            scene_name=state["scene_name"],
        )

        print("✅ Segmentation completed")
        print(
            f"📋 Found objects: {list(result['segmentation_data'].get('prompts', {}).keys())}"
        )

        # Asset cropping with transparency
        try:
            print("🔪 Starting asset cropping with transparency...")

            # masking_output 디렉토리 경로 (docker-compose.yml 볼륨 마운트 참고)
            segmentation_output_dir = f"/home/0in/workspace/Text2VR/masking_output/{state['scene_name']}"

            cropped_assets = crop_assets_with_transparency(
                panorama_path=panorama_path,
                segmentation_output_dir=segmentation_output_dir,
                scene_name=state['scene_name']
            )

            print(f"✅ Asset cropping completed: {len(cropped_assets)} asset types")
            print(f"📦 Cropped assets: {list(cropped_assets.keys())}")

        except Exception as crop_exc:
            print(f"⚠️ Asset cropping failed (non-critical): {crop_exc}")
            cropped_assets = {}

        # 세그멘테이션 작업 완료 후 컨테이너 중지 (VRAM 절약)
        try:
            import subprocess
            print("🛑 Stopping segmentation container to free VRAM...")
            subprocess.run(["docker", "stop", "text2vr_segmentation_api"],
                          capture_output=True, timeout=10)
            print("✅ Segmentation container stopped")
        except Exception as e:
            print(f"⚠️ Failed to stop segmentation container: {e}")

        # Update task status after segmentation completion
        try:
            if state.get("task_id"):
                scene_name = state["scene_name"]
                segmentation_results_path = f"/home/0in/workspace/Text2VR/masking_output/{scene_name}/results.json"
                segmentation_visualization_path = f"/home/0in/workspace/Text2VR/masking_output/{scene_name}/visualizations/panorama_visualization.png"

                task_manager.update_task_status(
                    task_id=state["task_id"],
                    status=TaskStatus.PROCESSING,
                    message="Segmentation completed, starting inpainting...",
                    segmentation_results_path=segmentation_results_path,
                    segmentation_visualization_path=segmentation_visualization_path
                )
                print(f"✅ Task manager updated for task_id: {state['task_id']}")
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

    # 세그멘테이션 데이터가 없으면 인페인팅 스킵
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

        # 컨테이너 경로로 변환
        container_panorama_path = panorama_path.replace(
            "/home/0in/workspace/Text2VR/data/", "/workspace/data/"
        )

        container_mask_dir = f"/workspace/masking_output/{scene_name}/masks"

        print(f"🔄 Container panorama path: {container_panorama_path}")
        print(f"🔄 Container mask dir: {container_mask_dir}")

        client = InpaintingAPIClient(base_url=settings.INPAINTING_API_URL)

        result_path = client.inpaint_panorama(
            panorama_path=container_panorama_path,
            mask_dir=container_mask_dir,
            scene_name=scene_name
        )

        print(f"✅ Inpainting completed: {result_path}")

        # 컨테이너 경로를 호스트 경로로 변환
        host_result_path = result_path.replace(
            "/workspace/inpainted_pano/", "/home/0in/workspace/Text2VR/inpainted_pano/"
        )

        # 인페인팅 작업 완료 후 컨테이너 중지 (VRAM 절약)
        try:
            import subprocess
            print("🛑 Stopping inpainting container to free VRAM...")
            subprocess.run(["docker", "stop", "text2vr_inpainting_api"],
                          capture_output=True, timeout=10)
            print("✅ Inpainting container stopped")
        except Exception as e:
            print(f"⚠️ Failed to stop inpainting container: {e}")

        # Update task status after inpainting completion
        try:
            if state.get("task_id") and host_result_path:
                task_manager.update_task_status(
                    task_id=state["task_id"],
                    status=TaskStatus.PROCESSING,
                    message="Inpainting completed, generating PLY...",
                    inpainted_panorama_path=host_result_path
                )
                print(f"✅ Task manager updated for task_id: {state['task_id']}")
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
        # 인페인팅 실패 시 원본 파노라마 경로 유지
        return {
            **state,
            "inpainted_panorama_path": state["panorama_path"],
            "messages": [
                HumanMessage(content=f"Inpainting failed (using original): {str(exc)}")
            ],
        }


def ply_generation_node(state: WorkflowState) -> WorkflowState:
    """Generate PLY point cloud from inpainted panorama"""

    # inpainted_panorama_path 또는 panorama_path 사용
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

        # 컨테이너 경로로 변환
        container_path = panorama_path.replace(
            "/home/0in/workspace/Text2VR/", "/workspace/"
        )

        print(f"🔄 Container path: {container_path}")

        # API 호출
        response = requests.post(
            f"{settings.DREAMSCENE_API_URL}/panorama_to_ply",
            json={
                "panorama_path": container_path,
                "output_name": "scene.ply"
            },
            timeout=300
        )

        if response.status_code == 200:
            result = response.json()
            container_ply_path = result["ply_path"]

            # 호스트 경로로 변환
            host_ply_path = container_ply_path.replace(
                "/workspace/output", "/home/0in/workspace/Text2VR/plyoutput"
            )

            print(f"✅ PLY generation completed: {host_ply_path}")

            return {
                **state,
                "ply_path": host_ply_path,
                "messages": [
                    HumanMessage(content=f"PLY file generated: {host_ply_path}")
                ],
            }
        else:
            error_msg = f"API error: {response.status_code} - {response.text}"
            print(f"❌ {error_msg}")
            return {
                **state,
                "ply_path": "",
                "messages": [
                    HumanMessage(content=f"PLY generation failed: {error_msg}")
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
