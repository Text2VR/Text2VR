"""
Panorama generation steps for the workflow
"""

import os
import uuid
import time
import glob
from typing import List, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from ...core.config import settings
from ...core.path_utils import to_host_path
from ...core.constants import PANORAMA_DEFAULTS
from ...models.workflow_state import WorkflowState
from ...clients.panorama import PanoramaAPIClient
from ...services.task_manager import task_manager
from ...models.panorama import TaskStatus


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
            use_self_refinement=state.get("use_self_refinement", PANORAMA_DEFAULTS.use_self_refinement),
            num_prompt=state.get("num_prompt", PANORAMA_DEFAULTS.num_prompt),
            max_rounds=state.get("max_rounds", PANORAMA_DEFAULTS.max_rounds),
        )

        print(f"📦 DreamScene360 API returned: {result_path}")

        # Convert container path to host path using our utility
        local_result_path = to_host_path(result_path)
        print(f"🔄 Converted path: {local_result_path}")

        if local_result_path:
            max_retries = 5
            for attempt in range(max_retries):
                if os.path.exists(local_result_path):
                    print(f"✅ Path exists: {local_result_path}")
                    break
                
                print(f"⚠️ Attempt {attempt + 1}/{max_retries}: Path does not exist yet: {local_result_path}")
                if attempt < max_retries - 1:
                    time.sleep(3)

            if not os.path.exists(local_result_path):
                print(f"🔍 Original path was: {result_path}")
                
                # Fallback search logic
                scene_name = state["scene_name"]
                scene_dir = settings.output_dir / scene_name
                
                if scene_dir.exists():
                    pano_files = list(scene_dir.glob("*.png"))
                    if pano_files:
                        local_result_path = str(pano_files[0])
                        print(f"✅ Found alternative file: {local_result_path}")
                    else:
                        print("❌ No PNG files found in scene directory")
                        local_result_path = None
                else:
                    print(f"❌ Scene directory does not exist: {scene_dir}")
                    local_result_path = None

        final_path = local_result_path or ""
        print(f"🎯 Final panorama path: {final_path}")

        # Update task status
        try:
            if final_path and state.get("task_id"):
                task_manager.update_task_status(
                    task_id=state["task_id"],
                    status=TaskStatus.PROCESSING,
                    message="Panorama generated, processing segmentation...",
                    panorama_path=final_path
                )
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
