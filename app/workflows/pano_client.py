#!/usr/bin/env python3
"""
Panorama Generation API Client for Text2VR pipeline
"""

import requests
import time
from typing import Optional, Dict, Any

from .defaults import PANORAMA_DEFAULTS


class PanoramaAPIClient:
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url

    def generate_panorama(
        self,
        text: str,
        scene_name: str,
        use_self_refinement: Optional[bool] = None,
        num_prompt: Optional[int] = None,
        max_rounds: Optional[int] = None
    ) -> str:
        """Generate panorama and return the file path"""

        # Start generation task
        actual_use_self_refinement = use_self_refinement if use_self_refinement is not None else PANORAMA_DEFAULTS.use_self_refinement
        actual_num_prompt = num_prompt if num_prompt is not None else PANORAMA_DEFAULTS.num_prompt
        actual_max_rounds = max_rounds if max_rounds is not None else PANORAMA_DEFAULTS.max_rounds

        response = requests.post(f"{self.base_url}/generate", json={
            "text": text,
            "scene_name": scene_name,
            "use_self_refinement": actual_use_self_refinement,
            "num_prompt": actual_num_prompt,
            "max_rounds": actual_max_rounds
        })
        response.raise_for_status()

        task_data = response.json()
        task_id = task_data["task_id"]
        print(f"✅ Task started: {task_id}, Self Refinement: {actual_use_self_refinement}")

        # Poll for completion
        while True:
            status_response = requests.get(f"{self.base_url}/status/{task_id}")
            status_response.raise_for_status()
            status = status_response.json()

            print(f"📊 Status: {status['status']} - {status['message']}")

            if status["status"] == "completed":
                return status["result_path"]
            elif status["status"] == "failed":
                raise Exception(f"Task failed: {status['message']}")

            time.sleep(PANORAMA_DEFAULTS.poll_interval)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python pano_client.py <text> <scene_name> [use_self_refinement] [num_prompt] [max_rounds]")
        sys.exit(1)

    text = sys.argv[1]
    scene_name = sys.argv[2]
    use_self_refinement = sys.argv[3].lower() == "true" if len(sys.argv) > 3 else PANORAMA_DEFAULTS.use_self_refinement
    num_prompt = int(sys.argv[4]) if len(sys.argv) > 4 else PANORAMA_DEFAULTS.num_prompt
    max_rounds = int(sys.argv[5]) if len(sys.argv) > 5 else PANORAMA_DEFAULTS.max_rounds

    client = PanoramaAPIClient()

    try:
        result_path = client.generate_panorama(
            text=text,
            scene_name=scene_name,
            use_self_refinement=use_self_refinement,
            num_prompt=num_prompt,
            max_rounds=max_rounds
        )
        print(f"🎉 Panorama generated: {result_path}")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
