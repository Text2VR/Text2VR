"""
Panorama Generation API Client for Text2VR pipeline
"""

import requests
import time
import logging
from typing import Optional
from ..core.constants import PANORAMA_DEFAULTS

logger = logging.getLogger(__name__)

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
        logger.info(f"✅ Task started: {task_id}, Self Refinement: {actual_use_self_refinement}")

        # Poll for completion
        while True:
            status_response = requests.get(f"{self.base_url}/status/{task_id}")
            status_response.raise_for_status()
            status = status_response.json()

            logger.info(f"📊 Status: {status['status']} - {status['message']}")

            if status["status"] == "completed":
                return status["result_path"]
            elif status["status"] == "failed":
                raise Exception(f"Task failed: {status['message']}")

            time.sleep(PANORAMA_DEFAULTS.poll_interval)
