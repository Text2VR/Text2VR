"""
Inpainting API Client for LangGraph workflows
"""

import time
import logging
from typing import Dict, Optional, Any

import requests
from ..core.constants import INPAINTING_DEFAULTS

logger = logging.getLogger(__name__)

class InpaintingAPIClient:
    """Client for communicating with the Background Inpainting API"""

    def __init__(self, base_url: str = "http://localhost:8003"):
        self.base_url = base_url.rstrip("/")

    def health_check(self) -> Dict[str, Any]:
        """Check if the API is healthy"""
        response = requests.get(f"{self.base_url}/health", timeout=10)
        response.raise_for_status()
        return response.json()

    def inpaint_panorama(
        self,
        panorama_path: str,
        mask_dir: str,
        scene_name: str,
        model_id: Optional[str] = None,
        prompt: Optional[str] = None,
        neg_prompt: Optional[str] = None,
        strength: Optional[float] = None,
        guidance: Optional[float] = None,
        steps: Optional[int] = None,
        wrap_pad: Optional[int] = None,
        dilate: Optional[int] = None,
        feather: Optional[int] = None,
        erase: Optional[str] = None,
        seed: Optional[int] = None,
        poll_interval: Optional[int] = None,
        timeout: Optional[int] = None
    ) -> str:
        """
        Request panorama inpainting and return the result path.
        """

        # 1. Request inpainting
        request_data = {
            "panorama_path": panorama_path,
            "mask_dir": mask_dir,
            "scene_name": scene_name,
            "model_id": model_id or INPAINTING_DEFAULTS.model_id,
            "prompt": prompt or INPAINTING_DEFAULTS.prompt,
            "neg_prompt": neg_prompt or INPAINTING_DEFAULTS.neg_prompt,
            "strength": strength if strength is not None else INPAINTING_DEFAULTS.strength,
            "guidance": guidance if guidance is not None else INPAINTING_DEFAULTS.guidance,
            "steps": steps if steps is not None else INPAINTING_DEFAULTS.steps,
            "wrap_pad": wrap_pad if wrap_pad is not None else INPAINTING_DEFAULTS.wrap_pad,
            "dilate": dilate if dilate is not None else INPAINTING_DEFAULTS.dilate,
            "feather": feather if feather is not None else INPAINTING_DEFAULTS.feather,
            "erase": erase or INPAINTING_DEFAULTS.erase,
            "seed": seed if seed is not None else INPAINTING_DEFAULTS.seed
        }

        logger.info(f"📤 Sending inpainting request for {scene_name}...")
        response = requests.post(
            f"{self.base_url}/inpaint",
            json=request_data,
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        task_id = result["task_id"]

        logger.info(f"✅ Inpainting task started: {task_id}")

        # 2. Poll status
        start_time = time.time()
        while True:
            elapsed = time.time() - start_time
            current_timeout = timeout if timeout is not None else INPAINTING_DEFAULTS.timeout
            if elapsed > current_timeout:
                raise TimeoutError(f"Inpainting timed out after {current_timeout}s")

            # Check status
            status_response = requests.get(
                f"{self.base_url}/status/{task_id}",
                timeout=10
            )
            status_response.raise_for_status()
            status_data = status_response.json()

            status = status_data["status"]
            message = status_data["message"]

            logger.info(f"📊 Inpainting Status: {status} - {message}")

            if status == "completed":
                result_path = status_data.get("result_path")
                if not result_path:
                    raise RuntimeError("Inpainting completed but no result path returned")
                logger.info(f"✅ Inpainting completed: {result_path}")
                return result_path

            elif status == "failed":
                raise RuntimeError(f"Inpainting failed: {message}")

            elif status in ("queued", "processing"):
                time.sleep(poll_interval if poll_interval is not None else INPAINTING_DEFAULTS.poll_interval)
                continue

            else:
                raise RuntimeError(f"Unknown status: {status}")
