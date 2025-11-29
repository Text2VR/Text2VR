"""
Segmentation API Client for Text2VR pipeline
"""

import time
import logging
from typing import Dict, Optional, Any

import requests
from ..core.constants import SEGMENTATION_DEFAULTS

logger = logging.getLogger(__name__)

class SegmentationAPIClient:
    def __init__(self, base_url: str = "http://localhost:8002"):
        self.base_url = base_url

    def segment_panorama(
        self,
        panorama_path: str,
        scene_name: str,
        sam_checkpoint: Optional[str] = None,
        openai_api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Segment panorama and return the results"""

        # Start segmentation task
        response = requests.post(f"{self.base_url}/segment", json={
            "panorama_path": panorama_path,
            "scene_name": scene_name,
            "sam_checkpoint": sam_checkpoint or SEGMENTATION_DEFAULTS.sam_checkpoint,
            "openai_api_key": openai_api_key or SEGMENTATION_DEFAULTS.openai_api_key,
        })
        response.raise_for_status()
        
        task_data = response.json()
        task_id = task_data["task_id"]
        logger.info(f"✅ Segmentation task started: {task_id}")
        
        # Poll for completion
        while True:
            status_response = requests.get(f"{self.base_url}/status/{task_id}")
            status_response.raise_for_status()
            status = status_response.json()
            
            logger.info(f"📊 Segmentation Status: {status['status']} - {status['message']}")
            
            if status["status"] == "completed":
                return {
                    "result_path": status["result_path"],
                    "segmentation_data": status["segmentation_data"]
                }
            elif status["status"] == "failed":
                raise Exception(f"Segmentation failed: {status['message']}")
            
            time.sleep(5)
