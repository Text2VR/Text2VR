#!/usr/bin/env python3
"""
Segmentation API Client for Text2VR pipeline
"""

import json
import time
from typing import Dict, Optional, Any

import requests

from .defaults import SEGMENTATION_DEFAULTS


class SegmentationAPIClient:
    def __init__(self, base_url: str = "http://localhost:8002"):
        self.base_url = base_url

    def segment_panorama(
        self,
        panorama_path: str,
        scene_name: str,
        sam_checkpoint: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        box_threshold: Optional[float] = None,
        text_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Segment panorama and return the results"""

        # Start segmentation task
        response = requests.post(f"{self.base_url}/segment", json={
            "panorama_path": panorama_path,
            "scene_name": scene_name,
            "sam_checkpoint": sam_checkpoint or SEGMENTATION_DEFAULTS.sam_checkpoint,
            "openai_api_key": openai_api_key or SEGMENTATION_DEFAULTS.openai_api_key,
            "box_threshold": box_threshold if box_threshold is not None else SEGMENTATION_DEFAULTS.box_threshold,
            "text_threshold": text_threshold if text_threshold is not None else SEGMENTATION_DEFAULTS.text_threshold
        })
        response.raise_for_status()
        
        task_data = response.json()
        task_id = task_data["task_id"]
        print(f"✅ Segmentation task started: {task_id}")
        
        # Poll for completion
        while True:
            status_response = requests.get(f"{self.base_url}/status/{task_id}")
            status_response.raise_for_status()
            status = status_response.json()
            
            print(f"📊 Segmentation Status: {status['status']} - {status['message']}")
            
            if status["status"] == "completed":
                return {
                    "result_path": status["result_path"],
                    "segmentation_data": status["segmentation_data"]
                }
            elif status["status"] == "failed":
                raise Exception(f"Segmentation failed: {status['message']}")
            
            time.sleep(5)  # Wait 5 seconds before checking again

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python segmentation_client.py <panorama_path> <scene_name>")
        sys.exit(1)
    
    panorama_path = sys.argv[1]
    scene_name = sys.argv[2]
    
    client = SegmentationAPIClient()
    
    try:
        result = client.segment_panorama(
            panorama_path=panorama_path,
            scene_name=scene_name
        )
        print(f"🎉 Segmentation completed: {result['result_path']}")
        print(f"📋 Data: {json.dumps(result['segmentation_data'], indent=2)}")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
