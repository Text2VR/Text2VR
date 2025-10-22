#!/usr/bin/env python3
"""
API Client for Text2VR pipeline
"""

import requests
import time
import json
import os
from typing import Optional

class PanoramaAPIClient:
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        
    def generate_panorama(self, text: str, scene_name: str, 
                         use_self_refinement: bool = False,
                         num_prompt: int = 3, max_rounds: int = 3) -> str:
        """Generate panorama and return the file path"""
        
        # Start generation task
        response = requests.post(f"{self.base_url}/generate", json={
            "text": text,
            "scene_name": scene_name,
            "use_self_refinement": use_self_refinement,
            "num_prompt": num_prompt,
            "max_rounds": max_rounds
        })
        response.raise_for_status()
        
        task_data = response.json()
        task_id = task_data["task_id"]
        print(f"✅ Task started: {task_id}, Self Refinement: {use_self_refinement}")
        
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
            
            time.sleep(5)  # Wait 5 seconds before checking again

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python api_client.py <text> <scene_name> [use_self_refinement] [num_prompt] [max_rounds]")
        sys.exit(1)
    
    text = sys.argv[1]
    scene_name = sys.argv[2]
    use_self_refinement = sys.argv[3].lower() == "true" if len(sys.argv) > 3 else False
    num_prompt = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    max_rounds = int(sys.argv[5]) if len(sys.argv) > 5 else 3
    
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