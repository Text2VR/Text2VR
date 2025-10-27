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
        # Basic parameters
        sam_checkpoint: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        box_threshold: Optional[float] = None,
        text_threshold: Optional[float] = None,
        # Advanced parameters
        max_prompts: Optional[int] = None,
        anchor_enable: Optional[bool] = None,
        min_area_ratio: Optional[float] = None,
        max_area_ratio: Optional[float] = None,
        # Exclusion controls
        exclusion_use_mask: Optional[bool] = None,
        exclusion_mask_dilate_px: Optional[int] = None,
        exclusion_overlap_drop: Optional[float] = None,
        exclusion_box_th: Optional[float] = None,
        exclusion_text_th: Optional[float] = None,
        exclusion_pad_ratio: Optional[float] = None,
        # Wrap NMS
        wrap_nms_iou: Optional[float] = None,
        # Floor filter
        enable_floor_filter: Optional[bool] = None,
        floor_band_ratio: Optional[float] = None,
        # Wrap mask merge & cleanup
        min_region_ratio: Optional[float] = None,
        close_kernel: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Segment panorama and return the results"""

        # Start segmentation task
        response = requests.post(f"{self.base_url}/segment", json={
            "panorama_path": panorama_path,
            "scene_name": scene_name,
            # Basic parameters
            "sam_checkpoint": sam_checkpoint or SEGMENTATION_DEFAULTS.sam_checkpoint,
            "openai_api_key": openai_api_key or SEGMENTATION_DEFAULTS.openai_api_key,
            "box_threshold": box_threshold if box_threshold is not None else SEGMENTATION_DEFAULTS.box_threshold,
            "text_threshold": text_threshold if text_threshold is not None else SEGMENTATION_DEFAULTS.text_threshold,
            # Advanced parameters
            "max_prompts": max_prompts if max_prompts is not None else SEGMENTATION_DEFAULTS.max_prompts,
            "anchor_enable": anchor_enable if anchor_enable is not None else SEGMENTATION_DEFAULTS.anchor_enable,
            "min_area_ratio": min_area_ratio if min_area_ratio is not None else SEGMENTATION_DEFAULTS.min_area_ratio,
            "max_area_ratio": max_area_ratio if max_area_ratio is not None else SEGMENTATION_DEFAULTS.max_area_ratio,
            # Exclusion controls
            "exclusion_use_mask": exclusion_use_mask if exclusion_use_mask is not None else SEGMENTATION_DEFAULTS.exclusion_use_mask,
            "exclusion_mask_dilate_px": exclusion_mask_dilate_px if exclusion_mask_dilate_px is not None else SEGMENTATION_DEFAULTS.exclusion_mask_dilate_px,
            "exclusion_overlap_drop": exclusion_overlap_drop if exclusion_overlap_drop is not None else SEGMENTATION_DEFAULTS.exclusion_overlap_drop,
            "exclusion_box_th": exclusion_box_th if exclusion_box_th is not None else SEGMENTATION_DEFAULTS.exclusion_box_th,
            "exclusion_text_th": exclusion_text_th if exclusion_text_th is not None else SEGMENTATION_DEFAULTS.exclusion_text_th,
            "exclusion_pad_ratio": exclusion_pad_ratio if exclusion_pad_ratio is not None else SEGMENTATION_DEFAULTS.exclusion_pad_ratio,
            # Wrap NMS
            "wrap_nms_iou": wrap_nms_iou if wrap_nms_iou is not None else SEGMENTATION_DEFAULTS.wrap_nms_iou,
            # Floor filter
            "enable_floor_filter": enable_floor_filter if enable_floor_filter is not None else SEGMENTATION_DEFAULTS.enable_floor_filter,
            "floor_band_ratio": floor_band_ratio if floor_band_ratio is not None else SEGMENTATION_DEFAULTS.floor_band_ratio,
            # Wrap mask merge & cleanup
            "min_region_ratio": min_region_ratio if min_region_ratio is not None else SEGMENTATION_DEFAULTS.min_region_ratio,
            "close_kernel": close_kernel if close_kernel is not None else SEGMENTATION_DEFAULTS.close_kernel,
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
