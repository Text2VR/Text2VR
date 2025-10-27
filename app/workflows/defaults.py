"""
Shared default parameter values for Text2VR workflow clients.

Centralising these values avoids scattering hard-coded numbers across multiple
clients/nodes and keeps them aligned with microservice request schemas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class SegmentationDefaults:
    panorama_path_prefix: str = "/home/capstoneproj0310/Text2VR/data/"
    container_path_prefix: str = "/app/host_data/"
    sam_checkpoint: str = "/app/checkpoints/sam_vit_h_4b8939.pth"
    openai_api_key: Optional[str] = None
    box_threshold: float = 0.20
    text_threshold: float = 0.15


@dataclass(frozen=True)
class InpaintingDefaults:
    panorama_path_prefix: str = "/home/capstoneproj0310/Text2VR/data/"
    container_path_prefix: str = "/workspace/data/"
    mask_dir_prefix: str = "/workspace/masking_output/"
    result_prefix: str = "/workspace/inpainted_pano/"

    model_id: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    prompt: str = (
        "a clean empty room background, photorealistic, seamless texture, 8k, sharp focus"
    )
    neg_prompt: str = (
        "objects, furniture, sofa, picture, chair, plant, lamp, table, blurry, hazy, watermark, text, signature, pillow"
    )
    strength: float = 0.95
    guidance: float = 7.5
    steps: int = 40
    wrap_pad: Optional[int] = None
    dilate: Optional[int] = None
    feather: int = 1
    erase: str = "gray"
    seed: int = 42
    poll_interval: int = 5
    timeout: int = 600


@dataclass(frozen=True)
class GaussianDefaults:
    panorama_path_prefix: str = "/home/capstoneproj0310/Text2VR/"
    container_path_prefix: str = "/workspace/"
    iterations: int = 100
    save_iterations: List[int] = field(default_factory=lambda: [50, 70])
    white_background: bool = False
    sh_degree: int = 3
    gen_res: int = 512
    request_timeout: int = 10000


SEGMENTATION_DEFAULTS = SegmentationDefaults()
INPAINTING_DEFAULTS = InpaintingDefaults()
GAUSSIAN_DEFAULTS = GaussianDefaults()
