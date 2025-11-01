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
    # Path conversion
    panorama_path_prefix: str = "/home/capstoneproj0310/Text2VR/data/"
    container_path_prefix: str = "/app/host_data/"

    # Basic segmentation parameters (API server defaults will be used for advanced params)
    sam_checkpoint: str = "/app/checkpoints/sam_vit_h_4b8939.pth"
    openai_api_key: Optional[str] = None

@dataclass(frozen=True)
class InpaintingDefaults:
    # Path conversion
    panorama_path_prefix: str = "/home/capstoneproj0310/Text2VR/data/"
    container_path_prefix: str = "/workspace/data/"
    mask_dir_prefix: str = "/workspace/masking_output/"
    result_prefix: str = "/workspace/inpainted_pano/"

    # Model parameters (aligned with API server)
    # model_id: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    model_id: str = "stabilityai/stable-diffusion-2-inpainting"
    prompt: str = "clean empty interior background, seamless walls and floor, photorealistic, matching lighting, no new objects"
    neg_prompt: str = "sofa, couch, armchair, chair, bench, text, watermark, logo, artifacts, distortion, blurry, people, signature"
    strength: float = 0.95
    guidance: float = 7.5
    steps: int = 40
    wrap_pad: Optional[int] = None  # None = auto
    dilate: Optional[int] = None    # None = auto
    feather: int = 0
    erase: str = "gray"
    seed: int = 42

    # Client-only parameters
    poll_interval: int = 5
    timeout: int = 600


@dataclass(frozen=True)
class GaussianDefaults:
    # Path conversion
    panorama_path_prefix: str = "/home/capstoneproj0310/Text2VR/"
    container_path_prefix: str = "/workspace/"

    # Training parameters (aligned with API server)
    iterations: int = 7000
    save_iterations: List[int] = field(default_factory=lambda: [5000, 7000])
    test_iterations: int = 7000
    no_perturb_loss: bool = True  # Disable perturbation loss for faster training

    white_background: bool = False
    sh_degree: int = 3
    gen_res: int = 512

    # Client-only parameters
    request_timeout: int = 10000


@dataclass(frozen=True)
class PanoramaDefaults:
    # Panorama generation parameters (aligned with API server)
    use_self_refinement: bool = True
    num_prompt: int = 2
    max_rounds: int = 2

    # Client-only parameters
    poll_interval: int = 5
    timeout: int = 600


@dataclass(frozen=True)
class TrellisDefaults:
    # Path conversion
    host_path_prefix: str = "/home/capstoneproj0310/Text2VR/seged_assets/"
    container_path_prefix: str = "/app/seged_assets/"

    # 3D generation parameters (aligned with API server)
    seed: int = 42
    simplify: float = 0.95
    texture_size: int = 1024
    ss_guidance_strength: float = 7.5
    ss_sampling_steps: int = 12
    slat_guidance_strength: float = 3.0
    slat_sampling_steps: int = 12

    # Client-only parameters
    timeout: int = 120


SEGMENTATION_DEFAULTS = SegmentationDefaults()
INPAINTING_DEFAULTS = InpaintingDefaults()
GAUSSIAN_DEFAULTS = GaussianDefaults()
PANORAMA_DEFAULTS = PanoramaDefaults()
TRELLIS_DEFAULTS = TrellisDefaults()
