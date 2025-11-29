"""
Shared constants and default parameter values for Text2VR workflow.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from .config import settings

@dataclass(frozen=True)
class SegmentationDefaults:
    # Basic segmentation parameters
    sam_checkpoint: str = "/app/checkpoints/sam_vit_h_4b8939.pth"
    openai_api_key: Optional[str] = None

@dataclass(frozen=True)
class InpaintingDefaults:
    # Model parameters
    model_id: str = "stabilityai/stable-diffusion-2-inpainting"
    prompt: str = "clean empty interior background, seamless walls and floor, photorealistic, matching lighting, no new objects"
    neg_prompt: str = "sofa, couch, armchair, chair, bench, text, watermark, logo, artifacts, distortion, blurry, people, signature"
    strength: float = 0.95
    guidance: float = 7.5
    steps: int = 40
    wrap_pad: Optional[int] = None
    dilate: Optional[int] = None
    feather: int = 0
    erase: str = "gray"
    seed: int = 42

    # Client-only parameters
    poll_interval: int = 5
    timeout: int = 600

@dataclass(frozen=True)
class GaussianDefaults:
    # Training parameters
    iterations: int = 100
    save_iterations: List[int] = field(default_factory=lambda: [50, 100])
    test_iterations: int = 100
    no_perturb_loss: bool = True
    white_background: bool = False
    sh_degree: int = 3
    gen_res: int = 512

    # Client-only parameters
    request_timeout: int = 10000

@dataclass(frozen=True)
class PanoramaDefaults:
    # Panorama generation parameters
    use_self_refinement: bool = False
    num_prompt: int = 2
    max_rounds: int = 2

    # Client-only parameters
    poll_interval: int = 5
    timeout: int = 600

@dataclass(frozen=True)
class TrellisDefaults:
    # 3D generation parameters
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
