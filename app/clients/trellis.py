"""
TRELLIS API Client for 3D Asset Generation
"""

import os
import logging
import requests
from pathlib import Path
from typing import Optional

from ..core.constants import TRELLIS_DEFAULTS

logger = logging.getLogger(__name__)

class TrellisAPIClient:
    """
    Client for interacting with TRELLIS 3D Asset Generation API.
    """

    def __init__(self, base_url: str = "http://localhost:8004", timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        logger.info(f"Initialized TrellisAPIClient with base_url: {self.base_url}")

    def health_check(self) -> dict:
        """Check if TRELLIS API is healthy"""
        try:
            logger.debug(f"Performing health check on {self.base_url}/health")
            response = requests.get(
                f"{self.base_url}/health",
                timeout=self.timeout
            )
            response.raise_for_status()
            health_data = response.json()
            logger.info(f"✅ TRELLIS API health check passed: {health_data.get('status', 'unknown')}")
            return health_data
        except requests.ConnectionError as e:
            logger.error(f"❌ Failed to connect to TRELLIS API: {e}")
            raise
        except requests.HTTPError as e:
            logger.error(f"❌ TRELLIS API health check failed with HTTP {e.response.status_code}")
            raise
        except requests.Timeout as e:
            logger.error(f"❌ TRELLIS API health check timed out: {e}")
            raise

    def generate_3d_asset(
        self,
        image_path: str,
        asset_name: str,
        output_path: str,
        seed: Optional[int] = None,
        texture_size: Optional[int] = None,
        simplify: Optional[float] = None,
        ss_guidance_strength: Optional[float] = None,
        ss_sampling_steps: Optional[int] = None,
        slat_guidance_strength: Optional[float] = None,
        slat_sampling_steps: Optional[int] = None,
        timeout: Optional[int] = None
    ) -> str:
        """
        Generate a 3D GLB asset from a 2D image using TRELLIS API.
        """
        # Validate input image exists
        image_path_obj = Path(image_path)
        if not image_path_obj.exists():
            error_msg = f"Input image not found: {image_path}"
            logger.error(f"❌ {error_msg}")
            raise FileNotFoundError(error_msg)

        if not image_path_obj.is_file():
            error_msg = f"Image path is not a file: {image_path}"
            logger.error(f"❌ {error_msg}")
            raise FileNotFoundError(error_msg)

        logger.info(f"🎯 Starting 3D generation for: {asset_name}")
        logger.debug(f"Input image: {image_path}")
        logger.debug(f"Output path: {output_path}")

        try:
            # Prepare multipart form data
            with open(image_path, 'rb') as f:
                files = {
                    'image': (image_path_obj.name, f, 'image/png')
                }

                data = {
                    'asset_name': asset_name,
                    'seed': seed if seed is not None else TRELLIS_DEFAULTS.seed,
                    'texture_size': texture_size if texture_size is not None else TRELLIS_DEFAULTS.texture_size,
                    'simplify': simplify if simplify is not None else TRELLIS_DEFAULTS.simplify,
                    'ss_guidance_strength': ss_guidance_strength if ss_guidance_strength is not None else TRELLIS_DEFAULTS.ss_guidance_strength,
                    'ss_sampling_steps': ss_sampling_steps if ss_sampling_steps is not None else TRELLIS_DEFAULTS.ss_sampling_steps,
                    'slat_guidance_strength': slat_guidance_strength if slat_guidance_strength is not None else TRELLIS_DEFAULTS.slat_guidance_strength,
                    'slat_sampling_steps': slat_sampling_steps if slat_sampling_steps is not None else TRELLIS_DEFAULTS.slat_sampling_steps
                }

                logger.info(f"📤 Uploading image to {self.base_url}/generate-direct")

                # Call TRELLIS API with streaming
                response = requests.post(
                    f"{self.base_url}/generate-direct",
                    files=files,
                    data=data,
                    timeout=timeout if timeout is not None else TRELLIS_DEFAULTS.timeout,
                    stream=True
                )

                # Check for HTTP errors
                try:
                    response.raise_for_status()
                except requests.HTTPError as e:
                    error_detail = response.text
                    logger.error(f"❌ API request failed with status {response.status_code}")
                    logger.error(f"Response: {error_detail}")
                    raise

                logger.debug(f"✅ API response received (status: {response.status_code})")

        except requests.Timeout:
            error_msg = f"API request timed out after {timeout} seconds"
            logger.error(f"❌ {error_msg}")
            raise

        except requests.ConnectionError as e:
            error_msg = f"Failed to connect to TRELLIS API: {e}"
            logger.error(f"❌ {error_msg}")
            raise

        # Create output directory and save GLB file
        output_path_obj = Path(output_path)
        try:
            output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            error_msg = f"Failed to create output directory: {e}"
            logger.error(f"❌ {error_msg}")
            raise IOError(error_msg)

        # Stream download the GLB file
        try:
            bytes_downloaded = 0
            with open(output_path_obj, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        bytes_downloaded += len(chunk)

            logger.info(f"💾 GLB file saved: {output_path}")

            # Verify output file
            if not output_path_obj.exists():
                error_msg = f"Output file was not created: {output_path}"
                logger.error(f"❌ {error_msg}")
                raise IOError(error_msg)

            file_size = output_path_obj.stat().st_size
            if file_size == 0:
                error_msg = f"Output file is empty: {output_path}"
                logger.error(f"❌ {error_msg}")
                raise IOError(error_msg)

            logger.info(f"✅ 3D generation completed successfully")
            return str(output_path_obj)

        except IOError as e:
            error_msg = f"Failed to save GLB file: {e}"
            logger.error(f"❌ {error_msg}")
            raise
