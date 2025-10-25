#!/usr/bin/env python3
"""
TRELLIS API Client for 3D Asset Generation

This module provides a Python client for the TRELLIS Image-to-3D API.
TRELLIS converts 2D images (typically with transparent backgrounds) into
3D GLB assets with realistic geometry and textures.

Author: Claude
Date: 2025-10-22
"""

import os
import logging
import requests
from pathlib import Path
from typing import Optional

# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add console handler if not already added
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class TrellisAPIClient:
    """
    Client for interacting with TRELLIS 3D Asset Generation API.

    The TRELLIS API accepts 2D images (preferably with transparent RGBA backgrounds)
    and generates high-quality 3D GLB files. This client handles file validation,
    API communication, and streaming downloads.

    Attributes:
        base_url (str): Base URL of the TRELLIS API server (default: http://localhost:8004)
        timeout (int): Default request timeout in seconds

    Example:
        >>> client = TrellisAPIClient(base_url="http://localhost:8004")
        >>> health = client.health_check()
        >>> glb_path = client.generate_3d_asset(
        ...     image_path="/path/to/asset.png",
        ...     asset_name="sofa",
        ...     output_path="/path/to/output/sofa.glb"
        ... )
    """

    def __init__(self, base_url: str = "http://localhost:8004", timeout: int = 30):
        """
        Initialize TRELLIS API Client.

        Args:
            base_url (str): Base URL of the TRELLIS API server.
                           Default: http://localhost:8004
            timeout (int): Default timeout for health check requests in seconds.
                          Default: 30
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        logger.info(f"Initialized TrellisAPIClient with base_url: {self.base_url}")

    def health_check(self) -> dict:
        """
        Check if TRELLIS API is healthy and ready to process requests.

        This endpoint verifies that:
        - The API server is running
        - The TRELLIS pipeline is loaded
        - GPU memory is available
        - The service is ready to accept generation requests

        Returns:
            dict: Health status information containing:
                - status (str): "healthy" or "unhealthy"
                - gpu_memory_used (float): GPU memory usage in GB
                - gpu_memory_total (float): Total GPU memory in GB
                - pipeline_loaded (bool): Whether TRELLIS pipeline is initialized

        Raises:
            requests.ConnectionError: If unable to connect to API server
            requests.HTTPError: If API returns an error status code
            requests.Timeout: If request exceeds timeout

        Example:
            >>> health = client.health_check()
            >>> print(f"Status: {health['status']}")
            >>> print(f"GPU Memory: {health['gpu_memory_used']:.2f}GB")
        """
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
        seed: int = 42,
        texture_size: int = 1024,
        simplify: float = 0.95,
        ss_guidance_strength: float = 7.5,
        ss_sampling_steps: int = 12,
        slat_guidance_strength: float = 3.0,
        slat_sampling_steps: int = 12,
        timeout: int = 120
    ) -> str:
        """
        Generate a 3D GLB asset from a 2D image using TRELLIS API.

        This method uploads an image to the TRELLIS API and receives back a
        3D model in GLB format. The GLB file is saved to the specified output path.

        The API uses two-stage image-to-3D generation:
        1. **Sparse Structure**: Initial 3D structure generation with guidance
        2. **SLAT (Sparse Latent Appearance Transfer)**: Texture and appearance refinement

        Args:
            image_path (str): Path to input image file. Must be a valid image file
                            (PNG, JPG, etc). RGBA PNG with transparent background
                            is recommended for best results.
            asset_name (str): Name for the generated asset. Used for logging and
                            GLB file metadata. Should be descriptive (e.g., "sofa", "plant").
            output_path (str): Full path where the output GLB file should be saved.
                             Parent directories will be created if they don't exist.
                             Should end with .glb extension.
            seed (int): Random seed for reproducible generation.
                       Default: 42
                       Use different seeds to generate variations of the same image.
            texture_size (int): Resolution of generated textures in pixels.
                               Options: 512, 1024, 2048
                               Higher values produce better quality but take longer.
                               Default: 1024
            simplify (float): Mesh simplification ratio between 0.0 and 1.0.
                             0.0 = maximum simplification (fewer polygons)
                             1.0 = no simplification (full detail)
                             Default: 0.95 (recommended for VR applications)
            ss_guidance_strength (float): Guidance strength for sparse structure generation.
                                         Higher values produce more structured geometry.
                                         Range: 0.0-15.0
                                         Default: 7.5
            ss_sampling_steps (int): Number of sampling steps for sparse structure.
                                    More steps improve quality but increase processing time.
                                    Range: 4-20
                                    Default: 12 (~30-40 seconds per asset)
            slat_guidance_strength (float): Guidance strength for texture/appearance refinement.
                                           Higher values produce more detailed textures.
                                           Range: 0.0-10.0
                                           Default: 3.0
            slat_sampling_steps (int): Number of sampling steps for texture refinement.
                                      More steps improve texture quality.
                                      Range: 4-20
                                      Default: 12
            timeout (int): Request timeout in seconds.
                          Default: 120 (2 minutes)
                          Increase for complex models or slower connections.

        Returns:
            str: Full path to the generated GLB file.

        Raises:
            FileNotFoundError: If input image file does not exist.
            requests.HTTPError: If API request fails with HTTP error.
            requests.Timeout: If request exceeds timeout duration.
            IOError: If unable to write output file.

        Example:
            >>> client = TrellisAPIClient()
            >>> glb_path = client.generate_3d_asset(
            ...     image_path="/path/to/sofa.png",
            ...     asset_name="sofa",
            ...     output_path="/output/sofa.glb",
            ...     seed=42,
            ...     texture_size=1024,
            ...     simplify=0.95
            ... )
            >>> print(f"Generated: {glb_path}")

        Note:
            - Processing time: typically 30-40 seconds per asset
            - VRAM requirement: 6-8GB
            - Output file size: 5-50MB depending on complexity and texture_size
            - Transparent backgrounds are recommended for best results
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
        logger.debug(f"Parameters: seed={seed}, texture_size={texture_size}, simplify={simplify}")

        try:
            # Prepare multipart form data
            with open(image_path, 'rb') as f:
                files = {
                    'image': (image_path_obj.name, f, 'image/png')
                }

                data = {
                    'asset_name': asset_name,
                    'seed': seed,
                    'texture_size': texture_size,
                    'simplify': simplify,
                    'ss_guidance_strength': ss_guidance_strength,
                    'ss_sampling_steps': ss_sampling_steps,
                    'slat_guidance_strength': slat_guidance_strength,
                    'slat_sampling_steps': slat_sampling_steps
                }

                logger.info(f"📤 Uploading image to {self.base_url}/generate-direct")

                # Call TRELLIS API with streaming
                response = requests.post(
                    f"{self.base_url}/generate-direct",
                    files=files,
                    data=data,
                    timeout=timeout,
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
            logger.debug(f"Created output directory: {output_path_obj.parent}")
        except OSError as e:
            error_msg = f"Failed to create output directory: {e}"
            logger.error(f"❌ {error_msg}")
            raise IOError(error_msg)

        # Stream download the GLB file
        try:
            bytes_downloaded = 0
            with open(output_path_obj, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # Filter out keep-alive chunks
                        f.write(chunk)
                        bytes_downloaded += len(chunk)

            logger.info(f"💾 GLB file saved: {output_path}")
            logger.debug(f"Downloaded {bytes_downloaded / (1024*1024):.2f}MB")

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
            logger.info(f"   Asset: {asset_name}")
            logger.info(f"   File size: {file_size / (1024*1024):.2f}MB")
            logger.info(f"   Output: {output_path}")

            return str(output_path_obj)

        except IOError as e:
            error_msg = f"Failed to save GLB file: {e}"
            logger.error(f"❌ {error_msg}")
            raise


def main():
    """
    CLI interface for TRELLIS API Client.

    Usage:
        python trellis_client.py <image_path> <asset_name> <output_path> [options]

    Examples:
        # Basic usage
        python trellis_client.py sofa.png sofa output/sofa.glb

        # With custom parameters
        python trellis_client.py sofa.png sofa output/sofa.glb \\
            --seed 123 \\
            --texture-size 2048 \\
            --simplify 0.9

    Options:
        --seed SEED                Random seed (default: 42)
        --texture-size SIZE        Texture resolution: 512, 1024, or 2048 (default: 1024)
        --simplify RATIO           Mesh simplification 0.0-1.0 (default: 0.95)
        --ss-guidance STRENGTH     Sparse structure guidance (default: 7.5)
        --ss-steps STEPS           Sparse structure steps (default: 12)
        --slat-guidance STRENGTH   SLAT guidance (default: 3.0)
        --slat-steps STEPS         SLAT steps (default: 12)
        --timeout SECONDS          Request timeout (default: 120)
        --base-url URL             TRELLIS API base URL (default: http://localhost:8004)
    """
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="TRELLIS API Client for 3D Asset Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python trellis_client.py sofa.png sofa output/sofa.glb

  # With custom parameters
  python trellis_client.py sofa.png sofa output/sofa.glb \\
    --seed 123 --texture-size 2048 --simplify 0.9

  # Health check only
  python trellis_client.py --health-check
        """
    )

    # Positional arguments
    parser.add_argument('image_path', nargs='?', help='Path to input image')
    parser.add_argument('asset_name', nargs='?', help='Asset name for the 3D model')
    parser.add_argument('output_path', nargs='?', help='Output path for GLB file')

    # Optional arguments
    parser.add_argument('--base-url', default='http://localhost:8004',
                       help='TRELLIS API base URL (default: http://localhost:8004)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--texture-size', type=int, default=1024, choices=[512, 1024, 2048],
                       help='Texture resolution (default: 1024)')
    parser.add_argument('--simplify', type=float, default=0.95,
                       help='Mesh simplification ratio 0.0-1.0 (default: 0.95)')
    parser.add_argument('--ss-guidance', type=float, default=7.5,
                       help='Sparse structure guidance strength (default: 7.5)')
    parser.add_argument('--ss-steps', type=int, default=12,
                       help='Sparse structure sampling steps (default: 12)')
    parser.add_argument('--slat-guidance', type=float, default=3.0,
                       help='SLAT guidance strength (default: 3.0)')
    parser.add_argument('--slat-steps', type=int, default=12,
                       help='SLAT sampling steps (default: 12)')
    parser.add_argument('--timeout', type=int, default=120,
                       help='Request timeout in seconds (default: 120)')
    parser.add_argument('--health-check', action='store_true',
                       help='Only perform health check and exit')

    args = parser.parse_args()

    # Initialize client
    client = TrellisAPIClient(base_url=args.base_url)

    try:
        # Health check
        logger.info("🔍 Checking TRELLIS API health...")
        health = client.health_check()
        logger.info(f"✅ TRELLIS API Status: {health.get('status', 'unknown')}")

        if 'gpu_memory_used' in health:
            logger.info(f"📊 GPU Memory: {health['gpu_memory_used']:.2f}GB / "
                       f"{health.get('gpu_memory_total', 'unknown')}GB")
        if 'pipeline_loaded' in health:
            logger.info(f"🔧 Pipeline: {'Loaded' if health['pipeline_loaded'] else 'Not loaded'}")

        # If only health check requested, exit
        if args.health_check:
            logger.info("Health check completed successfully")
            sys.exit(0)

        # Validate required arguments for generation
        if not args.image_path or not args.asset_name or not args.output_path:
            parser.print_help()
            logger.error("❌ Missing required arguments: image_path, asset_name, output_path")
            sys.exit(1)

        # Generate 3D asset
        logger.info(f"🎯 Generating 3D asset: {args.asset_name}")
        result_path = client.generate_3d_asset(
            image_path=args.image_path,
            asset_name=args.asset_name,
            output_path=args.output_path,
            seed=args.seed,
            texture_size=args.texture_size,
            simplify=args.simplify,
            ss_guidance_strength=args.ss_guidance,
            ss_sampling_steps=args.ss_steps,
            slat_guidance_strength=args.slat_guidance,
            slat_sampling_steps=args.slat_steps,
            timeout=args.timeout
        )
        logger.info(f"✅ 3D asset generated successfully: {result_path}")
        sys.exit(0)

    except FileNotFoundError as e:
        logger.error(f"❌ File error: {e}")
        sys.exit(1)
    except requests.ConnectionError as e:
        logger.error(f"❌ Connection error: {e}")
        logger.error("   Make sure TRELLIS API is running at the specified base-url")
        sys.exit(1)
    except requests.HTTPError as e:
        logger.error(f"❌ API error: {e}")
        sys.exit(1)
    except requests.Timeout as e:
        logger.error(f"❌ Timeout error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
