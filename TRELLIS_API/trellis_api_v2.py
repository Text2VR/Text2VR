#!/usr/bin/env python3
"""
Improved TRELLIS FastAPI with direct file upload/download
"""
import os
import sys
import asyncio
import logging
import tempfile
import io
from typing import Dict, Any, Optional
from pathlib import Path
import traceback

# Add TRELLIS path
sys.path.append('/app/TRELLIS')

from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from PIL import Image
import torch

# Import TRELLIS components
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="TRELLIS 3D Asset Generator V2", version="2.0.0")

# Global pipeline instance
pipeline = None

class GenerateRequest(BaseModel):
    image_path: str
    asset_name: str
    output_dir: str
    seed: int = 42
    simplify: float = 0.95
    texture_size: int = 1024
    ss_guidance_strength: float = 7.5
    ss_sampling_steps: int = 12
    slat_guidance_strength: float = 3.0
    slat_sampling_steps: int = 12

class GenerateResponse(BaseModel):
    status: str
    glb_path: str = ""
    asset_name: str = ""
    message: str = ""
    processing_time: float = 0.0

@app.on_event("startup")
async def startup_event():
    """Initialize TRELLIS pipeline on startup"""
    global pipeline
    try:
        logger.info("🚀 Loading TRELLIS pipeline...")
        
        # Set environment variables for optimal performance
        os.environ['SPCONV_ALGO'] = 'native'
        
        # Check GPU availability
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available - GPU required for TRELLIS")
        
        logger.info(f"🔧 GPU detected: {torch.cuda.get_device_name()}")
        logger.info(f"📊 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        
        # Load pipeline
        pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
        pipeline.cuda()
        
        logger.info("✅ TRELLIS pipeline loaded successfully!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize TRELLIS pipeline: {str(e)}")
        logger.error(traceback.format_exc())
        raise RuntimeError(f"Pipeline initialization failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    gpu_available = torch.cuda.is_available()
    pipeline_loaded = pipeline is not None
    
    return {
        "status": "healthy" if (gpu_available and pipeline_loaded) else "unhealthy",
        "gpu_available": gpu_available,
        "pipeline_loaded": pipeline_loaded,
        "gpu_memory_used": torch.cuda.memory_allocated() / 1024**3 if gpu_available else 0,
        "gpu_memory_cached": torch.cuda.memory_reserved() / 1024**3 if gpu_available else 0
    }

@app.post("/generate-direct")
async def generate_3d_asset_direct(
    image: UploadFile = File(...),
    asset_name: str = Form("generated_asset"),
    seed: int = Form(42),
    simplify: float = Form(0.95),
    texture_size: int = Form(1024),
    ss_guidance_strength: float = Form(7.5),
    ss_sampling_steps: int = Form(12),
    slat_guidance_strength: float = Form(3.0),
    slat_sampling_steps: int = Form(12)
):
    """
    Generate 3D asset from uploaded image - returns GLB file directly
    """
    import time
    start_time = time.time()
    
    try:
        logger.info(f"🎯 Starting direct 3D generation for: {asset_name}")
        
        # Validate pipeline
        if pipeline is None:
            raise HTTPException(status_code=500, detail="TRELLIS pipeline not initialized")
        
        # Validate image file
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        logger.info(f"📷 Received image: {image.filename} ({image.content_type})")
        
        # Log memory before processing
        if torch.cuda.is_available():
            logger.info(f"📊 GPU memory before: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
        
        # Load image from uploaded file
        try:
            image_content = await image.read()
            pil_image = Image.open(io.BytesIO(image_content)).convert('RGBA')
            logger.info(f"📷 Loaded image: {pil_image.size} pixels")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to process image: {str(e)}")
        
        # Run TRELLIS pipeline
        logger.info("🔄 Running TRELLIS pipeline...")
        try:
            outputs = pipeline.run(
                pil_image,
                seed=seed,
                formats=["gaussian", "mesh"],
                preprocess_image=True,
                sparse_structure_sampler_params={
                    "steps": ss_sampling_steps,
                    "cfg_strength": ss_guidance_strength,
                },
                slat_sampler_params={
                    "steps": slat_sampling_steps,
                    "cfg_strength": slat_guidance_strength,
                },
            )
            logger.info("✅ TRELLIS pipeline completed")
        except Exception as e:
            logger.error(f"❌ Pipeline execution failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {str(e)}")
        
        # Generate GLB file in memory
        logger.info("📦 Generating GLB file...")
        try:
            glb = postprocessing_utils.to_glb(
                outputs['gaussian'][0],
                outputs['mesh'][0],
                simplify=simplify,
                texture_size=texture_size,
                verbose=False
            )
            
            # Create temporary file for GLB
            with tempfile.NamedTemporaryFile(delete=False, suffix='.glb') as tmp_file:
                glb.export(tmp_file.name)
                glb_path = tmp_file.name
            
            logger.info(f"💾 GLB generated successfully")
            
        except Exception as e:
            logger.error(f"❌ GLB generation failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"GLB generation failed: {str(e)}")
        
        # Log memory after processing
        if torch.cuda.is_available():
            logger.info(f"📊 GPU memory after: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
            # Clean up GPU memory
            torch.cuda.empty_cache()
        
        processing_time = time.time() - start_time
        logger.info(f"⏱️  Processing completed in {processing_time:.2f}s")
        
        # Return GLB file as download
        def cleanup():
            try:
                os.unlink(glb_path)
            except:
                pass
        
        return FileResponse(
            glb_path,
            media_type="application/octet-stream",
            filename=f"{asset_name}.glb",
            background=cleanup
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# Keep original endpoint for backward compatibility
@app.post("/generate", response_model=GenerateResponse)
async def generate_3d_asset(request: GenerateRequest):
    """
    Generate 3D asset from input image (original file-based method)
    """
    import time
    start_time = time.time()
    
    try:
        logger.info(f"🎯 Starting 3D generation for: {request.asset_name}")
        
        # Validate pipeline
        if pipeline is None:
            raise HTTPException(status_code=500, detail="TRELLIS pipeline not initialized")
        
        # Validate input image
        if not os.path.exists(request.image_path):
            raise HTTPException(status_code=400, detail=f"Input image not found: {request.image_path}")
        
        # Create output directory
        os.makedirs(request.output_dir, exist_ok=True)
        
        # Log memory before processing
        if torch.cuda.is_available():
            logger.info(f"📊 GPU memory before: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
        
        # Load and validate image
        try:
            image = Image.open(request.image_path).convert('RGBA')
            logger.info(f"📷 Loaded image: {image.size} pixels")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to load image: {str(e)}")
        
        # Run TRELLIS pipeline
        logger.info("🔄 Running TRELLIS pipeline...")
        try:
            outputs = pipeline.run(
                image,
                seed=request.seed,
                formats=["gaussian", "mesh"],
                preprocess_image=True,
                sparse_structure_sampler_params={
                    "steps": request.ss_sampling_steps,
                    "cfg_strength": request.ss_guidance_strength,
                },
                slat_sampler_params={
                    "steps": request.slat_sampling_steps,
                    "cfg_strength": request.slat_guidance_strength,
                },
            )
            logger.info("✅ TRELLIS pipeline completed")
        except Exception as e:
            logger.error(f"❌ Pipeline execution failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {str(e)}")
        
        # Generate GLB file
        logger.info("📦 Generating GLB file...")
        try:
            glb = postprocessing_utils.to_glb(
                outputs['gaussian'][0],
                outputs['mesh'][0],
                simplify=request.simplify,
                texture_size=request.texture_size,
                verbose=False
            )
            
            # Export GLB
            glb_path = os.path.join(request.output_dir, f"{request.asset_name}.glb")
            glb.export(glb_path)
            
            logger.info(f"💾 GLB saved: {glb_path}")
            
        except Exception as e:
            logger.error(f"❌ GLB generation failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"GLB generation failed: {str(e)}")
        
        # Log memory after processing
        if torch.cuda.is_available():
            logger.info(f"📊 GPU memory after: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
            # Clean up GPU memory
            torch.cuda.empty_cache()
        
        processing_time = time.time() - start_time
        logger.info(f"⏱️  Processing completed in {processing_time:.2f}s")
        
        return GenerateResponse(
            status="success",
            glb_path=glb_path,
            asset_name=request.asset_name,
            message="3D asset generated successfully",
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "TRELLIS 3D Asset Generator V2",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "POST /generate-direct": "Upload image and download GLB directly",
            "POST /generate": "Generate 3D asset from file path (legacy)",
            "GET /health": "Health check",
            "GET /": "This information"
        },
        "new_features": [
            "Direct file upload/download",
            "No volume mounts required",
            "Single API call workflow"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)