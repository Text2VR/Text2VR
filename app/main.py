"""
Text2VR FastAPI Application
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .api.panorama import router as panorama_router
from .services.panorama_service import panorama_service


def setup_logging():
    """Configure application logging"""
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper()),
        format=settings.LOG_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set specific loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info(f"Starting Text2VR application on {settings.host}:{settings.port}")
    logger.info(f"Data directory: {settings.data_dir}")
    logger.info(f"Debug mode: {settings.debug}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Text2VR application")
    panorama_service.cleanup()


def create_app() -> FastAPI:
    """Create FastAPI application"""
    
    app = FastAPI(
        title="Text2VR API",
        description="API for generating VR from text descriptions",
        version="1.0.0",
        debug=settings.debug,
        lifespan=lifespan
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure properly for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(panorama_router)
    
    # Static files for web frontend
    web_dist_dir = Path(__file__).parent.parent / "web-dist"
    web_dir = Path(__file__).parent.parent / "web"
    
    # Use React build if available, otherwise fallback to static HTML
    if web_dist_dir.exists():
        app.mount("/", StaticFiles(directory=web_dist_dir, html=True), name="web")
    elif web_dir.exists():
        app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "service": "text2vr-api",
            "version": "1.0.0"
        }
    
    return app


# Create application instance
app = create_app()