"""
Configuration management for Text2VR
"""

import os
from pathlib import Path
from typing import Optional, List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # API Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    
    # Workspace Configuration
    # Default to the current user's home directory if not specified
    WORKSPACE_ROOT: Path = Path(os.getcwd())
    
    # Directories
    @property
    def output_dir(self) -> Path:
        return self.WORKSPACE_ROOT / "output"
        
    @property
    def workflows_dir(self) -> Path:
        return Path(__file__).parent.parent / "workflows"
        
    @property
    def dreamscene_dir(self) -> Path:
        return self.WORKSPACE_ROOT / "DREAMSCENE360"
    
    # External APIs
    DREAMSCENE_API_URL: str = "http://localhost:8001"
    SEGMENTATION_API_URL: str = "http://localhost:8002"
    INPAINTING_API_URL: str = "http://localhost:8003"
    TRELLIS_API_URL: str = "http://localhost:8004"
    
    OPENAI_API_KEY: str
    OPENAI_BASE_URL: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o"
    OPENAI_TEMPERATURE: float = 0.7
    
    # Task Management
    TASK_TIMEOUT: int = 3600
    STATUS_CHECK_INTERVAL: int = 2

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    class Config:
        env_file = Path(__file__).parent.parent.parent / ".env"
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields in .env

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_task_dir(self, scene_name: str) -> Path:
        """Get the task-specific output directory"""
        return self.output_dir / scene_name

    def get_task_paths(self, scene_name: str) -> dict:
        """Get all paths for a specific task"""
        task_dir = self.get_task_dir(scene_name)
        return {
            "task_dir": task_dir,
            "panorama": task_dir / "panorama.png",
            "stitch": task_dir / "stitch",
            "masking": task_dir / "masking",
            "inpainted": task_dir / "inpainted.png",
            "assets": task_dir / "assets",
            "ply": task_dir / "ply",
            "3d": task_dir / "3d",
        }

    def ensure_task_dirs(self, scene_name: str) -> dict:
        """Create all directories for a task and return paths"""
        paths = self.get_task_paths(scene_name)
        for key, path in paths.items():
            if key not in ["panorama", "inpainted"]:  # 파일은 제외
                path.mkdir(parents=True, exist_ok=True)
        return paths


# Global settings instance
settings = Settings()
