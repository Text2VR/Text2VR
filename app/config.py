"""
Configuration management for Text2VR
"""

from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # API Configuration
    host: str
    port: int
    debug: bool
    
    # Directories
    data_dir: Path = Path("/home/0in/workspace/Text2VR/data")
    workflows_dir: Path = Path(__file__).parent / "workflows"
    dreamscene_dir: Path = Path("/home/0in/workspace/Text2VR/DREAMSCENE360")
    
    # External APIs
    DREAMSCENE_API_URL: str
    SEGMENTATION_API_URL: str = "http://localhost:8002"
    INPAINTING_API_URL: str = "http://localhost:8003"
    TRELLIS_API_URL: str = "http://localhost:8004"
    OPENAI_API_KEY: str
    OPENAI_BASE_URL: Optional[str] = None
    OPENAI_MODEL: str
    OPENAI_TEMPERATURE: float
    
    # Task Management
    TASK_TIMEOUT: int
    STATUS_CHECK_INTERVAL: int

    # Mock Pipeline Mode
    MOCK_PIPELINE_MODE: bool = False
    MOCK_PANORAMA_PATH: str = "/home/capstoneproj0310/Text2VR/demo_assets/panorama.png"
    MOCK_SEGMENTATION_DIR: str = "/home/capstoneproj0310/Text2VR/demo_assets/segmented"
    MOCK_SEGMENTATION_VIZ_PATH: str = "/home/capstoneproj0310/Text2VR/demo_assets/segmentation_viz.png"
    MOCK_INPAINTED_PATH: str = "/home/capstoneproj0310/Text2VR/demo_assets/inpainted.png"

    # Mock Pipeline Timing (in seconds)
    MOCK_PANORAMA_DELAY: int = 3
    MOCK_SEGMENTATION_DELAY: int = 4
    MOCK_INPAINTING_DELAY: int = 5

    # Logging
    LOG_LEVEL: str
    LOG_FORMAT: str
    
    class Config:
        env_file = Path(__file__).parent.parent / ".env"
        case_sensitive = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure data directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()