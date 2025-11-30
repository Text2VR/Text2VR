"""
Service for managing Docker containers
"""

import subprocess
import logging
import time
from typing import Optional, List
from ..core.exceptions import DockerServiceError

logger = logging.getLogger(__name__)

class DockerService:
    """Service for interacting with Docker containers"""
    
    def start_container(self, container_name: str, timeout: int = 10) -> bool:
        """Start a docker container if it's not running"""
        try:
            # Check if running
            if self.is_container_running(container_name):
                logger.info(f"Container {container_name} is already running")
                return True
                
            logger.info(f"Starting container: {container_name}")
            result = subprocess.run(
                ["docker", "start", container_name],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if result.returncode != 0:
                logger.error(f"Failed to start container {container_name}: {result.stderr}")
                return False
                
            # Wait a bit for the service inside to be ready
            time.sleep(2)
            return True
            
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout starting container {container_name}")
            return False
        except Exception as e:
            logger.error(f"Error starting container {container_name}: {e}")
            raise DockerServiceError(f"Failed to start container {container_name}") from e

    def stop_container(self, container_name: str, timeout: int = 10) -> bool:
        """Stop a docker container"""
        try:
            logger.info(f"Stopping container: {container_name}")
            result = subprocess.run(
                ["docker", "stop", container_name],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if result.returncode != 0:
                logger.warning(f"Failed to stop container {container_name}: {result.stderr}")
                return False
                
            return True
            
        except subprocess.TimeoutExpired:
            logger.warning(f"Timeout stopping container {container_name}")
            return False
        except Exception as e:
            logger.error(f"Error stopping container {container_name}: {e}")
            return False

    def is_container_running(self, container_name: str) -> bool:
        """Check if a container is running"""
        try:
            result = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Running}}", container_name],
                capture_output=True,
                text=True
            )
            return result.stdout.strip() == "true"
        except Exception:
            return False

# Global docker service instance
docker_service = DockerService()
