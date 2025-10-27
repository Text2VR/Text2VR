"""
Inpainting API Client for LangGraph workflows
"""

import time
from typing import Dict, Optional, Any

import requests

from .defaults import INPAINTING_DEFAULTS


class InpaintingAPIClient:
    """Client for communicating with the Background Inpainting API"""

    def __init__(self, base_url: str = "http://localhost:8003"):
        self.base_url = base_url.rstrip("/")

    def health_check(self) -> Dict[str, Any]:
        """Check if the API is healthy"""
        response = requests.get(f"{self.base_url}/health", timeout=10)
        response.raise_for_status()
        return response.json()

    def inpaint_panorama(
        self,
        panorama_path: str,
        mask_dir: str,
        scene_name: str,
        model_id: Optional[str] = None,
        prompt: Optional[str] = None,
        neg_prompt: Optional[str] = None,
        strength: Optional[float] = None,
        guidance: Optional[float] = None,
        steps: Optional[int] = None,
        wrap_pad: Optional[int] = None,
        dilate: Optional[int] = None,
        feather: Optional[int] = None,
        erase: Optional[str] = None,
        seed: Optional[int] = None,
        poll_interval: Optional[int] = None,
        timeout: Optional[int] = None
    ) -> str:
        """
        파노라마 인페인팅을 요청하고 결과 경로를 반환합니다.

        Args:
            panorama_path: 원본 파노라마 이미지 경로 (컨테이너 내부 경로)
            mask_dir: 마스크 디렉토리 경로 (컨테이너 내부 경로)
            scene_name: 씬 이름
            model_id: Hugging Face 모델 ID
            prompt: 포지티브 프롬프트
            neg_prompt: 네거티브 프롬프트
            strength: 인페인팅 강도 (0.0 ~ 1.0)
            guidance: CFG 스케일
            steps: 추론 스텝 수
            wrap_pad: 수평 패딩 (None = auto)
            dilate: 마스크 확장 (None = auto)
            feather: 마스크 페더링
            erase: 마스크 영역 사전 지우기 ("none", "gray", "black")
            seed: 랜덤 시드
            poll_interval: 상태 확인 간격 (초)
            timeout: 최대 대기 시간 (초)

        Returns:
            인페인팅된 파노라마 경로

        Raises:
            requests.HTTPError: API 요청 실패
            TimeoutError: 타임아웃
            RuntimeError: 인페인팅 실패
        """

        # 1. 인페인팅 요청
        request_data = {
            "panorama_path": panorama_path,
            "mask_dir": mask_dir,
            "scene_name": scene_name,
            "model_id": model_id or INPAINTING_DEFAULTS.model_id,
            "prompt": prompt or INPAINTING_DEFAULTS.prompt,
            "neg_prompt": neg_prompt or INPAINTING_DEFAULTS.neg_prompt,
            "strength": strength if strength is not None else INPAINTING_DEFAULTS.strength,
            "guidance": guidance if guidance is not None else INPAINTING_DEFAULTS.guidance,
            "steps": steps if steps is not None else INPAINTING_DEFAULTS.steps,
            "wrap_pad": wrap_pad if wrap_pad is not None else INPAINTING_DEFAULTS.wrap_pad,
            "dilate": dilate if dilate is not None else INPAINTING_DEFAULTS.dilate,
            "feather": feather if feather is not None else INPAINTING_DEFAULTS.feather,
            "erase": erase or INPAINTING_DEFAULTS.erase,
            "seed": seed if seed is not None else INPAINTING_DEFAULTS.seed
        }

        print(f"📤 Sending inpainting request for {scene_name}...")
        response = requests.post(
            f"{self.base_url}/inpaint",
            json=request_data,
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        task_id = result["task_id"]

        print(f"✅ Inpainting task started: {task_id}")

        # 2. 상태 폴링
        start_time = time.time()
        while True:
            elapsed = time.time() - start_time
            current_timeout = timeout if timeout is not None else INPAINTING_DEFAULTS.timeout
            if elapsed > current_timeout:
                raise TimeoutError(f"Inpainting timed out after {current_timeout}s")

            # 상태 확인
            status_response = requests.get(
                f"{self.base_url}/status/{task_id}",
                timeout=10
            )
            status_response.raise_for_status()
            status_data = status_response.json()

            status = status_data["status"]
            message = status_data["message"]

            print(f"📊 Inpainting Status: {status} - {message}")

            if status == "completed":
                result_path = status_data.get("result_path")
                if not result_path:
                    raise RuntimeError("Inpainting completed but no result path returned")
                print(f"✅ Inpainting completed: {result_path}")
                return result_path

            elif status == "failed":
                raise RuntimeError(f"Inpainting failed: {message}")

            elif status in ("queued", "processing"):
                time.sleep(poll_interval if poll_interval is not None else INPAINTING_DEFAULTS.poll_interval)
                continue

            else:
                raise RuntimeError(f"Unknown status: {status}")


# Example usage
if __name__ == "__main__":
    client = InpaintingAPIClient("http://localhost:8003")

    # Health check
    health = client.health_check()
    print(f"Health: {health}")

    # Inpaint panorama
    result_path = client.inpaint_panorama(
        panorama_path="/workspace/data/scene_test/panorama.png",
        mask_dir="/workspace/masking_output/scene_test/masks",
        scene_name="scene_test"
    )
    print(f"Result: {result_path}")
