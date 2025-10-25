# TRELLIS 3D Asset Generation 통합 계획

**작성일**: 2025-10-22
**상태**: 검증 완료 ✅
**목적**: LangGraph 워크플로우에 TRELLIS API를 통합하여 세그멘테이션된 오브젝트를 3D GLB 에셋으로 자동 변환

---

## 📋 목차

1. [개요](#개요)
2. [현재 워크플로우 분석](#현재-워크플로우-분석)
3. [TRELLIS 통합 목표](#trellis-통합-목표)
4. [검증 완료 사항](#검증-완료-사항)
5. [구현 계획](#구현-계획)
6. [디렉토리 구조](#디렉토리-구조)
7. [VRAM 관리 전략](#vram-관리-전략)
8. [구현 체크리스트](#구현-체크리스트)
9. [예상 이슈 및 해결방안](#예상-이슈-및-해결방안)

---

## 개요

### 목적
세그멘테이션된 2D 오브젝트 이미지를 TRELLIS API를 통해 3D GLB 에셋으로 자동 변환하여, VR 씬 구성에 필요한 3D 에셋을 자동 생성합니다.

### 기대 효과
- **자동화**: 수동 3D 모델링 없이 2D 이미지에서 3D 에셋 생성
- **일관성**: 파노라마에서 추출한 오브젝트를 동일한 스타일의 3D 에셋으로 변환
- **효율성**: 워크플로우 내에서 원스톱으로 2D→3D 변환 처리

---

## 현재 워크플로우 분석

### 기존 LangGraph 워크플로우

```
query_rewrite → panorama_generation → segmentation → inpainting → ply_generation → END
```

#### 각 노드의 역할

1. **query_rewrite** (`app/workflows/nodes.py:25-61`)
   - 사용자 입력을 파노라마 생성에 적합하게 재작성
   - OpenAI LLM 사용
   - Scene name 생성 (UUID 기반)

2. **panorama_generation** (`app/workflows/nodes.py:64-221`)
   - DreamScene360 API 호출
   - 360도 파노라마 이미지 생성
   - 경로: `/home/0in/workspace/Text2VR/data/{scene_name}/panorama.png`

3. **segmentation** (`app/workflows/nodes.py:224-328`)
   - SAM + GroundingDINO로 오브젝트 세그멘테이션
   - 마스크 생성: `/home/0in/workspace/Text2VR/masking_output/{scene_name}/masks/`
   - **크롭된 에셋 생성**: `/home/0in/workspace/Text2VR/seged_assets/{scene_name}/`
   - 투명 배경(RGBA) PNG 파일로 저장
   - 완료 후 컨테이너 중지 (VRAM 절약)

4. **inpainting** (`app/workflows/nodes.py:331-425`)
   - Stable Diffusion으로 배경 인페인팅
   - 오브젝트 제거된 파노라마 생성
   - 완료 후 컨테이너 중지 (VRAM 절약)

5. **ply_generation** (`app/workflows/nodes.py:428-500`)
   - 인페인팅된 파노라마를 PLY 포인트 클라우드로 변환
   - Depth estimation 기반

### 현재 State 구조 (`app/workflows/states.py`)

```python
class WorkflowState(TypedDict):
    task_id: str
    user_input: str
    rewritten_query: str
    scene_name: str
    panorama_path: str
    segmentation_data: Dict[str, object]
    inpainted_panorama_path: str
    ply_path: str
    messages: Annotated[List[BaseMessage], operator.add]
```

### 크롭된 에셋 생성 로직 (`app/workflows/asset_cropper.py:102-198`)

✅ **검증 완료**: `crop_assets_with_transparency()` 함수가 이미 구현되어 있음

**특징**:
- 투명 배경(RGBA) PNG 생성
- Bounding box 기반 크롭
- 출력 경로: `/home/0in/workspace/Text2VR/seged_assets/{scene_name}/{asset_name}.png`
- 반환값: `Dict[str, List[str]]` (asset_name → 파일 경로 리스트)

**검증된 동작**:
```python
# nodes.py:265-269에서 호출
cropped_assets = crop_assets_with_transparency(
    panorama_path=panorama_path,
    segmentation_output_dir=segmentation_output_dir,
    scene_name=state['scene_name']
)
# 결과: {"sofa": ["/path/to/sofa.png"], "plant": ["/path/to/plant.png"], ...}
```

---

## TRELLIS 통합 목표

### 새로운 워크플로우

```
query_rewrite → panorama_generation → segmentation → asset_3d_generation → inpainting → ply_generation → END
                                                              ↑ 새로 추가
```

### asset_3d_generation 노드의 역할

1. `cropped_assets`에서 세그멘테이션된 에셋 이미지 가져오기
2. 각 에셋 이미지를 TRELLIS API에 전송
3. GLB 파일 다운로드 및 저장
4. State에 `asset_3d_paths` 업데이트
5. TRELLIS 컨테이너 중지 (VRAM 해제)

### 기대 입출력

**입력**:
- `state['cropped_assets']`: `{"sofa": ["/path/to/sofa.png"], ...}`
- `state['scene_name']`: `"scene_abc123de"`

**출력**:
- `state['asset_3d_paths']`: `{"sofa": "/path/to/3d_assets/scene_abc123de/sofa.glb", ...}`

---

## 검증 완료 사항

### ✅ 1. Docker 이미지 존재 확인

```bash
$ docker images | grep trellis
trellis    v1    5745868dd8d4    6 weeks ago    43.2GB
```

**결과**: TRELLIS 이미지가 로컬에 존재함 (trellis:v1)

### ✅ 2. TRELLIS API 검증

**사용할 API**: `trellis_api_v2.py`

**엔드포인트**: `POST /generate-direct`
- 파일 업로드 방식 (multipart/form-data)
- 직접 GLB 파일 다운로드
- 볼륨 마운트 불필요

**코드 위치**: `/home/0in/workspace/Text2VR/TRELLIS_API/trellis_api_v2.py:98-213`

**파라미터**:
```python
image: UploadFile           # 입력 이미지 (RGBA PNG)
asset_name: str = "generated_asset"
seed: int = 42
simplify: float = 0.95
texture_size: int = 1024
ss_guidance_strength: float = 7.5
ss_sampling_steps: int = 12
slat_guidance_strength: float = 3.0
slat_sampling_steps: int = 12
```

### ✅ 3. 디렉토리 구조 확인

**기존 디렉토리**:
```
/home/0in/workspace/Text2VR/
├── data/                    # 파노라마 저장
│   └── {scene_name}/
│       └── panorama.png
├── masking_output/          # 세그멘테이션 결과
│   └── {scene_name}/
│       ├── masks/           # 마스크 PNG
│       └── results.json
├── seged_assets/            # ✅ 이미 존재 - 크롭된 에셋
│   └── {scene_name}/
│       ├── sofa.png
│       ├── plant.png
│       └── ...
└── output/                  # 기타 출력
    └── {scene_name}/
```

**추가 필요 디렉토리**:
```
/home/0in/workspace/Text2VR/
└── output/
    └── 3d_assets/           # 새로 생성 필요
        └── {scene_name}/
            ├── sofa.glb
            ├── plant.glb
            └── ...
```

### ✅ 4. 경로 매핑 검증

**호스트 ↔ 컨테이너 경로 매핑**:

| 항목 | 호스트 경로 | 컨테이너 경로 |
|------|------------|--------------|
| 크롭된 에셋 | `/home/0in/workspace/Text2VR/seged_assets/{scene}` | `/app/seged_assets/{scene}` |
| 3D 에셋 저장 | `/home/0in/workspace/Text2VR/output/3d_assets/{scene}` | `/app/output/3d_assets/{scene}` |
| 캐시 | `/home/0in/workspace/Text2VR/cache/hf` | `/root/.cache/huggingface` |

### ✅ 5. VRAM 사용량 확인

| 서비스 | VRAM (idle) | VRAM (processing) | 비고 |
|--------|-------------|-------------------|------|
| DreamScene360 | - | ~8-10GB | 파노라마 생성 |
| Segmentation | - | ~6GB | SAM + GroundingDINO |
| Inpainting | - | ~6GB | Stable Diffusion |
| **TRELLIS** | **5.3GB** | **6-8GB** | Image-to-3D |

**결론**: 동시 실행 시 VRAM 부족 → 각 단계 완료 후 컨테이너 중지 필요

---

## 구현 계획

### Task 1: docker-compose.yml에 TRELLIS 서비스 추가

**파일**: `docker-compose.yml`

**추가할 서비스**:

```yaml
services:
  # ... 기존 서비스들 ...

  # TRELLIS 3D Asset Generation API
  trellis-api:
    image: trellis:v1
    container_name: text2vr_trellis_api
    working_dir: /app
    ports:
      - "8004:8000"
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - SPCONV_ALGO=native
    ipc: host
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - ./TRELLIS_API/trellis_api_v2.py:/app/trellis_api_v2.py:ro
      - ./seged_assets:/app/seged_assets:ro  # 크롭된 에셋 읽기
      - ./output/3d_assets:/app/output/3d_assets  # GLB 저장
      - ./cache/hf:/root/.cache/huggingface
      - ./cache/torch:/root/.cache/torch
    command: ["python", "/app/trellis_api_v2.py"]
    restart: unless-stopped
```

**포트 할당**: 8004 (다른 서비스와 충돌 방지)

---

### Task 2: WorkflowState에 필드 추가

**파일**: `app/workflows/states.py`

**수정**:

```python
class WorkflowState(TypedDict):
    """Represents the shared state that flows through the LangGraph workflow."""

    task_id: str
    user_input: str
    rewritten_query: str
    scene_name: str
    panorama_path: str
    segmentation_data: Dict[str, object]
    inpainted_panorama_path: str
    ply_path: str

    # 추가: 크롭된 에셋 경로
    cropped_assets: Dict[str, List[str]]  # {"sofa": ["/path/to/sofa.png"], ...}

    # 추가: 3D 에셋 경로
    asset_3d_paths: Dict[str, str]  # {"sofa": "/path/to/sofa.glb", ...}

    messages: Annotated[List[BaseMessage], operator.add]
```

**참고**: `cropped_assets`는 `nodes.py:309`에서 이미 사용 중이므로 타입 정의만 추가

---

### Task 3: TRELLIS API Client 작성

**파일**: `app/workflows/trellis_client.py` (새로 생성)

**구현**:

```python
#!/usr/bin/env python3
"""
TRELLIS API Client for 3D asset generation
"""

import requests
from typing import Optional
from pathlib import Path


class TrellisAPIClient:
    def __init__(self, base_url: str = "http://localhost:8004"):
        self.base_url = base_url

    def health_check(self) -> dict:
        """Check if TRELLIS API is healthy and ready"""
        response = requests.get(f"{self.base_url}/health", timeout=10)
        response.raise_for_status()
        return response.json()

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
        Generate 3D GLB asset from image using TRELLIS API

        Args:
            image_path: Path to input image (RGBA PNG with transparent background)
            asset_name: Name for the asset (used in filename)
            output_path: Full path where GLB file should be saved
            seed: Random seed for reproducibility
            texture_size: Texture resolution (512, 1024, or 2048)
            simplify: Mesh simplification ratio (0.0-1.0)
            ss_guidance_strength: Sparse structure guidance strength
            ss_sampling_steps: Sparse structure sampling steps
            slat_guidance_strength: SLAT guidance strength
            slat_sampling_steps: SLAT sampling steps
            timeout: Request timeout in seconds

        Returns:
            Path to generated GLB file

        Raises:
            FileNotFoundError: If input image doesn't exist
            requests.HTTPError: If API request fails
        """
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Input image not found: {image_path}")

        # Prepare multipart form data
        with open(image_path, 'rb') as f:
            files = {'image': (Path(image_path).name, f, 'image/png')}

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

            # Call TRELLIS API
            response = requests.post(
                f"{self.base_url}/generate-direct",
                files=files,
                data=data,
                timeout=timeout,
                stream=True
            )
            response.raise_for_status()

        # Save GLB file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return str(output_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print("Usage: python trellis_client.py <image_path> <asset_name> <output_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    asset_name = sys.argv[2]
    output_path = sys.argv[3]

    client = TrellisAPIClient()

    try:
        # Health check
        health = client.health_check()
        print(f"✅ TRELLIS API Status: {health['status']}")
        print(f"📊 GPU Memory: {health['gpu_memory_used']:.2f}GB")

        # Generate 3D asset
        print(f"🎯 Generating 3D asset for: {asset_name}")
        result_path = client.generate_3d_asset(
            image_path=image_path,
            asset_name=asset_name,
            output_path=output_path
        )
        print(f"✅ 3D asset saved: {result_path}")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
```

---

### Task 4: asset_3d_generation_node 구현

**파일**: `app/workflows/nodes.py`

**추가할 노드**:

```python
def asset_3d_generation_node(state: WorkflowState) -> WorkflowState:
    """
    Generate 3D GLB assets from cropped segmentation images using TRELLIS API
    """
    from .trellis_client import TrellisAPIClient
    import subprocess
    import time

    cropped_assets = state.get("cropped_assets", {})

    if not cropped_assets:
        print("⚠️ No cropped assets found, skipping 3D generation")
        return {
            **state,
            "asset_3d_paths": {},
            "messages": [
                HumanMessage(content="No assets to convert to 3D")
            ],
        }

    try:
        print(f"🎲 Starting 3D asset generation for {len(cropped_assets)} assets")

        # TRELLIS 컨테이너 시작 (이미 실행 중이면 무시됨)
        try:
            subprocess.run(
                ["docker", "start", "text2vr_trellis_api"],
                capture_output=True,
                timeout=10
            )
            print("🚀 TRELLIS container started")
            # 파이프라인 로딩 대기 (약 10초)
            time.sleep(10)
        except Exception as e:
            print(f"⚠️ Failed to start TRELLIS container: {e}")

        client = TrellisAPIClient(base_url=settings.TRELLIS_API_URL)

        # Health check
        try:
            health = client.health_check()
            if health['status'] != 'healthy':
                raise Exception(f"TRELLIS API not healthy: {health}")
            print(f"✅ TRELLIS API ready (GPU memory: {health['gpu_memory_used']:.2f}GB)")
        except Exception as e:
            raise Exception(f"TRELLIS API health check failed: {e}")

        asset_3d_paths = {}
        scene_name = state["scene_name"]

        # 각 에셋에 대해 3D 생성
        for asset_name, image_paths in cropped_assets.items():
            if not image_paths:
                continue

            # 첫 번째 이미지 사용 (보통 하나만 있음)
            image_path = image_paths[0]

            print(f"🎯 Generating 3D for: {asset_name}")

            # 출력 경로 설정
            output_dir = f"/home/0in/workspace/Text2VR/output/3d_assets/{scene_name}"
            output_path = f"{output_dir}/{asset_name}.glb"

            try:
                # TRELLIS API 호출
                result_path = client.generate_3d_asset(
                    image_path=image_path,
                    asset_name=asset_name,
                    output_path=output_path,
                    timeout=120  # 2분 타임아웃
                )

                asset_3d_paths[asset_name] = result_path
                print(f"✅ 3D asset created: {result_path}")

            except Exception as asset_exc:
                print(f"❌ Failed to generate 3D for {asset_name}: {asset_exc}")
                # 계속 진행 (일부 실패해도 나머지 처리)
                continue

        # TRELLIS 컨테이너 중지 (VRAM 해제)
        try:
            print("🛑 Stopping TRELLIS container to free VRAM...")
            subprocess.run(
                ["docker", "stop", "text2vr_trellis_api"],
                capture_output=True,
                timeout=10
            )
            print("✅ TRELLIS container stopped")
        except Exception as e:
            print(f"⚠️ Failed to stop TRELLIS container: {e}")

        # Task manager 업데이트
        try:
            if state.get("task_id") and asset_3d_paths:
                task_manager.update_task_status(
                    task_id=state["task_id"],
                    status=TaskStatus.PROCESSING,
                    message=f"3D assets generated ({len(asset_3d_paths)} assets), starting inpainting...",
                )
                print(f"✅ Task manager updated for task_id: {state['task_id']}")
        except Exception as tm_exc:
            print(f"⚠️ Failed to update task manager: {tm_exc}")

        print(f"🎉 3D generation completed: {len(asset_3d_paths)}/{len(cropped_assets)} assets")

        return {
            **state,
            "asset_3d_paths": asset_3d_paths,
            "messages": [
                HumanMessage(
                    content=f"3D assets generated: {list(asset_3d_paths.keys())}"
                )
            ],
        }

    except Exception as exc:
        print(f"❌ 3D asset generation failed: {exc}")
        return {
            **state,
            "asset_3d_paths": {},
            "messages": [
                HumanMessage(content=f"3D generation failed: {str(exc)}")
            ],
        }
```

---

### Task 5: Workflow에 노드 추가

**파일**: `app/workflows/workflow.py`

**수정**:

```python
from .nodes import (
    panorama_generation_node,
    query_rewrite_node,
    segmentation_node,
    asset_3d_generation_node,  # 추가
    inpainting_node,
    ply_generation_node,
)

def create_workflow():
    """Compile and return the LangGraph workflow for panorama generation."""

    workflow = StateGraph(WorkflowState)
    workflow.add_node("query_rewrite", query_rewrite_node)
    workflow.add_node("panorama_generation", panorama_generation_node)
    workflow.add_node("segmentation", segmentation_node)
    workflow.add_node("asset_3d_generation", asset_3d_generation_node)  # 추가
    workflow.add_node("inpainting", inpainting_node)
    workflow.add_node("ply_generation", ply_generation_node)

    workflow.set_entry_point("query_rewrite")
    workflow.add_edge("query_rewrite", "panorama_generation")
    workflow.add_edge("panorama_generation", "segmentation")
    workflow.add_edge("segmentation", "asset_3d_generation")  # 추가
    workflow.add_edge("asset_3d_generation", "inpainting")    # 수정
    workflow.add_edge("inpainting", "ply_generation")
    workflow.add_edge("ply_generation", END)

    return workflow.compile()
```

**langgraph_workflow.py 수정**:

```python
from .nodes import (
    panorama_generation_node,
    query_rewrite_node,
    segmentation_node,
    asset_3d_generation_node,  # 추가
)

__all__ = [
    "WorkflowState",
    "create_workflow",
    "query_rewrite_node",
    "panorama_generation_node",
    "segmentation_node",
    "asset_3d_generation_node",  # 추가
]
```

---

### Task 6: config.py에 TRELLIS API URL 추가

**파일**: `app/config.py`

**수정**:

```python
class Settings(BaseSettings):
    """Application settings"""

    # ... 기존 필드 ...

    # External APIs
    DREAMSCENE_API_URL: str
    SEGMENTATION_API_URL: str = "http://localhost:8002"
    INPAINTING_API_URL: str = "http://localhost:8003"
    TRELLIS_API_URL: str = "http://localhost:8004"  # 추가

    # ... 나머지 ...
```

---

### Task 7: 디렉토리 생성

**실행**:

```bash
mkdir -p /home/0in/workspace/Text2VR/output/3d_assets
```

---

## 디렉토리 구조

### 최종 디렉토리 구조

```
/home/0in/workspace/Text2VR/
├── app/
│   └── workflows/
│       ├── states.py                    # State 정의 (수정)
│       ├── nodes.py                     # 노드 구현 (추가: asset_3d_generation_node)
│       ├── workflow.py                  # 워크플로우 조립 (수정)
│       ├── langgraph_workflow.py        # 호환성 레이어 (수정)
│       ├── trellis_client.py           # TRELLIS API 클라이언트 (새로 생성)
│       ├── segmentation_client.py      # 기존
│       ├── inpainting_client.py        # 기존
│       └── asset_cropper.py            # 기존
├── config.py                            # 설정 (수정)
├── docker-compose.yml                   # Docker 서비스 정의 (수정)
├── TRELLIS_API/
│   ├── trellis_api.py
│   └── trellis_api_v2.py               # 사용할 API
├── data/
│   └── {scene_name}/
│       └── panorama.png
├── masking_output/
│   └── {scene_name}/
│       ├── masks/
│       │   ├── sofa.png
│       │   └── plant.png
│       └── results.json
├── seged_assets/                        # ✅ 크롭된 에셋 (투명 배경)
│   └── {scene_name}/
│       ├── sofa.png
│       └── plant.png
├── output/
│   └── 3d_assets/                       # ✨ 새로 생성: 3D GLB 에셋
│       └── {scene_name}/
│           ├── sofa.glb
│           └── plant.glb
└── docs/
    └── trellis-integration-plan.md      # 이 문서
```

---

## VRAM 관리 전략

### VRAM 사용량 분석

| 단계 | 서비스 | VRAM (processing) | 컨테이너 상태 |
|------|--------|-------------------|--------------|
| 1. Query Rewrite | - | 0GB | - |
| 2. Panorama Generation | DreamScene360 | 8-10GB | 실행 중 |
| 3. Segmentation | ASSET_SEG | 6GB | 실행 중 → 완료 후 중지 ✅ |
| **4. 3D Generation** | **TRELLIS** | **6-8GB** | **시작 → 완료 후 중지** ✅ |
| 5. Inpainting | BG_INPAINT | 6GB | 시작 → 완료 후 중지 ✅ |
| 6. PLY Generation | DreamScene360 | - | 재사용 |

### 컨테이너 중지 패턴 (이미 구현됨)

**Segmentation 완료 후** (`nodes.py:279-286`):
```python
subprocess.run(["docker", "stop", "text2vr_segmentation_api"],
               capture_output=True, timeout=10)
```

**Inpainting 완료 후** (`nodes.py:386-393`):
```python
subprocess.run(["docker", "stop", "text2vr_inpainting_api"],
               capture_output=True, timeout=10)
```

**TRELLIS 완료 후** (새로 구현):
```python
subprocess.run(["docker", "stop", "text2vr_trellis_api"],
               capture_output=True, timeout=10)
```

### 권장 GPU 사양

- **최소**: 12GB VRAM (RTX 3080 Ti, RTX 4070 Ti)
- **권장**: 16GB+ VRAM (RTX 4080, A4000, A5000)
- **이상적**: 24GB+ VRAM (RTX 4090, A6000)

---

## 구현 체크리스트

### Phase 1: 환경 준비

- [x] TRELLIS Docker 이미지 확인 (`trellis:v1` 존재)
- [ ] 3D assets 디렉토리 생성 (`mkdir -p output/3d_assets`)
- [ ] docker-compose.yml에 TRELLIS 서비스 추가
- [ ] Docker Compose로 TRELLIS 컨테이너 시작 테스트

### Phase 2: 코드 구현

- [ ] `app/config.py`: TRELLIS_API_URL 추가
- [ ] `app/workflows/states.py`: WorkflowState에 필드 추가
  - [ ] `cropped_assets: Dict[str, List[str]]` 타입 정의
  - [ ] `asset_3d_paths: Dict[str, str]` 추가
- [ ] `app/workflows/trellis_client.py`: API 클라이언트 작성
  - [ ] `health_check()` 메서드
  - [ ] `generate_3d_asset()` 메서드
- [ ] `app/workflows/nodes.py`: asset_3d_generation_node 추가
  - [ ] 컨테이너 시작 로직
  - [ ] Health check
  - [ ] 각 에셋 반복 처리
  - [ ] GLB 파일 저장
  - [ ] 컨테이너 중지
- [ ] `app/workflows/workflow.py`: 워크플로우에 노드 추가
  - [ ] `add_node("asset_3d_generation", ...)`
  - [ ] 엣지 수정 (`segmentation` → `asset_3d_generation` → `inpainting`)
- [ ] `app/workflows/langgraph_workflow.py`: export 업데이트

### Phase 3: 테스트

- [ ] TRELLIS API 단독 테스트 (curl 또는 Python 스크립트)
- [ ] TrellisAPIClient 단위 테스트
- [ ] asset_3d_generation_node 단독 실행
- [ ] 전체 워크플로우 통합 테스트
- [ ] VRAM 사용량 모니터링 (`nvidia-smi`)
- [ ] 에러 케이스 테스트
  - [ ] 이미지 파일 없음
  - [ ] API 연결 실패
  - [ ] 일부 에셋 생성 실패

### Phase 4: 문서화 및 배포

- [x] 통합 계획 문서 작성 (이 문서)
- [ ] README 업데이트
- [ ] API 사용 예제 추가
- [ ] 트러블슈팅 가이드 작성

---

## 예상 이슈 및 해결방안

### 이슈 1: TRELLIS 컨테이너 시작 지연

**문제**: 파이프라인 로딩에 시간이 걸림 (약 10초)

**해결방안**:
```python
subprocess.run(["docker", "start", "text2vr_trellis_api"])
time.sleep(10)  # 로딩 대기

# Health check로 확인
for _ in range(30):  # 최대 30번 시도 (30초)
    try:
        health = client.health_check()
        if health['status'] == 'healthy':
            break
    except:
        time.sleep(1)
```

### 이슈 2: 처리 시간 증가

**문제**: 에셋당 30-40초 소요, 3개면 2분 추가

**해결방안**:
- 병렬 처리 고려 (여러 GPU 사용 시)
- 또는 순차 처리하며 사용자에게 진행 상황 업데이트

### 이슈 3: 일부 에셋 생성 실패

**문제**: 특정 에셋이 TRELLIS에서 실패할 수 있음

**해결방안**:
```python
for asset_name, image_paths in cropped_assets.items():
    try:
        result = client.generate_3d_asset(...)
        asset_3d_paths[asset_name] = result
    except Exception as e:
        print(f"⚠️ Failed for {asset_name}: {e}")
        continue  # 다음 에셋 계속 처리
```

### 이슈 4: 투명 배경 처리

**문제**: TRELLIS가 투명 배경을 제대로 처리하지 못할 수 있음

**검증 완료**:
- `crop_assets_with_transparency()`가 RGBA PNG 생성 ✅
- TRELLIS API가 RGBA 입력 받음 (`trellis_api_v2.py:136`) ✅

### 이슈 5: VRAM 부족

**문제**: 여러 서비스 동시 실행 시 OOM

**해결방안**:
- ✅ 이미 구현된 패턴: 각 단계 완료 후 컨테이너 중지
- TRELLIS도 동일하게 적용

### 이슈 6: 볼륨 마운트 권한

**문제**: Docker 볼륨 마운트 시 권한 문제

**해결방안**:
```bash
# 디렉토리 권한 설정
chmod -R 755 /home/0in/workspace/Text2VR/output/3d_assets
```

---

## 참고 자료

### 관련 파일

- TRELLIS API 가이드: `/home/0in/workspace/TRELLIS_API_Guide.md`
- TRELLIS API v2: `/home/0in/workspace/Text2VR/TRELLIS_API/trellis_api_v2.py`
- 현재 워크플로우: `/home/0in/workspace/Text2VR/app/workflows/`
- Asset Cropper: `/home/0in/workspace/Text2VR/app/workflows/asset_cropper.py`

### API 엔드포인트

| 서비스 | 포트 | 엔드포인트 |
|--------|------|-----------|
| DreamScene360 | 8001 | `/generate`, `/panorama_to_ply` |
| Segmentation | 8002 | `/segment`, `/status/{task_id}` |
| Inpainting | 8003 | `/inpaint`, `/status/{task_id}` |
| **TRELLIS** | **8004** | `/generate-direct`, `/health` |

---

## 버전 히스토리

- **v1.0** (2025-10-22): 초안 작성 및 검증 완료
  - Docker 이미지 확인 완료
  - 디렉토리 구조 검증 완료
  - API 엔드포인트 검증 완료
  - VRAM 관리 전략 수립 완료

---

## 다음 단계

1. **즉시 실행 가능**:
   - `mkdir -p /home/0in/workspace/Text2VR/output/3d_assets`
   - docker-compose.yml 수정
   - config.py 수정

2. **코드 구현** (1-2시간):
   - trellis_client.py 작성
   - asset_3d_generation_node 구현
   - workflow.py 수정

3. **테스트** (30분):
   - 단위 테스트
   - 통합 테스트
   - VRAM 모니터링

4. **배포**:
   - Git commit
   - 문서 업데이트

---

**작성자**: Claude (llmops-expert agent)
**검증자**: 0in
**승인 상태**: 검증 완료 ✅
