# Text2VR 컨테이너 API 파라미터 정리

동료들이 Docker 컨테이너로 제공되는 각 FastAPI 서비스를 손쉽게 실험할 수 있도록, 지원하는 엔드포인트와 요청 파라미터를 한눈에 볼 수 있게 정리했습니다. 표에 없는 필드는 현재 백엔드에서 무시하거나 지원하지 않는 값입니다.

> 기본 호스트: `http://localhost:<포트>` (docker-compose 포트 기준)

---

## DreamScene360 Panorama API (`panorama-api`, 포트 8001)

| 메서드 | 경로 | 설명 | 요청 파라미터 (JSON Body) | 기본값/비고 |
|--------|------|------|---------------------------|-------------|
| POST | `/generate` | 파노라마 생성 | `text` *(필수)* | 장면 설명 텍스트 |
|  |  |  | `scene_name` | 지정하지 않으면 `scene_<uuid>` |
|  |  |  | `use_self_refinement` | `false` |
|  |  |  | `num_prompt` | `3` |
|  |  |  | `max_rounds` | `3` |
| POST | `/panorama_to_ply` | 파노라마 → PLY 변환 | `panorama_path` *(필수)* | 컨테이너 내부 경로 (`/workspace/...`) |
|  |  |  | `output_name` | 기본 `panorama_pointcloud.ply` |
| POST | `/train_gaussian` | Gaussian Splatting 학습 | `panorama_path` *(필수)* | 컨테이너 내부 경로 |
|  |  |  | `scene_name` | 없으면 `gaussian_scene_<uuid>` |
|  |  |  | `iterations` | `100` |
|  |  |  | `save_iterations` | `[50, 100]` |
|  |  |  | `gen_res` | `512` *(현재 사용 안 함)* |
|  |  |  | `white_background` | `false` |
|  |  |  | `sh_degree` | `3` |
| GET | `/health` | 상태 확인 | - | JSON 헬스 체크 |
| GET | `/status/{task_id}` | 작업 상태 | - | Path 파라미터만 사용 |
| GET | `/result/{task_id}` | 결과 파노라마 다운로드 | - | Path 파라미터만 사용 |
| GET | `/tasks` | 전체 작업 목록 | - |  |

---

## Asset Segmentation API (`segmentation-api`, 포트 8002)

| 메서드 | 경로 | 설명 | 요청 파라미터 (JSON Body) | 기본값/비고 |
|--------|------|------|---------------------------|-------------|
| POST | `/segment` | 파노라마 세그멘테이션 | `panorama_path` *(필수)* | 컨테이너 내부 경로 (`/app/host_data/...`) |
|  |  |  | `scene_name` *(필수)* | 출력 디렉토리명 |
|  |  |  | `sam_checkpoint` | `/app/checkpoints/sam_vit_h_4b8939.pth` |
|  |  |  | `openai_api_key` | 기본은 환경변수 `OPENAI_API_KEY` |
|  |  |  | `box_threshold` | `0.20` |
|  |  |  | `text_threshold` | `0.15` |
| GET | `/health` | 상태 확인 | - |  |
| GET | `/status/{task_id}` | 작업 상태 | - | Path 파라미터만 사용 |
| GET | `/result/{task_id}` | 세그멘테이션 JSON 다운로드 | - | Path 파라미터만 사용 |
| GET | `/tasks` | 전체 작업 목록 | - |  |

---

## Background Inpainting API (`inpainting-api`, 포트 8003)

| 메서드 | 경로 | 설명 | 요청 파라미터 (JSON Body) | 기본값/비고 |
|--------|------|------|---------------------------|-------------|
| POST | `/inpaint` | 배경 인페인팅 | `panorama_path` *(필수)* | 컨테이너 내부 경로 (`/workspace/data/...`) |
|  |  |  | `mask_dir` *(필수)* | 마스크 디렉토리 (`/workspace/masking_output/...`) |
|  |  |  | `scene_name` *(필수)* | 결과 디렉토리명 |
|  |  |  | `model_id` | `"diffusers/stable-diffusion-xl-1.0-inpainting-0.1"` |
|  |  |  | `prompt` | `"clean empty interior background, …"` |
|  |  |  | `neg_prompt` | `"sofa, couch, armchair, …"` |
|  |  |  | `strength` | `0.94` |
|  |  |  | `guidance` | `5.0` |
|  |  |  | `steps` | `40` |
|  |  |  | `wrap_pad` | `null` (자동) |
|  |  |  | `dilate` | `null` (자동) |
|  |  |  | `feather` | `0` |
|  |  |  | `erase` | `"gray"` (`"none"`/`"black"` 가능) |
|  |  |  | `seed` | `0` |
| GET | `/health` | 상태 확인 | - | `cuda_available`, `model_loaded` 포함 |
| GET | `/status/{task_id}` | 작업 상태 | - | Path 파라미터만 사용 |
| GET | `/result/{task_id}` | 인페인팅 결과 이미지 | - | Path 파라미터만 사용 |
| GET | `/tasks` | 전체 작업 목록 | - |  |

---

## TRELLIS 3D Asset API (`trellis-api`, 포트 8004)

### 파일 업로드 방식 (`POST /generate-direct`)

- **폼 필드** (multipart/form-data):
  - `image` *(필수)*: 업로드 이미지 파일
  - `asset_name`: 기본 `"generated_asset"`
  - `seed`: 기본 `42`
  - `simplify`: 기본 `0.95` (0.0~1.0)
  - `texture_size`: 기본 `1024` (512/1024/2048)
  - `ss_guidance_strength`: 기본 `7.5`
  - `ss_sampling_steps`: 기본 `12`
  - `slat_guidance_strength`: 기본 `3.0`
  - `slat_sampling_steps`: 기본 `12`

### 경로 기반 방식 (`POST /generate`)

- **JSON Body**:
  - `image_path` *(필수)*: 컨테이너 내부 이미지 경로 (`/app/...`)
  - `asset_name` *(필수)*
  - `output_dir` *(필수)*: GLB 저장 경로
  - `seed`, `simplify`, `texture_size`, `ss_guidance_strength`, `ss_sampling_steps`, `slat_guidance_strength`, `slat_sampling_steps`: 업로드 방식과 동일 기본값

### 기타 엔드포인트

| 메서드 | 경로 | 설명 | 비고 |
|--------|------|------|------|
| GET | `/health` | 파이프라인 상태 확인 | GPU 메모리, 로드 상태 반환 |

---

## 사용 예시

```bash
# DreamScene360 파노라마 생성
curl -X POST http://localhost:8001/generate \
  -H "Content-Type: application/json" \
  -d '{
        "text": "sunny beach boardwalk with cafes",
        "scene_name": "scene_demo",
        "use_self_refinement": true,
        "num_prompt": 4,
        "max_rounds": 2
      }'

# 세그멘테이션 실행
curl -X POST http://localhost:8002/segment \
  -H "Content-Type: application/json" \
  -d '{
        "panorama_path": "/app/host_data/scene_demo/panorama.png",
        "scene_name": "scene_demo",
        "box_threshold": 0.25,
        "text_threshold": 0.2
      }'

# 인페인팅 실행
curl -X POST http://localhost:8003/inpaint \
  -H "Content-Type: application/json" \
  -d '{
        "panorama_path": "/workspace/data/scene_demo/panorama.png",
        "mask_dir": "/workspace/masking_output/scene_demo/masks",
        "scene_name": "scene_demo",
        "strength": 0.9,
        "guidance": 6.5,
        "steps": 50
      }'

# TRELLIS 파일 업로드 예시
curl -X POST http://localhost:8004/generate-direct \
  -F "image=@./seged_assets/scene_demo/chair.png" \
  -F "asset_name=chair" \
  -F "texture_size=2048" \
  -o chair.glb
```

필요에 따라 위 파라미터 값을 조정하면서 실험하면 됩니다. 현재 FastAPI 모델에 정의돼 있지 않은 값은 백엔드에서 받아주지 않으니, 새로운 파라미터가 필요하면 먼저 모델/엔드포인트부터 확장해야 합니다.  
공통으로 쓰는 기본값은 `app/workflows/defaults.py`에 정리돼 있으니 참고하거나 수정 시 한 곳만 업데이트하면 됩니다.
