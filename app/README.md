## Execution
Front, Back, Docker Container 실행 방법

- docker pull로 아래 3가지 이미지 다운 후 진행
    - 0in11/text2vr-dreamscene360
    - 0in11/text2vr-bg_inpaint 
    - 0in11/text2vr-asset_seg

- 총 3개의 터미널 창을 띄워놓고 각 터미널 당 아래의 방식을 하나씩 진행
    1. ~/Text2VR 경로에서 docker compose up -d 실행

    2-1. ~/Text2VR 경로에서 source myenv/bin/activate 명령어를 통해 myenv 가상환경 활성화
    2-2. 가상환경이 활성화된 ~/Text2VR 경로에서 uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload 명령어 실행 
    2.3. "INFO:     Application startup complete." -> 이 문구가 뜨면 FastAPI 서버 구동 성공

    3. ~/Text2VR/src 경로에서 npm run dev 실행

## Host Outputs

각 파이프라인 단계에서 생성되는 산출물은 호스트 파일시스템에 다음 경로로 저장

- **Panorama**  
  - 메인 결과: `data/<scene_name>/panorama.png`  
  - 실패 시 최신 백업 탐색: `stitch_output/im_*.png`
- **Segmentation**  
  - 마스크·결과 JSON: `masking_output/<scene_name>/`  
  - 투명 배경 에셋 크롭: `seged_assets/<scene_name>/*.png`
- **Background Inpainting**  
  - 인페인팅 결과: `inpainted_pano/<scene_name>/inpainted_panorama.png`
- **3D Asset Generation (TRELLIS)**  
  - GLB 파일: `output/3d_assets/<scene_name>/<asset>.glb`
- **Gaussian Splatting**  
  - 학습 PLY: `plyoutput/<scene_name>/point_cloud_iter_*.ply`

필요 시 `app/workflows/nodes.py`와 `app/workflows/defaults.py`에서 경로 기본값을 확인하거나 수정 가능

## API Request Parameters

마이크로서비스에 전달되는 주요 파라미터와 기본값은 `app/workflows/defaults.py`에서 중앙 관리. 각 서비스가 실제로 사용하는 필드는 아래와 같음.

- **Panorama API (`dreamscene360`)**
  - `POST /generate`: `text`, `scene_name`, `use_self_refinement`, `num_prompt`, `max_rounds`
  - `POST /panorama_to_ply`: `panorama_path`, `output_name`
  - `POST /train_gaussian`: `panorama_path`, `scene_name`, `iterations`, `save_iterations`, `white_background`, `sh_degree`, `gen_res`
  - 기타 상태/결과 GET 엔드포인트는 경로 파라미터만 사용

- **Segmentation API**
  - `POST /segment`: `panorama_path`, `scene_name`, `sam_checkpoint`, `openai_api_key`, `box_threshold`, `text_threshold`
  - GET 엔드포인트는 경로 파라미터만 사용

- **Inpainting API**
  - `POST /inpaint`: `panorama_path`, `mask_dir`, `scene_name`, `model_id`, `prompt`, `neg_prompt`, `strength`, `guidance`, `steps`, `wrap_pad`, `dilate`, `feather`, `erase`, `seed`
  - GET 엔드포인트는 경로 파라미터만 사용

- **TRELLIS 3D API**
  - `POST /generate-direct` (업로드 방식): 이미지 파일 + `asset_name`, `seed`, `simplify`, `texture_size`, `ss_guidance_strength`, `ss_sampling_steps`, `slat_guidance_strength`, `slat_sampling_steps`
  - `POST /generate` (경로 방식): `image_path`, `asset_name`, `output_dir` + 동일한 파라미터 세트
  - `GET /health`: 추가 파라미터 없음

- **Unity Export (FastAPI 내부)**
  - `GET /unity/{scene_name}/assets/{asset_name}.glb`: TRELLIS가 생성한 GLB 다운로드
  - `GET /unity/{scene_name}/scene.ply?iteration=5000`: Gaussian PLY 다운로드 (미지정 시 최신/기본 파일 반환)
  - `GET /unity/latest/assets.zip`: 가장 최근 완료된 작업의 GLB를 ZIP으로 반환  
  - `GET /unity/latest/scene.ply`: 가장 최근 완료된 작업의 PLY 반환

필요한 값을 변경하려면 우선 해당 서비스의 FastAPI 모델을 확장한 뒤, `defaults.py`와 관련 클라이언트를 함께 업데이트해야 전체 파이프라인에 반영됨.(파라미터 변수 추가할 때만 해당)
