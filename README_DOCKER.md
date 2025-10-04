# Flash-Sculptor Docker 실행 가이드

## 🚀 빠른 시작

### 1. 통합 실행 (권장)
```bash
cd /home/suyeon8235/Flash-Sculptor
./setup_and_run.sh
```

### 2. 단계별 실행

#### Docker 이미지 빌드
```bash
./build_docker.sh
```

#### 컨테이너 실행 (대화형)
```bash
./run_docker.sh
```

#### 컨테이너 실행 (백그라운드)
```bash
./run_docker_daemon.sh
```

## 📁 마운트된 폴더들

| 호스트 경로 | 컨테이너 경로 | 설명 |
|------------|--------------|------|
| `/home/suyeon8235/Flash-Sculptor/results` | `/app/Flash-Sculptor/results` | Flash-Sculptor & Text2VR 결과물 |
| `/home/suyeon8235/Text2VR` | `/workspace/Text2VR` | Text2VR 전체 프로젝트 |
| `/home/suyeon8235/bedroom` | `/app/Flash-Sculptor/bedroom` | bedroom 데이터 |
| `/home/suyeon8235` | `/workspace` | 전체 홈 디렉토리 |

## 🔧 컨테이너 접근

### 대화형 모드
```bash
./run_docker.sh
```

### 백그라운드 모드에서 접근
```bash
docker exec -it flashsculptor-daemon /bin/bash
```

## 📋 Text2VR 결과 복사

Text2VR 파이프라인 실행 후 Flash-Sculptor 형식으로 복사:

```bash
./copy_text2vr_results.sh [SCENE_NAME]
```

예시:
```bash
./copy_text2vr_results.sh bedroom
```

## 🛑 컨테이너 정리

### 백그라운드 컨테이너 중지
```bash
docker stop flashsculptor-daemon
```

### 모든 컨테이너 정리
```bash
docker stop flashsculptor-daemon flashsculptor-container 2>/dev/null || true
docker rm flashsculptor-daemon flashsculptor-container 2>/dev/null || true
```

## 📊 결과 확인

Flash-Sculptor 결과는 다음 경로에서 확인할 수 있습니다:

```
/home/suyeon8235/Flash-Sculptor/results/[SCENE_NAME]/
├── background/
│   ├── point_cloud.ply
│   ├── background_recover.png
│   └── mask_0.png
├── SAM/
│   ├── automatic_label_output.jpg
│   ├── label.json
│   └── ...
├── Single/
│   ├── object_0.png
│   └── ...
└── 2DImage.png
```

## 🔍 문제 해결

### GPU 인식 안됨
```bash
# NVIDIA Docker 설치 확인
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

### 권한 문제
```bash
# 스크립트 실행 권한 부여
chmod +x *.sh
```

### 컨테이너 재시작
```bash
docker stop flashsculptor-daemon
docker rm flashsculptor-daemon
./run_docker_daemon.sh
```
