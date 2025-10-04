#!/bin/bash

# Flash-Sculptor Docker 컨테이너 실행 스크립트

echo "🚀 Starting Flash-Sculptor Docker container..."

# 결과 폴더 생성 (호스트)
mkdir -p /home/suyeon8235/Flash-Sculptor/results

# Docker 컨테이너 실행 (마운트 포함)
docker run -it --gpus all \
  --name flashsculptor-container \
  -v /home/suyeon8235/Flash-Sculptor/results:/app/Flash-Sculptor/results \
  -v /home/suyeon8235/Text2VR:/workspace/Text2VR \
  -v /home/suyeon8235/bedroom:/app/Flash-Sculptor/bedroom \
  -v /home/suyeon8235:/workspace \
  --rm \
  flashsculptor:latest

echo "✅ Container stopped"
