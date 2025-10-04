#!/bin/bash

# Flash-Sculptor Docker 컨테이너 백그라운드 실행 스크립트

echo "🚀 Starting Flash-Sculptor Docker container in background..."

# 기존 컨테이너 정리
docker stop flashsculptor-daemon 2>/dev/null || true
docker rm flashsculptor-daemon 2>/dev/null || true

# 결과 폴더 생성 (호스트)
mkdir -p /home/suyeon8235/Flash-Sculptor/results

# Docker 컨테이너 백그라운드 실행
docker run -d --gpus all \
  --name flashsculptor-daemon \
  -v /home/suyeon8235/Flash-Sculptor/results:/app/Flash-Sculptor/results \
  -v /home/suyeon8235/Text2VR:/workspace/Text2VR \
  -v /home/suyeon8235/bedroom:/app/Flash-Sculptor/bedroom \
  -v /home/suyeon8235:/workspace \
  flashsculptor:latest \
  tail -f /dev/null

if [ $? -eq 0 ]; then
    echo "✅ Flash-Sculptor container started in background!"
    echo "📦 Container name: flashsculptor-daemon"
    echo ""
    echo "🔧 To access the container:"
    echo "   docker exec -it flashsculptor-daemon /bin/bash"
    echo ""
    echo "🛑 To stop the container:"
    echo "   docker stop flashsculptor-daemon"
else
    echo "❌ Failed to start container!"
    exit 1
fi
