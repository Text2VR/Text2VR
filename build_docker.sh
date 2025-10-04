#!/bin/bash

# Flash-Sculptor Docker 이미지 빌드 스크립트

echo "🔨 Building Flash-Sculptor Docker image..."

# Docker 이미지 빌드
docker build -t flashsculptor:latest .

if [ $? -eq 0 ]; then
    echo "✅ Docker image built successfully!"
    echo "📦 Image name: flashsculptor:latest"
    echo ""
    echo "🚀 To run the container, use:"
    echo "   ./run_docker.sh"
else
    echo "❌ Docker build failed!"
    exit 1
fi
