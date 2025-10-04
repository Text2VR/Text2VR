#!/bin/bash

# Flash-Sculptor 통합 설정 및 실행 스크립트

echo "🔧 Flash-Sculptor Setup & Run Script"
echo "======================================"

# 1. Docker 이미지 빌드
echo "📦 Step 1: Building Docker image..."
./build_docker.sh

if [ $? -ne 0 ]; then
    echo "❌ Build failed! Exiting..."
    exit 1
fi

echo ""
echo "🎯 Choose run mode:"
echo "1) Interactive mode (recommended for development)"
echo "2) Background daemon mode (for services)"
echo "3) Exit"
echo ""
read -p "Enter choice (1-3): " choice

case $choice in
    1)
        echo "🚀 Starting interactive container..."
        ./run_docker.sh
        ;;
    2)
        echo "🚀 Starting background daemon..."
        ./run_docker_daemon.sh
        ;;
    3)
        echo "👋 Exiting..."
        exit 0
        ;;
    *)
        echo "❌ Invalid choice!"
        exit 1
        ;;
esac
