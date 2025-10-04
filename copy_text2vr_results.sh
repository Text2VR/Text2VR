#!/bin/bash

# Text2VR 결과를 Flash-Sculptor 형식으로 복사하는 스크립트

SCENE_NAME=${1:-"bedroom"}

echo "📁 Copying Text2VR results to Flash-Sculptor format..."
echo "Scene: ${SCENE_NAME}"

# 소스 경로
TEXT2VR_RESULTS="/home/suyeon8235/Flash-Sculptor/results/${SCENE_NAME}"
FLASH_SCULPTOR_RESULTS="/home/suyeon8235/Flash-Sculptor/results/${SCENE_NAME}"

# 대상 디렉토리 생성
mkdir -p "${FLASH_SCULPTOR_RESULTS}/background"
mkdir -p "${FLASH_SCULPTOR_RESULTS}/SAM"
mkdir -p "${FLASH_SCULPTOR_RESULTS}/Single"

# 파일 복사
if [ -d "${TEXT2VR_RESULTS}" ]; then
    echo "✅ Copying files from ${TEXT2VR_RESULTS}..."
    
    # 2D 이미지
    if [ -f "${TEXT2VR_RESULTS}/2DImage.png" ]; then
        cp "${TEXT2VR_RESULTS}/2DImage.png" "${FLASH_SCULPTOR_RESULTS}/"
        echo "  ✅ 2DImage.png"
    fi
    
    # 배경 파일들
    if [ -f "${TEXT2VR_RESULTS}/background/background_recover.png" ]; then
        cp "${TEXT2VR_RESULTS}/background/background_recover.png" "${FLASH_SCULPTOR_RESULTS}/background/"
        echo "  ✅ background_recover.png"
    fi
    
    if [ -f "${TEXT2VR_RESULTS}/background/point_cloud.ply" ]; then
        cp "${TEXT2VR_RESULTS}/background/point_cloud.ply" "${FLASH_SCULPTOR_RESULTS}/background/"
        echo "  ✅ point_cloud.ply"
    fi
    
    if [ -f "${TEXT2VR_RESULTS}/background/mask_0.png" ]; then
        cp "${TEXT2VR_RESULTS}/background/mask_0.png" "${FLASH_SCULPTOR_RESULTS}/background/"
        echo "  ✅ mask_0.png"
    fi
    
    # SAM 결과들
    if [ -d "${TEXT2VR_RESULTS}/SAM" ]; then
        cp -r "${TEXT2VR_RESULTS}/SAM"/* "${FLASH_SCULPTOR_RESULTS}/SAM/"
        echo "  ✅ SAM results"
    fi
    
    # Single 객체들
    if [ -d "${TEXT2VR_RESULTS}/Single" ]; then
        cp -r "${TEXT2VR_RESULTS}/Single"/* "${FLASH_SCULPTOR_RESULTS}/Single/"
        echo "  ✅ Single objects"
    fi
    
    echo "🎉 Copy completed!"
    echo "📁 Results available at: ${FLASH_SCULPTOR_RESULTS}"
else
    echo "❌ Text2VR results not found at: ${TEXT2VR_RESULTS}"
    echo "💡 Make sure to run Text2VR pipeline first!"
    exit 1
fi
