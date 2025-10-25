"""Asset cropping utilities for extracting segmented objects from panorama."""

import os
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np


def crop_assets_from_panorama(
    panorama_path: str,
    segmentation_output_dir: str,
    results_json_path: str = None
) -> Dict[str, List[str]]:
    """
    파노라마에서 세그멘테이션된 asset들을 각각 추출합니다.

    Args:
        panorama_path: 원본 파노라마 이미지 경로
        segmentation_output_dir: 세그멘테이션 결과가 저장된 디렉토리
        results_json_path: results.json 파일 경로 (optional)

    Returns:
        각 asset별로 추출된 이미지 경로들의 딕셔너리
        예: {"sofa": ["output/cropped/sofa_0.png"], "plant": ["output/cropped/plant_0.png", "output/cropped/plant_1.png"]}
    """

    if not os.path.exists(panorama_path):
        raise FileNotFoundError(f"Panorama not found: {panorama_path}")

    # 파노라마 이미지 로드
    panorama = cv2.imread(panorama_path, cv2.IMREAD_UNCHANGED)
    if panorama is None:
        raise ValueError(f"Failed to load panorama: {panorama_path}")

    print(f"📸 Loaded panorama: {panorama.shape}")

    # 출력 디렉토리 설정
    output_dir = os.path.join(segmentation_output_dir, "cropped_assets")
    os.makedirs(output_dir, exist_ok=True)

    # 마스크 디렉토리
    masks_dir = os.path.join(segmentation_output_dir, "masks")
    if not os.path.exists(masks_dir):
        raise FileNotFoundError(f"Masks directory not found: {masks_dir}")

    cropped_files = {}

    # 마스크 파일들 처리
    mask_files = sorted(Path(masks_dir).glob("*.png"))

    for mask_path in mask_files:
        asset_name = mask_path.stem  # 파일명에서 확장자 제거

        print(f"🔍 Processing {asset_name}...")

        # 마스크 로드
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"⚠️ Failed to load mask: {mask_path}")
            continue

        # 마스크 크기가 파노라마와 다르면 리사이즈
        if mask.shape != panorama.shape[:2]:
            mask = cv2.resize(mask, (panorama.shape[1], panorama.shape[0]))

        # 마스크를 3채널로 변환 (RGB용)
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) if len(panorama.shape) == 3 else mask

        # 마스크 적용 (마스크가 있는 부분만 유지)
        masked_image = cv2.bitwise_and(panorama, mask_3ch)

        # 마스크 영역의 bounding box 찾기
        coords = cv2.findNonZero(mask)
        if coords is None:
            print(f"⚠️ Empty mask for {asset_name}")
            continue

        x, y, w, h = cv2.boundingRect(coords)

        # Bounding box로 크롭
        cropped = masked_image[y:y+h, x:x+w]

        # 저장
        output_path = os.path.join(output_dir, f"{asset_name}.png")
        cv2.imwrite(output_path, cropped)

        # 결과 기록
        if asset_name not in cropped_files:
            cropped_files[asset_name] = []
        cropped_files[asset_name].append(output_path)

        print(f"✅ Saved: {output_path} (size: {w}x{h})")

    print(f"\n🎉 Cropped {len(cropped_files)} asset types")
    print(f"📁 Output directory: {output_dir}")

    return cropped_files


def crop_assets_with_transparency(
    panorama_path: str,
    segmentation_output_dir: str,
    scene_name: str = None,
) -> Dict[str, List[str]]:
    """
    파노라마에서 세그멘테이션된 asset들을 투명 배경으로 추출합니다.

    Args:
        panorama_path: 원본 파노라마 이미지 경로
        segmentation_output_dir: 세그멘테이션 결과가 저장된 디렉토리
        scene_name: 씬 이름 (폴더 구분용)

    Returns:
        각 asset별로 추출된 PNG 이미지 경로들의 딕셔너리 (알파 채널 포함)
    """

    if not os.path.exists(panorama_path):
        raise FileNotFoundError(f"Panorama not found: {panorama_path}")

    # 파노라마 이미지 로드
    panorama = cv2.imread(panorama_path, cv2.IMREAD_UNCHANGED)
    if panorama is None:
        raise ValueError(f"Failed to load panorama: {panorama_path}")

    # BGR을 BGRA로 변환 (알파 채널 추가)
    if panorama.shape[2] == 3:
        panorama = cv2.cvtColor(panorama, cv2.COLOR_BGR2BGRA)

    print(f"📸 Loaded panorama: {panorama.shape}")

    # 출력 디렉토리 설정 - seged_assets/{scene_name} 구조
    base_output_dir = "/home/0in/workspace/Text2VR/seged_assets"
    if scene_name:
        output_dir = os.path.join(base_output_dir, scene_name)
    else:
        output_dir = os.path.join(base_output_dir, "default")

    os.makedirs(output_dir, exist_ok=True)

    # 마스크 디렉토리
    masks_dir = os.path.join(segmentation_output_dir, "masks")
    if not os.path.exists(masks_dir):
        raise FileNotFoundError(f"Masks directory not found: {masks_dir}")

    cropped_files = {}

    # 마스크 파일들 처리
    mask_files = sorted(Path(masks_dir).glob("*.png"))

    for mask_path in mask_files:
        asset_name = mask_path.stem

        print(f"🔍 Processing {asset_name} with transparency...")

        # 마스크 로드
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"⚠️ Failed to load mask: {mask_path}")
            continue

        # 마스크 크기가 파노라마와 다르면 리사이즈
        if mask.shape != panorama.shape[:2]:
            mask = cv2.resize(mask, (panorama.shape[1], panorama.shape[0]))

        # 파노라마 복사
        result = panorama.copy()

        # 알파 채널에 마스크 적용
        result[:, :, 3] = mask

        # 마스크 영역의 bounding box 찾기
        coords = cv2.findNonZero(mask)
        if coords is None:
            print(f"⚠️ Empty mask for {asset_name}")
            continue

        x, y, w, h = cv2.boundingRect(coords)

        # Bounding box로 크롭
        cropped = result[y:y+h, x:x+w]

        # 저장
        output_path = os.path.join(output_dir, f"{asset_name}.png")
        cv2.imwrite(output_path, cropped)

        # 결과 기록
        if asset_name not in cropped_files:
            cropped_files[asset_name] = []
        cropped_files[asset_name].append(output_path)

        print(f"✅ Saved with transparency: {output_path} (size: {w}x{h})")

    print(f"\n🎉 Cropped {len(cropped_files)} asset types with transparency")
    print(f"📁 Output directory: {output_dir}")

    return cropped_files
