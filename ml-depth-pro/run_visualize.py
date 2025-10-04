from PIL import Image
import depth_pro
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import cv2

def visualize_depth(depth, output_path, title="Depth Map"):
    """Depth map을 다양한 방식으로 시각화하고 저장"""
    
    # 1. Grayscale 시각화
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.imshow(depth, cmap='gray')
    plt.colorbar(label="Normalized Depth")
    plt.title(f"{title} (Grayscale)")
    plt.axis("off")
    
    # 2. Plasma colormap 시각화
    plt.subplot(2, 2, 2)
    plt.imshow(depth, cmap='plasma')
    plt.colorbar(label="Normalized Depth")
    plt.title(f"{title} (Plasma)")
    plt.axis("off")
    
    # 3. Inferno colormap 시각화
    plt.subplot(2, 2, 3)
    plt.imshow(depth, cmap='inferno')
    plt.colorbar(label="Normalized Depth")
    plt.title(f"{title} (Inferno)")
    plt.axis("off")
    
    # 4. Viridis colormap 시각화
    plt.subplot(2, 2, 4)
    plt.imshow(depth, cmap='viridis')
    plt.colorbar(label="Normalized Depth")
    plt.title(f"{title} (Viridis)")
    plt.axis("off")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # 개별 이미지로도 저장
    base_name = os.path.splitext(output_path)[0]
    
    # Grayscale 저장
    depth_array_255 = (depth * 255).astype(np.uint8)
    Image.fromarray(depth_array_255, mode='L').save(f"{base_name}_gray.png")
    
    # Colormap으로 저장
    depth_colormap_plasma = cv2.applyColorMap(depth_array_255, cv2.COLORMAP_PLASMA)
    cv2.imwrite(f"{base_name}_plasma.png", depth_colormap_plasma)
    
    depth_colormap_inferno = cv2.applyColorMap(depth_array_255, cv2.COLORMAP_INFERNO)
    cv2.imwrite(f"{base_name}_inferno.png", depth_colormap_inferno)
    
    print(f"시각화 완료: {output_path}")
    print(f"개별 이미지 저장: {base_name}_gray.png, {base_name}_plasma.png, {base_name}_inferno.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Estimation with Visualization')
    parser.add_argument("--task_name", type=str, required=True, help="Task name")
    parser.add_argument("--image_path", type=str, default=None, help="Custom image path (optional)")
    args = parser.parse_args()

    # Load model and preprocessing transform
    model, transform = depth_pro.create_model_and_transforms()
    model.eval()

    # 이미지 경로 설정
    if args.image_path:
        image_path = args.image_path
        print(f"사용자 지정 이미지 사용: {image_path}")
    else:
        image_path = os.path.join("../results", args.task_name, "2DImage.png")
        print(f"기본 이미지 경로 사용: {image_path}")

    # 이미지 존재 확인
    if not os.path.exists(image_path):
        print(f"오류: 이미지 파일을 찾을 수 없습니다: {image_path}")
        exit(1)

    # Load and preprocess an image.
    image, _, f_px = depth_pro.load_rgb(image_path)
    image = transform(image)

    # Run inference.
    prediction = model.infer(image, f_px=f_px)
    depth = prediction["depth"]  # Depth in [m].
    focallength_px = prediction["focallength_px"]  # Focal length in pixels.

    # Normalize depth to range [0,1].
    depth = np.array(depth)
    normalized_depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))

    # 출력 디렉토리 생성
    if args.image_path:
        # 사용자 지정 이미지인 경우 workspace에 저장 (Docker 볼륨 마운트)
        output_dir = "/workspace/depth_output"
        os.makedirs(output_dir, exist_ok=True)
        task_name = os.path.splitext(os.path.basename(args.image_path))[0]
    else:
        output_dir = os.path.join("../results", args.task_name, "depth")
        os.makedirs(output_dir, exist_ok=True)
        task_name = args.task_name

    # .npy 파일 저장
    output_path = os.path.join(output_dir, f"{task_name}_pred.npy")
    np.save(output_path, normalized_depth)
    print(f"Depth 데이터 저장: {output_path}")

    # 시각화
    viz_output_path = os.path.join(output_dir, f"{task_name}_depth_visualization.png")
    visualize_depth(normalized_depth, viz_output_path, f"Depth Map - {task_name}")

    print("Depth 추정 및 시각화 완료!")
