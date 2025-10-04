import os
import argparse
os.environ['SPCONV_ALGO'] = 'native'
os.environ['ATTN_BACKEND'] = 'xformers'
import imageio
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

def rotate_x(coords, angle):
    rad = np.radians(angle)
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(rad), -np.sin(rad)],
        [0, np.sin(rad),  np.cos(rad)]
    ])
    return np.dot(coords, rotation_matrix.T)

def rotate_x180(original_ply, output_ply_path, x_angle=180):
    ply_data = PlyData.read(original_ply)
    vertices = ply_data['vertex'].data
    coords = np.vstack((vertices['x'], vertices['y'], vertices['z'])).T
    coords_rotated = rotate_x(coords, x_angle)
    new_vertices = vertices.copy()
    new_vertices['x'] = coords_rotated[:, 0]
    new_vertices['y'] = coords_rotated[:, 1]
    new_vertices['z'] = coords_rotated[:, 2]
    new_ply_data = PlyData([PlyElement.describe(new_vertices, 'vertex')], text=ply_data.text)
    new_ply_data.write(output_ply_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3D Reconstruction')
    parser.add_argument("--task_name", type=str, required=True, help="Task name")
    args = parser.parse_args()

    pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
    pipeline.cuda()

    input_folder = os.path.join("../results", args.task_name, "Single")
    output_folder = os.path.join("../results", args.task_name, "Single3D")
    os.makedirs(output_folder, exist_ok=True)

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for image_file in image_files:
        image = Image.open(os.path.join(input_folder, image_file))
        outputs = pipeline.run(
            image,
            seed=1,
            # 필요시 샘플러 파라미터를 열어 조정
            # sparse_structure_sampler_params={"steps": 12, "cfg_strength": 7.5},
            # slat_sampler_params={"steps": 12, "cfg_strength": 3},
        )

        file_name, _ = os.path.splitext(image_file)
        ply_path = os.path.join(output_folder, f"{file_name}.ply")
        glb_path = os.path.join(output_folder, f"{file_name}.glb")

        # —— PLY 저장 (기존) ——
        outputs['gaussian'][0].save_ply(ply_path)
        rotate_x180(ply_path, ply_path)  # x축 180도 회전

        # —— GLB 저장 (추가) ——
        # mesh가 없을 수도 있으므로 안전하게 꺼냄
        mesh_list = outputs.get('mesh', None)
        mesh0 = mesh_list[0] if mesh_list and len(mesh_list) > 0 else None

        # to_glb: (gaussian, mesh)를 받아 GLB 씬 생성
        glb_scene = postprocessing_utils.to_glb(
            outputs['gaussian'][0],
            mesh0,
            simplify=0.95,     # 삼각형 간소화 비율(필요 없으면 0.0 또는 None)
            texture_size=1024  # 텍스처 해상도
        )
        # 파일명 동일, 확장자만 .glb
        glb_scene.export(glb_path)

        print(f"Generated PLY: {ply_path}")
        print(f"Generated GLB: {glb_path}")
