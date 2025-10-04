#!/usr/bin/env python3
import os
import glob
import numpy as np

def load_ply_points_xyz(ply_path):
    try:
        import trimesh
        mesh = trimesh.load(ply_path, process=False)
        if hasattr(mesh, 'vertices'):
            pts = np.asarray(mesh.vertices)
        elif isinstance(mesh, trimesh.points.PointCloud):
            pts = np.asarray(mesh.vertices)
        else:
            raise RuntimeError("PLY loaded but vertices not found.")
        return pts
    except Exception:
        # fallback: plyfile
        from plyfile import PlyData
        plydata = PlyData.read(ply_path)
        v = plydata['vertex'].data
        x = np.asarray(v['x'], dtype=np.float32)
        y = np.asarray(v['y'], dtype=np.float32)
        z = np.asarray(v['z'], dtype=np.float32)
        return np.stack([x, y, z], axis=1)

def union_sam_masks(mask_dir):
    npys = sorted([p for p in glob.glob(os.path.join(mask_dir, 'mask_*.npy'))])
    if not npys:
        raise FileNotFoundError(f"No SAM masks found at {mask_dir}/mask_*.npy")
    mask = np.load(npys[0])
    for p in npys[1:]:
        mask |= np.load(p)
    mask = mask.astype(np.uint8)
    # (1, H, W) → (H, W)
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    return mask

def robust_range(values, zscore_thr=3.0):
    """값들에서 z-score로 이상치 제거 후 (max - min) 반환"""
    if values.size == 0:
        return 0.0
    mean = values.mean()
    std = values.std() + 1e-8
    z = np.abs((values - mean) / std)
    keep = z < zscore_thr
    vals = values[keep]
    if vals.size == 0:
        vals = values  # 전부 제거되면 원본으로
    return vals.max() - vals.min()

def main(task_name):
    root = f"results/{task_name}"

    ply_path   = os.path.join(root, "background", "point_cloud.ply")
    depth_path = os.path.join(root, "depth", "2DImage_pred.npy")
    mask_dir   = os.path.join(root, "SAM")

    print(f"[INFO] Processing task: {task_name}")
    print(f"[INFO] Root directory: {root}")
    print(f"[INFO] PLY path: {ply_path}")
    print(f"[INFO] Depth path: {depth_path}")
    print(f"[INFO] Mask directory: {mask_dir}")

    # Check if files exist
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"PLY file not found: {ply_path}")
    if not os.path.exists(depth_path):
        raise FileNotFoundError(f"Depth file not found: {depth_path}")
    if not os.path.exists(mask_dir):
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    # 1) 3D 포인트에서 x/y/z 범위
    print("[INFO] Loading point cloud...")
    pts = load_ply_points_xyz(ply_path)   # (N, 3)
    x_range = pts[:,0].max() - pts[:,0].min()
    y_range = pts[:,1].max() - pts[:,1].min()
    z_range = pts[:,2].max() - pts[:,2].min()
    
    print(f"[INFO] 3D ranges - X: {x_range:.6f}, Y: {y_range:.6f}, Z: {z_range:.6f}")

    # 2) 마스크 영역의 깊이 변화량(range_z)
    print("[INFO] Loading masks and depth...")
    mask  = union_sam_masks(mask_dir)     # (H, W)
    depth = np.load(depth_path)           # (H, W) 가정
    valid_depth = depth[mask == 1]
    range_z_img = robust_range(valid_depth, zscore_thr=3.0)
    
    print(f"[INFO] Image depth range: {range_z_img:.6f}")
    print(f"[INFO] Valid depth points: {len(valid_depth)}")

    # 3) 축 스케일 계산
    if x_range <= 1e-8:
        raise ZeroDivisionError("x_range is zero; point cloud may be degenerate.")
    axis_scale_z = range_z_img * (z_range / x_range)

    out = np.array([axis_scale_z], dtype=np.float32)
    out_path = os.path.join(root, "axis_scale.npy")
    np.save(out_path, out)
    print(f"[OK] Saved axis_scale.npy at {out_path} (value = {out[0]:.6f})")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python make_axis_scale.py <task_name>")
        print("Example: python make_axis_scale.py bedroom")
        sys.exit(1)
    main(sys.argv[1])

