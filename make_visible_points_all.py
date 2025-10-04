#!/usr/bin/env python3
import numpy as np
from plyfile import PlyData
import sys

def main(task_name):
    ply = f"./results/{task_name}/background/point_cloud.ply"
    
    print(f"[INFO] Processing task: {task_name}")
    print(f"[INFO] PLY file: {ply}")
    
    # Check if PLY file exists
    import os
    if not os.path.exists(ply):
        raise FileNotFoundError(f"PLY file not found: {ply}")
    
    # Read PLY and get vertex count
    N = PlyData.read(ply)['vertex'].count
    print(f"[INFO] Total vertices: {N}")
    
    # Create array of all point indices
    visible_points = np.arange(N, dtype=np.int64)
    
    # Save to file
    output_path = f"./results/{task_name}/visible_points.npy"
    np.save(output_path, visible_points)
    print(f"[OK] Saved visible_points.npy at {output_path} (count = {N})")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python make_visible_points_all.py <task_name>")
        print("Example: python make_visible_points_all.py bedroom")
        sys.exit(1)
    main(sys.argv[1])

