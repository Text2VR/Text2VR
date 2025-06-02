import torch
from worldgen import WorldGen
import open3d as o3d

prompt = "A Minecraft-style Room with Chests and Torches"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Gaussian Splat
worldgen = WorldGen(mode="t2s", device=device)
splat = worldgen.generate_world(prompt)
splat.save("output_gaussian.ply")

# Mesh
mesh = worldgen.generate_world(prompt, return_mesh=True)
o3d.io.write_triangle_mesh("output_mesh.ply", mesh)
