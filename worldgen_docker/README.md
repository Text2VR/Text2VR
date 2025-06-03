# 🌍 WorldGen Docker Example — Text-to-3D Mesh (.ply)

This folder ships a CUDA 12.1 Conda-based image that runs the full **WorldGen + FLUX.1-dev** pipeline.

---

## ✨ Highlights
- CUDA 12.1 + cuDNN 8 base (`nvcr.io/nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04`)
- Miniconda (Py 3.11) + Mamba
- PyTorch 2.x (cu128) + PyTorch3D
- Auto-clone of `WorldGen` & `UniK3D`
- Builds the CUDA KNN kernel at container start
- Generates both Gaussian-splat and mesh `.ply` files

---

## 📁 Folder Layout
```
worldgen_docker/
├── Dockerfile
├── generate_both.py
└── output/           # mapped to /app/WorldGen/output
```

---

## 🚀 Quick Start

```bash
# git clone --recurse-submodules https://github.com/<your-repo>.git

# in docker image, miniconda and mamba may not be needed !!!

cd worldgen_docker
docker build --no-cache -t worldgen .
docker run --gpus all --rm \
  -v $(pwd)/output:/app/WorldGen/output \
  worldgen
```

---

## 🔧 Inside the Container (optional)

```bash
docker run --gpus all -it --rm \
  -v $(pwd)/output:/app/WorldGen/output \
  worldgen bash
```

```bash
# (re)compile kernels
bash submodules/UniK3D/unik3d/ops/knn/compile.sh

# gated model access
pip install huggingface_hub
huggingface-cli login   # use a Read-scoped token
```

---
## 🔥 RUN
```
python generate_both.py
```

---

## 📂 Output Example
```
output/
├── output_mesh.ply
└── output_gaussian.ply
```

---

## 🛠 Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `GatedRepoError` | FLUX.1-dev requires approval | Click **Request access**, wait, rerun |
| `libGL.so.1` missing | OpenGL runtime | `apt-get install libgl1` (pre-installed) |
| Slow first run | Large model download | Mount HF cache: `-v ~/.cache/huggingface:/root/.cache/huggingface` |

---

## 🔗 References
- WorldGen — <https://github.com/ZiYang-xie/WorldGen>  
- FLUX.1-dev — <https://huggingface.co/black-forest-labs/FLUX.1-dev>
