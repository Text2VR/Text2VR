# ⚡ Flash3D Docker Example — Feed-Forward 3D Scene from One Image

CUDA 11.8 container reproducing **Flash3D** (3DV 2025, arXiv 2406.04343) for both *inference* **and** *training* on RealEstate10K.

---

## ✨ Highlights
| Component | Version |
|-----------|---------|
| CUDA      | 11.8 |
| Python    | 3.10 |
| PyTorch   | 2.2.2 + cu118 |
| GCC       | 11.2+ |
| Extras    | git, ninja, ffmpeg, unzip |

---

## 📁 Folder Layout
```
flash3d_docker/
├── Dockerfile
├── demo_infer.py
├── assets/sample.jpg
└── output/
```

---

## 🚀 Quick Demo (Inference Only)

```bash
git clone --recurse-submodules https://github.com/<your-repo>.git
cd flash3d_docker
docker build --no-cache -t flash3d .

docker run --gpus all --rm \
  -v $(pwd)/output:/workspace/output \
  flash3d \
  python demo_infer.py \
     --input assets/sample.jpg \
     --out   output/scene
```

---

## 📚 Full Pipeline — RealEstate10K

### 1 Dataset (takes days)
```bash
python datasets/download_realestate10k.py -d data/RealEstate10K -o data/RealEstate10K -m train
python datasets/download_realestate10k.py -d data/RealEstate10K -o data/RealEstate10K -m test
sh   datasets/dowload_realestate10k_colmap.sh
python -m datasets.preprocess_realestate10k -d data/RealEstate10K -s train
python -m datasets.preprocess_realestate10k -d data/RealEstate10K -s test
```

### 2 Pre-trained Model & Evaluation
```bash
python -m misc.download_pretrained_models -o exp/re10k_v2
sh evaluate.sh exp/re10k_v2
```

### 3 Training (single GPU)
```bash
python train.py \
  +experiment=layered_re10k \
  model.depth.version=v1 \
  train.logging=false
```

### 4 Training (multi-GPU)
```bash
sh train.sh          # edit configs/hydra/cluster as needed
```

---

## 📂 Output Example
```
output/
└── scene.ply
```

---

## 🛠 Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `CUDA error: invalid device ordinal` | GPU not visible | Install NVIDIA Container Toolkit & use `--gpus all` |
| `ModuleNotFoundError: flash3d` | ops not built | `pip install -e .` or `python setup.py install --force` |
| Dataset script needs `ffmpeg` | missing | `apt-get install ffmpeg` (pre-installed) |
| Slow dataset download | single stream | Install aria2 and edit script |

---

## 🔗 References
- Paper — <https://arxiv.org/abs/2406.04343>  
- Code  — <https://github.com/eldar/flash3d>