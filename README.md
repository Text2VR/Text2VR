# 🐳 Text & Image → 3D — Docker Collection

GPU-ready Docker environments for state-of-the-art 3D generation / reconstruction:

| Folder | Model | Task | CUDA | Framework |
|--------|-------|------|------|-----------|
| `worldgen_docker/` | WorldGen + FLUX.1-dev | Text → 3D Mesh (.ply) | 12.1 | PyTorch cu128 |
| `flash3d_docker/`  | Flash3D | Single image → 3D Scene | 11.8 | PyTorch 2.2.2 |

---

## ⚡ Quick Tour

```bash
git clone --recurse-submodules https://github.com/<your-repo>.git
cd <your-repo>

# WorldGen demo
cd worldgen_docker
docker build -t worldgen .
docker run --gpus all --rm -v $(pwd)/output:/app/WorldGen/output worldgen
cd ..

# Flash3D demo
cd flash3d_docker
docker build -t flash3d .
docker run --gpus all -it \
  -v $(pwd)/data:/workspace/flash3d/data \
  -v $(pwd)/output:/workspace/output \
  --name flash3d_container flash3d_autorun

python evaluate.py \
  +experiment=layered_re10k \
  +dataset.crop_border=true \
  dataset.test_split_path=splits/re10k_mine_filtered/test_files.txt \
  model.depth.version=v1 \
  ++eval.save_vis=true
```

---

## ➕ Add a New Model

1. Create `<model>_docker/` with `Dockerfile`, scripts, `output/`.  
2. Write `<model>_docker/README.md`.  
3. Add the folder to the table above.  
4. Follow this CLI pattern:

```bash
docker run --gpus all <image> <entrypoint> --input ... --out output/<name>
```

---

<!--## License

Each sub-folder inherits its upstream license. This meta-repo is MIT-licensed.-->
