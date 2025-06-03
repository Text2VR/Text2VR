# 🌀 DreamScene360 Docker Usage Guide

## 🚀 Overview

DreamScene360 is a powerful text-to-3D pipeline that generates immersive 360° panoramic scenes and reconstructs them into 3D Gaussian splats. This README outlines how to use the Docker-based workflow, from image build to scene generation and viewing, using an L4 GPU-enabled GCP VM.

---

## 🛠️ Environment Setup (Docker)

### 1. Build the Docker Image (CUDA 12.4)

```bash
docker system prune -af --volumes  # (optional cleanup)
docker build -t dreamscene360:cu124 .
```

### 2. Prepare Host Workspace

```bash
mkdir -p ~/dreamscene360_docker/{data,output,pre_checkpoints}
```

### 3. Download Checkpoints to Host

* Download `omnidata_dpt_depth_v2.ckpt` from this [Dropbox folder](https://www.dropbox.com/scl/fo/348s01x0trt0yxb934cwe/h?rlkey=a96g2incso7g53evzamzo0j0y&dl=0) and place it in `~/dreamscene360_docker/pre_checkpoints`.

---

## 🧊 Run the Container

```bash
docker run --gpus all -it \
  -v ~/dreamscene360_docker/data:/workspace/DreamScene360/data \
  -v ~/dreamscene360_docker/output:/workspace/DreamScene360/output \
  -v ~/dreamscene360_docker/pre_checkpoints:/workspace/DreamScene360/pre_checkpoints \
  dreamscene360:cu124
```

---

## 🔧 Inside the Container – First-Time Setup

### 1. Compile CUDA Extensions (L4: SM=8.9)

```bash
export TORCH_CUDA_ARCH_LIST=8.9
pip install submodules/diff-gaussian-rasterization-depth
pip install submodules/simple-knn
```

### 2. Download Text2Pano Models

```bash
cd stitch_diffusion/pretrained_model
wget https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.safetensors -O stable-diffusion-2-1-base.safetensors
cd ../vae
wget https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.ckpt -O stablediffusion.vae.pt
cd ..
python download_lora.py
cd ../..
```

---

## 🧠 Example: Text-to-3D Scene Generation

### 1. Write Prompt

```bash
cd /workspace/DreamScene360/data
mkdir -p indoor_livingroom
echo "A spacious modern living room with white marble floors, two gray fabric sofas facing each other, and a small wooden coffee table with a green potted plant by the window. Warm orange evening sunlight filters through, casting a cozy and inviting glow on the walls." \
  > indoor_livingroom/indoor_livingroom_PROMPT.txt

mkdir -p outdoor_park
echo "A large urban park with lush green grass and tall trees surrounding a central fountain, distant city skyscrapers visible on the skyline, bright midday sunlight, gentle breeze rustling leaves, with children playing near the fountain, creating a refreshing and lively scene." \
  > outdoor_park/outdoor_park_PROMPT.txt
```

### 2. Run Training (with optional GPT-4V refinement)

```bash
export OPENAI_API_KEY=<Your_OpenAI_GPT4V_Key>
python train.py -s data/Italy_text -m output/italy_demo \
  --self_refinement --api_key $OPENAI_API_KEY --num_prompt 2 --max_rounds 2
```

> To skip GPT-4V refinement, omit `--self_refinement` and `--api_key`.

### 3. Export PLY (if not already saved)

```bash
python tools/export_ply.py \
  -i output/italy_demo/iteration_9000/gaussians.pkl \
  -o output/italy_demo/scene.ply
```

---

## 🔍 Dummy .ply File Usage (Testing Viewer)

If you want to test the viewer without running training, you can use a pre-made dummy `.ply` file:

```bash
wget -O output/dummy/scene.ply https://huggingface.co/your-user-or-test/dummy-scene/resolve/main/scene.ply
```

Then launch the viewer (after viewer build):

```bash
./SIBR_viewers/build/bin/SIBR_gaussianViewer_app -m output/dummy
```

---

## 🎥 Render Perspective Views

```bash
python render.py -s data/Italy_text -m output/italy_demo --iteration 9000
```

---

## 👀 Interactive Viewer (Ubuntu)

```bash
sudo apt install -y libglew-dev libassimp-dev libboost-all-dev libgtk-3-dev libopencv-dev \
  libglfw3-dev libavdevice-dev libavcodec-dev libeigen3-dev libxxf86vm-dev libembree-dev

cd SIBR_viewers
cmake -Bbuild . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j24 --target install
cd ..

./SIBR_viewers/build/bin/SIBR_gaussianViewer_app -m output/italy_demo
```

Use `WASD`/`IJKLUO` keys to navigate, or switch to Trackball mode via GUI.

---

## 📜 Citation

```bibtex
@inproceedings{zhou2024dreamscene360,
  title={Dreamscene360: Unconstrained text-to-3d scene generation with panoramic gaussian splatting},
  author={Zhou, Shijie and Fan, Zhiwen and Xu, Dejia and Chang, Haoran and Chari, Pradyumna and Bharadwaj, Tejas and You, Suya and Wang, Zhangyang and Kadambi, Achuta},
  booktitle={European Conference on Computer Vision},
  pages={324--342},
  year={2024},
  organization={Springer}
}
```

---

## 🙏 Acknowledgements

This project builds on [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/), [PERF](https://github.com/perf-project/PeRF), [Idea2Img](https://github.com/zyang-ur/Idea2Img), and [StitchDiffusion](https://github.com/littlewhitesea/StitchDiffusion). Many thanks to the authors for their contributions to the community.
