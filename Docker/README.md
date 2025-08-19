# 🌀 DreamScene360 Development Guide (for Text2VR Project)

## 🚀 Overview

This guide outlines the Docker-based development workflow for the `DreamScene360` module within the `Text2VR` project. This setup uses volume mounting to sync your local source code with the container in real-time, allowing for efficient development and testing on a GCP VM with an NVIDIA L4 GPU.

---

## 🛠️ 1. Environment Setup (on the Host VM)

### 1.1. Directory Structure

Ensure the `Dockerfile` is located at the root of your `Text2VR` repository. Then, create the necessary directories for outputs and pretrained models from the repository root.

```bash
# Run from the root of the Text2VR repository
# e.g., /home/ckdals1380/Text2VR

mkdir -p ./output/dreamscene360
mkdir -p ./pre_checkpoints
```

Your porject structure should look like this:
```bash
Text2VR/
├── Docker/                             # <-- Directory for Dockerfiles
│   └── Dockerfile                      # <-- e.g.) DreamScene360 Dockerfile
├── DreamScene360/                      # <-- Your DreamScene360 source code
├── output/
│   └── dreamscene360/                  # <-- Generated scenes will be saved here
└── pre_checkpoints/
    └── big-lama.ckpt                   # <-- Pretrained models will be placed here
    └── omnidata_dpt_depth_v2.ckpt      # <-- Pretrained models will be placed here
    └── monidata_dpt_normal_v2.ckpt     # <-- Pretrained models will be placed here
```

### 1.2. Download Checkpoints
Download the omnidata_dpt_depth_v2.ckpt file from the official from this [Dropbox folder](https://www.dropbox.com/scl/fo/348s01x0trt0yxb934cwe/h?rlkey=a96g2incso7g53evzamzo0j0y&dl=0) and place it in `Text2VR/pre_checkpoints/` directory.

### 1.3. Build the Docker Image
From the root directory of your `Text2VR` repository, run the following command to build the base development environment image.

```bash
docker build -t text2vr-dev:latest -f Docker/Dockerfile .
```

---

## 🧊 2. Run the Development Container
This is the most important step. Run this command from the root of your `Text2VR` repository to start the container.

```bash
docker run --gpus all -it --name text2vr_container \
  -v "$(pwd)/DreamScene360:/workspace/DreamScene360" \
  -v "$(pwd)/output/dreamscene360:/workspace/DreamScene360/output" \
  -v "$(pwd)/pre_checkpoints:/workspace/DreamScene360/pre_checkpoints" \
  text2vr-dev:latest
```

> TIP
```bash
docker builder prune
docker start text2vr_container # text2vr_isolated_test_container
docker exec -it text2vr_container /bin/bash
docker exec -it text2vr_isolated_test_container /bin/bash
docker rm text2vr_container
```

### Explanation of Volume Mounts (-v):

* `-v "$(pwd)/DreamScene360:/workspace/DreamScene360"`: This is the key part. It maps your local `DreamScene360` source code folder to the `/workspace/DreamScene360` directory inside the container. **Any code you edit on the VM will be instantly reflected inside the container.**
* The other `-v` flags map the output and checkpoint folders in the same way, ensuring that generated files and models are saved directly to your host VM.
* `--rm`: This flag automatically cleans up and removes the container when you exit, keeping your system tidy.

---

## 🔧 3. Inside the Container: First-Time Setup
After the container starts, you'll be inside a `bash` shell. You only need to run this setup once per image unless you rebuild it.

```bash
# if no python or wget, then
echo 'eval "$(micromamba shell hook --shell bash)"' >> /root/.bashrc && echo 'micromamba activate dev' >> /root/.bashrc

# You are now inside the container, at /workspace/DreamScene360
# Download pretrained models for the Text-to-Pano pipeline.
cd stitch_diffusion/pretrained_model
wget https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.safetensors -O stable-diffusion-2-1-base.safetensors
cd ../vae
wget https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.ckpt -O stablediffusion.vae.pt
cd ..
python download_lora.py
cd ..
```

Result:
```bash
/workspace/DreamScene360
├── data/
│   ├── indoor_livingroom/
│   │   └── indoor_livingroom_PROMPT.txt
│   └── outdoor_park/
│       └── outdoor_park_PROMPT.txt
├── pre_checkpoints/
│    └── big-lama.ckpt                   
│    └── omnidata_dpt_depth_v2.ckpt      
│    └── monidata_dpt_normal_v2.ckpt     
├── stitch_diffusion/
│   └── pretrained_model/
│       ├── v2-1_512-ema-pruned.safetensors
│       └── vae-ft-mse-840000-ema-pruned.ckpt
│   └── download_lora.py → LoRA models (installed)
└── ...
```
---

## 4. Example: Text-to-3D Scene Generation

The following commands are run inside the container. The working directory is `/workspace/DreamScene360`.

### 4.1. Write Prompt

The `data` directory is mounted from your host, so any files created here will persist.

```bash
# The current directory is /workspace/DreamScene360
cd /workspace/DreamScene360/data
mkdir -p indoor_livingroom
echo "A spacious modern living room with white marble floors, two gray fabric sofas facing each other, and a small wooden coffee table with a green potted plant by the window. Warm orange evening sunlight filters through, casting a cozy and inviting glow on the walls." \
  > indoor_livingroom/indoor_livingroom_PROMPT.txt

# or
echo "A 360 equirectangular photo of a minimalist and spacious living room. In the center, there is a single modern leather sofa. The room has plain white walls, a smooth light gray concrete floor, and no other furniture or decorations. The scene is brightly lit by soft, natural light from a large window, with no harsh shadows. photorealistic, 8k, sharp focus." \
  > indoor_livingroom/indoor_livingroom_PROMPT.txt


mkdir -p outdoor_park
echo "A large urban park with lush green grass and tall trees surrounding a central fountain, distant city skyscrapers visible on the skyline, bright midday sunlight, gentle breeze rustling leaves, with children playing near the fountain, creating a refreshing and lively scene." \
  > outdoor_park/outdoor_park_PROMPT.txt
```

### 4.2. Run Training
You can run the training process with or without GPT-4V refinement for prompts.

**To run with GPT-4V refinement:**

```bash
# Make sure to set your OpenAI API key
export OPENAI_API_KEY=<Your_OpenAI_GPT4V_Key>

python train.py -s data/Italy_text -m output/Italy \
 —self_refinement —api_key "yourt-api-key" —num_prompt 2 —max_rounds 2 —debug
```

**To run without GPT-4V refinement:**
```bash
cd /workspace/DreamScene360

# (1) Inddor Livingroom
python train.py \
  -s data/indoor_livingroom \
  -m output/indoor_livingroom_demo

# (2) Outdoor Park
python train.py \
  -s data/outdoor_park \
  -m output/outdoor_park_demo
```

### 4.3. Export PLY File
After training is complete, export the Gaussian Splatting data to a `.ply` file.
```bash
python tools/export_ply.py \
  -i output/indoor_livingroom_demo/iteration_9000/gaussians.pkl \
  -o output/indoor_livingroom_demo/scene.ply

python tools/export_ply.py \
  -i output/outdoor_park_demo/iteration_9000/gaussians.pkl \
  -o output/outdoor_park_demo/scene.ply
```

---

## 5. Compile and Run the Interactive Viewer
The system dependencies for the SIBR viewer are already installed in the Docker image. You only need to compile the viewer source code once.

### 5.1. Compile the SIBR Viewer (One-Time Task)
```bash
# Run from /workspace/DreamScene360
cd SIBR_viewers
cmake -Bbuild . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j24 --target install
cd ..
```

### 5.2. Launch the Viewer
```bash
# Example for the living room scene
./SIBR_viewers/install/bin/SIBR_gaussianViewer_app -m output/indoor_livingroom_demo
```
*Use WASD/IJKLUO keys to navigate, or switch to Trackball mode via GUI.*

---

## 6. Render Perspective Views
You can also render out specific camera views from your trained model.
```bash
python render.py -s data/indoor_livingroom -m output/indoor_livingroom_demo --iteration 9000
```

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
