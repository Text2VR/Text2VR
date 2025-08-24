# 🌀 DreamScene360 Service

## 🚀 Overview

This service encapsulates the original **DreamScene360** project. Its primary responsibilities within the Text2VR pipeline are:
1.  Generating a 360° panoramic image from a text prompt, optionally enhanced with GPT-4V self-refinement.
2.  (Future) Training the final hybrid 3D model using the generated panorama and segmentation masks from the `segmentation_service`.

This service runs in a dedicated Docker environment with specific, legacy library versions (`diffusers==0.10.2`, `transformers==4.26.0`) to ensure compatibility with the original codebase and pretrained models.

---

## 🛠️ Standalone Usage for Development

All dependencies are managed by the `Dockerfile` in this directory, which is orchestrated by the main `docker-compose.yml` at the project root. To work on this service in isolation, use the following commands from the **root of the Text2VR repository**.

### 1. Build the Service Image
If you've made changes to this service's `Dockerfile` or `requirements.txt`, you must rebuild its image.

```bash
docker-compose build dreamscene360
```

### 2. Run an Interactive Session
This command starts the service container and gives you a `bash` shell inside it. This is the primary method for debugging and development.

```bash
docker-compose run --rm dreamscene360 /bin/bash
```

### 3. Running Scripts Manually
You are now inside the container at `/workspace/dreamscene360_code`. You can run any of the original scripts for testing or debugging.

#### 3.1. First-Time Setup Inside Container
If you are running a fresh container, you must download the Stable Diffusion checkpoints required by the panorama generation script. The working directory is `/workspace/dreamscene360_code`.
```bash
# Inside the container, from /workspace/dreamscene360_code
cd stitch_diffusion/pretrained_model
wget [https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.safetensors](https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.safetensors) -O stable-diffusion-2-1-base.safetensors

cd ../vae
wget [https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.ckpt](https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.ckpt) -O stablediffusion.vae.pt

cd ..
python download_lora.py
cd ..
```

#### 3.2. How to Run train.py Manually
**Step 1: Create a Prompt File**
The script reads the scene prompt from a text file. You need to create this file first.
```bash
# Inside the container, from /workspace/dreamscene360_code
cd /workspace/dreamscene360_code/data
mkdir -p indoor_livingroom
echo "A 360 equirectangular photo of a minimalist and spacious living room. In the center, there is a single modern leather sofa. The room has plain white walls, a smooth light gray concrete floor, and no other furniture or decorations. The scene is brightly lit by soft, natural light from a large window, with no harsh shadows. photorealistic, 8k, sharp focus." \
  > indoor_livingroom/indoor_livingroom_PROMPT.txt

# outdoor view
mkdir -p outdoor_park
echo "A large urban park with lush green grass and tall trees surrounding a central fountain, distant city skyscrapers visible on the skyline, bright midday sunlight, gentle breeze rustling leaves, with children playing near the fountain, creating a refreshing and lively scene." \
  > outdoor_park/outdoor_park_PROMPT.txt
```

**Step 2: Run Training**
The output will be saved to `/workspace/output/` inside the container, which is mapped to `Text2VR/output/` on your host machine.

**To run with GPT-4V self-refinement:**
* You must pass your API key.
```bash
# Inside the container
python train.py \
  -s data/indoor_livingroom \
  -m output/indoor_livingroom_demo \
--self_refinement —api_key "yourt-api-key" \
--num_prompt 2 --max_rounds 2 --debug"
```

**To run without GPT-4V self-refinement:**
```bash
# Inside the container
python train.py \
  -s data/indoor_livingroom \
  -m output/indoor_livingroom_demo
```
---
## 4. Compile and Run the Interactive Viewer
The system dependencies for the SIBR viewer are already installed in the Docker image. You only need to compile the viewer source code once.

### 4.1. Compile the SIBR Viewer (One-Time Task)
```bash
# Run from /workspace/DreamScene360
cd SIBR_viewers
cmake -Bbuild . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j24 --target install
cd ..
```

### 4.2. Launch the Viewer
```bash
# Example for the living room scene
./SIBR_viewers/install/bin/SIBR_gaussianViewer_app -m output/indoor_livingroom_demo
```
*Use WASD/IJKLUO keys to navigate, or switch to Trackball mode via GUI.*

---

## 5. Render Perspective Views
You can also render out specific camera views from your trained model.
```bash
python render.py -s data/indoor_livingroom -m output/indoor_livingroom_demo --iteration 9000
```
## 🙏 Acknowledgements

This project builds on [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/), [PERF](https://github.com/perf-project/PeRF), [Idea2Img](https://github.com/zyang-ur/Idea2Img), and [StitchDiffusion](https://github.com/littlewhitesea/StitchDiffusion). Many thanks to the authors for their contributions to the community.
