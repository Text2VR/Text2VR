# ==================================================================
# DOCKERFILE FOR DREAMSCENE360 SERVICE
# This environment is specifically tailored to run the original
# DreamScene360 training script with its specific dependencies.
# ==================================================================

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# --- Stage 1: System Setup ---
# (No changes from the original file)
ARG DEBIAN_FRONTEND=noninteractive
ARG GPU_ARCH=89
ARG TORCH_ARCH_LIST=8.9
ENV PYTHONUNBUFFERED=1 \
    TCNN_CUDA_ARCHITECTURES=${GPU_ARCH} \
    TORCH_CUDA_ARCH_LIST=${TORCH_ARCH_LIST} \
    CUDA_HOME=/usr/local/cuda \
    PATH=${CUDA_HOME}/bin:${PATH} \
    LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git wget curl ca-certificates \
    cmake ninja-build pkg-config \
    libglib2.0-0 libxext6 libsm6 libxrender1 libgl1 \
    libglew-dev libassimp-dev libboost-all-dev libgtk-3-dev \
    libopencv-dev libglfw3-dev libavdevice-dev libavcodec-dev \
    libeigen3-dev libxxf86vm-dev libembree-dev libglm-dev \
    gcc-12 g++-12 \
 && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 60 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 60

# --- Stage 2: Python Environment & Project Setup ---
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=$MAMBA_ROOT_PREFIX/bin:$PATH
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba && \
    micromamba create -y -n dev python=3.10 && \
    micromamba clean -a -y

SHELL ["micromamba", "run", "-n", "dev", "/bin/bash", "-c"]
    
# --- Stage 3: Install Dependencies & Compile Modules (cache-friendly order) ---
#0) Set working directory
WORKDIR /workspace/DREAMSCENE360

# 1) install PyTorch first (fixed CUDA wheel))
RUN pip install --no-cache-dir \
    torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --extra-index-url https://download.pytorch.org/whl/cu124

# 2) ONLY requirements first → cache-friendly
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Now, copy the rest of the source code
COPY . .

# 4) Install local and git packages specific to DreamScene360
RUN pip install --no-build-isolation ./submodules/diff-gaussian-rasterization-depth
RUN pip install --no-build-isolation ./submodules/simple-knn
RUN pip install --no-build-isolation git+https://github.com/NVlabs/tiny-cuda-nn/@d9e5274da3630ff747b67ed6188c838f4ad671ef#subdirectory=bindings/torch


# --- Stage 3.5: Download Pretrained Models for Text-to-Panorama if they don't exist ---
# This ensures that models are present without re-downloading them on every build.
# RUN \
#     SD_MODEL="stitch_diffusion/pretrained_model/stable-diffusion-2-1-base.safetensors" && \
#     VAE_MODEL="stitch_diffusion/vae/stablediffusion.vae.pt" && \
#     LORA_MODEL="stitch_diffusion/lora.safetensors" && \
#     \
#     mkdir -p stitch_diffusion/pretrained_model stitch_diffusion/vae && \
#     \
#     if [ ! -f "$SD_MODEL" ]; then \
#         echo "--> Stable Diffusion model not found. Downloading..." && \
#         wget https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.safetensors -O "$SD_MODEL"; \
#     else \
#         echo "--> Stable Diffusion model found. Skipping download."; \
#     fi && \
#     \
#     if [ ! -f "$VAE_MODEL" ]; then \
#         echo "--> VAE model not found. Downloading..." && \
#         wget https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.ckpt -O "$VAE_MODEL"; \
#     else \
#         echo "--> VAE model found. Skipping download."; \
#     fi && \
#     \
#     if [ ! -f "$LORA_MODEL" ]; then \
#         echo "--> LoRA model not found. Running download script..." && \
#         cd stitch_diffusion && python download_lora.py && cd ..; \
#     else \
#         echo "--> LoRA model found. Skipping download."; \
#     fi

# Compile the SIBR viewer
# RUN cd SIBR_viewers && \
#     (cmake -Bbuild . -DCMAKE_BUILD_TYPE=Release -G Ninja && ninja -C build install -j$(nproc)) || \
#     (cmake -Bbuild . -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc) --target install)

# --- Stage 4: Finalization ---
# Add auto-activation to the bash startup script for interactive sessions
RUN echo 'eval "$(micromamba shell hook --shell bash)"' >> /root/.bashrc && \
    echo 'micromamba activate dev' >> /root/.bashrc

CMD ["/bin/bash"]
