# syntax=docker/dockerfile:1

############################################
# DreamScene360 – GPU-enabled Docker image #
#   CUDA 12.x / Ubuntu 20.04 LTS           #
############################################

# 1) CUDA base that matches the PyTorch-cu124 wheels
# NOTE: As of June 2025, the *cudnn* flavour for 12.4 is not yet on Docker Hub.
#       Use the standard devel image – PyTorch wheels already bundle cuDNN.
FROM nvidia/cuda:12.4.0-devel-ubuntu20.04

# ---------- Basic packages ----------
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git wget curl ca-certificates \
        cmake ninja-build pkg-config \
        libglib2.0-0 libxext6 libsm6 libxrender1 libgl1 \
        libglew-dev libassimp-dev libboost-all-dev libgtk-3-dev \
        libopencv-dev libglfw3-dev libavdevice-dev libavcodec-dev \
        libeigen3-dev libxxf86vm-dev libembree-dev libglm-dev \
    && rm -rf /var/lib/apt/lists/*

# ---------- Micromamba + Python 3.10 ----------
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=$MAMBA_ROOT_PREFIX/bin:$PATH
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba && \
    micromamba create -y -n dreamscene360 python=3.8 && \
    micromamba clean -a -y

SHELL ["bash", "-c"]
ENV CONDA_DEFAULT_ENV=dreamscene360
ENV PATH=$MAMBA_ROOT_PREFIX/envs/dreamscene360/bin:$PATH

# ---------- PyTorch 2.4.0 (cu124) + tiny-cuda-nn ----------
# tiny-cuda-nn needs to know target SM; set a safe default (T4 = 7.5).
# Override at build time:  docker build --build-arg TCNN_ARCH=80 …
ARG TCNN_ARCH="75"
ENV TCNN_CUDA_ARCHITECTURES=${TCNN_ARCH}
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
        --index-url https://download.pytorch.org/whl/cu124 && \
    pip install --no-cache-dir git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# ---------- Clone DreamScene360 ----------
# Some submodules are referenced via SSH (git@github.com:...). Convert them to HTTPS first
RUN git config --global url."https://github.com/".insteadOf "git@github.com:" && \
    git clone --recursive --depth 1 https://github.com/ShijieZhou-UCLA/DreamScene360.git /workspace/DreamScene360

WORKDIR /workspace/DreamScene360

# ---------- Python requirements & CUDA rasterizers ----------
RUN pip install --no-cache-dir -r requirements.txt

# ---------- Workspace directories ----------
RUN mkdir -p /workspace/DreamScene360/{data,output,pre_checkpoints}

CMD ["/bin/bash"]
