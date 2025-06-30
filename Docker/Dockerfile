# syntax=docker/dockerfile:1
################################################################################
# Dockerfile for Text2VR Project - DreamScene360 Development Environment
#
# This Dockerfile creates a BASE ENVIRONMENT with all necessary dependencies
# to run DreamScene360.
# The source code is NOT included in the image; it will be MOUNTED at runtime
# for a flexible and efficient development workflow.
#
# Base: Ubuntu 22.04 LTS
# CUDA: 12.4.1
################################################################################


# Use a recent official CUDA image with Ubuntu 22.04 for better compatibility.
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04


# --- Stage 1: Install Base Packages & System Dependencies ---
# Install essential tools and all libraries required to build the SIBR viewers.
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git wget curl ca-certificates \
        cmake ninja-build pkg-config \
        libglib2.0-0 libxext6 libsm6 libxrender1 libgl1 \
        libglew-dev libassimp-dev libboost-all-dev libgtk-3-dev \
        libopencv-dev libglfw3-dev libavdevice-dev libavcodec-dev \
        libeigen3-dev libxxf86vm-dev libembree-dev libglm-dev \
    && rm -rf /var/lib/apt/lists/*


# --- Stage 2: Setup Python Environment via Micromamba ---
# Use Python 3.10 for modern library compatibility.
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=$MAMBA_ROOT_PREFIX/bin:$PATH
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba && \
    micromamba create -y -n dev python=3.10 && \
    micromamba clean -a -y

# Set the default shell to use the created conda environment.
SHELL ["bash", "-c"]
ENV CONDA_DEFAULT_ENV=dev
ENV PATH=$MAMBA_ROOT_PREFIX/envs/dev/bin:$PATH


# --- Stage 3: Install Core Python Libraries ---
# Upgrade pip and install PyTorch compatible with the host's CUDA version.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
        --index-url https://download.pytorch.org/whl/cu121

# Pre-install dependencies from the requirements.txt to speed up container startup.
# This assumes the Dockerfile is at the root of the Text2VR repo.
WORKDIR /tmp
COPY ./DreamScene360/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# --- Stage 4: Final Setup ---
# Set the final working directory. The source code will be mounted here.
WORKDIR /workspace/DreamScene360

# Set the default command to start an interactive bash session.
CMD ["/bin/bash"]