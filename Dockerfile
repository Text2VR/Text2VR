# Flash Sculptor Dockerfile
# Multi-stage build for optimization

# Stage 1: Base image with CUDA support
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS base

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# CUDA toolchain 경로 (devel 이미지에선 명시해두면 편함)
ENV CUDA_HOME=/usr/local/cuda-12.1
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    libglew-dev \
    libassimp-dev \
    libboost-all-dev \
    libgtk-3-dev \
    libopencv-dev \
    libglfw3-dev \
    libavdevice-dev \
    libavcodec-dev \
    libeigen3-dev \
    libxxf86vm-dev \
    libembree-dev \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python (Ubuntu 22.04 already has python3)
RUN [ -e /usr/bin/python ] || ln -s /usr/bin/python3 /usr/bin/python

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh
ENV PATH=/opt/conda/bin:${PATH}

# Accept defaults TOS explicitly (for non-interactive build)
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Remove defaults channel and set preferred channels (avoid TOS issues)
RUN conda config --system --remove-key default_channels || true && \
    conda config --system --remove-key channels || true && \
    conda config --system --add channels pytorch && \
    conda config --system --add channels nvidia && \
    conda config --system --add channels conda-forge && \
    conda config --system --set channel_priority strict

# Stage 2: Python environment setup
FROM base as python-env

# Create conda environment (channels already configured above)
COPY environment.yaml /tmp/environment.yaml
RUN conda env create -f /tmp/environment.yaml && conda clean -afy

# Activate environment
ENV CONDA_DEFAULT_ENV=flashsculptor
ENV CONDA_PREFIX=/opt/conda/envs/flashsculptor
ENV PATH=${CONDA_PREFIX}/bin:${PATH}

# Install additional CUDA-specific packages
RUN pip install xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121
RUN pip install spconv-cu120  # not needed for basic segmentation
RUN pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html  # not needed for basic segmentation

# Stage 3: Application setup
FROM python-env as app

# Set working directory
WORKDIR /app

# Clone Flash-Sculptor repository (suyeon branch)
RUN git clone https://github.com/Text2VR/Text2VR.git /app/Flash-Sculptor && \
    cd /app/Flash-Sculptor && \
    git checkout suyeon

# Copy requirements.txt and install dependencies
COPY requirements.txt /app/Flash-Sculptor/requirements.txt

# Clone and install SAM
RUN git clone https://github.com/facebookresearch/segment-anything.git /tmp/segment_anything
RUN cd /tmp/segment_anything && python -m pip install -e .

# Install PyTorch first (required for GroundingDINO)
RUN pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

# Clone and install GroundingDINO
RUN git clone https://github.com/IDEA-Research/GroundingDINO.git /tmp/GroundingDINO
ENV AM_I_DOCKER=False
ENV BUILD_WITH_CUDA=True
# ENV CUDA_HOME=/usr/local/cuda-12.1
# RUN cd /tmp/GroundingDINO && pip install -e .

# Install grounded-sam-osx
RUN git clone https://github.com/linjing7/grounded-sam-osx.git && \
    cd grounded-sam-osx && \
    bash install.sh

# Install recognize-anything
RUN git clone https://github.com/xinyu1205/recognize-anything.git && \
    pip install -r ./recognize-anything/requirements.txt && \
    pip install -e ./recognize-anything/

# Install transformers for segment_new.py (uses HuggingFace GroundingDINO)
RUN pip install 'transformers>=4.45.0'

# Install mmcv-full with pre-compiled wheel (avoid build issues)
RUN pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4.0/index.html

# Install requirements.txt with error handling
RUN pip install -r /app/Flash-Sculptor/requirements.txt || echo "Some packages failed to install, continuing..."

# Install additional required packages
RUN pip install git+https://github.com/EasternJournalist/utils3d

# Install ml-depth-pro dependencies
RUN pip install timm==0.9.12 pillow_heif
RUN mkdir -p /tmp/extensions
RUN git clone https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast && \
    pip install /tmp/extensions/nvdiffrast
RUN git clone https://github.com/autonomousvision/mip-splatting.git /tmp/extensions/mip-splatting && \
    TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6+PTX" pip install /tmp/extensions/mip-splatting/submodules/diff-gaussian-rasterization/

# Install Depth-Pro (ml-depth-pro)
RUN cd /app/Flash-Sculptor/ml-depth-pro && \
    pip install -e . && \
    cd /app

# Install detectron2 (commented out - not needed for basic segmentation)
# RUN python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Build MSDA operations (commented out - not needed for basic segmentation)
# RUN cd VistaDream/tools/OneFormer/oneformer/modeling/pixel_decoder/ops && \
#     sh make.sh && \
#     cd /app

# Download pretrained models for VistaDream (commented out - not needed for basic segmentation)
# RUN cd VistaDream && \
#     bash download_weights.sh && \
#     cd /app

# # Download SAM models
# RUN wget -O sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth && \
#     wget -O sam_vit_l_0b3195.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth && \
#     wget -O sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# Download SAM models
RUN wget -O sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# Download GroundingDINO model
RUN wget -O groundingdino_swint_ogc.pth https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# Download ml-depth-pro pretrained models
RUN cd /app/Flash-Sculptor/ml-depth-pro && \
    bash get_pretrained_models.sh && \
    cd /app

# # Download RAM models (using resolve/main for proper download)
# RUN wget -O ram_swin_large_14m.pth \
#       "https://huggingface.co/xinyu1205/recognize-anything-plus-model/resolve/main/ram_swin_large_14m.pth?download=true" && \
#     wget -O ram_plus_swin_large_14m.pth \
#       "https://huggingface.co/xinyu1205/recognize-anything-plus-model/resolve/main/ram_plus_swin_large_14m.pth?download=true" && \
#     wget -O tag2text_swin_14m.pth \
#       "https://huggingface.co/spaces/xinyu1205/Recognize_Anything-Tag2Text/resolve/main/tag2text_swin_14m.pth?download=true"

# Create results directory
RUN mkdir -p /app/Flash-Sculptor/results

# Set environment variables
ENV PYTHONPATH=/app/Flash-Sculptor:${PYTHONPATH}
ENV CUDA_VISIBLE_DEVICES=0

# Expose port for potential web interface
EXPOSE 8000

# Default command
CMD ["/bin/bash"]
