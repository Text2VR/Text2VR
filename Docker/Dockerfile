# ==================================================================
# ULTIMATE DOCKERFILE FOR TEXT2VR - DREAMSCENE360 (FINAL VERSION)
#
# This file performs all functions of the reference Dockerfile while
# being tailored to a local development workflow with modified code.
# It solves the 'torch' dependency error by staging the installation.
# ==================================================================

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# --- Stage 1: System Setup (from reference Dockerfile) ---

# Set build-time and runtime environment variables for CUDA compilation.
ARG DEBIAN_FRONTEND=noninteractive
ARG GPU_ARCH=89
ARG TORCH_ARCH_LIST=8.9
ENV PYTHONUNBUFFERED=1 \
    TCNN_CUDA_ARCHITECTURES=${GPU_ARCH} \
    TORCH_CUDA_ARCH_LIST=${TORCH_ARCH_LIST} \
    CUDA_HOME=/usr/local/cuda \
    PATH=${CUDA_HOME}/bin:${PATH} \
    LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install all system packages and specific compilers.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git wget curl ca-certificates \
    cmake ninja-build pkg-config \
    libglib2.0-0 libxext6 libsm6 libxrender1 libgl1 \
    libglew-dev libassimp-dev libboost-all-dev libgtk-3-dev \
    libopencv-dev libglfw3-dev libavdevice-dev libavcodec-dev \
    libeigen3-dev libxxf86vm-dev libembree-dev libglm-dev \
    gcc-12 g++-12 \
 && rm -rf /var/lib/apt/lists/*

# Set default C/C++ compilers.
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 60 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 60

# --- Stage 2: Python Environment & Project Setup ---

# Use Micromamba for a clean, isolated Python 3.10 environment.
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=$MAMBA_ROOT_PREFIX/bin:$PATH
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba && \
    micromamba create -y -n dev python=3.10 && \
    micromamba clean -a -y

# Activate the conda environment for all subsequent commands.
SHELL ["micromamba", "run", "-n", "dev", "/bin/bash", "-c"]

# Create a workspace and COPY your entire local project into it.
WORKDIR /workspace
COPY . .

# --- Stage 3: Install Dependencies & Compile Modules ---

# Stage 3.1: Install PyTorch first.
# This ensures it is available for all subsequent build steps.
RUN pip install --no-cache-dir \
    torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --extra-index-url https://download.pytorch.org/whl/cu124

# Stage 3.2: Install all other pre-built packages from your requirements.txt.
# We filter out local file and git dependencies to install them later.
COPY DreamScene360/requirements.txt /tmp/requirements.txt
RUN grep -v "file://" /tmp/requirements.txt | grep -v "git+" > /tmp/pypi_requirements.txt
RUN pip install --no-cache-dir -r /tmp/pypi_requirements.txt

# Stage 3.3: Now, install the local and git packages that require compilation.
RUN pip install --no-build-isolation ./DreamScene360/submodules/diff-gaussian-rasterization-depth
RUN pip install --no-build-isolation ./DreamScene360/submodules/simple-knn

RUN pip install --no-build-isolation git+https://github.com/IDEA-Research/GroundingDINO.git@856dde20aee659246248e20734ef9ba5214f5e44
RUN pip install --no-build-isolation git+https://github.com/facebookresearch/segment-anything.git@dca509fe793f601edb92606367a655c15ac00fdf
RUN pip install --no-build-isolation git+https://github.com/NVlabs/tiny-cuda-nn/@d9e5274da3630ff747b67ed6188c838f4ad671ef#subdirectory=bindings/torch

# Stage 3.4: Compile the SIBR viewer using the robust Ninja build command.
RUN cd DreamScene360/SIBR_viewers && \
    (cmake -Bbuild . -DCMAKE_BUILD_TYPE=Release -G Ninja && ninja -C build install -j$(nproc)) || \
    (cmake -Bbuild . -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc) --target install)

# --- Stage 4: Finalization ---

# Create the environment test script from the reference Dockerfile.
RUN echo '#!/bin/bash\n\
echo "=== DreamScene360 Environment Test ==="\n\
echo "GPU Info:"\n\
nvidia-smi\n\
echo ""\n\
echo "PyTorch GPU:"\n\
python -c "import torch; print(f\"CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}\")"\n\
echo ""\n\
echo "Core modules check:"\n\
python -c "import tinycudann" && echo "✅ tiny-cuda-nn" || echo "❌ tiny-cuda-nn"\n\
python -c "from diff_gaussian_rasterization import GaussianRasterizationSettings" && echo "✅ diff-gaussian-rasterization" || echo "❌ diff-gaussian-rasterization"\n\
python -c "import simple_knn" && echo "✅ simple-knn" || echo "❌ simple-knn"\n\
echo ""\n\
echo "🎯 Environment is ready!"' > /workspace/test_env.sh && \
    chmod +x /workspace/test_env.sh
    
# Add auto-activation to the bash startup script.
# This ensures any new interactive bash shell will start inside the 'dev' env.
# shell initialization and auto-activation to the bash startup script.
RUN echo 'eval "$(micromamba shell hook --shell bash)"' >> /root/.bashrc && \
    echo 'micromamba activate dev' >> /root/.bashrc

# Set the final working directory and default command.
WORKDIR /workspace/DreamScene360
CMD ["/bin/bash"]