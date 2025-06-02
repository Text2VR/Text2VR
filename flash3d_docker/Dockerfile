# 1. CUDA 11.8 & cuDNN 포함된 기본 이미지
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# 2. 기본 설정
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# 3. 필수 패키지 + Python 3.10 + GCC 11 + ninja
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    add-apt-repository ppa:ubuntu-toolchain-r/test && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3.10-dev \
    gcc-11 g++-11 \
    build-essential \
    git wget curl nano bzip2 ca-certificates \
    ninja-build \
    libglib2.0-0 libxext6 libsm6 libxrender1 libarchive13 libgl1 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    rm -rf /var/lib/apt/lists/*

# 4. 환경 변수: CUDA 아키텍처
ENV TORCH_CUDA_ARCH_LIST="7.5"

# 5. pip 설치 및 업그레이드
RUN python -m ensurepip && \
    python -m pip install --upgrade pip setuptools wheel


# 6. 작업 디렉토리
WORKDIR /workspace

# 7. Flash3D 클론
RUN git clone --recursive https://github.com/eldar/flash3d.git
WORKDIR /workspace/flash3d

# 8. PyTorch 및 기타 requirements 설치
RUN pip install -r requirements-torch.txt --extra-index-url https://download.pytorch.org/whl/cu118 && \
    pip install -r requirements.txt

# 9. 엔트리포인트
CMD ["/bin/bash"]
