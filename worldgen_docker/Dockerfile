# NVIDIA CUDA 베이스 이미지 설정
# WorldGen 레포지토리의 PyTorch cu128 요구사항에 맞춰 CUDA 12.x 버전을 사용합니다.
# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/tags 에서
# "12.1.0-cudnn8-devel-ubuntu20.04"와 같은 태그 중 최신 버전을 확인하고 필요에 따라 변경할 수 있습니다.
# PyTorch cu128은 일반적으로 CUDA 12.1, 12.2, 12.3 등과 잘 작동합니다.



# [1] CUDA 베이스 이미지
FROM nvcr.io/nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04

LABEL maintainer="Changmin <ckdals1380@gmail.com>"

# [2] 환경 변수
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/opt/conda/bin:${PATH}"

# [3] 기본 패키지
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    curl \
    vim \
    nano \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    libarchive13 \
    libgl1 && \
    rm -rf /var/lib/apt/lists/*

# [4] Miniconda 설치 (필요 없을 수도 있음)
ARG CONDA_VERSION="py311_23.11.0-2"
ARG MINICONDA_INSTALLER_URL="https://repo.anaconda.com/miniconda/Miniconda3-${CONDA_VERSION}-Linux-x86_64.sh"
RUN wget ${MINICONDA_INSTALLER_URL} -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    /opt/conda/bin/conda clean --all -y

# [5] mamba 설치 (안정 버전, 필요 없을 수도 있음)
RUN /opt/conda/bin/conda install -y -c conda-forge mamba=1.4.2 && \
    /opt/conda/bin/mamba update -y --all

# [6] PyTorch 설치 (cu128)
RUN /opt/conda/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# [7] PyTorch3D 정확한 버전 설치
RUN /opt/conda/bin/pip install --upgrade pip && \
    pip install "git+https://github.com/facebookresearch/pytorch3d.git"

# [8] 기타 라이브러리 설치
RUN pip install open3d iopath

# [9] WorldGen 클론 + 서브모듈 포함
WORKDIR /app
RUN git clone --recurse-submodules https://github.com/ZiYang-xie/WorldGen.git

# [10] WorldGen 설치
WORKDIR /app/WorldGen
RUN pip install -e submodules/UniK3D
RUN pip install .

# [11] generate_both.py 복사
COPY generate_both.py /app/WorldGen/generate_both.py

# [12] PYTHONPATH 설정
ENV PYTHONPATH="/app/WorldGen/submodules/UniK3D"

# [13] 실행 (KNN CUDA kernel 컴파일 후 .ply 생성)
WORKDIR /app/WorldGen
CMD ["bash", "-c", "bash submodules/UniK3D/unik3d/ops/knn/compile.sh && python generate_both.py"]
