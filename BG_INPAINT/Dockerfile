# ==================================================================
# DIFFUSION INPAINTING (Stable Diffusion Inpainting via diffusers)
# This environment installs the latest libraries required for
# 
# ==================================================================
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime
WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential libgl1-mesa-glx libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -U pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

COPY inpaint_panorama.py .

ENV HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface

# Optional (nice for interactive debugging). Safe to keep because `run` overrides it.
CMD ["/bin/bash"]
