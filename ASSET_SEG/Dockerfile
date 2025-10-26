# ==================================================================
# DOCKERFILE FOR ASSET SEGMENTATION
# This environment installs the latest libraries required for
# GroundingDINO, and SAM.
# ==================================================================

# Use a standard PyTorch base image with CUDA support
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY segment_panorama.py .

# CMD ["python", "segment_panorama.py"]
