#!/bin/bash

# Activate micromamba environment
eval "$(micromamba shell hook --shell bash)"
micromamba activate dev

# Install FastAPI and uvicorn if not already installed
pip install fastapi uvicorn python-multipart

# Start the API server
cd /workspace/DREAMSCENE360
python api_server.py