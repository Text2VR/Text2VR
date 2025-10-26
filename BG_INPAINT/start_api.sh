#!/bin/bash

echo "🚀 Starting Background Inpainting API Server..."

cd /workspace

# Run the API server
python3 -m uvicorn api_server:app --host 0.0.0.0 --port 8003 --log-level info
