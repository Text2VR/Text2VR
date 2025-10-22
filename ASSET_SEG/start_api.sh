#!/bin/bash
set -e

echo "🚀 Starting Asset Segmentation API..."

# Change to working directory
cd /app

# Set Python path
export PYTHONPATH="/app:$PYTHONPATH"

# Start the API server
python api_server.py