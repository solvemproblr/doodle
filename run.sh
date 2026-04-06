#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="doodle-to-image"
CONTAINER_NAME="doodle-to-image"
PORT="${PORT:-7860}"
HF_TOKEN="${HF_TOKEN:-}"

# Persistent volume for model weights (~30 GB on first download)
DATA_DIR="${DATA_DIR:-$HOME/.doodle-to-image-data}"
mkdir -p "$DATA_DIR"

echo "==> Building Docker image..."
docker build -t "$IMAGE_NAME" .

# Stop previous run if still alive
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

echo "==> Starting container (GPU)..."
echo "    UI will be at http://localhost:${PORT}"
echo "    Model cache: ${DATA_DIR}"

docker run \
    --name "$CONTAINER_NAME" \
    --gpus all \
    -p "${PORT}:7860" \
    -v "${DATA_DIR}:/data" \
    -e HF_TOKEN="${HF_TOKEN}" \
    "$IMAGE_NAME"
