#!/usr/bin/env bash
# Run HuggingFace text-embeddings-inference (TEI) on the GPU, bound to LOOPBACK only.
#
# Prereqs (one-time, needs sudo):
#   sudo pacman -S nvidia-container-toolkit
#   sudo nvidia-ctk runtime configure --runtime=docker
#   sudo systemctl restart docker
#   sudo usermod -aG docker "$USER"   # then log out/in (or run this script with sudo)
#
# Usage: ./scripts/run-tei.sh
# Env:   TEI_MODEL (default nomic-ai/nomic-embed-text-v1.5), TEI_PORT (8080),
#        TEI_IMAGE, HF_CACHE (~/.cache/huggingface), TEI_CONTAINER (name)
set -euo pipefail

MODEL="${TEI_MODEL:-nomic-ai/nomic-embed-text-v1.5}"
PORT="${TEI_PORT:-8080}"
IMAGE="${TEI_IMAGE:-ghcr.io/huggingface/text-embeddings-inference:latest}"
CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
NAME="${TEI_CONTAINER:-opencode-hermes-tei}"

mkdir -p "$CACHE"

echo "Starting TEI: model=$MODEL  bind=127.0.0.1:$PORT  image=$IMAGE"
exec docker run --rm --name "$NAME" --gpus all \
  -p "127.0.0.1:${PORT}:80" \
  -v "${CACHE}:/data" \
  "${IMAGE}" \
  --model-id "${MODEL}" \
  --hostname 0.0.0.0
