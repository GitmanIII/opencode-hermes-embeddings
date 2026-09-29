#!/usr/bin/env bash
# Run HuggingFace text-embeddings-inference (TEI) on the GPU, bound to LOOPBACK only.
#
# Prereqs (one-time, needs sudo):
#   sudo pacman -S nvidia-container-toolkit
#   sudo nvidia-ctk runtime configure --runtime=docker
#   sudo systemctl restart docker
#   sudo usermod -aG docker "$USER"   # then log out/in
#
# Usage:
#   ./scripts/run-tei.sh            # foreground (Ctrl-C to stop)
#   ./scripts/run-tei.sh --detach   # background, --restart unless-stopped (auto-start on boot)
#
# Env: TEI_MODEL, TEI_PORT (8080), TEI_IMAGE, HF_CACHE (~/.cache/huggingface), TEI_CONTAINER
set -euo pipefail

DETACH=0
case "${1:-}" in -d|--detach) DETACH=1 ;; "") ;; *) echo "unknown arg: $1" >&2; exit 2 ;; esac
[ "${TEI_DETACH:-0}" = "1" ] && DETACH=1

MODEL="${TEI_MODEL:-nomic-ai/nomic-embed-text-v1.5}"
PORT="${TEI_PORT:-8080}"
IMAGE="${TEI_IMAGE:-ghcr.io/huggingface/text-embeddings-inference:latest}"
CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
NAME="${TEI_CONTAINER:-opencode-hermes-tei}"

mkdir -p "$CACHE"

# Host port is bound to loopback only; 0.0.0.0:80 is the container-internal bind.
COMMON=(--gpus all -p "127.0.0.1:${PORT}:80" -v "${CACHE}:/data" --name "$NAME"
        "$IMAGE" --model-id "$MODEL" --hostname 0.0.0.0)

if [ "$DETACH" = 1 ]; then
  if docker inspect "$NAME" >/dev/null 2>&1; then
    if [ "$(docker inspect -f '{{.State.Running}}' "$NAME")" = "true" ]; then
      echo "already running: $NAME (http://127.0.0.1:${PORT})"
    else
      docker start "$NAME" >/dev/null
      echo "restarted: $NAME (http://127.0.0.1:${PORT})"
    fi
  else
    docker run -d --restart unless-stopped "${COMMON[@]}" >/dev/null
    echo "started (detached, restart=unless-stopped): $NAME  -> http://127.0.0.1:${PORT}"
  fi
else
  if docker inspect "$NAME" >/dev/null 2>&1; then
    echo "container '$NAME' already exists — use --detach, or stop it first: ./scripts/stop-tei.sh" >&2
    exit 1
  fi
  echo "Starting TEI (foreground): model=$MODEL  bind=127.0.0.1:$PORT"
  exec docker run --rm "${COMMON[@]}"
fi
