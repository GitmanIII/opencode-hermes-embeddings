#!/usr/bin/env bash
# Create the self-contained embedder venv (no Docker). Needs `uv`.
# PyPI torch wheels bundle the CUDA runtime, so only the NVIDIA driver is needed.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$HERE/.venv"

uv venv "$VENV" --python 3.12
uv pip install --python "$VENV/bin/python" \
  torch sentence-transformers fastapi "uvicorn[standard]" einops

echo
echo "venv ready: $VENV"
echo "run it:     ./scripts/run-venv.sh        (foreground)"
echo "            ./scripts/run-venv.sh --detach  (background)"
