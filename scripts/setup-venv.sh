#!/usr/bin/env bash
# Create the self-contained embedder venv (no Docker). Needs `uv`.
# PyPI torch wheels bundle the CUDA runtime, so only the NVIDIA driver is needed.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$HERE/.venv"

uv venv "$VENV" --python 3.12
# Pin Transformers <5: nomic-embed-text-v1.5's remote modeling code calls
# get_extended_attention_mask(), which Transformers 5 removed (TEI uses its own
# Rust impl, so it is unaffected).
uv pip install --python "$VENV/bin/python" \
  torch "sentence-transformers==3.4.1" "transformers==4.49.0" fastapi "uvicorn[standard]" einops

echo
echo "venv ready: $VENV"
echo "run it:     ./scripts/run-venv.sh        (foreground)"
echo "            ./scripts/run-venv.sh --detach  (background)"
