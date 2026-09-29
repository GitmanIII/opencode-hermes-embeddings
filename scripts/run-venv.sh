#!/usr/bin/env bash
# Run the venv embeddings server (loopback only).
# Usage: ./scripts/run-venv.sh [--detach]
# Env:   EMBED_PORT (8080), EMBED_MODEL, EMBED_DEVICE (cuda)
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
PY="$HERE/.venv/bin/python"

if [ ! -x "$PY" ]; then
  echo "venv missing — run ./scripts/setup-venv.sh first" >&2
  exit 1
fi

export EMBED_HOST="${EMBED_HOST:-127.0.0.1}"
export EMBED_PORT="${EMBED_PORT:-8080}"

DETACH=0
case "${1:-}" in -d|--detach) DETACH=1 ;; "") ;; *) echo "unknown arg: $1" >&2; exit 2 ;; esac

if [ "$DETACH" = 1 ]; then
  PIDFILE="$HERE/.venv-server.pid"
  nohup "$PY" "$HERE/server/venv_server.py" >"$HERE/.venv-server.log" 2>&1 &
  echo $! > "$PIDFILE"
  echo "started (pid $(cat "$PIDFILE")) -> http://127.0.0.1:$EMBED_PORT  (log: .venv-server.log)"
else
  echo "Starting venv embedder on 127.0.0.1:$EMBED_PORT (model=$EMBED_MODEL)"
  exec "$PY" "$HERE/server/venv_server.py"
fi
