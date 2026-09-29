#!/usr/bin/env bash
# Stop the detached venv embedder server.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
PIDFILE="$HERE/.venv-server.pid"
if [ ! -f "$PIDFILE" ]; then echo "no pidfile (not running detached)"; exit 0; fi
PID="$(cat "$PIDFILE")"
if kill "$PID" 2>/dev/null; then echo "stopped pid $PID"; else echo "pid $PID not running"; fi
rm -f "$PIDFILE"
