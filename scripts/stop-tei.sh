#!/usr/bin/env bash
# Stop the opencode-hermes TEI container. Pass --remove to delete it too.
set -euo pipefail
NAME="${TEI_CONTAINER:-opencode-hermes-tei}"
if ! docker inspect "$NAME" >/dev/null 2>&1; then
  echo "no container '$NAME'"
  exit 0
fi
docker stop "$NAME" >/dev/null && echo "stopped $NAME"
if [ "${1:-}" = "--remove" ] || [ "${1:-}" = "-r" ]; then
  docker rm "$NAME" >/dev/null && echo "removed $NAME"
fi
