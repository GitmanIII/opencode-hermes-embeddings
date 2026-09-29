#!/usr/bin/env python3
"""Latency comparison of OpenAI-compatible embedding endpoints.

Usage:
    python scripts/bench.py [url ...]
    # default http://127.0.0.1:8080 ; compare e.g. TEI :8080 vs venv :8081

Env: EMBED_MODEL, BENCH_N (50), BENCH_BATCH (1)
"""
import json
import os
import sys
import time
import urllib.request

MODEL = os.environ.get("EMBED_MODEL", "nomic-ai/nomic-embed-text-v1.5")
N = int(os.environ.get("BENCH_N", "50"))
BATCH = int(os.environ.get("BENCH_BATCH", "1"))
TEXTS = [f"note {i}: nightly backup archives snapshots to cold storage at 03:00" for i in range(BATCH)]


def call(url: str) -> float:
    body = json.dumps({"input": TEXTS if BATCH > 1 else TEXTS[0], "model": MODEL}).encode()
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/embeddings", data=body, headers={"content-type": "application/json"}
    )
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=30) as r:
        r.read()
    return (time.perf_counter() - t) * 1000


def bench(url: str) -> tuple[float, float, float]:
    call(url)  # warm-up
    lat = sorted(call(url) for _ in range(N))
    return lat[len(lat) // 2], lat[max(0, int(len(lat) * 0.95) - 1)], lat[0]


def main() -> None:
    targets = sys.argv[1:] or ["http://127.0.0.1:8080"]
    print(f"model={MODEL}  n={N}  batch={BATCH}")
    for u in targets:
        try:
            p50, p95, lo = bench(u)
            print(f"{u:34s} p50={p50:7.2f}ms  p95={p95:7.2f}ms  min={lo:6.2f}ms")
        except Exception as e:  # noqa: BLE001
            print(f"{u:34s} ERROR: {e}")


if __name__ == "__main__":
    main()
