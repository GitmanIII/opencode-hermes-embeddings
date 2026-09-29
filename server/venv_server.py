#!/usr/bin/env python3
"""Local OpenAI-compatible embeddings server (GPU) — the no-Docker alternative.

Self-contained: a uv venv with sentence-transformers on CUDA, exposing the same
POST /v1/embeddings shape as TEI, so the opencode-hermes provider needs no
changes — just point `endpoint` here.

Bind is 127.0.0.1 only. Do NOT add retrieval prefixes here: the provider applies
them (search_document: / search_query:).

Run: python venv_server.py     (env: EMBED_MODEL, EMBED_HOST, EMBED_PORT, EMBED_DEVICE)
"""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

MODEL = os.environ.get("EMBED_MODEL", "nomic-ai/nomic-embed-text-v1.5")
HOST = os.environ.get("EMBED_HOST", "127.0.0.1")
PORT = int(os.environ.get("EMBED_PORT", "8080"))
DEVICE = os.environ.get("EMBED_DEVICE", "cuda")

_model: SentenceTransformer | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _model
    _model = SentenceTransformer(MODEL, device=DEVICE, trust_remote_code=True)
    yield
    _model = None


app = FastAPI(lifespan=lifespan)


class EmbeddingsRequest(BaseModel):
    input: str | list[str]
    model: str | None = None
    encoding_format: str | None = None


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL, "device": DEVICE, "ready": _model is not None}


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": MODEL, "object": "model"}]}


@app.post("/v1/embeddings")
def embeddings(req: EmbeddingsRequest):
    assert _model is not None, "model not loaded"
    inputs = req.input if isinstance(req.input, list) else [req.input]
    vectors = _model.encode(list(inputs), normalize_embeddings=True, convert_to_numpy=True)
    data = [
        {"object": "embedding", "index": i, "embedding": [float(x) for x in v]}
        for i, v in enumerate(vectors)
    ]
    return {"object": "list", "data": data, "model": MODEL, "usage": {"prompt_tokens": 0, "total_tokens": 0}}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
