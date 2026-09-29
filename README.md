# opencode-hermes-embeddings

An **embeddings-backed long-term memory provider for [opencode-hermes](https://github.com/GitmanIII/opencode-hermes)** — semantic recall on your own GPU via [HuggingFace text-embeddings-inference (TEI)](https://github.com/huggingface/text-embeddings-inference).

It plugs into opencode-hermes's external **provider slot** (module-spec loading). It **complements** the built-in memory instead of replacing it:

- **Built-in memory** (opencode-hermes' `MEMORY.md`/`USER.md`) stays the small, always-on curated set — injected whole.
- **This provider** is the unbounded, **semantic** recall store — notes are embedded and the most relevant are injected automatically before each turn.

Unlike the usual CPU-only local-ONNX or cloud-API setups, embeddings here run **locally on the GPU** behind a fast OpenAI-compatible endpoint.

## How it works

- `add` / `onMemoryWrite` → embed the note (`document` side) and store the vector.
- `prefetch(query)` → embed the user message (`query` side), cosine-search the store, inject the top matches as a `<provider-memory>` block.
- `search` / `forget` → the `provider_memory` tool.
- **Scoped**: notes are tagged `global` or `project:<id>`; a query only sees `global` + the **current** project — no cross-project bleed.
- Vectors live in SQLite (BLOBs) with **brute-force cosine** — no native extension, sub-ms at personal scale.

## Requirements

- [opencode-hermes](https://github.com/GitmanIII/opencode-hermes) **>= v1.0.2** (external provider loading + `providerOptions`/`projectId`).
- A running **TEI** server with a CUDA GPU.
- Bun (opencode-hermes runtime).

## 1. Run TEI (GPU)

**Docker (recommended):**
```bash
docker run --gpus all -p 8080:80 -v "$HOME/.cache/huggingface:/data" \
  ghcr.io/huggingface/text-embeddings-inference:latest \
  --model-id nomic-ai/nomic-embed-text-v1.5
```

**Binary:** download a CUDA build from the [TEI releases](https://github.com/huggingface/text-embeddings-inference/releases), then:
```bash
text-embeddings-router --model-id nomic-ai/nomic-embed-text-v1.5 --port 8080
```

Verify:
```bash
curl -s localhost:8080/v1/embeddings -H 'content-type: application/json' \
  -d '{"input":"hello","model":"nomic-ai/nomic-embed-text-v1.5"}' | head -c 200
```

> `nomic-embed-text-v1.5` supports 8192 tokens and **requires** the `search_document:` / `search_query:` prefixes — this provider applies them for you (`prefixes: true` by default). Swapping models? Set `prefixes: false` for models that don't use them (e.g. `bge-m3`).

## 2. Install this provider

```bash
git clone https://github.com/GitmanIII/opencode-hermes-embeddings.git ~/opencode-hermes-embeddings
```

Point opencode-hermes at it — `~/.config/opencode/opencode-hermes.json`:
```jsonc
{
  "provider": "file:///home/YOU/opencode-hermes-embeddings/src/provider.ts",
  "providerOptions": {
    "endpoint": "http://127.0.0.1:8080",
    "model": "nomic-ai/nomic-embed-text-v1.5",
    "topK": 5,
    "minScore": 0.35
  }
}
```

Restart opencode. The log (`opencode-hermes.log`) should show `provider=embeddings`.

## Options (`providerOptions`)

| option | default | meaning |
|---|---|---|
| `endpoint` | `http://127.0.0.1:8080` | TEI base URL |
| `model` | `nomic-ai/nomic-embed-text-v1.5` | model id (must match TEI) |
| `dims` | auto | vector dims (informational; read from responses) |
| `prefixes` | `true` | apply doc/query retrieval prefixes |
| `docPrefix` / `queryPrefix` | nomic | override prefixes |
| `topK` | `5` | notes injected per turn / max search results |
| `minScore` | `0` | minimum cosine to inject |
| `dbPath` | `<memoryRoot>/embeddings.sqlite` | vector store path |

## Testing

```bash
bun run test
```

13 hermetic checks using an injectable fake embedder (no TEI needed): cosine, add/search ranking, prefetch block, project isolation, global mirroring + dedupe, forget.

## License

MIT © GitmanIII. Compatible with opencode-hermes (module-spec external provider).
