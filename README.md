# opencode-hermes-embeddings

An **embeddings-backed long-term memory provider for [opencode-hermes](https://github.com/GitmanIII/opencode-hermes)** — semantic recall on your own GPU via [HuggingFace text-embeddings-inference (TEI)](https://github.com/huggingface/text-embeddings-inference).

It plugs into opencode-hermes's external **provider slot** (module-spec loading). It **complements** the built-in memory instead of replacing it:

- **Built-in memory** (opencode-hermes' `MEMORY.md`/`USER.md`) stays the small, always-on curated set — injected whole.
- **This provider** is the unbounded, **semantic** recall store — notes are embedded and the most relevant are injected automatically before each turn.

Unlike the usual CPU-only local-ONNX or cloud-API setups, embeddings here run **locally on the GPU** behind a fast OpenAI-compatible endpoint.

## How it works

- `add` / `onMemoryWrite` → embed the note (`document` side) and store the vector. Built-in writes are mirrored: `replace` deletes the superseded text and stores the new one, `remove` propagates the deletion, and `demote` keeps the fact (append-only). `add` is idempotent per (text, scope) and re-checks after the embed await, so concurrent identical adds don't create duplicates; resurrecting a re-added fact clears only its **tombstoned** rows, so it can't clobber a live note raced in concurrently.
- `prefetch(query)` → embed the user message (`query` side), cosine-search the store, inject the top matches as a `<provider-memory>` block.
- `search` / `forget` → the `provider_memory` tool.
- **Scoped**: notes are tagged `global` or `project:<id>`; a query only sees `global` + the **current** project — no cross-project bleed.
- `reconcile(canonical)` — the **dream** invoked on idle by opencode-hermes: notes made obsolete by a current canonical fact are **superseded** (tombstoned, excluded from recall); near-duplicates (`cosine ≥ duplicateThreshold`) are collapsed, and an optional `judge` resolves an ambiguous band. opencode-hermes passes `hardDelete`, so each dream GCs its tombstones (the store stays bounded). All new canonical facts are embedded in **one batched request** (was one per fact), and canonical facts are deduped across `MEMORY.md`/`USER.md`.
- Vectors live in SQLite (BLOBs) with **brute-force cosine** and no native extension. Search **streams row-by-row with zero-copy float32 views** (no per-row allocation): measured ~3 µs/note (100k notes ≈ 0.34 s) at a flat ~60 MB RSS — ~4.5× faster and no memory blow-up vs the earlier materialize-then-map path. Identical text (a note mirrored into both scopes) is deduped **before** the top-K cut so it can't crowd out a distinct hit. Vectors whose `dims` don't match the query are skipped, so switching models/quantizations can't silently produce garbage scores — and a one-time warning is logged (once per query dims) so the skipped notes aren't silently lost from recall.

## Requirements

- [opencode-hermes](https://github.com/GitmanIII/opencode-hermes) **>= v0.5.0** (external provider loading + `providerOptions`/`projectId`); the **dream** (`reconcile`) needs **>= v0.7.0**.
- A running **TEI** server with a CUDA GPU.
- Bun (opencode-hermes runtime).

## 1. Run TEI (GPU, Docker)

One-time setup (needs sudo):
```bash
sudo pacman -S nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
sudo usermod -aG docker "$USER"    # then log out/in (or run docker with sudo)
```

Then run TEI — **bound to loopback only** (`127.0.0.1`, never the LAN):
```bash
./scripts/run-tei.sh
# or manually:
docker run --rm --gpus all -p 127.0.0.1:8080:80 -v "$HOME/.cache/huggingface:/data" \
  ghcr.io/huggingface/text-embeddings-inference:latest \
  --model-id nomic-ai/nomic-embed-text-v1.5
```

> `-p 127.0.0.1:8080:80` (not `-p 8080:80`) — the latter publishes on all interfaces.

**Binary (no Docker):** download a CUDA build from the [TEI releases](https://github.com/huggingface/text-embeddings-inference/releases); it links against the CUDA runtime externally, so you must provide matching `libcudart`/`libcublas` (e.g. the `nvidia-*-cu12` pip wheels) and set `LD_LIBRARY_PATH`. This is why Docker + `nvidia-container-toolkit` is recommended.

Verify:
```bash
curl -s localhost:8080/v1/embeddings -H 'content-type: application/json' \
  -d '{"input":"hello","model":"nomic-ai/nomic-embed-text-v1.5"}' | head -c 200
```

> `nomic-embed-text-v1.5` supports 8192 tokens and **requires** the `search_document:` / `search_query:` prefixes — this provider applies them for you (`prefixes: true` by default). Swapping models? Set `prefixes: false` for models that don't use them (e.g. `bge-m3`).

## Keep it running (automatic)

TEI is a service the provider connects to; it should be up whenever opencode runs.

**Detached container (recommended):**
```bash
sudo systemctl enable docker      # start the Docker daemon at boot
./scripts/run-tei.sh --detach     # named container, --restart unless-stopped
./scripts/stop-tei.sh             # add --remove to delete the container
```
TEI then starts with Docker on every boot. (`docker.socket` alone is socket-activated, so the daemon starts on first use — enable `docker.service` for reliable boot start.)

**`systemd --user` unit (optional, starts at login):**
```bash
./scripts/run-tei.sh --detach      # create the container once
mkdir -p ~/.config/systemd/user
cp systemd/opencode-hermes-tei.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now opencode-hermes-tei.service
```
Caveat: the **user manager must already be in the `docker` group**. If you were added to `docker` after the manager started — common with `Linger=yes`, where the manager starts at boot and persists — it stays stale and the unit fails. Refresh with a reboot, or `sudo systemctl restart user@$(id -u).service`. (An unprivileged user manager can't add the group itself, so `SupplementaryGroups=docker` doesn't help — it fails with `216/GROUP`.) If you just want automatic operation, prefer the restart-policy route above.

## Alternative embedder: local venv (no Docker)

TEI+Docker is the fastest, but needs the NVIDIA container toolkit. If you'd rather stay Docker-free, the repo ships an equivalent OpenAI-compatible server running `sentence-transformers` on CUDA in a self-contained `uv` venv:

```bash
./scripts/setup-venv.sh          # once: creates .venv (torch cu + sentence-transformers)
./scripts/run-venv.sh            # foreground (127.0.0.1:8080)
./scripts/run-venv.sh --detach   # background
./scripts/stop-venv.sh
```
The provider is unchanged — just point `providerOptions.endpoint` at it. (No prefixes here either; the provider adds them.)

Trade-offs vs TEI: same model/endpoint, but Python's `import torch` makes **startup slower** and there's no container. Run both on different ports and compare latency:

```bash
./scripts/run-tei.sh --detach                    # TEI on :8080
EMBED_PORT=8081 ./scripts/run-venv.sh --detach   # venv on :8081
python scripts/bench.py http://127.0.0.1:8080 http://127.0.0.1:8081
```

Measured on an RTX 3090 (nomic-embed-text-v1.5, single short note):

| backend | cold start | p50 | p95 | VRAM |
|---|---|---|---|---|
| **TEI** (Docker) | ~5–8 s | **1.73 ms** | 1.79 ms | **562 MiB** |
| **Ollama** (GGUF) | warm | 5.39 ms | 6.35 ms | 620 MiB |
| **venv** (sentence-transformers) | ~6–8 s | 6.12 ms | 6.88 ms | 902 MiB |

All are imperceptible for this workload (a few embeds per turn). TEI wins on latency/VRAM; no-Docker routes (Ollama, venv) trade a little speed. Note the venv route pins `transformers<5` (nomic's remote code predates Transformers 5).

## Other backends (Ollama, llama.cpp)

Anything that serves OpenAI `/v1/embeddings` works — set `endpoint` + `model`, no code changes:

**Ollama** (GGUF via llama.cpp):
```bash
ollama pull nomic-embed-text
ollama serve                       # http://127.0.0.1:11434
```
```jsonc
"providerOptions": { "endpoint": "http://127.0.0.1:11434/v1", "model": "nomic-embed-text", "prefixes": true }
```

**llama.cpp `llama-server`** (GGUF, no daemon):
```bash
llama-server --embedding -m nomic-embed-text-v1.5.Q8_0.gguf --host 127.0.0.1 --port 8080
```
```jsonc
"providerOptions": { "endpoint": "http://127.0.0.1:8080/v1", "model": "nomic-embed-text-v1.5", "prefixes": true }
```

Ollama and llama.cpp **share the same engine**, so they perform nearly identically — that's why neither gets its own script. Measured (RTX 3090, nomic-embed-text): **Ollama p50 5.39 ms / p95 6.35 ms, 620 MiB VRAM, 100 % GPU**; expect llama.cpp to be similar.

Note: vectors are close but not identical across engines/quantizations, so **use one backend per store** (or re-embed when switching).

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
    "minScore": 0.58
  }
}
```

Restart opencode. The log (`opencode-hermes.log`) should show `provider=embeddings`.

> **`minScore` — tune per model.** It's the cosine floor a note must clear to be injected; `0` disables the gate (topK always injected). Score scales differ by model: `nomic-embed-text` compresses high, so its *unrelated* text still scores ~0.49–0.56 and truly relevant hits ~0.61–0.84. Measured on a real store, **≈0.58 is the full gate** for nomic (drops every unrelated probe, keeps every relevant top hit), so that is the default — matching the shipped default model. A lower-scale model needs a lower value (or `0` to disable).

## Options (`providerOptions`)

| option | default | meaning |
|---|---|---|
| `endpoint` | `http://127.0.0.1:8080` | TEI base URL |
| `model` | `nomic-ai/nomic-embed-text-v1.5` | model id (must match TEI) |
| `dims` | auto | vector dims (informational; read from responses) |
| `prefixes` | `true` | apply doc/query retrieval prefixes |
| `docPrefix` / `queryPrefix` | nomic | override prefixes |
| `topK` | `5` | notes injected per turn / max search results |
| `minScore` | `0.58` | minimum cosine to inject; **model-specific** (default matches nomic's full gate; 0 = off) |
| `recencyWeight` | `0` | blend cosine with a recency term (0 = pure cosine) |
| `recencyHalfLifeDays` | `30` | recency half-life, in days |
| `canonicalWeight` | `0` | additive score boost for canonical notes |
| `duplicateThreshold` | `0.92` | dream: cosine ≥ this collapses a near-duplicate |
| `ambiguousThreshold` | `0.8` | dream: cosine ≥ this consults the judge |
| `dbPath` | `<memoryRoot>/embeddings.sqlite` | vector store path |

## Verified (RTX 3090)

End-to-end, with opencode-hermes wired to this provider and TEI running on the GPU:

- **Explicit semantic search** — `provider_memory search "colour preference"` returned *"My favourite test colour is aubergine-42"* at **score 0.76**.
- **Cross-session semantic recall** — in a *new* session, asking *"when do we copy data off-site?"* surfaced *"The nightly backup archives snapshots to cold storage at 03:00."* (no keyword overlap).
- **Latency** — TEI `openai_embed` on GPU: **~1.2 ms** for a short note, **~4–18 ms** for longer prompts; 768-dim vectors; **~0.6 GB VRAM**.
- **Scoping** — a note added in a project is visible only in that project (+ `global`); verified by the isolation tests.

Try it:
```
provider_memory add "The nightly backup archives snapshots to cold storage at 03:00."
provider_memory search "when do we copy data off-site?"
```

## Testing

```bash
bun run test
```

41 hermetic checks using an injectable fake embedder (no TEI needed): cosine, recency ranking, dims guard + mismatch warning, add/search, concurrent-add dedupe, tombstone-only cleanup, prefetch block, project isolation, global mirroring + dedupe (before the top-K cut), replace/remove/demote propagation, dream reconcile (near-dup collapse, judge band, canonical, GC, tombstone resurrection, batched multi-fact pass), volume search, forget.

## Roadmap

- **Add-time near-duplicate guard** (next) — on `add`/mirror, a top-1 cosine check at/above `duplicateThreshold` skips or supersedes a near-duplicate, bounding growth at the source (O(N) per write, no idle pass). Complements the canonical dream.
- **ANN index (`sqlite-vec`)** — *deferred, trigger-gated.* Brute-force streaming is ~3 µs/note with flat memory, so an approximate index is only worth it once measured need appears: store **> ~50k notes** or search **p95 > ~50 ms**. Preferred route keeps vectors in SQLite via a loadable extension (adds a binary dependency and uses approximate recall), so it needs its own benchmark before adopting.
- **Store-wide contradiction pass** — full note-vs-note supersession needs ANN candidate generation; not viable as an O(N²) JS pass at personal-store scale.

## License

MIT © GitmanIII. Compatible with opencode-hermes (module-spec external provider).
