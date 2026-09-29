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

- [opencode-hermes](https://github.com/GitmanIII/opencode-hermes) **>= v0.5.0** (external provider loading + `providerOptions`/`projectId`).
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

13 hermetic checks using an injectable fake embedder (no TEI needed): cosine, add/search ranking, prefetch block, project isolation, global mirroring + dedupe, forget.

## License

MIT © GitmanIII. Compatible with opencode-hermes (module-spec external provider).
