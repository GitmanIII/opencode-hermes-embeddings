/**
 * Embeddings-backed MemoryProvider for opencode-hermes.
 *
 * Structurally compatible with opencode-hermes' `MemoryProvider` interface
 * (same method surface), so opencode-hermes can load it as an external provider
 * via `provider: "file:///…/src/provider.ts"` (module-spec loading).
 *
 * Complements the built-in two-file memory: the built-in stays the small,
 * always-on curated set; this provider is the unbounded, semantic recall store.
 * Notes are scoped (`global` or `project:<id>`) so prefetch never bleeds facts
 * across projects.
 */
import * as path from "node:path";
import { TeiEmbedder, type EmbedKind, type Embedder } from "./embedder.ts";
import { VectorStore, type ScoredNote } from "./store.ts";

export type ProviderContext = {
  memoryRoot: string;
  providerPath?: string;
  prefetchLimit?: number;
  options?: Record<string, unknown>;
  projectId?: string;
};

export type EmbeddingsOptions = {
  endpoint?: string;
  model?: string;
  dims?: number;
  prefixes?: boolean;
  docPrefix?: string;
  queryPrefix?: string;
  topK?: number;
  minScore?: number;
  dbPath?: string;
};

export type ProviderHit = { id: string; text: string; score: number };

const DEFAULT_ENDPOINT = "http://127.0.0.1:8080";
const DEFAULT_MODEL = "nomic-ai/nomic-embed-text-v1.5";
const GLOBAL_SCOPE = "global";

export class EmbeddingsMemoryProvider {
  readonly name = "embeddings";
  private store!: VectorStore;
  private embedder!: Embedder;
  private topK = 5;
  private minScore = 0;
  private projectId: string | null = null;

  /** `embedder` is injectable for tests. */
  constructor(private opts: EmbeddingsOptions = {}, embedder?: Embedder) {
    if (embedder) this.embedder = embedder;
  }

  async initialize(ctx: ProviderContext): Promise<void> {
    const o = { ...(ctx.options ?? {}), ...this.opts } as EmbeddingsOptions;
    if (!this.embedder) {
      this.embedder = new TeiEmbedder({
        endpoint: o.endpoint ?? DEFAULT_ENDPOINT,
        model: o.model ?? DEFAULT_MODEL,
        dims: o.dims,
        prefixes: o.prefixes ?? true,
        docPrefix: o.docPrefix,
        queryPrefix: o.queryPrefix,
      });
    }
    this.topK = o.topK ?? ctx.prefetchLimit ?? 5;
    this.minScore = o.minScore ?? 0;
    this.projectId = ctx.projectId ?? null;
    const dbPath = o.dbPath ?? path.join(ctx.memoryRoot, "embeddings.sqlite");
    this.store = new VectorStore(dbPath);
  }

  /** opencode-hermes calls this per session so recall is project-scoped. */
  setProject(projectId: string | null): void {
    this.projectId = projectId ?? null;
  }

  private projectScope(): string {
    return this.projectId ? `project:${this.projectId}` : GLOBAL_SCOPE;
  }

  /** Visible scopes for a query: global + current project (never other projects). */
  private queryScopes(): string[] {
    return [...new Set([GLOBAL_SCOPE, this.projectScope()])];
  }

  systemPromptBlock(): string {
    return "A semantic memory provider (embeddings) is active: relevant notes are recalled automatically before each turn; use the provider_memory tool to search or add notes.";
  }

  async add(content: string, tags?: string[]): Promise<{ id: string }> {
    const text = content.trim();
    if (!text) throw new Error("add requires content");
    const [vector] = await this.embedder.embed([text], "document");
    const scope = tags?.includes("global") ? GLOBAL_SCOPE : this.projectScope();
    const id = `em_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
    this.store.add(id, text, vector, scope);
    return { id };
  }

  async search(query: string, limit?: number): Promise<ProviderHit[]> {
    if (!query.trim()) return [];
    const [qv] = await this.embedder.embed([query], "query");
    return this.store.search(qv, this.queryScopes(), limit ?? this.topK, this.minScore);
  }

  async prefetch(query: string): Promise<{ text: string; hits: number }> {
    const hits = await this.search(query, this.topK);
    if (!hits.length) return { text: "", hits: 0 };
    const body = hits.map((h) => `• ${h.text.slice(0, 400)}`).join("\n");
    return { text: `<provider-memory source="embeddings">\nRelevant notes from past sessions:\n${body}\n</provider-memory>`, hits: hits.length };
  }

  forget(id: string): boolean {
    this.store.delete(id);
    return true;
  }

  /** Mirror built-in memory writes (which are global) into the semantic store. */
  async onMemoryWrite(action: "add" | "replace" | "remove", content: string): Promise<void> {
    if (action === "remove") return;
    const text = content.trim();
    if (!text) return;
    if (this.store.hasText(text, GLOBAL_SCOPE)) return;
    const [vector] = await this.embedder.embed([text], "document");
    this.store.add(`em_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`, text, vector, GLOBAL_SCOPE);
  }

  shutdown(): void {
    this.store?.close();
  }

  // Exposed for the provider_memory tool via opencode-hermes.
  embedKindSample(): EmbedKind {
    return "document";
  }
}

/** Factory used by opencode-hermes module-spec loading. */
export function createProvider(options: EmbeddingsOptions = {}): EmbeddingsMemoryProvider {
  return new EmbeddingsMemoryProvider(options);
}

export default createProvider;
