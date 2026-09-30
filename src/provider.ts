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
import { VectorStore, type ScoredNote, type SearchOptions } from "./store.ts";

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
  /** Blend cosine with a recency term (0 = pure cosine, the default). */
  recencyWeight?: number;
  /** Recency half-life in days (default 30). */
  recencyHalfLifeDays?: number;
  /** Additive score boost for notes mirroring a current canonical fact. */
  canonicalWeight?: number;
  /** Cosine at/above which a note is a duplicate (dream reconcile; default 0.92). */
  duplicateThreshold?: number;
  /** Cosine at/above which the dream judge is consulted (default 0.8). */
  ambiguousThreshold?: number;
};

/** Dream reconciliation options (structurally matches opencode-hermes). */
export type ReconcileOptions = {
  judge?: (canonicalText: string, noteText: string) => Promise<boolean>;
  duplicateThreshold?: number;
  ambiguousThreshold?: number;
  hardDelete?: boolean;
  now?: number;
};
export type ReconcileStats = { canonical: number; added: number; superseded: number; judged: number; removed: number };

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
  private recencyWeight = 0;
  private recencyHalfLifeDays = 30;
  private canonicalWeight = 0;
  private duplicateThreshold = 0.92;
  private ambiguousThreshold = 0.8;
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
    this.recencyWeight = o.recencyWeight ?? 0;
    this.recencyHalfLifeDays = o.recencyHalfLifeDays ?? 30;
    this.canonicalWeight = o.canonicalWeight ?? 0;
    this.duplicateThreshold = o.duplicateThreshold ?? 0.92;
    this.ambiguousThreshold = o.ambiguousThreshold ?? 0.8;
    this.projectId = ctx.projectId ?? null;
    const dbPath = o.dbPath ?? path.join(ctx.memoryRoot, "embeddings.sqlite");
    this.store = new VectorStore(dbPath);
  }

  /** opencode-hermes calls this per session so recall is project-scoped. */
  setProject(projectId: string | null): void {
    this.projectId = projectId ?? null;
  }

  private newId(): string {
    return `em_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
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
    const scope = tags?.includes("global") ? GLOBAL_SCOPE : this.projectScope();
    const existing = this.store.findByText(text, scope);
    if (existing) return { id: existing }; // idempotent
    const [vector] = await this.embedder.embed([text], "document");
    const id = this.newId();
    this.store.add(id, text, vector, scope);
    return { id };
  }

  async search(query: string, limit?: number): Promise<ProviderHit[]> {
    if (!query.trim()) return [];
    const [qv] = await this.embedder.embed([query], "query");
    const opts: SearchOptions = { dedupeByText: true };
    if (this.recencyWeight > 0) {
      opts.recencyWeight = this.recencyWeight;
      opts.halfLifeMs = this.recencyHalfLifeDays * 86_400_000;
    }
    if (this.canonicalWeight > 0) opts.canonicalBoost = this.canonicalWeight;
    return this.store.search(qv, this.queryScopes(), limit ?? this.topK, this.minScore, opts);
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

  /**
   * Mirror built-in memory writes (which are global) into the semantic store.
   * `remove`/`replace` propagate the deletion so wrong or superseded facts stop
   * being recalled; `demote` keeps the fact (the append-only path).
   */
  async onMemoryWrite(action: "add" | "replace" | "remove" | "demote", content: string, oldText?: string): Promise<void> {
    if (action === "demote") return;
    if (action === "remove") {
      const t = content.trim();
      if (t) this.store.deleteByText(t, GLOBAL_SCOPE);
      return;
    }
    if (action === "replace") {
      const o = (oldText ?? "").trim();
      if (o) this.store.deleteByText(o, GLOBAL_SCOPE);
    }
    const text = content.trim();
    if (!text) return;
    if (this.store.findByText(text, GLOBAL_SCOPE)) return;
    const [vector] = await this.embedder.embed([text], "document");
    this.store.add(this.newId(), text, vector, GLOBAL_SCOPE);
  }

  /**
   * "Dream": reconcile the semantic store against the current canonical facts.
   * Bounded (one embed + a store scan per canonical entry). Near-duplicate notes
   * (cosine >= duplicateThreshold) are superseded (soft-deleted) by the canonical
   * note; an ambiguous band can be resolved by the optional `judge`. With
   * `hardDelete`, tombstones are physically removed afterwards (GC).
   */
  async reconcile(canonical: string[], opts: ReconcileOptions = {}): Promise<ReconcileStats> {
    const scope = GLOBAL_SCOPE;
    const dup = opts.duplicateThreshold ?? this.duplicateThreshold;
    const amb = opts.ambiguousThreshold ?? this.ambiguousThreshold;
    const now = opts.now ?? Date.now();
    const stats: ReconcileStats = { canonical: 0, added: 0, superseded: 0, judged: 0, removed: 0 };

    this.store.clearCanonical(scope);

    // Pass 1: canonical facts already in the store are re-flagged in place.
    // Dedupe first so a fact present in both MEMORY.md and USER.md is added once.
    const facts = [...new Set(canonical.map((c) => c.trim()).filter(Boolean))];
    const pending: string[] = [];
    for (const text of facts) {
      const id = this.store.findByText(text, scope);
      if (id) {
        this.store.clearSuperseded(id);
        this.store.setCanonical(id, true);
        stats.canonical++;
      } else {
        pending.push(text);
      }
    }

    // Pass 2: embed all new facts in ONE request, then supersede near-duplicates.
    // (Batching turns N GPU round-trips into one; the GPU already batches.)
    const vectors = pending.length ? await this.embedder.embed(pending, "document") : [];
    for (let i = 0; i < pending.length; i++) {
      const text = pending[i];
      const vector = vectors[i];
      let superseded = this.store.search(vector, [scope], 10, dup).filter((h) => !h.canonical);
      if (!superseded.length && opts.judge) {
        const top = this.store.search(vector, [scope], 1, amb).find((h) => !h.canonical);
        if (top) {
          stats.judged++;
          if (await opts.judge(text, top.text)) superseded = [top];
        }
      }
      const id = this.newId();
      this.store.add(id, text, vector, scope, now);
      this.store.setCanonical(id, true);
      stats.added++;
      for (const h of superseded) {
        this.store.supersede(h.id, id);
        stats.superseded++;
      }
      stats.canonical++;
    }

    if (opts.hardDelete) stats.removed = this.store.deleteSuperseded(scope);
    return stats;
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
