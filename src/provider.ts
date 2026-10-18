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
  /** Host log sink (used when `options.log` is absent — config JSON can't carry functions). */
  log?: (msg: string) => void;
  /** Host warn sink (used when `options.warn` is absent — config JSON can't carry functions). */
  warn?: (msg: string) => void;
};

export type EmbeddingsOptions = {
  endpoint?: string;
  model?: string;
  dims?: number;
  prefixes?: boolean;
  docPrefix?: string;
  queryPrefix?: string;
  topK?: number;
  /**
   * Minimum cosine similarity for a note to be injected/returned (default 0.58).
   * Score scales are model-specific: the shipped default model nomic-embed-text
   * compresses high (measured on a live store: unrelated text ~0.49–0.56,
   * relevant ~0.61–0.84), so its full gate is ~0.58 — the default — while a
   * lower-scale model needs a lower value. 0 disables the gate (topK always
   * injected). A `warn` callback is used for one-time diagnostics.
   */
  minScore?: number;
  /**
   * Sink for one-time diagnostics (default `console.warn`). Not settable via
   * config JSON; injectable for programmatic use and tests.
   */
  warn?: (msg: string) => void;
  dbPath?: string;
  /** Blend cosine with a recency term (0 = pure cosine, the default). */
  recencyWeight?: number;
  /** Recency half-life in days (default 30). */
  recencyHalfLifeDays?: number;
  /** Additive score boost for notes mirroring a current canonical fact. */
  canonicalWeight?: number;
  /**
   * Characters of each recalled note injected into the prompt (default 400);
   * longer notes are truncated in recall, so their tail can never be surfaced.
   * 0 disables truncation. At the default, truncation binds long before the
   * embedding dilutes: measured on nomic, a query for a fact stated at the top of
   * a note still scores 0.69 (1200 chars), 0.65 (2000) and 0.58 (5000) against the
   * 0.58 gate. Notes longer than this also warn at add time.
   */
  prefetchCharsPerNote?: number;
  /**
   * Include canonical notes (mirrors of MEMORY.md/USER.md) in prefetch. Default
   * `false`: they are already injected whole by the host every turn, so recalling
   * them again only spends tokens. The `provider_memory` search tool is unaffected.
   */
  prefetchIncludeCanonical?: boolean;
  /** Cosine at/above which a note is a duplicate (dream reconcile; default 0.92). */
  duplicateThreshold?: number;
  /** Cosine at/above which the dream judge is consulted (default 0.8). */
  ambiguousThreshold?: number;
  /**
   * Add-time near-duplicate guard (default true): skip an `add`/mirror whose
   * nearest active note in the same scope scores >= `duplicateThreshold`. Bounds
   * store growth at the source (one extra scan per write) instead of waiting for
   * the idle dream. `false` (or a non-positive `duplicateThreshold`) disables it.
   */
  dedupeOnWrite?: boolean;
  /**
   * Accelerated search via the optional `sqlite-vec` extension: `"auto"`
   * (default) uses it when installed, else falls back to brute force; `true`
   * warns when unavailable; `false` forces brute force. Results are identical
   * (vec0 is an **exact** KNN — SIMD-accelerated, *not* approximate).
   */
  accelerate?: "auto" | boolean;
  /** Info sink for the one-line startup report (default `console.log`). */
  log?: (msg: string) => void;
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
/**
 * Relevance gate default. The shipped default model is nomic-embed-text, whose
 * scale compresses high: measured on a real store, unrelated probes top out at
 * ~0.49–0.56 and relevant hits at ~0.61–0.84, so ~0.58 drops every unrelated
 * probe while keeping every relevant one. Other models need a lower value
 * (see README); 0 disables the gate.
 */
const DEFAULT_MIN_SCORE = 0.58;
const DEFAULT_PREFETCH_CHARS_PER_NOTE = 400;
/**
 * A legacy project id used by the host for the home/global directory. Notes were
 * then stored under `project:global`, which no query scope matches (queries see
 * `global` + the current project), so they were silently unreachable. It is
 * normalized to the global scope, and existing rows are moved on initialize().
 */
const LEGACY_PROJECT_GLOBAL = "global";

export class EmbeddingsMemoryProvider {
  readonly name = "embeddings";
  private store!: VectorStore;
  private embedder!: Embedder;
  private topK = 5;
  private minScore = DEFAULT_MIN_SCORE;
  private recencyWeight = 0;
  private recencyHalfLifeDays = 30;
  private canonicalWeight = 0;
  private duplicateThreshold = 0.92;
  private ambiguousThreshold = 0.8;
  private dedupeOnWrite = true;
  private prefetchCharsPerNote = DEFAULT_PREFETCH_CHARS_PER_NOTE;
  private prefetchIncludeCanonical = false;
  private accelerate: "auto" | boolean = "auto";
  private projectId: string | null = null;
  private warn: (msg: string) => void = console.warn;
  private log: (msg: string) => void = console.log;
  /** Query dims already checked for legacy mismatches (warn at most once each). */
  private warnedDims = new Set<number>();
  /** Long-note guidance is emitted at most once per process (log hygiene). */
  private warnedLongNote = false;

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
    this.minScore = o.minScore ?? DEFAULT_MIN_SCORE;
    this.recencyWeight = o.recencyWeight ?? 0;
    this.recencyHalfLifeDays = o.recencyHalfLifeDays ?? 30;
    this.canonicalWeight = o.canonicalWeight ?? 0;
    this.duplicateThreshold = o.duplicateThreshold ?? 0.92;
    this.ambiguousThreshold = o.ambiguousThreshold ?? 0.8;
    this.dedupeOnWrite = o.dedupeOnWrite ?? true;
    this.prefetchCharsPerNote = o.prefetchCharsPerNote ?? DEFAULT_PREFETCH_CHARS_PER_NOTE;
    this.prefetchIncludeCanonical = o.prefetchIncludeCanonical ?? false;
    this.accelerate = o.accelerate ?? "auto";
    this.projectId = ctx.projectId ?? null;
    // `warn`/`log` preference: an explicit option (only possible in-process),
    // then the host's context sink, then the console. Config JSON can't carry
    // functions, so the context sink is what actually makes the startup report
    // and dims warnings visible under opencode-hermes.
    this.warn = typeof o.warn === "function" ? o.warn : typeof ctx.warn === "function" ? ctx.warn : console.warn;
    this.log = typeof o.log === "function" ? o.log : typeof ctx.log === "function" ? ctx.log : console.log;
    const dbPath = o.dbPath ?? path.join(ctx.memoryRoot, "embeddings.sqlite");
    this.store = new VectorStore(dbPath);
    // Optional sqlite-vec accelerator: exact KNN, SIMD (~3x faster, ~25%
    // smaller). Falls back to brute force when the optional package isn't
    // installed. Not approximate — results are identical.
    if (this.accelerate !== false) {
      const ok = await this.store.initVecIndex();
      if (!ok && this.accelerate === true) this.warn("embeddings: sqlite-vec not available; falling back to brute-force search.");
    }
    // Repair notes stranded under a legacy project id: `project:global` is not in
    // any query scope, so those notes could never be recalled. Runs after
    // initVecIndex so the derived index is rebuilt with the new partition.
    const moved = this.store.reassignScope(`project:${LEGACY_PROJECT_GLOBAL}`, GLOBAL_SCOPE);
    if (moved) {
      this.warn(
        `embeddings: moved ${moved} note(s) from the legacy scope 'project:${LEGACY_PROJECT_GLOBAL}' to '${GLOBAL_SCOPE}' (they were previously unreachable).`,
      );
    }
    // Observable startup report: which search index is active.
    this.log(`embeddings: search index = ${this.store?.acceleratedIndex ? "sqlite-vec (exact KNN, SIMD)" : "brute-force"}`);
  }

  /** The active search path: `"sqlite-vec"` (exact SIMD KNN) or `"brute-force"`. */
  searchIndex(): "sqlite-vec" | "brute-force" {
    return this.store?.acceleratedIndex ? "sqlite-vec" : "brute-force";
  }

  /** opencode-hermes calls this per session so recall is project-scoped. */
  setProject(projectId: string | null): void {
    this.projectId = projectId ?? null;
  }

  private newId(): string {
    return `em_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
  }

  private projectScope(): string {
    // A project id of "global" (the host's home/global directory) must not become
    // the unreachable scope `project:global`.
    if (!this.projectId || this.projectId === LEGACY_PROJECT_GLOBAL) return GLOBAL_SCOPE;
    return `project:${this.projectId}`;
  }

  /** Visible scopes for a query: global + current project (never other projects). */
  private queryScopes(): string[] {
    return [...new Set([GLOBAL_SCOPE, this.projectScope()])];
  }

  systemPromptBlock(): string {
    return "A semantic memory provider (embeddings) is active: relevant notes are recalled automatically before each turn; use the provider_memory tool to search or add notes.";
  }

  /**
   * Nearest active note in `scope` when it is a near-duplicate of `vector`
   * (cosine >= `duplicateThreshold`) — its id, else null. The add-time guard:
   * skipping the write keeps the existing (often canonical) note and bounds
   * growth. Disabled by `dedupeOnWrite: false` or a non-positive threshold.
   */
  private nearDuplicateId(vector: number[], scope: string): string | null {
    if (!this.dedupeOnWrite || this.duplicateThreshold <= 0) return null;
    return this.store.search(vector, [scope], 1, this.duplicateThreshold)[0]?.id ?? null;
  }

  async add(content: string, tags?: string[]): Promise<{ id: string }> {
    const text = content.trim();
    if (!text) throw new Error("add requires content");
    this.noteLengthGuard(text);
    const scope = tags?.includes("global") ? GLOBAL_SCOPE : this.projectScope();
    const existing = this.store.findByText(text, scope);
    if (existing) return { id: existing }; // idempotent
    // Re-adding text whose only remaining rows are dream tombstones: drop the
    // stale tombstones so the freshly embedded note is the single live copy.
    this.store.deleteTombstonesByText(text, scope);
    const [vector] = await this.embedder.embed([text], "document");
    // The check-then-insert above is not atomic across the embed await: a second
    // caller may have inserted the same text meanwhile. Reuse its row instead of
    // creating a duplicate.
    const raced = this.store.findByText(text, scope);
    if (raced) return { id: raced };
    // Near-duplicate of an existing note: keep it rather than storing a
    // paraphrase. The idle dream still reconciles against canonical memory, so
    // freshness relative to MEMORY.md/USER.md is unaffected.
    const near = this.nearDuplicateId(vector, scope);
    if (near) return { id: near };
    const id = this.newId();
    this.store.add(id, text, vector, scope);
    return { id };
  }

  async search(query: string, limit?: number): Promise<ProviderHit[]> {
    return this.runSearch(query, limit ?? this.topK, {});
  }

  /** Shared search; `extra` carries per-caller store options (e.g. prefetch). */
  private async runSearch(query: string, limit: number, extra: SearchOptions): Promise<ProviderHit[]> {
    if (!query.trim()) return [];
    const [qv] = await this.embedder.embed([query], "query");
    // The `dims` filter silently drops notes embedded by a different model, so
    // recall can look empty after a model/quantization switch. Say so once per
    // query dimension instead of failing silently.
    if (!this.warnedDims.has(qv.length)) {
      this.warnedDims.add(qv.length);
      const stale = this.store.mismatchedDims(this.queryScopes(), qv.length);
      if (stale.length) {
        this.warn(
          `embeddings: note vectors with dimension(s) [${stale.join(", ")}] don't match the current model's ${qv.length}; those notes are skipped in recall (re-embed or clear the store to restore them).`,
        );
      }
    }
    const opts: SearchOptions = { dedupeByText: true, ...extra };
    if (this.recencyWeight > 0) {
      opts.recencyWeight = this.recencyWeight;
      opts.halfLifeMs = this.recencyHalfLifeDays * 86_400_000;
    }
    if (this.canonicalWeight > 0) opts.canonicalBoost = this.canonicalWeight;
    return this.store.search(qv, this.queryScopes(), limit, this.minScore, opts);
  }

  async prefetch(query: string): Promise<{ text: string; hits: number }> {
    // Canonical notes mirror the host's always-injected MEMORY/USER files, so
    // recalling them again only spends tokens (see prefetchIncludeCanonical).
    const hits = await this.runSearch(query, this.topK, { excludeCanonical: !this.prefetchIncludeCanonical });
    if (!hits.length) return { text: "", hits: 0 };
    const cap = this.prefetchCharsPerNote;
    const body = hits.map((h) => `• ${cap > 0 ? h.text.slice(0, cap) : h.text}`).join("\n");
    return { text: `<provider-memory source="embeddings">\nRelevant notes from past sessions:\n${body}\n</provider-memory>`, hits: hits.length };
  }

  /**
   * One-time guidance: a note longer than the injected slice is silently
   * truncated in recall, so its tail can never be surfaced.
   */
  private noteLengthGuard(text: string): void {
    if (this.warnedLongNote || this.prefetchCharsPerNote <= 0 || text.length <= this.prefetchCharsPerNote) return;
    this.warnedLongNote = true;
    this.warn(
      `embeddings: note is ${text.length} chars but prefetch injects only the first ${this.prefetchCharsPerNote}, so its tail can never be recalled. ` +
        `Prefer shorter notes; raising prefetchCharsPerNote works up to ~2k chars (past that a single mean-pooled vector dilutes and recall drops off).`,
    );
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
    this.noteLengthGuard(text);
    if (this.store.findByText(text, GLOBAL_SCOPE)) return;
    this.store.deleteTombstonesByText(text, GLOBAL_SCOPE); // drop dream tombstones, if any
    const [vector] = await this.embedder.embed([text], "document");
    if (this.store.findByText(text, GLOBAL_SCOPE)) return; // raced with a concurrent mirror
    if (this.nearDuplicateId(vector, GLOBAL_SCOPE)) return; // skip a paraphrase
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
