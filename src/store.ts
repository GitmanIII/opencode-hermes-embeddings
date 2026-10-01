/**
 * Vector store: SQLite (bun:sqlite), float32 vectors as BLOBs, cosine search.
 *
 * Two interchangeable search paths:
 *  - **brute force** (always available): stream rows with zero-copy float32
 *    views and score in JS. Fine to tens of thousands of notes.
 *  - **sqlite-vec** (optional, auto-detected): the `vec0` extension does an
 *    exact SIMD KNN in C, ~3x faster and ~25% smaller, at ~2x ingest cost. It is
 *    loaded only if the (optional) `sqlite-vec` package is installed; otherwise
 *    we fall back to brute force with identical results.
 *
 * The `memo` table is the source of truth; `vec_<dims>` tables are a derived
 * index that can be rebuilt from `memo` at any time (`rebuildAnn`).
 */
import { Database } from "bun:sqlite";

export type StoredNote = {
  id: string;
  text: string;
  scope: string;
  created_at: number;
  /** If set, this note was superseded by another (soft-deleted / tombstoned). */
  superseded_by: string | null;
  /** 1 when the note mirrors a current canonical fact (built-in memory). */
  canonical: number;
};
export type ScoredNote = StoredNote & { score: number };

/**
 * Search tuning: recency blend, canonical boost, and whether to include
 * tombstoned (superseded) notes — excluded by default.
 */
export type SearchOptions = {
  recencyWeight?: number;
  halfLifeMs?: number;
  now?: number;
  canonicalBoost?: number;
  includeSuperseded?: boolean;
  /** Keep only the best-scoring note per exact text (before the `limit` cut). */
  dedupeByText?: boolean;
};

/** Rows loaded from `memo` for scoring. */
type MemoRow = {
  id: string;
  text: string;
  scope: string;
  created_at: number;
  superseded_by: string | null;
  canonical: number;
  vector: Uint8Array;
};

export class VectorStore {
  private db: Database;
  /** True once the sqlite-vec extension is loaded and the index is usable. */
  private annEnabled = false;
  /** Dims for which a `vec_<dims>` table has been created this session. */
  private vecTables = new Set<number>();

  constructor(path: string) {
    this.db = new Database(path);
    // WAL + a busy timeout so several OpenCode processes can share one store
    // without SQLITE_BUSY errors.
    this.db.run(`PRAGMA journal_mode = WAL;`);
    this.db.run(`PRAGMA busy_timeout = 5000;`);
    this.db.run(
      `CREATE TABLE IF NOT EXISTS memo (
         id TEXT PRIMARY KEY,
         text TEXT NOT NULL,
         vector BLOB NOT NULL,
         dims INTEGER NOT NULL,
         scope TEXT NOT NULL,
         created_at INTEGER NOT NULL,
         superseded_by TEXT,
         canonical INTEGER NOT NULL DEFAULT 0
       );`,
    );
    // Small key/value table for derived-index bookkeeping.
    this.db.run(`CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);`);
    this.migrate();
    this.db.run(`CREATE INDEX IF NOT EXISTS memo_scope_idx ON memo (scope);`);
  }

  /** Add columns to stores created before tombstones/canonical existed. */
  private migrate(): void {
    const cols = new Set((this.db.query(`PRAGMA table_info(memo)`).all() as { name: string }[]).map((c) => c.name));
    try {
      if (!cols.has("superseded_by")) this.db.run(`ALTER TABLE memo ADD COLUMN superseded_by TEXT`);
      if (!cols.has("canonical")) this.db.run(`ALTER TABLE memo ADD COLUMN canonical INTEGER NOT NULL DEFAULT 0`);
    } catch (err) {
      // A concurrent process may have migrated between our PRAGMA read and here.
      if (!/duplicate column name/i.test(String(err))) throw err;
    }
  }

  /** Whether the sqlite-vec accelerated path is active. */
  get ann(): boolean {
    return this.annEnabled;
  }

  /**
   * Load the optional `sqlite-vec` extension and make the derived index usable.
   * Returns true when active, false when unavailable (the caller then keeps the
   * brute-force path). Best-effort: never throws.
   *
   * On the first enable (or after the index is dropped) the index is rebuilt
   * from `memo` once, then kept in sync incrementally.
   */
  async initAnn(): Promise<boolean> {
    if (this.annEnabled) return true;
    try {
      // Non-literal spec: don't make TS/bundlers hard-resolve the optional dep.
      const spec = "sqlite-vec";
      const mod = (await import(spec)) as { getLoadablePath?: () => string };
      if (typeof mod?.getLoadablePath !== "function") throw new Error("sqlite-vec: no getLoadablePath");
      this.db.loadExtension(mod.getLoadablePath());
      this.annEnabled = true;
      const built = (this.db.query(`SELECT value FROM meta WHERE key = 'ann_built'`).get() as { value?: string } | null)?.value;
      if (built !== "1") this.rebuildAnn();
      else for (const d of this.activeDims()) this.ensureVecTable(d);
      return true;
    } catch {
      this.annEnabled = false;
      return false;
    }
  }

  // ─── Derived vec0 index ───

  /** Distinct active vector dims (used to pre-create vec tables). */
  private activeDims(): number[] {
    return (this.db.query(`SELECT DISTINCT dims FROM memo WHERE superseded_by IS NULL`).all() as { dims: number }[]).map((r) => r.dims);
  }

  private ensureVecTable(dims: number): void {
    if (this.vecTables.has(dims)) return;
    this.db.run(
      `CREATE VIRTUAL TABLE IF NOT EXISTS "vec_${dims}" USING vec0(id TEXT PRIMARY KEY, scope TEXT PARTITION KEY, embedding float[${dims}] distance_metric=cosine)`,
    );
    this.vecTables.add(dims);
  }

  private vecInsert(id: string, vec: Float32Array, scope: string, dims: number): void {
    this.ensureVecTable(dims);
    const buf = Buffer.from(vec.buffer, vec.byteOffset, vec.byteLength);
    try {
      this.db.run(`INSERT INTO "vec_${dims}"(id, scope, embedding) VALUES (?, ?, ?)`, [id, scope, buf]);
    } catch (err) {
      // vec0 has no upsert: replace by deleting the stale row first.
      if (!/unique/i.test(String(err))) throw err;
      this.db.run(`DELETE FROM "vec_${dims}" WHERE id = ?`, [id]);
      this.db.run(`INSERT INTO "vec_${dims}"(id, scope, embedding) VALUES (?, ?, ?)`, [id, scope, buf]);
    }
  }

  /** Remove ids from whichever vec tables hold them (uses dims from `memo`). */
  private vecDelete(ids: { id: string; dims: number }[]): void {
    if (!this.annEnabled || !ids.length) return;
    const byDims = new Map<number, string[]>();
    for (const { id, dims } of ids) {
      if (!byDims.has(dims)) byDims.set(dims, []);
      byDims.get(dims)!.push(id);
    }
    for (const [dims, list] of byDims) {
      this.ensureVecTable(dims);
      this.db.run(`DELETE FROM "vec_${dims}" WHERE id IN (${list.map(() => "?").join(",")})`, list);
    }
  }

  /**
   * Rebuild the derived index from `memo` (active notes only). Safe to call any
   * time the index looks stale; also runs automatically on first enable.
   */
  rebuildAnn(): number {
    if (!this.annEnabled) return 0;
    for (const d of this.vecTables) this.db.run(`DROP TABLE IF EXISTS "vec_${d}"`);
    this.vecTables.clear();
    let n = 0;
    const rows = this.db.query(`SELECT id, vector, dims, scope FROM memo WHERE superseded_by IS NULL`).iterate() as IterableIterator<{
      id: string; vector: Uint8Array; dims: number; scope: string;
    }>;
    for (const r of rows) {
      const v = r.vector.byteOffset % 4 === 0
        ? new Float32Array(r.vector.buffer, r.vector.byteOffset, r.vector.byteLength >> 2)
        : new Float32Array(r.vector.slice().buffer, 0, r.vector.byteLength >> 2);
      this.vecInsert(r.id, v, r.scope, r.dims);
      n++;
    }
    this.db.run(`INSERT OR REPLACE INTO meta(key, value) VALUES ('ann_built', '1')`);
    return n;
  }

  add(id: string, text: string, vector: number[], scope: string, createdAt = Date.now()): void {
    const f32 = new Float32Array(vector);
    this.db.run(`INSERT OR REPLACE INTO memo (id, text, vector, dims, scope, created_at) VALUES (?, ?, ?, ?, ?, ?)`, [
      id,
      text,
      Buffer.from(f32.buffer),
      vector.length,
      scope,
      createdAt,
    ]);
    if (this.annEnabled) this.vecInsert(id, f32, scope, vector.length);
  }

  /**
   * Find an ACTIVE (non-tombstoned) note by exact text. Tombstoned notes are
   * intentionally invisible: re-adding their text must create/resurrect a
   * recallable note rather than silently resolving to a superseded row.
   */
  findByText(text: string, scope: string): string | null {
    const row = this.db
      .query(`SELECT id FROM memo WHERE text = ? AND scope = ? AND superseded_by IS NULL LIMIT 1`)
      .get(text, scope) as { id: string } | null;
    return row?.id ?? null;
  }

  /**
   * Distinct active vector dims within `scopes` that differ from `currentDims`
   * (e.g. notes embedded by a previous model). `search` skips these; this lets
   * the caller warn instead of silently losing recall after a model switch.
   */
  mismatchedDims(scopes: string[], currentDims: number): number[] {
    const unique = [...new Set(scopes.filter(Boolean))];
    const where = unique.length ? `scope IN (${unique.map(() => "?").join(",")})` : `1=1`;
    const rows = this.db
      .query(`SELECT DISTINCT dims FROM memo WHERE ${where} AND superseded_by IS NULL AND dims <> ?`)
      .all(...unique, currentDims) as { dims: number }[];
    return rows.map((r) => r.dims);
  }

  /**
   * Cosine search. Uses the sqlite-vec index when it is active and the query
   * cannot diverge from the brute-force result (i.e. no recency/canonical boost
   * and superseded notes excluded); otherwise it streams rows and scores in JS.
   */
  search(queryVector: number[], scopes: string[], limit: number, minScore: number, opts?: SearchOptions): ScoredNote[] {
    const boosts = (opts?.recencyWeight ?? 0) > 0 || (opts?.canonicalBoost ?? 0) > 0;
    if (this.annEnabled && !opts?.includeSuperseded && !boosts) {
      const viaAnn = this.searchAnn(queryVector, scopes, limit, minScore, opts);
      if (viaAnn) return viaAnn;
    }
    return this.searchBrute(queryVector, scopes, limit, minScore, opts);
  }

  /** sqlite-vec KNN path; returns null when no vec table exists for the dims. */
  private searchAnn(queryVector: number[], scopes: string[], limit: number, minScore: number, opts?: SearchOptions): ScoredNote[] | null {
    const q = Float32Array.from(queryVector);
    const dims = q.length;
    if (!this.vecTables.has(dims)) return null;
    const unique = [...new Set(scopes.filter(Boolean))];
    if (!unique.length) return [];
    const k = Math.max(limit * 25, 200);
    const qbuf = Buffer.from(q.buffer, q.byteOffset, q.byteLength);
    const dist = new Map<string, number>();
    for (const scope of unique) {
      const rows = this.db
        .query(`SELECT id, distance FROM "vec_${dims}" WHERE embedding MATCH ? AND scope = ? ORDER BY distance LIMIT ?`)
        .all(qbuf, scope, k) as { id: string; distance: number }[];
      for (const r of rows) dist.set(r.id, r.distance);
    }
    // Nothing from the index (empty/stale, or genuinely none): let the caller
    // fall back to brute force so a missing index can never hide notes.
    if (!dist.size) return null;
    const ids = [...dist.keys()];
    const rows = this.db
      .query(`SELECT id, text, scope, created_at, superseded_by, canonical FROM memo WHERE id IN (${ids.map(() => "?").join(",")})`)
      .all(...ids) as Omit<MemoRow, "vector">[];
    const scored: ScoredNote[] = [];
    for (const r of rows) {
      if (r.superseded_by !== null) continue;
      const score = 1 - (dist.get(r.id) ?? 1); // vec0 cosine distance = 1 - cosine
      if (score >= minScore) scored.push({ ...r, score });
    }
    return this.finalize(scored, limit, opts);
  }

  /**
   * Brute-force cosine search, streamed row-by-row with zero-copy float32 views
   * (no per-row `Array.from`). Memory stays flat at personal scale.
   */
  private searchBrute(queryVector: number[], scopes: string[], limit: number, minScore: number, opts?: SearchOptions): ScoredNote[] {
    const q = Float32Array.from(queryVector);
    let qn = 0;
    for (let i = 0; i < q.length; i++) qn += q[i] * q[i];
    qn = Math.sqrt(qn);

    const unique = [...new Set(scopes.filter(Boolean))];
    const where = unique.length ? `scope IN (${unique.map(() => "?").join(",")})` : `1=1`;
    // `dims = ?` skips vectors from a different model/quantization. Comparing a
    // query against a mismatched vector would otherwise produce meaningless
    // scores (the dot product just runs over the shorter length).
    const filter = `${opts?.includeSuperseded ? "" : " AND superseded_by IS NULL"} AND dims = ?`;
    const stmt = this.db.query(
      `SELECT id, text, scope, created_at, superseded_by, canonical, vector FROM memo WHERE ${where}${filter}`,
    );
    const rows = stmt.iterate(...unique, q.length) as IterableIterator<MemoRow>;

    const scored: ScoredNote[] = [];
    for (const r of rows) {
      // Zero-copy view into the BLOB; fall back to a copy if the offset isn't
      // 4-byte aligned (Float32Array requires alignment).
      const v =
        r.vector.byteOffset % 4 === 0
          ? new Float32Array(r.vector.buffer, r.vector.byteOffset, r.vector.byteLength >> 2)
          : new Float32Array(r.vector.slice().buffer, 0, r.vector.byteLength >> 2);
      const n = Math.min(q.length, v.length);
      let dot = 0;
      let vn2 = 0;
      for (let i = 0; i < n; i++) {
        const x = q[i];
        const y = v[i];
        dot += x * y;
        vn2 += y * y;
      }
      const denom = qn * Math.sqrt(vn2);
      const score = denom > 0 ? dot / denom : 0;
      if (score >= minScore) {
        scored.push({ id: r.id, text: r.text, scope: r.scope, created_at: r.created_at, superseded_by: r.superseded_by, canonical: r.canonical, score });
      }
    }
    return this.finalize(scored, limit, opts);
  }

  /** Shared post-processing: boosts, cross-scope text dedupe, sort, top-K. */
  private finalize(scored: ScoredNote[], limit: number, opts?: SearchOptions): ScoredNote[] {
    const weight = opts?.recencyWeight ?? 0;
    if (weight > 0) {
      const now = opts?.now ?? Date.now();
      const halfLife = Math.max(1, opts?.halfLifeMs ?? 30 * 86_400_000);
      for (const s of scored) s.score += weight * Math.pow(0.5, Math.max(0, now - s.created_at) / halfLife);
    }
    const boost = opts?.canonicalBoost ?? 0;
    if (boost > 0) for (const s of scored) if (s.canonical) s.score += boost;
    // Dedupe identical text (e.g. mirrored into both global and project scope)
    // before the limit cut, so duplicates can't crowd out distinct hits.
    let results = scored;
    if (opts?.dedupeByText && scored.length > 1) {
      const best = new Map<string, ScoredNote>();
      for (const s of scored) {
        const key = s.text.trim().toLowerCase();
        const prev = best.get(key);
        if (!prev || s.score > prev.score) best.set(key, s);
      }
      results = [...best.values()];
    }
    results.sort((a, b) => b.score - a.score || b.created_at - a.created_at);
    return results.slice(0, Math.max(1, limit));
  }

  delete(id: string): void {
    const dims = this.dimsOf([id]);
    this.db.run(`DELETE FROM memo WHERE id = ?`, [id]);
    this.vecDelete(dims);
  }

  /** Delete notes by exact text (optionally within a scope); returns rows removed. */
  deleteByText(text: string, scope?: string): number {
    const t = (text ?? "").trim();
    if (!t) return 0;
    const ids = scope
      ? (this.db.query(`SELECT id, dims FROM memo WHERE text = ? AND scope = ?`).all(t, scope) as { id: string; dims: number }[])
      : (this.db.query(`SELECT id, dims FROM memo WHERE text = ?`).all(t) as { id: string; dims: number }[]);
    const res = scope
      ? this.db.run(`DELETE FROM memo WHERE text = ? AND scope = ?`, [t, scope])
      : this.db.run(`DELETE FROM memo WHERE text = ?`, [t]);
    this.vecDelete(ids);
    return res.changes;
  }

  /**
   * Delete only tombstoned (superseded) rows for exact text. Used when
   * resurrecting a re-added fact: a concurrent process may have inserted a live
   * note for the same text during the embed await, and a blanket deleteByText
   * would clobber it.
   */
  deleteTombstonesByText(text: string, scope?: string): number {
    const t = (text ?? "").trim();
    if (!t) return 0;
    const ids = scope
      ? (this.db.query(`SELECT id, dims FROM memo WHERE text = ? AND scope = ? AND superseded_by IS NOT NULL`).all(t, scope) as { id: string; dims: number }[])
      : (this.db.query(`SELECT id, dims FROM memo WHERE text = ? AND superseded_by IS NOT NULL`).all(t) as { id: string; dims: number }[]);
    const res = scope
      ? this.db.run(`DELETE FROM memo WHERE text = ? AND scope = ? AND superseded_by IS NOT NULL`, [t, scope])
      : this.db.run(`DELETE FROM memo WHERE text = ? AND superseded_by IS NOT NULL`, [t]);
    this.vecDelete(ids);
    return res.changes;
  }

  // ─── Dream support: tombstones + canonical flags ───

  /** Soft-delete: mark `id` as superseded by `byId`. */
  supersede(id: string, byId: string): number {
    const dims = this.dimsOf([id]);
    const changes = this.db.run(`UPDATE memo SET superseded_by = ? WHERE id = ?`, [byId, id]).changes;
    this.vecDelete(dims); // superseded notes leave recall
    return changes;
  }

  clearSuperseded(id: string): number {
    const changes = this.db.run(`UPDATE memo SET superseded_by = NULL WHERE id = ?`, [id]).changes;
    if (this.annEnabled) this.reindexOne(id);
    return changes;
  }

  setCanonical(id: string, flag: boolean): number {
    return this.db.run(`UPDATE memo SET canonical = ? WHERE id = ?`, [flag ? 1 : 0, id]).changes;
  }

  clearCanonical(scope?: string): number {
    const res = scope
      ? this.db.run(`UPDATE memo SET canonical = 0 WHERE scope = ? AND canonical <> 0`, [scope])
      : this.db.run(`UPDATE memo SET canonical = 0 WHERE canonical <> 0`);
    return res.changes;
  }

  /** Hard-delete tombstoned notes (GC); returns rows removed. */
  deleteSuperseded(scope?: string): number {
    const ids = scope
      ? (this.db.query(`SELECT id, dims FROM memo WHERE superseded_by IS NOT NULL AND scope = ?`).all(scope) as { id: string; dims: number }[])
      : (this.db.query(`SELECT id, dims FROM memo WHERE superseded_by IS NOT NULL`).all() as { id: string; dims: number }[]);
    const res = scope
      ? this.db.run(`DELETE FROM memo WHERE superseded_by IS NOT NULL AND scope = ?`, [scope])
      : this.db.run(`DELETE FROM memo WHERE superseded_by IS NOT NULL`);
    this.vecDelete(ids);
    return res.changes;
  }

  /** dims for a set of ids (for the vec index). */
  private dimsOf(ids: string[]): { id: string; dims: number }[] {
    if (!ids.length) return [];
    return this.db
      .query(`SELECT id, dims FROM memo WHERE id IN (${ids.map(() => "?").join(",")})`)
      .all(...ids) as { id: string; dims: number }[];
  }

  /** (Re)insert a single active note into the vec index. */
  private reindexOne(id: string): void {
    const r = this.db.query(`SELECT vector, dims, scope FROM memo WHERE id = ? AND superseded_by IS NULL`).get(id) as
      | { vector: Uint8Array; dims: number; scope: string }
      | null;
    if (!r) return;
    const v = r.vector.byteOffset % 4 === 0
      ? new Float32Array(r.vector.buffer, r.vector.byteOffset, r.vector.byteLength >> 2)
      : new Float32Array(r.vector.slice().buffer, 0, r.vector.byteLength >> 2);
    this.vecInsert(id, v, r.scope, r.dims);
  }

  close(): void {
    this.db.close();
  }
}
