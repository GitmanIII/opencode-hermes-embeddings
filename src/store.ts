/**
 * Vector store: SQLite (bun:sqlite) with float32 vectors as BLOBs and
 * brute-force cosine search. No native extension needed — for the scale of a
 * personal memory store (thousands of notes) this is sub-millisecond and has
 * zero external dependencies.
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

export class VectorStore {
  private db: Database;

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

  add(id: string, text: string, vector: number[], scope: string, createdAt = Date.now()): void {
    const buf = Buffer.from(new Float32Array(vector).buffer);
    this.db.run(`INSERT OR REPLACE INTO memo (id, text, vector, dims, scope, created_at) VALUES (?, ?, ?, ?, ?, ?)`, [
      id,
      text,
      buf,
      vector.length,
      scope,
      createdAt,
    ]);
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
   * Brute-force cosine search, streamed row-by-row with zero-copy float32 views
   * (no per-row `Array.from`). Memory stays flat and throughput is ~6x a
   * materialize-then-map approach at personal scale (see tests/perf notes).
   */
  search(queryVector: number[], scopes: string[], limit: number, minScore: number, opts?: SearchOptions): ScoredNote[] {
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
    const rows = stmt.iterate(...unique, q.length) as IterableIterator<{
      id: string;
      text: string;
      scope: string;
      created_at: number;
      superseded_by: string | null;
      canonical: number;
      vector: Uint8Array;
    }>;

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
    this.db.run(`DELETE FROM memo WHERE id = ?`, [id]);
  }

  /** Delete notes by exact text (optionally within a scope); returns rows removed. */
  deleteByText(text: string, scope?: string): number {
    const t = (text ?? "").trim();
    if (!t) return 0;
    const res = scope
      ? this.db.run(`DELETE FROM memo WHERE text = ? AND scope = ?`, [t, scope])
      : this.db.run(`DELETE FROM memo WHERE text = ?`, [t]);
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
    const res = scope
      ? this.db.run(`DELETE FROM memo WHERE text = ? AND scope = ? AND superseded_by IS NOT NULL`, [t, scope])
      : this.db.run(`DELETE FROM memo WHERE text = ? AND superseded_by IS NOT NULL`, [t]);
    return res.changes;
  }

  // ─── Dream support: tombstones + canonical flags ───

  /** Soft-delete: mark `id` as superseded by `byId`. */
  supersede(id: string, byId: string): number {
    return this.db.run(`UPDATE memo SET superseded_by = ? WHERE id = ?`, [byId, id]).changes;
  }

  clearSuperseded(id: string): number {
    return this.db.run(`UPDATE memo SET superseded_by = NULL WHERE id = ?`, [id]).changes;
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
    const res = scope
      ? this.db.run(`DELETE FROM memo WHERE superseded_by IS NOT NULL AND scope = ?`, [scope])
      : this.db.run(`DELETE FROM memo WHERE superseded_by IS NOT NULL`);
    return res.changes;
  }

  close(): void {
    this.db.close();
  }
}
