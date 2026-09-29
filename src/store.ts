/**
 * Vector store: SQLite (bun:sqlite) with float32 vectors as BLOBs and
 * brute-force cosine search. No native extension needed — for the scale of a
 * personal memory store (thousands of notes) this is sub-millisecond and has
 * zero external dependencies.
 */
import { Database } from "bun:sqlite";
import { cosine } from "./embedder.ts";

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
};

export class VectorStore {
  private db: Database;

  constructor(path: string) {
    this.db = new Database(path);
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
    if (!cols.has("superseded_by")) this.db.run(`ALTER TABLE memo ADD COLUMN superseded_by TEXT`);
    if (!cols.has("canonical")) this.db.run(`ALTER TABLE memo ADD COLUMN canonical INTEGER NOT NULL DEFAULT 0`);
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

  /** Notes visible in the given scopes (e.g. ["global", "project:foo"]). */
  inScopes(scopes: string[], includeSuperseded = false): { note: StoredNote; vector: number[] }[] {
    const unique = [...new Set(scopes.filter(Boolean))];
    const where = unique.length ? `scope IN (${unique.map(() => "?").join(",")})` : `1=1`;
    const filter = includeSuperseded ? "" : ` AND superseded_by IS NULL`;
    const sql = `SELECT id, text, vector, scope, created_at, superseded_by, canonical FROM memo WHERE ${where}${filter}`;
    const rows = (
      unique.length ? this.db.query(sql).all(...unique) : this.db.query(sql).all()
    ) as { id: string; text: string; vector: Uint8Array; scope: string; created_at: number; superseded_by: string | null; canonical: number }[];
    return rows.map((r) => ({
      note: { id: r.id, text: r.text, scope: r.scope, created_at: r.created_at, superseded_by: r.superseded_by, canonical: r.canonical },
      vector: toFloat32(r.vector),
    }));
  }

  findByText(text: string, scope: string): string | null {
    const row = this.db.query(`SELECT id FROM memo WHERE text = ? AND scope = ? LIMIT 1`).get(text, scope) as { id: string } | null;
    return row?.id ?? null;
  }

  search(queryVector: number[], scopes: string[], limit: number, minScore: number, opts?: SearchOptions): ScoredNote[] {
    const scored: ScoredNote[] = [];
    for (const { note, vector } of this.inScopes(scopes, opts?.includeSuperseded)) {
      const score = cosine(queryVector, vector);
      if (score >= minScore) scored.push({ ...note, score });
    }
    const weight = opts?.recencyWeight ?? 0;
    if (weight > 0) {
      const now = opts?.now ?? Date.now();
      const halfLife = Math.max(1, opts?.halfLifeMs ?? 30 * 86_400_000);
      for (const s of scored) s.score += weight * Math.pow(0.5, Math.max(0, now - s.created_at) / halfLife);
    }
    const boost = opts?.canonicalBoost ?? 0;
    if (boost > 0) for (const s of scored) if (s.canonical) s.score += boost;
    scored.sort((a, b) => b.score - a.score || b.created_at - a.created_at);
    return scored.slice(0, Math.max(1, limit));
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

function toFloat32(blob: Uint8Array): number[] {
  const f = new Float32Array(blob.buffer, blob.byteOffset, Math.floor(blob.byteLength / 4));
  return Array.from(f);
}
