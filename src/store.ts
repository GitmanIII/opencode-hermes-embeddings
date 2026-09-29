/**
 * Vector store: SQLite (bun:sqlite) with float32 vectors as BLOBs and
 * brute-force cosine search. No native extension needed — for the scale of a
 * personal memory store (thousands of notes) this is sub-millisecond and has
 * zero external dependencies.
 */
import { Database } from "bun:sqlite";
import { cosine } from "./embedder.ts";

export type StoredNote = { id: string; text: string; scope: string; created_at: number };
export type ScoredNote = StoredNote & { score: number };
/** Optional ranking adjustment blending cosine with a recency term. */
export type RankOptions = { recencyWeight?: number; halfLifeMs?: number; now?: number };

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
         created_at INTEGER NOT NULL
       );`,
    );
    this.db.run(`CREATE INDEX IF NOT EXISTS memo_scope_idx ON memo (scope);`);
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
  inScopes(scopes: string[]): { note: StoredNote; vector: number[] }[] {
    const unique = [...new Set(scopes.filter(Boolean))];
    const rows = (
      unique.length
        ? this.db.query(`SELECT id, text, vector, scope, created_at FROM memo WHERE scope IN (${unique.map(() => "?").join(",")})`).all(...unique)
        : this.db.query(`SELECT id, text, vector, scope, created_at FROM memo`).all()
    ) as { id: string; text: string; vector: Uint8Array; scope: string; created_at: number }[];
    return rows.map((r) => ({
      note: { id: r.id, text: r.text, scope: r.scope, created_at: r.created_at },
      vector: toFloat32(r.vector),
    }));
  }

  findByText(text: string, scope: string): string | null {
    const row = this.db.query(`SELECT id FROM memo WHERE text = ? AND scope = ? LIMIT 1`).get(text, scope) as { id: string } | null;
    return row?.id ?? null;
  }

  search(queryVector: number[], scopes: string[], limit: number, minScore: number, rank?: RankOptions): ScoredNote[] {
    const scored: ScoredNote[] = [];
    for (const { note, vector } of this.inScopes(scopes)) {
      const score = cosine(queryVector, vector);
      if (score >= minScore) scored.push({ ...note, score });
    }
    const weight = rank?.recencyWeight ?? 0;
    if (weight > 0) {
      const now = rank?.now ?? Date.now();
      const halfLife = Math.max(1, rank?.halfLifeMs ?? 30 * 86_400_000);
      for (const s of scored) s.score += weight * Math.pow(0.5, Math.max(0, now - s.created_at) / halfLife);
    }
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

  close(): void {
    this.db.close();
  }
}

function toFloat32(blob: Uint8Array): number[] {
  const f = new Float32Array(blob.buffer, blob.byteOffset, Math.floor(blob.byteLength / 4));
  return Array.from(f);
}
