/**
 * Embedder abstraction + HTTP client for HuggingFace text-embeddings-inference.
 *
 * TEI exposes an OpenAI-compatible POST {endpoint}/v1/embeddings and runs the
 * model on the GPU. The embedder is injectable so the provider can be tested
 * without a live server.
 */

export type EmbedKind = "document" | "query";

export interface Embedder {
  dims(): number;
  embed(texts: string[], kind: EmbedKind): Promise<number[][]>;
}

export type TeiConfig = {
  endpoint: string;
  model: string;
  /** Apply the model's retrieval prefixes (e.g. nomic's search_document/search_query). */
  prefixes: boolean;
  docPrefix?: string;
  queryPrefix?: string;
  dims?: number;
  timeoutMs?: number;
};

// Defaults match nomic-embed-text-v1.5, which REQUIRES these prefixes.
export const NOMIC_DOC_PREFIX = "search_document: ";
export const NOMIC_QUERY_PREFIX = "search_query: ";

export class TeiEmbedder implements Embedder {
  private observedDims = 0;

  constructor(private cfg: TeiConfig) {}

  dims(): number {
    return this.cfg.dims ?? this.observedDims;
  }

  private prefix(kind: EmbedKind, text: string): string {
    if (!this.cfg.prefixes) return text;
    const p = kind === "query" ? (this.cfg.queryPrefix ?? NOMIC_QUERY_PREFIX) : (this.cfg.docPrefix ?? NOMIC_DOC_PREFIX);
    return `${p}${text}`;
  }

  async embed(texts: string[], kind: EmbedKind): Promise<number[][]> {
    if (texts.length === 0) return [];
    const input = texts.map((t) => this.prefix(kind, t));
    const url = `${this.cfg.endpoint.replace(/\/+$/, "")}/v1/embeddings`;
    let res: Response;
    try {
      res = await fetch(url, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ input, model: this.cfg.model, encoding_format: "float" }),
        signal: AbortSignal.timeout(this.cfg.timeoutMs ?? 15_000),
      });
    } catch (err) {
      throw new Error(`TEI unreachable at ${url}: ${String(err)}`);
    }
    if (!res.ok) {
      const body = await res.text().catch(() => "");
      throw new Error(`TEI ${res.status} at ${url}: ${body.slice(0, 200)}`);
    }
    const json = (await res.json()) as { data?: { index?: number; embedding: number[] }[] };
    const data = (json.data ?? []).slice().sort((a, b) => (a.index ?? 0) - (b.index ?? 0)).map((d) => d.embedding);
    if (data.length !== texts.length) throw new Error(`TEI returned ${data.length} embeddings for ${texts.length} inputs`);
    if (data[0]) this.observedDims = data[0].length;
    return data;
  }
}

export function cosine(a: number[], b: number[]): number {
  const n = Math.min(a.length, b.length);
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < n; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  if (na === 0 || nb === 0) return 0;
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}
