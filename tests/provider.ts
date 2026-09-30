/**
 * opencode-hermes-embeddings — provider harness (hermetic).
 * Uses an injectable FakeEmbedder (hashed bag-of-words) so we test add/search/
 * prefetch/scope/forget/mirror without a live TEI server.
 */
import * as fs from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";
import { cosine, type Embedder, type EmbedKind } from "../src/embedder.ts";
import { EmbeddingsMemoryProvider } from "../src/provider.ts";
import { VectorStore } from "../src/store.ts";

let passed = 0;
let failed = 0;
const assert = (name: string, cond: boolean, detail = "") => {
  if (cond) {
    passed++;
    console.log(`✅ ${name}`);
  } else {
    failed++;
    console.log(`❌ ${name} ${detail}`);
  }
};

// Deterministic hashed bag-of-words embedder (dims=64), normalized.
const DIMS = 64;
class FakeEmbedder implements Embedder {
  dims() {
    return DIMS;
  }
  async embed(texts: string[], _kind: EmbedKind): Promise<number[][]> {
    return texts.map((t) => {
      const v = new Array(DIMS).fill(0);
      for (const tok of t.toLowerCase().split(/[^a-z0-9]+/).filter((x) => x.length >= 2)) {
        let h = 0;
        for (const c of tok) h = (h * 31 + c.charCodeAt(0)) >>> 0;
        v[h % DIMS] += 1;
      }
      const norm = Math.sqrt(v.reduce((a, b) => a + b * b, 0)) || 1;
      return v.map((x) => x / norm);
    });
  }
}

const TMP = await fs.mkdtemp(path.join(os.tmpdir(), "ohe-emb-"));

// cosine sanity
assert("cosine of identical vectors is 1", Math.abs(cosine([1, 2, 3], [1, 2, 3]) - 1) < 1e-9);
assert("cosine of orthogonal vectors is 0", Math.abs(cosine([1, 0], [0, 1])) < 1e-9);

// recency ranking: a slightly-less-relevant but newer note can outrank an older one
const rstore = new VectorStore(path.join(TMP, "recency.sqlite"));
rstore.add("old", "old note", [1, 0.01], "global", Date.now() - 10 * 86_400_000);
rstore.add("new", "new note", [0.9, 0.1], "global", Date.now());
const byScore = rstore.search([1, 0], ["global"], 5, 0);
const byRank = rstore.search([1, 0], ["global"], 5, 0, { recencyWeight: 0.05, halfLifeMs: 86_400_000 });
assert("without recency the higher-cosine note wins", byScore[0].id === "old", JSON.stringify(byScore.map((h) => h.id)));
assert("with recency the newer note wins", byRank[0].id === "new", JSON.stringify(byRank.map((h) => h.id)));
rstore.close();

// dims guard: vectors from a different model/quantization are skipped, not
// scored against a shorter-longer dot product (silent garbage recall).
const dstore = new VectorStore(path.join(TMP, "dims.sqlite"));
dstore.add("d4", "old model note", [1, 0, 0, 0], "global");
dstore.add("d2", "current model note", [1, 0], "global");
const dimHits = dstore.search([1, 0], ["global"], 5, -1);
assert("search ignores notes with mismatched vector dims", dimHits.length === 1 && dimHits[0].id === "d2", JSON.stringify(dimHits.map((h) => h.id)));
dstore.close();

// Search streams row-by-row (no per-row Array.from): a planted nearest note wins at volume.
const scale = new VectorStore(path.join(TMP, "scale.sqlite"));
const SDIM = 32;
const tvec = new Array(SDIM).fill(0);
tvec[0] = 1;
for (let i = 0; i < 20_000; i++) scale.add(`s${i}`, `note ${i}`, Array.from({ length: SDIM }, () => Math.random()), "global");
scale.add("target", "planted target note", tvec, "global", Date.now() + 1000);
const top = scale.search(tvec, ["global"], 1, -1)[0];
assert("search streams at 20k notes and finds the nearest", top?.id === "target", JSON.stringify(top && { id: top.id, score: top.score }));
scale.close();

const provider = new EmbeddingsMemoryProvider({ minScore: 0.01, topK: 5 }, new FakeEmbedder());
await provider.initialize({ memoryRoot: TMP, prefetchLimit: 5, projectId: "projA" });
assert("provider name", provider.name === "embeddings");
assert("systemPromptBlock mentions embeddings", provider.systemPromptBlock().includes("embeddings"));

// add + search within a project
const a = await provider.add("Alpha beta gamma detector notes");
await provider.add("Completely unrelated postgres backup procedure");
const hits = await provider.search("alpha beta detector", 5);
assert("add returns an id", a.id.startsWith("em_"));
assert("search ranks the relevant note first", hits.length >= 1 && hits[0].text.includes("Alpha beta gamma"), JSON.stringify(hits.map((h) => h.score.toFixed(2))));

// idempotent add + cross-scope dedupe
const a2 = await provider.add("Alpha beta gamma detector notes");
assert("add is idempotent for same text+scope", a2.id === a.id && (await provider.search("alpha beta detector", 20)).filter((h) => h.text.includes("Alpha beta gamma")).length === 1, JSON.stringify(a2));
await provider.add("shared duplicate across scopes marker");
await provider.onMemoryWrite("add", "shared duplicate across scopes marker"); // global copy
const dup = await provider.search("shared duplicate across scopes marker", 20);
assert("search dedupes identical text across scopes", dup.filter((h) => h.text === "shared duplicate across scopes marker").length === 1, JSON.stringify(dup.map((h) => h.text)));

const pf = await provider.prefetch("alpha beta detector");
assert("prefetch returns a provider-memory block with the note", pf.hits >= 1 && pf.text.includes("<provider-memory") && pf.text.includes("Alpha beta gamma"));

// scoping: project A notes invisible in project B
await provider.add("projA-only secret token zzzz");
provider.setProject("projB");
const cross = await provider.search("projA-only secret zzzz", 5);
assert("project A notes are NOT visible in project B", !cross.some((h) => h.text.includes("projA-only")), JSON.stringify(cross.map((h) => h.text)));

// global mirror is visible everywhere
await provider.onMemoryWrite("add", "global shared fact about caches");
const g = await provider.search("global shared fact caches", 5);
assert("global mirrored note visible in project B", g.some((h) => h.text.includes("global shared fact")));

// Dedupe happens before the limit cut: a text mirrored into two scopes must not
// consume two of the top-K slots and crowd out a distinct hit.
const dupLimit = await provider.search("shared duplicate across scopes marker", 2);
assert(
  "dedupe happens before the limit cut",
  dupLimit.length === 2 && dupLimit.filter((h) => h.text === "shared duplicate across scopes marker").length === 1,
  JSON.stringify(dupLimit.map((h) => h.text)),
);

// mirror dedupe + replace/remove propagation + demote keeps
await provider.onMemoryWrite("add", "global shared fact about caches");
assert("mirror dedupes identical global text", (await provider.search("global shared fact caches", 20)).filter((h) => h.text === "global shared fact about caches").length === 1);
await provider.onMemoryWrite("add", "global fact that will be replaced");
await provider.onMemoryWrite("replace", "global fact about cache invalidation", "global fact that will be replaced");
assert("mirror replace deletes the old global text", !(await provider.search("global fact that will be replaced", 20)).some((h) => h.text === "global fact that will be replaced"));
assert("mirror replace adds the new global text", (await provider.search("global fact cache invalidation", 20)).some((h) => h.text === "global fact about cache invalidation"));
await provider.onMemoryWrite("add", "global fact that will be demoted");
await provider.onMemoryWrite("demote", "global fact that will be demoted");
assert("mirror demote keeps the global text", (await provider.search("global fact demoted", 20)).some((h) => h.text === "global fact that will be demoted"));
await provider.onMemoryWrite("remove", "global shared fact about caches");
assert("mirror remove deletes the global text", !(await provider.search("global shared fact caches", 20)).some((h) => h.text.includes("global shared fact")));

// forget
const target = (await provider.search("global fact demoted", 20)).find((h) => h.text === "global fact that will be demoted")!.id;
provider.forget(target);
assert("forget removes the note", !(await provider.search("global fact demoted", 20)).some((h) => h.id === target));

// scopes isolated at the file level: store path exists
assert("store file created", await fs.access(path.join(TMP, "embeddings.sqlite")).then(() => true).catch(() => false));

// ── Dream: reconcile the store against canonical facts ──
// Each provider gets its own store file so these cases don't interfere.
let dreamN = 0;
const mkProvider = async (opts: Record<string, unknown> = {}) => {
  const p = new EmbeddingsMemoryProvider(
    { minScore: 0.01, topK: 5, dbPath: path.join(TMP, `dream-${++dreamN}.sqlite`), ...opts },
    new FakeEmbedder(),
  );
  await p.initialize({ memoryRoot: TMP, prefetchLimit: 5, projectId: "projDream" });
  return p;
};

// deterministic near-duplicate collapse
const pd = await mkProvider({ duplicateThreshold: 0.6, ambiguousThreshold: 0.5 });
await pd.onMemoryWrite("add", "detector alpha beta gamma");
await pd.onMemoryWrite("add", "unrelated postgres backup procedure");
const dStats = await pd.reconcile(["detector alpha beta delta"]);
assert("dream supersedes a near-duplicate note", dStats.superseded === 1 && dStats.added === 1 && dStats.canonical === 1, JSON.stringify(dStats));
const afterCanon = await pd.search("detector alpha beta", 10);
assert("dream hides the superseded note", !afterCanon.some((h) => h.text === "detector alpha beta gamma"), JSON.stringify(afterCanon.map((h) => h.text)));
assert("dream keeps the canonical note", afterCanon.some((h) => h.text === "detector alpha beta delta"));
assert("dream leaves distinct notes alone", (await pd.search("unrelated postgres backup", 5)).some((h) => h.text === "unrelated postgres backup procedure"));
// Re-adding a fact whose only row is a dream tombstone must resurrect it, not
// silently resolve to the hidden note.
await pd.onMemoryWrite("add", "detector alpha beta gamma");
assert("re-adding a superseded note resurrects it", (await pd.search("detector alpha beta gamma", 10)).some((h) => h.text === "detector alpha beta gamma"));
pd.shutdown();

// the judge resolves the ambiguous band
const pj = await mkProvider({ duplicateThreshold: 0.99, ambiguousThreshold: 0.5 });
await pj.onMemoryWrite("add", "detector alpha beta gamma");
let judgeCalls = 0;
const jStats = await pj.reconcile(["detector alpha beta delta"], { judge: async () => { judgeCalls++; return true; } });
assert("dream consults the judge in the ambiguous band", judgeCalls === 1 && jStats.judged === 1, JSON.stringify(jStats));
assert("dream applies the judge's verdict", jStats.superseded === 1 && !(await pj.search("detector alpha beta gamma", 10)).some((h) => h.text === "detector alpha beta gamma"));
pj.shutdown();

// the judge can keep a distinct fact
const pk = await mkProvider({ duplicateThreshold: 0.99, ambiguousThreshold: 0.5 });
await pk.onMemoryWrite("add", "detector alpha beta gamma");
const kStats = await pk.reconcile(["detector alpha beta delta"], { judge: async () => false });
assert("dream keeps a note the judge says is distinct", kStats.superseded === 0 && (await pk.search("detector alpha beta gamma", 10)).some((h) => h.text === "detector alpha beta gamma"));
pk.shutdown();

// exact canonical is not duplicated; hardDelete GCs tombstones
const pg = await mkProvider({ duplicateThreshold: 0.6 });
await pg.onMemoryWrite("add", "detector alpha beta gamma");
await pg.reconcile(["detector alpha beta delta"]); // supersedes the near-duplicate
const gStats = await pg.reconcile(["detector alpha beta delta"], { hardDelete: true });
assert("dream does not duplicate an exact canonical fact", gStats.added === 0 && gStats.canonical === 1, JSON.stringify(gStats));
assert("dream GC removes tombstones", gStats.removed >= 1, JSON.stringify(gStats));
pg.shutdown();

// multiple canonical facts in one reconcile are embedded in a single batched
// request (2 new + 1 exact match); the exact match is not re-added.
const pm = await mkProvider({ duplicateThreshold: 0.99 });
await pm.onMemoryWrite("add", "gamma delta epsilon unique");
const mStats = await pm.reconcile(["alpha beta one", "gamma delta epsilon unique", "zeta eta two"]);
assert(
  "dream handles multiple canonical facts in one batched pass",
  mStats.canonical === 3 && mStats.added === 2 && mStats.superseded === 0,
  JSON.stringify(mStats),
);
pm.shutdown();

provider.shutdown();
await fs.rm(TMP, { recursive: true, force: true });
console.log(`\n${passed} passed, ${failed} failed`);
if (failed > 0) process.exit(1);
