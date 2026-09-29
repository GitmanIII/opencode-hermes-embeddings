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

provider.shutdown();
await fs.rm(TMP, { recursive: true, force: true });
console.log(`\n${passed} passed, ${failed} failed`);
if (failed > 0) process.exit(1);
