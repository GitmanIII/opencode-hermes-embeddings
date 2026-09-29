export { TeiEmbedder, cosine, NOMIC_DOC_PREFIX, NOMIC_QUERY_PREFIX } from "./embedder.ts";
export type { Embedder, EmbedKind, TeiConfig } from "./embedder.ts";
export { VectorStore } from "./store.ts";
export { EmbeddingsMemoryProvider, createProvider } from "./provider.ts";
export type { EmbeddingsOptions, ProviderContext, ProviderHit } from "./provider.ts";
export { createProvider as default } from "./provider.ts";
