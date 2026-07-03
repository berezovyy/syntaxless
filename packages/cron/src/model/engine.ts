/**
 * Lazy model engine lifecycle.
 *
 * Nothing here runs at import time. The first {@link getEngine} or
 * {@link preload} call dynamically imports `inference.ts` (Phase 16), and only
 * then does the weight artifact decode. Concurrent calls share one in-flight
 * promise per configuration, a rejected initialization clears its cache entry so
 * a later call retries, and {@link dispose} drops every cached engine.
 *
 * No browser globals, no top-level await, no network. Self-hosted bytes are
 * passed in through {@link EngineConfig}; the bundled default works offline.
 */
import type { Predictions } from "./inference.js";
import type { RawToken } from "./tokenizer.js";

/** Engine configuration. `weights` overrides the bundled artifact. */
export interface EngineConfig {
  /** Caller-provided, self-hosted artifact bytes. */
  weights?: Uint8Array;
}

/** The prepared inference engine handed to the parser. */
export interface InferenceEngine {
  inferTokens(tokens: readonly RawToken[], debug?: boolean): Predictions;
  inferRows(rows: Uint16Array | readonly number[], count: number, debug?: boolean): Predictions;
}

type InferenceModule = typeof import("./inference.js");

const engines = new Map<string, Promise<InferenceEngine>>();

function keyOf(config?: EngineConfig): string {
  const bytes = config?.weights;

  if (bytes === undefined) return "default";
  let hash = 2166136261;

  for (let index = 0; index < bytes.length; index++) {
    hash = Math.imul(hash ^ bytes[index]!, 16777619);
  }

  return `weights:${bytes.length}:${(hash >>> 0).toString(16)}`;
}

async function createEngine(config?: EngineConfig): Promise<InferenceEngine> {
  // Resolve (and version-check) the artifact before paying for the model.
  const { resolveWeights } = await import("./assets.js");

  resolveWeights(config?.weights);
  const inference: InferenceModule = await import("./inference.js");

  return {
    inferTokens: (tokens, debug) => inference.inferTokens(tokens, debug),
    inferRows: (rows, count, debug) => inference.inferRows(rows, count, debug),
  };
}

/**
 * Return the engine for `config`, importing the inference module on first use.
 * The same promise is returned for the same configuration; a rejection clears
 * the cache entry so a later call retries.
 */
export function getEngine(config?: EngineConfig): Promise<InferenceEngine> {
  const key = keyOf(config);
  const cached = engines.get(key);

  if (cached !== undefined) return cached;

  const pending = createEngine(config);

  engines.set(key, pending);

  pending.catch(() => {
    if (engines.get(key) === pending) engines.delete(key);
  });

  return pending;
}

/**
 * Prepare the engine for `config`. Resolves only once inference is ready, and
 * shares the default promise with {@link getEngine} and the parser.
 */
export async function preload(config?: EngineConfig): Promise<void> {
  await getEngine(config);
}

/** Drop every cached engine. A later `getEngine` initializes again. */
export function dispose(): void {
  engines.clear();
}
