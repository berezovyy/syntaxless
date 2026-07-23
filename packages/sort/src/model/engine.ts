/** Lazy model initialization. Only this module owns the public engine cache. */
import type { Predictions } from "./inference.js";
import type { RawToken } from "./tokenizer.js";

export interface InferenceEngine {
  readonly confidenceFloor: number;
  inferTokens(tokens: readonly RawToken[], debug?: boolean): Predictions;
  inferRows(rows: Uint16Array | readonly number[], count: number, debug?: boolean): Predictions;
}

export interface EngineConfig {
  /** UTF-8 JSON artifact bytes. Treat the buffer as immutable after first use. */
  weights?: Uint8Array;
}

let bundled: Promise<InferenceEngine> | undefined;
let overrides = new WeakMap<Uint8Array, Promise<InferenceEngine>>();

async function createEngine(bytes?: Uint8Array): Promise<InferenceEngine> {
  const { assertFormatVersion, decodeWeightsBytes } = await import("./assets.js");

  const artifact =
    bytes === undefined ? (await import("./weights.gen.js")).weights : decodeWeightsBytes(bytes);

  if (artifact === null) throw new TypeError("no promoted sort model is available");
  assertFormatVersion(artifact);
  const { createInference } = await import("./inference.js");

  return createInference(artifact);
}

/**
 * Share initialization for the bundled model or the same caller-owned buffer.
 * Snapshot custom bytes before awaiting imports; a failed load can be retried.
 * Different buffers own different engines, without content hashes or collisions.
 */
export function getEngine(config?: EngineConfig): Promise<InferenceEngine> {
  const bytes = config?.weights;

  if (bytes === undefined) {
    if (bundled !== undefined) return bundled;
    const pending = createEngine();

    bundled = pending;

    pending.catch(() => {
      if (bundled === pending) bundled = undefined;
    });

    return pending;
  }

  const cached = overrides.get(bytes);

  if (cached !== undefined) return cached;
  const pending = createEngine(new Uint8Array(bytes));
  const cache = overrides;

  cache.set(bytes, pending);

  pending.catch(() => {
    if (cache.get(bytes) === pending) cache.delete(bytes);
  });

  return pending;
}

/** Prepare the same engine used by parsing. */
export async function preload(config?: EngineConfig): Promise<void> {
  await getEngine(config);
}

/** Release cached engines. References held by callers remain usable. */
export function dispose(): void {
  bundled = undefined;
  overrides = new WeakMap();
}
