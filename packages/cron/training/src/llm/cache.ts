/**
 * Content-addressed response cache for opencode corpus runs.
 *
 * The model is not deterministic, so a run is only replayable if each response
 * is pinned to the exact request that produced it. The key covers the model,
 * the provider, a hash of the fully rendered prompt, and the prompt's schema
 * version; the value is the raw model response. Nothing else is stored, and the
 * cache directory (`data/llm/cache/`) is untracked.
 */
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

export interface CacheKeyInput {
  /** Model id (e.g. `claude-sonnet-4-5`). */
  model: string;
  /** Provider id (e.g. `anthropic`). Never a key or token. */
  provider: string;
  /** `sha256` of the fully rendered prompt. */
  promptHash: string;
  /** Prompt front-matter `schemaVersion`. */
  schemaVersion: number;
}

/**
 * Key a response by `sha256(model + provider + promptHash + schemaVersion)`.
 * Fields are NUL-separated so `("ab", "c")` cannot collide with `("a", "bc")`.
 */
export function cacheKey(input: CacheKeyInput): string {
  return createHash("sha256")
    .update([input.model, input.provider, input.promptHash, String(input.schemaVersion)].join("\0"))
    .digest("hex");
}

/** Default cache directory, resolved from this module, not the working dir. */
export function defaultCacheDir(): string {
  return join(dirname(fileURLToPath(import.meta.url)), "..", "..", "data", "llm", "cache");
}

export function cachePath(key: string, dir = defaultCacheDir()): string {
  return join(dir, `${key}.json`);
}

/** Read a cached response, or `undefined` when the entry is absent or corrupt. */
export function readCache(key: string, dir = defaultCacheDir()): string | undefined {
  const path = cachePath(key, dir);

  if (!existsSync(path)) return undefined;
  const parsed = JSON.parse(readFileSync(path, "utf8")) as unknown;

  if (typeof parsed !== "object" || parsed === null) return undefined;
  const response = (parsed as { response?: unknown }).response;

  return typeof response === "string" ? response : undefined;
}

/** Persist a raw response under its content key. Returns the written path. */
export function writeCache(key: string, response: string, dir = defaultCacheDir()): string {
  const path = cachePath(key, dir);

  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, `${JSON.stringify({ response }, null, 2)}\n`);

  return path;
}
