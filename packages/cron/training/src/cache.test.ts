import { expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { cacheKey, readCache, writeCache } from "./llm/cache";

const BASE = {
  model: "claude-sonnet-4-5",
  provider: "anthropic",
  promptHash: "a".repeat(64),
  schemaVersion: 1,
};

test("identical inputs produce the same key", () => {
  expect(cacheKey(BASE)).toBe(cacheKey({ ...BASE }));
});

test("a changed prompt, model, provider, or schema version changes the key", () => {
  expect(cacheKey({ ...BASE, promptHash: "b".repeat(64) })).not.toBe(cacheKey(BASE));
  expect(cacheKey({ ...BASE, model: "other-model" })).not.toBe(cacheKey(BASE));
  expect(cacheKey({ ...BASE, provider: "other-provider" })).not.toBe(cacheKey(BASE));
  expect(cacheKey({ ...BASE, schemaVersion: 2 })).not.toBe(cacheKey(BASE));
});

test("keys are field-separated, not raw concatenation", () => {
  const left = cacheKey({ ...BASE, model: "ab", provider: "c" });
  const right = cacheKey({ ...BASE, model: "a", provider: "bc" });

  expect(left).not.toBe(right);
});

test("a written entry reads back, and misses read as undefined", () => {
  const dir = mkdtempSync(join(tmpdir(), "corpus-cache-"));

  try {
    const key = cacheKey(BASE);

    expect(readCache(key, dir)).toBeUndefined();
    writeCache(key, '{"text":"x","intent":null,"output":"negative"}\n', dir);
    expect(readCache(key, dir)).toBe('{"text":"x","intent":null,"output":"negative"}\n');
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
