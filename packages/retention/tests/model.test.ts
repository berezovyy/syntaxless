import { expect, test } from "bun:test";

import { compile, dispose, parse, preload } from "../src/index.js";
import type { CompileOptions } from "../src/index.js";

const reference = "keep daily backups for a week and monthly backups for a year";

test(
  "the bundled model parses the reference " + "request and previews it only with context",
  async () => {
    const result = await parse(reference);

    expect(result).toMatchObject({
      ok: true,
      source: "model",
      intent: {
        tiers: [
          { bucket: "day", window: { kind: "calendar", unit: "day", count: 7 } },
          { bucket: "month", window: { kind: "calendar", unit: "month", count: 12 } },
        ],
      },
    });

    if (!result.ok) throw new Error("reference did not parse");

    expect(compile(result.intent, {} as CompileOptions)).toMatchObject({
      supported: false,
      code: "missing-context",
    });

    expect(
      compile(result.intent, {
        timeZone: "UTC",
        referenceInstant: Date.parse("2024-03-15T12:00:00Z"),
        backups: [
          { id: "current", timestamp: Date.parse("2024-03-15T00:00:00Z") },
          { id: "old", timestamp: Date.parse("2022-01-01T00:00:00Z") },
        ],
      }),
    ).toEqual({
      supported: true,
      keep: [{ id: "current", tierIndexes: [0, 1] }],
      remove: ["old"],
    });
  },
);

test(
  "questions, uncertain wording and multiple representatives refuse, " + "with a positive control",
  async () => {
    expect((await parse("keep daily backups for a week")).ok).toBe(true);

    for (const text of [
      "keep daily backups for a week?",
      "maybe keep daily backups for a week",
      "keep two backups per day for a week",
      "",
    ]) {
      expect((await parse(text)).ok).toBe(false);
    }
  },
);

test("incompatible or malformed custom weights refuse instead of throwing", async () => {
  const encode = (value: unknown) => new TextEncoder().encode(JSON.stringify(value));

  for (const weights of [
    encode({ version: 99 }),
    encode([]),
    new TextEncoder().encode("{not json"),
  ]) {
    expect(await parse(reference, { weights })).toMatchObject({
      ok: false,
      code: "incompatible-model",
    });
  }
});

test("dispose clears the cache and a later preload restores parsing", async () => {
  await preload();
  dispose();
  await preload();
  expect((await parse(reference)).ok).toBe(true);
});
