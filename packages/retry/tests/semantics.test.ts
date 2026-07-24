import { expect, test } from "bun:test";

import { compile } from "../src/compile.js";
import type { RetryIntent } from "../src/types.js";

const policy: RetryIntent = {
  retries: 3,
  strategy: "exponential",
  initialDelayMs: 1000,
  multiplier: 2,
  defaults: [],
};

function delays(intent: RetryIntent, seed?: number): readonly number[] {
  const result = compile(intent, seed === undefined ? {} : { seed });

  if (!result.supported) throw new Error(result.reason);

  return result.delaysMs;
}

test(
  "retry count excludes the initial attempt, " +
    "and exact exponential results round once before cap",
  () => {
    expect(delays(policy)).toEqual([1000, 2000, 4000]);

    expect(compile(Object.assign(Object.create(policy), { defaults: [] }))).toMatchObject({
      supported: false,
      code: "invalid-intent",
    });

    expect(delays({ ...policy, retries: 0 })).toEqual([]);

    expect(delays({ ...policy, retries: 5, maxDelayMs: 3000 })).toEqual([
      1000, 2000, 3000, 3000, 3000,
    ]);

    expect(delays({ ...policy, maxDelayMs: 500 })).toEqual([500, 500, 500]);
    expect(delays({ ...policy, maxDelayMs: 0 })).toEqual([0, 0, 0]);

    expect(delays({ ...policy, initialDelayMs: 1, multiplier: 1.5, retries: 5 })).toEqual([
      1, 2, 3, 4, 6,
    ]);

    const constant: RetryIntent = {
      retries: 4,
      strategy: "constant",
      initialDelayMs: 250,
      defaults: [],
    };

    expect(delays(constant)).toEqual([250, 250, 250, 250]);

    // This exact decimal ratio is greater than 1, even near floating-point boundaries.
    expect(
      delays({
        ...policy,
        initialDelayMs: 1,
        multiplier: 1.0000000000000002,
        retries: 3,
      }),
    ).toEqual([1, 2, 2]);

    expect(
      compile({ ...policy, initialDelayMs: Number.MAX_SAFE_INTEGER, retries: 2 }),
    ).toMatchObject({ supported: false, code: "overflow" });

    expect(
      delays({
        ...policy,
        initialDelayMs: Number.MAX_SAFE_INTEGER,
        retries: 1000,
        maxDelayMs: 100,
      }),
    ).toHaveLength(1000);
  },
);

test("seeded jitter is reproducible and stays inside exact bounds and the final cap", () => {
  for (const jitter of [
    { kind: "full" },
    { kind: "equal" },
    { kind: "proportional", fraction: 0.25 },
  ] as const) {
    const intent = { ...policy, retries: 6, maxDelayMs: 3500, jitter };

    const a = delays(intent, 42);
    const b = delays(intent, 42);
    const different = delays(intent, 17);

    expect(a).toEqual(b);
    expect(a).not.toEqual(different);
    const base = delays({ ...policy, retries: 6, maxDelayMs: 3500 });

    for (const [i, delay] of a.entries()) {
      const d = base[i]!;

      const low =
        jitter.kind === "full"
          ? 0
          : jitter.kind === "equal"
            ? Math.ceil(d / 2)
            : Math.ceil(d * 0.75);

      const high = jitter.kind === "proportional" ? Math.min(3500, Math.floor(d * 1.25)) : d;

      expect(Number.isSafeInteger(delay) && delay >= low && delay <= high).toBe(true);
    }
  }

  expect(
    delays(
      {
        ...policy,
        initialDelayMs: 1,
        jitter: { kind: "proportional", fraction: 1e-7 },
      },
      0,
    ),
  ).toEqual([1, 2, 4]);

  expect(compile({ ...policy, jitter: { kind: "full" } })).toMatchObject({
    supported: false,
    code: "missing-seed",
  });

  expect(compile({ ...policy, jitter: { kind: "full" } }, { seed: -1 })).toMatchObject({
    supported: false,
    code: "invalid-input",
  });

  expect(
    compile(
      {
        ...policy,
        retries: 1,
        initialDelayMs: Number.MAX_SAFE_INTEGER,
        jitter: { kind: "proportional", fraction: 1 },
      },
      { seed: 1 },
    ),
  ).toMatchObject({ supported: false, code: "overflow" });
});
