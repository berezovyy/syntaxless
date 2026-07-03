/**
 * Unit tests for the promotion gate's pure core.
 *
 * The engine-backed run is exercised by the `gate` script; these tests pin the
 * decision arithmetic so a rejected candidate can never be argued into an
 * acceptance by moving a threshold.
 */
import { describe, expect, test } from "bun:test";

import { assertSameArtifact, decide, familyGuards, proportionGuard } from "./gate";

describe("proportionGuard", () => {
  test("accepts a regression inside the tolerance band", () => {
    const guard = proportionGuard({ total: 100, correct: 89 }, { total: 100, correct: 90 });

    expect(guard.passed).toBe(true);
    expect(guard.reason).toBeNull();
    expect(guard.support).toBe(100);
    expect(guard.delta).toBeCloseTo(0.01, 10);
    expect(guard.tolerance).toBeGreaterThan(guard.delta);
  });

  test("rejects a regression beyond the tolerance band", () => {
    const guard = proportionGuard({ total: 100, correct: 50 }, { total: 100, correct: 90 });

    expect(guard.passed).toBe(false);
    expect(guard.reason).toBe("regression");
    expect(guard.delta).toBeCloseTo(0.4, 10);
    expect(guard.tolerance).toBeLessThan(guard.delta);
  });

  test("rejects a support mismatch", () => {
    const guard = proportionGuard({ total: 100, correct: 80 }, { total: 90, correct: 80 });

    expect(guard.passed).toBe(false);
    expect(guard.reason).toBe("support mismatch");
  });

  test("rejects support below thirty", () => {
    const guard = proportionGuard({ total: 20, correct: 18 }, { total: 20, correct: 19 });

    expect(guard.passed).toBe(false);
    expect(guard.reason).toBe("insufficient support");
    expect(guard.support).toBe(20);
  });
});

describe("familyGuards", () => {
  test("flags a family missing from one side as a support mismatch", () => {
    const guards = familyGuards(
      { terse: { total: 40, correct: 30 } },
      { terse: { total: 40, correct: 30 }, bounds: { total: 40, correct: 20 } },
    );

    const bounds = guards.find((guard) => guard.family === "bounds");

    expect(bounds?.passed).toBe(false);
    expect(bounds?.reason).toBe("support mismatch");
  });
});

describe("decide", () => {
  test("rejects equal counts with no strict improvement", () => {
    const decision = decide(
      { total: 100, correct: 80, byFamily: { terse: { total: 40, correct: 30 } } },
      { total: 100, correct: 80, byFamily: { terse: { total: 40, correct: 30 } } },
    );

    expect(decision.accepted).toBe(false);
    expect(decision.improvement).toBe(0);
    expect(decision.failures).toContain("reserved-carrier exact sequences did not improve");
  });

  test("accepts a strict improvement when every guard passes", () => {
    const decision = decide(
      { total: 100, correct: 85, byFamily: { terse: { total: 40, correct: 32 } } },
      { total: 100, correct: 80, byFamily: { terse: { total: 40, correct: 30 } } },
    );

    expect(decision.accepted).toBe(true);
    expect(decision.improvement).toBeCloseTo(0.05, 10);
    expect(decision.failures).toEqual([]);
  });

  test("rejects a candidate that regresses a family", () => {
    const decision = decide(
      { total: 100, correct: 90, byFamily: { terse: { total: 40, correct: 10 } } },
      { total: 100, correct: 80, byFamily: { terse: { total: 40, correct: 40 } } },
    );

    expect(decision.accepted).toBe(false);
    expect(decision.failures.some((failure) => failure.startsWith("terse:"))).toBe(true);
  });

  test("rejects relative improvement that remains below an absolute floor", () => {
    const decision = decide(
      { total: 100, correct: 30, byFamily: { terse: { total: 40, correct: 12 } } },
      { total: 100, correct: 20, byFamily: { terse: { total: 40, correct: 10 } } },
    );

    expect(decision.accepted).toBe(false);
    expect(decision.failures).toContain("overall accuracy below 0.5");
  });
});

describe("assertSameArtifact", () => {
  test("returns the shared digest when every report agrees", () => {
    expect(assertSameArtifact([{ artifactSha256: "abc" }, { artifactSha256: "abc" }])).toBe("abc");
  });

  test("throws when reports carry mixed artifactSha256", () => {
    expect(() =>
      assertSameArtifact([{ artifactSha256: "abc" }, { artifactSha256: "def" }]),
    ).toThrow(/artifact mismatch/);
  });
});
