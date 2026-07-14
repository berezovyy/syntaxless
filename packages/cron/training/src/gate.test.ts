import { describe, expect, test } from "bun:test";

import {
  assertSameArtifact,
  decide,
  decidePromotion,
  familyGuards,
  parseArgs,
  proportionGuard,
} from "./gate";
import type { GateScores, PromotionMetrics } from "./gate";

describe("proportionGuard", () => {
  test("accepts a regression inside the tolerance band", () => {
    const guard = proportionGuard({ total: 100, correct: 89 }, { total: 100, correct: 90 });

    expect(guard.passed).toBe(true);
    expect(guard.reason).toBeNull();
    expect(guard.support).toBe(100);
    expect(guard.delta).toBeCloseTo(0.01, 10);
    // 1.96 * sqrt(((12/102)*(90/102) + (11/102)*(91/102)) / 100).
    expect(guard.tolerance).toBeCloseTo(0.08765807712468944, 10);
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

  test("requires at least thirty samples per family", () => {
    const guard = proportionGuard({ total: 29, correct: 29 }, { total: 29, correct: 29 });

    expect(guard.passed).toBe(false);
    expect(guard.reason).toBe("insufficient support");
    expect(guard.support).toBe(29);

    expect(proportionGuard({ total: 30, correct: 30 }, { total: 30, correct: 30 }).passed).toBe(
      true,
    );
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

function scores(correct = 98, overrides: Partial<GateScores> = {}): GateScores {
  return {
    total: 100,
    correct,
    byFamily: {
      terse: { total: 40, correct: 39 },
      interval: { total: 60, correct: correct - 39 },
    },
    selective: {
      silentMeaningChangeRate: 0,
      negativeFalseAcceptanceRate: 0,
      typoInvariance: 1,
    },
    ...overrides,
  };
}

const ledger: PromotionMetrics = {
  H: "98/100",
  EXEC: "98/100",
  "FA@cov": "0@0.9",
  O: "100/100",
};

const baseline = { overlayOn: scores(96), overlayOff: scores(96) };

function promote(candidate: GateScores, metrics: Partial<PromotionMetrics> = ledger) {
  return decidePromotion({ overlayOn: candidate, overlayOff: candidate }, baseline, metrics);
}

test("a strictly better candidate with complete evidence is promoted", () => {
  const decision = promote(scores());

  expect(decision.accepted).toBe(true);
  expect(decision.failures).toEqual([]);
  expect(decision.improvement).toBeCloseTo(0.02, 10);

  expect(decision.guards.map(({ family, passed }) => ({ family, passed }))).toEqual([
    { family: "interval", passed: true },
    { family: "terse", passed: true },
  ]);
});

test("an improvement exactly at the overall floor is accepted", () => {
  expect(decide(scores(95), scores(94))).toMatchObject({
    accepted: true,
    failures: [],
  });
});

test("overall improvement cannot hide a statistically significant family regression", () => {
  const before = scores(18800, {
    total: 20000,
    byFamily: {
      terse: { total: 10000, correct: 9700 },
      interval: { total: 10000, correct: 9100 },
    },
  });

  const candidate = scores(19000, {
    total: 20000,
    byFamily: {
      terse: { total: 10000, correct: 9000 },
      interval: { total: 10000, correct: 10000 },
    },
  });

  // Both absolute floors pass; the family regression alone blocks promotion.
  expect(decide(candidate, before)).toMatchObject({
    accepted: false,
    failures: ["terse: regression", "bare expressions: regression"],
  });
});

test.each([
  {
    name: "equal score",
    candidate: scores(96),
    reason: "reserved-carrier exact sequences did not improve",
  },
  {
    name: "different support",
    candidate: scores(98, { total: 101 }),
    reason: "evaluation support mismatch",
  },
  {
    name: "below overall floor",
    candidate: scores(94),
    reason: "overall accuracy below 0.95",
  },
])("decide rejects $name", ({ candidate, reason }) => {
  // A lower baseline isolates the absolute floor from strict improvement.
  const before = reason.startsWith("overall") ? scores(93) : scores(96);
  const decision = decide(candidate, before);

  expect(decision.accepted).toBe(false);
  expect(decision.failures).toEqual([reason]);
});

test.each([
  {
    name: "silent changes",
    field: "silentMeaningChangeRate",
    boundary: 0.005,
    failing: 0.00501,
    reason: "silent meaning-change rate above 0.005",
  },
  {
    name: "false accepts",
    field: "negativeFalseAcceptanceRate",
    boundary: 0.01,
    failing: 0.01001,
    reason: "negative false-acceptance rate above 0.01",
  },
  {
    name: "typo invariance",
    field: "typoInvariance",
    boundary: 0.95,
    failing: 0.94999,
    reason: "typo invariance below 0.95",
  },
] as const)("promotion boundary: $name", ({ field, boundary, failing, reason }) => {
  for (const [value, accepted] of [
    [boundary, true],
    [failing, false],
  ] as const) {
    const candidate = scores();

    candidate.selective = { ...candidate.selective!, [field]: value };

    expect(promote(candidate)).toMatchObject({
      accepted,
      failures: accepted ? [] : [`overlay-off ${reason}`],
    });
  }
});

test("overall improvement cannot hide a family below its absolute floor", () => {
  const before = scores(970, {
    total: 1000,
    byFamily: {
      terse: { total: 100, correct: 90 },
      interval: { total: 900, correct: 880 },
    },
  });

  const candidate = scores(980, {
    total: 1000,
    byFamily: {
      terse: { total: 100, correct: 89 },
      interval: { total: 900, correct: 891 },
    },
  });

  expect(decide(candidate, before)).toMatchObject({
    accepted: false,
    failures: ["terse: accuracy below 0.9"],
  });

  candidate.byFamily.terse!.correct = 90;
  candidate.byFamily.interval!.correct = 890;
  expect(decide(candidate, before).accepted).toBe(true);
});

test.each(["H", "EXEC", "FA@cov", "O"] as const)(
  "missing or blank %s cannot promote an otherwise passing candidate",
  (key) => {
    for (const value of [undefined, "", "  "]) {
      const metrics = { ...ledger } as Partial<PromotionMetrics>;

      if (value === undefined) delete metrics[key];
      else metrics[key] = value;

      expect(promote(scores(), metrics)).toMatchObject({
        accepted: false,
        failures: [`gate refused: missing ${key}`],
      });
    }
  },
);

test("an empty ledger is not an implied pass", () => {
  expect(promote(scores(), {})).toMatchObject({
    accepted: false,
    failures: ["gate refused: missing H, EXEC, FA@cov, O"],
  });
});

test("reserved-carrier improvement alone is insufficient evidence", () => {
  expect(promote(scores(), { ...ledger, H: "1/100", EXEC: "1/100" })).toMatchObject({
    accepted: false,
    failures: ["G2-only improvement is not a promotion pass"],
  });
});

describe("parseArgs promotion metrics", () => {
  test("reads H/EXEC/FA@cov/O from the gate CLI", () => {
    const options = parseArgs([
      "--h",
      "10/20",
      "--exec",
      "3/10",
      "--fa-at-cov",
      "0.01@0.9",
      "--o",
      "10/10",
    ]);

    expect(options.metrics).toEqual({
      H: "10/20",
      EXEC: "3/10",
      "FA@cov": "0.01@0.9",
      O: "10/10",
    });
  });
});

test("perfect overlay-on cannot rescue a failing overlay-off candidate", () => {
  const decision = decidePromotion(
    { overlayOn: scores(100), overlayOff: scores(94) },
    { overlayOn: scores(93), overlayOff: scores(93) },
    ledger,
  );

  expect(decision).toMatchObject({
    accepted: false,
    failures: ["overlay-off overall accuracy below 0.95"],
  });
});

test.each([
  {
    field: "silentMeaningChangeRate",
    value: 0.001,
    reason: "overlay-off silent meaning-change worse than overlay-on",
  },
  {
    field: "negativeFalseAcceptanceRate",
    value: 0.001,
    reason: "overlay-off negative false-accept worse than overlay-on",
  },
] as const)(
  "overlay-off cannot worsen $field even within the absolute limit",
  ({ field, value, reason }) => {
    const overlayOff = scores();

    overlayOff.selective = { ...overlayOff.selective!, [field]: value };
    const decision = decidePromotion({ overlayOn: scores(), overlayOff }, baseline, ledger);

    expect(decision).toMatchObject({ accepted: false, failures: [reason] });
  },
);

test("missing overlay-off fails despite passing overlay-on and complete evidence", () => {
  expect(decidePromotion({ overlayOn: scores() }, baseline, ledger)).toMatchObject({
    accepted: false,
    failures: ["missing overlay-off"],
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
