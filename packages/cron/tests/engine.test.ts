import { afterEach, expect, test } from "bun:test";

import type { WeightsArtifact } from "../src/model/decode";
import { dispose, getEngine } from "../src/model/engine";
import { createInference } from "../src/model/inference";
import { LABEL_COUNT, labelId, LABELS } from "../src/model/labels";
import { tokenize } from "../src/model/tokenizer";
import { weights } from "../src/model/weights.gen";
import { parse } from "../src/parser/parse";

function bytes(artifact: unknown): Uint8Array {
  return new TextEncoder().encode(JSON.stringify(artifact));
}

// Keep the real encoder; replace only the role head with a constant classifier.
function roleWeights(role: number): Uint8Array {
  const weight = "f".repeat(LABEL_COUNT * weights.hidden);
  const bias = LABELS.map((_, id) => (id === role ? "p" : "f")).join("");

  return bytes({
    ...weights,
    roleClasses: LABEL_COUNT,
    labels: LABELS,
    q: weights.q + weight + bias,
    segments: [
      ...weights.segments.filter((segment) => !segment.name.startsWith("role_head.")),
      {
        name: "role_head.weight",
        offset: weights.q.length,
        length: weight.length,
        shape: [LABEL_COUNT, weights.hidden],
        scale: 1,
      },
      {
        name: "role_head.bias",
        offset: weights.q.length + weight.length,
        length: bias.length,
        shape: [LABEL_COUNT],
        scale: 1,
      },
    ],
  });
}

afterEach(dispose);

test("typed vocabularies cannot fall back to legacy parsing when their metadata differs", () => {
  expect(() => createInference({ ...weights, labels: LABELS })).toThrow("typed role vocabulary");

  const current = JSON.parse(
    new TextDecoder().decode(roleWeights(labelId.CONTEXT_NUMBER)),
  ) as WeightsArtifact;

  expect(() => createInference({ ...current, labels: weights.labels })).toThrow(
    "typed role vocabulary",
  );

  expect(() => createInference({ ...weights, labels: [...weights.labels].reverse() })).toThrow(
    "typed role vocabulary",
  );
});

test("custom engines run their own weights without changing other engines", async () => {
  const tokens = tokenize("every weekday at 9");
  const bundled = await getEngine();
  const baseline = bundled.inferTokens(tokens, true);

  const [monday, friday] = await Promise.all([
    getEngine({ weights: roleWeights(labelId.WEEKDAY_MON) }),
    getEngine({ weights: roleWeights(labelId.WEEKDAY_FRI) }),
  ]);

  expect(Array.from(monday.inferTokens(tokens).labels)).toEqual(
    tokens.map(() => labelId.WEEKDAY_MON),
  );

  expect(Array.from(friday.inferTokens(tokens).labels)).toEqual(
    tokens.map(() => labelId.WEEKDAY_FRI),
  );

  expect(bundled.inferTokens(tokens, true)).toEqual(baseline);

  expect(Array.from(monday.inferTokens(tokens).labels)).toEqual(
    tokens.map(() => labelId.WEEKDAY_MON),
  );
});

test("public parse uses the selected artifact's calibration", async () => {
  const text = "every weekday at 9am";
  const reject = bytes({ ...weights, energyThreshold: -1e9 });
  const accept = bytes({ ...weights, energyThreshold: 1e9 });

  expect(await parse(text, { weights: reject })).toMatchObject({
    ok: false,
    code: "unsafe-context",
    diagnostics: [expect.objectContaining({ code: "calibrated-abstain" })],
  });

  expect(await parse(text, { weights: accept })).toMatchObject({
    ok: true,
    intent: {
      frequency: { unit: "week", interval: 1 },
      weekdays: ["monday", "tuesday", "wednesday", "thursday", "friday"],
      time: { hour: 9, minute: 0 },
    },
  });

  expect(await parse(text)).toMatchObject({ ok: true });
});

test("public parse reports the calibration criterion that actually rejected", async () => {
  const custom = bytes({
    ...weights,
    energyThreshold: 1e9,
    mahalanobisMean: new Array(weights.hidden).fill(0),
    mahalanobisVariance: new Array(weights.hidden).fill(1),
    mahalanobisThreshold: 0,
  });

  const result = await parse("every weekday at 9am", { weights: custom });

  expect(result).toMatchObject({
    ok: false,
    code: "unsafe-context",
    reason: expect.stringContaining("Mahalanobis distance"),
    diagnostics: [
      expect.objectContaining({
        code: "calibrated-abstain",
        message: "Mahalanobis distance exceeded the conformal threshold",
      }),
    ],
  });
});

test("tensor dimensions must match the model before an engine is returned", async () => {
  for (const name of ["role_head.bias", "role_head.weight", "depthwise.weight"]) {
    const custom = bytes({
      ...weights,
      segments: weights.segments.map((segment) => {
        if (segment.name !== name) return segment;

        if (name === "role_head.bias") {
          return {
            ...segment,
            length: weights.roleClasses - 1,
            shape: [weights.roleClasses - 1],
          };
        }

        // Equal element counts still describe different operators.
        return { ...segment, shape: [...segment.shape].reverse() };
      }),
    });

    await expect(getEngine({ weights: custom })).rejects.toThrow(name);
  }
});

test("optional heads must be complete and agree with their metadata", async () => {
  const orphan = bytes({
    ...weights,
    segments: weights.segments.filter((segment) => segment.name !== "actionable_head.weight"),
  });

  await expect(getEngine({ weights: orphan })).rejects.toThrow("actionable_head.bias");
  const unadvertised = bytes({ ...weights, scheduleKindClasses: 0 });

  await expect(getEngine({ weights: unadvertised })).rejects.toThrow("schedule_kind_head");
});

test("calibration vectors must describe the complete hidden state", async () => {
  for (const [field, value] of [
    ["mahalanobisMean", []],
    ["mahalanobisMean", new Array(weights.hidden).fill(null)],
    ["mahalanobisVariance", [1]],
    ["mahalanobisVariance", new Array(weights.hidden).fill(-1)],
    ["mahalanobisVariance", new Array(weights.hidden).fill("1")],
  ] as const) {
    await expect(getEngine({ weights: bytes({ ...weights, [field]: value }) })).rejects.toThrow(
      field,
    );
  }
});

test("unsupported artifacts reject before inference and a failed initialization can retry", async () => {
  const custom = bytes({ version: 2 });
  const first = getEngine({ weights: custom });

  await expect(first).rejects.toMatchObject({
    name: "unsupported-weights-version",
    expected: 1,
    actual: 2,
  });

  const retry = getEngine({ weights: custom });

  expect(retry).not.toBe(first);
  await expect(retry).rejects.toHaveProperty("name", "unsupported-weights-version");
});
