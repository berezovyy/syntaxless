import { expect, test } from "bun:test";

import { LABEL_COUNT } from "../../src/model/labels";
import { tokenize } from "../../src/model/tokenizer";
import { loadTestEngine } from "../tests/helpers/candidate-engine";
import { conformalThreshold, fitEnergyThreshold } from "./calibrate-energy";

test("conformalThreshold is the FA quantile of negatives", () => {
  expect(conformalThreshold([])).toBeNull();
  expect(conformalThreshold([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.1)).toBe(2);
});

test("missing pooled states do not shift positive and negative calibration labels", async () => {
  const states = [undefined, [3, 5], [7, 9]];
  let index = 0;

  const fit = await fitEnergyThreshold(
    {
      tokenize,
      inferTokens: () => {
        const state = states[index++];

        return {
          labels: new Uint8Array(),
          ...(state === undefined ? {} : { pooled: Float32Array.from(state) }),
        };
      },
    },
    [
      { text: "missing pooled state", negative: true },
      { text: "positive example", negative: false },
      { text: "negative example", negative: true },
    ],
  );

  expect(fit.negatives).toBe(2);
  expect(fit.positives).toBe(1);
  expect(fit.mahalanobisMean).toEqual([3, 5]);
  expect(fit.mahalanobisVariance).toEqual([1, 1]);
  expect(fit.mahalanobisThreshold).toBeCloseTo(Math.sqrt(32), 6);
});

test(
  "calibration derives its threshold from the candidate logits",
  async () => {
    const engine = await loadTestEngine();

    try {
      const fit = await fitEnergyThreshold(engine, [
        { text: "I like pizza and hiking", negative: true },
        { text: "the sky is blue today", negative: true },
        { text: "every monday", negative: false },
        { text: "remind me every day at 9am", negative: false },
      ]);

      expect(fit.negatives).toBe(2);
      expect(fit.positives).toBe(2);
      expect(fit.predictedRoles).toBe(true);
      expect(fit.mahalanobisMean.length).toBeGreaterThan(1);
      expect(fit.mahalanobisMean.every((value) => value === 0)).toBe(false);
      // The fixture predicts one role with logit 10 and all other roles with
      // logit 0. This independent formula detects a constant/incorrect energy.
      expect(fit.energyThreshold).toBeCloseTo(-Math.log(Math.exp(10) + LABEL_COUNT - 1), 6);
    } finally {
      await engine.dispose();
    }
  },
  { timeout: 60_000 },
);
