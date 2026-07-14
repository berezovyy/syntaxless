import { expect, test } from "bun:test";

import { inferTokens } from "../model/inference";
import { labelId } from "../model/labels";
import { tokenize } from "../model/tokenizer";
import { logSumExp, rejectOnEnergy, sequenceEnergy, tokenEnergy } from "./abstain";
import {
  energyFromPredictions,
  parseWithModel,
  rejectWithCalibratedEnergy,
  speechActFromSpanLabels,
} from "./model";

test("energy is low for a peaked role distribution and high for a uniform one", () => {
  const peaked = [10, 0, 0, 0];
  const uniform = [0, 0, 0, 0];

  expect(tokenEnergy(peaked)).toBeLessThan(tokenEnergy(uniform));
  expect(logSumExp(peaked)).toBeGreaterThan(logSumExp(uniform) - 1e-6);
});

test("rejectOnEnergy uses the calibrated threshold, not a speech-act id", () => {
  expect(rejectOnEnergy(0.1, 0.5)).toBe(false);
  expect(rejectOnEnergy(0.9, 0.5)).toBe(true);
  expect(rejectOnEnergy(0.9, undefined)).toBe(false);
});

test("energyFromPredictions reads tagger logits from inferTokens", () => {
  const request = inferTokens(tokenize("every monday"), true);
  const negative = inferTokens(tokenize("I like pizza and hiking"), true);

  expect(request.logits).toBeDefined();
  expect(negative.logits).toBeDefined();
  expect(request.logits!.length).toBeGreaterThan(0);
  const requestEnergy = energyFromPredictions(request);
  const negativeEnergy = energyFromPredictions(negative);

  expect(Number.isFinite(requestEnergy)).toBe(true);
  expect(Number.isFinite(negativeEnergy)).toBe(true);
  expect(requestEnergy).not.toBe(negativeEnergy);
});

test("parseWithModel rejects a real-negative on energy and does not energy-reject a request", async () => {
  const negativeText = "I like pizza and hiking";
  const requestText = "every monday";
  const negativeEnergy = energyFromPredictions(inferTokens(tokenize(negativeText), true));
  const requestEnergy = energyFromPredictions(inferTokens(tokenize(requestText), true));

  const negative = await parseWithModel(negativeText, {
    energyThreshold: negativeEnergy - 1,
  });

  expect(negative.ok).toBe(false);

  if (negative.ok === false) {
    expect(negative.code).toBe("unsafe-context");
    expect(negative.diagnostics[0]?.code).toBe("calibrated-abstain");
  }

  const request = await parseWithModel(requestText, {
    energyThreshold: requestEnergy + 1,
  });

  expect(request.diagnostics.some((item) => item.code === "calibrated-abstain")).toBe(false);
  expect(rejectWithCalibratedEnergy(negativeEnergy, negativeEnergy - 1)).toBeDefined();
});

test("speechActFromSpanLabels is request on content, none on filler", () => {
  const tokens = tokenize("every monday").filter((token) => token.kind !== 3);

  const request = tokens.map((token) => ({
    token,
    role: token.text === "monday" ? labelId.WEEKDAY_MON : labelId.RECUR,
    boundary: token.text === "every",
    score: 1,
    actionable: true,
  }));

  expect(speechActFromSpanLabels(request)).toBe("request");

  const filler = tokens.map((token) => ({
    token,
    role: labelId.O,
    boundary: false,
    score: 1,
    actionable: false,
  }));

  expect(speechActFromSpanLabels(filler)).toBe("none");
});

test("sequenceEnergy averages content tokens and ignores O", () => {
  const logits = [
    [0, 0, 0, 0],
    [8, -2, -2, -2],
  ];

  const roles = [labelId.O, labelId.WEEKDAY_MON];
  const energy = sequenceEnergy(logits, roles);

  expect(energy).toBeCloseTo(tokenEnergy(logits[1]!), 5);
});
