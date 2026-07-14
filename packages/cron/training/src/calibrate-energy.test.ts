import { expect, test } from "bun:test";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { conformalThreshold, fitEnergyThreshold } from "./calibrate-energy";
import { loadEngine } from "./evaluate";

test("conformalThreshold is the FA quantile of negatives", () => {
  expect(conformalThreshold([])).toBeNull();
  expect(conformalThreshold([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.1)).toBe(2);
});

test(
  "fitEnergyThreshold uses inferTokens logits on a real-negative and a request",
  async () => {
    const engine = await loadEngine(
      "packages/cron/dist",
      "packages/cron/training/data/synth/experiments/a1/weights-seed-20260913.gen.ts",
    );

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
      expect(fit.energyThreshold).not.toBeNull();
      const dir = mkdtempSync(join(tmpdir(), "cron-energy-"));

      writeFileSync(join(dir, "fit.json"), JSON.stringify(fit));
    } finally {
      await engine.dispose();
    }
  },
  { timeout: 60_000 },
);
