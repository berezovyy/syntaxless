/**
 * F.3: conformal energy threshold on predicted-role energy of real negatives
 * vs positives. Uses the shipped inferTokens path, not gold role labels.
 */
import { readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import type { Predictions } from "../../src/model/inference";
import { energyFromPredictions, mahalanobisFromPredictions } from "../../src/parser/model";
import { loadEngine, resolveRepo } from "./evaluate";
import type { Engine } from "./evaluate";

export interface EnergyExample {
  text: string;
  negative: boolean;
}

export interface EnergyFit {
  count: number;
  negatives: number;
  positives: number;
  energyThreshold: number | null;
  mahalanobisMean: number[];
  mahalanobisVariance: number[];
  mahalanobisThreshold: number | null;
  predictedRoles: true;
  mean: number;
  variance: number;
}

export function conformalThreshold(
  negativeScores: readonly number[],
  faTarget = 0.01,
): number | null {
  if (negativeScores.length === 0) return null;
  const ordered = [...negativeScores].sort((left, right) => left - right);
  const index = Math.min(Math.floor(faTarget * ordered.length), ordered.length - 1);

  return ordered[Math.max(index, 0)]!;
}

function asPredictions(value: ReturnType<Engine["inferTokens"]>): Predictions {
  return value as Predictions;
}

export async function fitEnergyThreshold(
  engine: Engine,
  examples: readonly EnergyExample[],
  faTarget = 0.01,
): Promise<EnergyFit> {
  const energies: number[] = [];
  const pooled: number[][] = [];
  const isNeg: boolean[] = [];

  for (const example of examples) {
    const tokens = engine.tokenize(example.text);
    const predictions = asPredictions(engine.inferTokens(tokens));

    energies.push(energyFromPredictions(predictions));
    if (predictions.pooled !== undefined) pooled.push(Array.from(predictions.pooled));
    isNeg.push(example.negative);
  }

  const negEnergy = energies.filter((_, index) => isNeg[index]);
  const posPooled = pooled.filter((_, index) => !isNeg[index]);
  const hidden = posPooled[0]?.length ?? pooled[0]?.length ?? 0;
  const mean = new Array<number>(hidden).fill(0);
  const variance = new Array<number>(hidden).fill(1);

  if (posPooled.length > 0 && hidden > 0) {
    for (const row of posPooled) {
      for (let index = 0; index < hidden; index++) mean[index] = mean[index]! + row[index]!;
    }

    for (let index = 0; index < hidden; index++) mean[index] = mean[index]! / posPooled.length;

    for (const row of posPooled) {
      for (let index = 0; index < hidden; index++) {
        const delta = row[index]! - mean[index]!;

        variance[index] = variance[index]! + delta * delta;
      }
    }

    for (let index = 0; index < hidden; index++) {
      variance[index] = Math.max(variance[index]! / posPooled.length, 1e-8);
    }
  }

  const mahals = pooled.map((row) => {
    const fake = { pooled: Float32Array.from(row) } as Predictions;

    return mahalanobisFromPredictions(fake, mean, variance) ?? 0;
  });

  const negMahal = mahals.filter((_, index) => isNeg[index]);

  const meanEnergy =
    energies.length === 0 ? 0 : energies.reduce((sum, value) => sum + value, 0) / energies.length;

  const varianceEnergy =
    energies.length === 0
      ? 1
      : energies.reduce((sum, value) => sum + (value - meanEnergy) ** 2, 0) / energies.length;

  return {
    count: examples.length,
    negatives: negEnergy.length,
    positives: examples.length - negEnergy.length,
    energyThreshold: conformalThreshold(negEnergy, faTarget),
    mahalanobisMean: mean,
    mahalanobisVariance: variance,
    mahalanobisThreshold: conformalThreshold(negMahal, faTarget),
    predictedRoles: true,
    mean: meanEnergy,
    variance: varianceEnergy,
  };
}

export function loadHumanEnergyExamples(
  dir = resolveRepo("packages/cron/evaluation"),
  partitions: readonly string[] = ["development", "calibration"],
): EnergyExample[] {
  const allowed = new Set(partitions);

  const files = readdirSync(dir).filter(
    (name) => name.startsWith("human-v") && name.endsWith(".jsonl"),
  );

  const examples: EnergyExample[] = [];

  for (const name of files.sort()) {
    const text = readFileSync(join(dir, name), "utf8");

    for (const line of text.split("\n")) {
      if (line.trim().length === 0) continue;

      const record = JSON.parse(line) as {
        partition?: string;
        text?: string;
        intent?: unknown;
      };

      if (!allowed.has(String(record.partition))) continue;
      if (typeof record.text !== "string") continue;

      examples.push({
        text: record.text,
        negative: record.intent === null,
      });
    }
  }

  return examples;
}

if (import.meta.main) {
  const argv = process.argv.slice(2);
  let weights: string | undefined;
  let out: string | undefined;

  for (let index = 0; index < argv.length; index++) {
    const arg = argv[index]!;

    if (arg === "--weights") weights = argv[++index];
    else if (arg === "--out") out = argv[++index];
  }

  const engine = await loadEngine("packages/cron/dist", weights);

  try {
    const examples = loadHumanEnergyExamples();
    const fit = await fitEnergyThreshold(engine, examples);
    const body = `${JSON.stringify(fit, null, 2)}\n`;

    if (out !== undefined) writeFileSync(resolveRepo(out), body);
    process.stdout.write(body);
  } finally {
    await engine.dispose();
  }
}
