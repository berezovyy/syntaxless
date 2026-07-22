/** Calibrate intent refusal on validation only, then freeze the whole artifact. */
import { createHash } from "node:crypto";
import { cpSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";

import { parse } from "../../src/parser/parse.js";
import type { ParseResult } from "../../src/parser/parse.js";
import { evaluateRecords, readCorpus } from "./evaluate.js";
import type { CorpusRecord } from "./generate.js";

export const sha256 = (bytes: string | Uint8Array): string =>
  createHash("sha256").update(bytes).digest("hex");

function atFloor(result: ParseResult, floor: number): ParseResult {
  return result.ok && result.confidence < floor
    ? {
        ok: false,
        code: "low-confidence",
        reason: "below calibrated floor",
        confidence: result.confidence,
        diagnostics: [],
        source: "model",
      }
    : result;
}

export async function selectFloor(
  rows: readonly CorpusRecord[],
  predictions: readonly ParseResult[],
) {
  if (!rows.length || rows.length !== predictions.length) {
    throw new Error("calibration rows and predictions must match");
  }

  if (
    rows.some(
      (row) =>
        row.split !== "validation" ||
        (row as { trainingEligible?: boolean }).trainingEligible === false,
    )
  ) {
    throw new Error("calibration requires training-eligible validation rows only");
  }

  const thresholds = [
    ...new Set([
      0,
      0.5,
      1,
      ...predictions.filter((result) => result.ok).map((result) => result.confidence),
    ]),
  ].sort((a, b) => a - b);

  let best: { floor: number; report: Awaited<ReturnType<typeof evaluateRecords>> } | undefined;

  for (const floor of thresholds) {
    const report = await evaluateRecords(rows, async (_, index) =>
      atFloor(predictions[index]!, floor),
    );

    if (
      report.wrongAll > 0.005 ||
      report.wrongAccepted > 0.005 ||
      report.negativeFalseAcceptance > 0.01
    ) {
      continue;
    }

    // Keep the established 0.5 floor when validation supplies no evidence for a
    // change. Raising it to the smallest observed positive margin overfits a
    // finite sample and gratuitously refuses unseen, otherwise correct inputs.
    if (
      !best ||
      report.positiveExact > best.report.positiveExact ||
      (report.positiveExact === best.report.positiveExact &&
        Math.abs(floor - 0.5) < Math.abs(best.floor - 0.5))
    ) {
      best = { floor, report };
    }
  }

  if (!best?.report.syntheticGatePassed) {
    throw new Error("no validation confidence floor passes the declared quality gate");
  }

  return best;
}

export async function freezeCandidate(corpus: string, candidate: string, output: string) {
  const rows = readCorpus(corpus).filter((row) => row.split === "validation");
  const original = readFileSync(join(candidate, "weights.json"));
  const artifact = JSON.parse(original.toString());
  const weights = new TextEncoder().encode(JSON.stringify({ ...artifact, confidenceFloor: 0 }));
  const predictions: ParseResult[] = [];

  for (const row of rows) predictions.push(await parse(row.text, { source: "model", weights }));
  const { floor, report } = await selectFloor(rows, predictions);
  const frozen = JSON.stringify({ ...artifact, confidenceFloor: floor }, null, 2) + "\n";

  mkdirSync(output); // A frozen candidate is immutable; don't overwrite a previous freeze.
  cpSync(join(candidate, "parity"), join(output, "parity"), { recursive: true });
  writeFileSync(join(output, "weights.json"), frozen);
  const header = `/** Frozen with validation-only calibration. SHA-256: ${sha256(frozen)} */\n`;
  const source = `${header}export const weights = ${frozen.trim()} as const;\n`;

  writeFileSync(join(output, "weights.gen.ts"), source);

  const manifest = {
    artifactFileSha256: sha256(frozen),
    sourceFileSha256: sha256(source),
    originalArtifactFileSha256: sha256(original),
    corpusSha256: sha256(readFileSync(corpus)),
    calibration: {
      split: "validation",
      rows: rows.length,
      confidenceFloor: floor,
      objective:
        "maximum exact intent under wrong-meaning and false-acceptance limits; " +
        "minimum deviation from the established 0.5 floor breaks ties",
      report,
    },
    acceptanceUsed: false,
    createdAt: new Date().toISOString(),
  };

  writeFileSync(join(output, "freeze.json"), JSON.stringify(manifest, null, 2) + "\n");

  return {
    output,
    confidenceFloor: floor,
    artifactFileSha256: manifest.artifactFileSha256,
  };
}

if (import.meta.main) {
  const [corpus, candidate, output] = process.argv.slice(2);

  if (!corpus || !candidate || !output) {
    throw new Error("usage: freeze <corpus> <candidate> <new-output-directory>");
  }

  console.log(await freezeCandidate(resolve(corpus), resolve(candidate), resolve(output)));
}
