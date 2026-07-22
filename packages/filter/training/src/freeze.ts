import { copyFileSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import { parse } from "../../src/parse.js";
import type { ParseResult } from "../../src/parse.js";
import { hash, readRows } from "./corpus.js";
import { evaluate } from "./evaluate.js";

export async function freeze(corpus: string, candidate: string, destination: string) {
  const rows = readRows(corpus).filter((row) => row.split === "validation");

  if (rows.some((row) => row.trainingEligible === false)) {
    throw new Error("reserved rows cannot calibrate");
  }

  const original = readFileSync(join(candidate, "weights.json"));
  const artifact = JSON.parse(original.toString());

  const zero = Buffer.from(JSON.stringify({ ...artifact, confidenceFloor: 0 }));
  const predictions: ParseResult[] = [];

  for (const row of rows) {
    predictions.push(await parse(row.text, { ...row.context, weights: zero }));
  }

  const floors = [
    ...new Set([
      0,
      0.5,
      1,
      ...predictions.filter((p) => p.ok).map((p) => (p.ok ? p.confidence : 0)),
    ]),
  ];

  let selected: { floor: number; report: Awaited<ReturnType<typeof evaluate>> } | undefined;

  for (const floor of floors) {
    const report = await evaluate(rows, async (_, i) => {
      const p = predictions[i]!;

      return p.ok && p.confidence < floor
        ? { ok: false, code: "low-confidence", reason: "below floor", source: "model" }
        : p;
    });

    if (
      report.syntheticGatePassed &&
      (!selected ||
        report.positiveExact > selected.report.positiveExact ||
        (report.positiveExact === selected.report.positiveExact &&
          Math.abs(floor - 0.5) < Math.abs(selected.floor - 0.5)))
    ) {
      selected = { floor, report };
    }
  }

  if (!selected) throw new Error("validation calibration cannot satisfy the semantic gate");
  artifact.confidenceFloor = selected.floor;

  const payload = JSON.stringify(artifact) + "\n";

  const module =
    "/* Frozen filter candidate; validation-only calibration. */\nexport const weights = " +
    JSON.stringify(artifact) +
    " as const;\n";

  mkdirSync(destination);
  mkdirSync(join(destination, "parity"));
  writeFileSync(join(destination, "weights.json"), payload);
  writeFileSync(join(destination, "weights.gen.ts"), module);

  for (const name of readdirSync(join(candidate, "parity"))) {
    copyFileSync(join(candidate, "parity", name), join(destination, "parity", name));
  }

  const report = {
    artifactFileSha256: hash(payload),
    sourceFileSha256: hash(module),
    originalArtifactFileSha256: hash(original),
    corpusSha256: hash(readFileSync(corpus)),
    createdAt: new Date().toISOString(),
    calibration: {
      confidenceFloor: selected.floor,
      split: "validation",
      report: selected.report,
    },
    acceptanceUsed: false,
  };

  writeFileSync(join(destination, "freeze.json"), JSON.stringify(report, null, 2) + "\n");

  return report;
}
