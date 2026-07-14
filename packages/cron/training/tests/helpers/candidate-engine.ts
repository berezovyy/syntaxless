import { writeFileSync } from "node:fs";
import { join } from "node:path";

import type { WeightsArtifact } from "../../../src/model/decode";
import { LABEL_COUNT, labelId, LABELS } from "../../../src/model/labels";
import { weights as legacy } from "../../../src/model/weights.gen";
import { loadEngine } from "../../src/evaluate";
import { tempDir } from "./temp-dir";

/**
 * Exercise real candidate loading and inference without a local training run.
 * Reuse the committed encoder with a deterministic typed role head. These are
 * test weights, not a trained candidate or evidence of parsing accuracy.
 */
export async function loadTestEngine() {
  const roleWeight = "f".repeat(LABEL_COUNT * legacy.hidden); // int6 zero

  const roleBias = Array.from({ length: LABEL_COUNT }, (_, id) =>
    id === labelId.WEEKDAY_MON ? "p" : "f",
  ).join("");

  const weights: WeightsArtifact = {
    ...legacy,
    roleClasses: LABEL_COUNT,
    labels: LABELS,
    q: legacy.q + roleWeight + roleBias,
    segments: [
      ...legacy.segments.filter((segment) => !segment.name.startsWith("role_head.")),
      {
        name: "role_head.weight",
        offset: legacy.q.length,
        length: roleWeight.length,
        shape: [LABEL_COUNT, legacy.hidden],
        scale: 1,
      },
      {
        name: "role_head.bias",
        offset: legacy.q.length + roleWeight.length,
        length: roleBias.length,
        shape: [LABEL_COUNT],
        scale: 1,
      },
    ],
  };

  const directory = tempDir("syntaxless-candidate-");
  const path = join(directory, "weights.gen.ts");

  writeFileSync(path, `export const weights = ${JSON.stringify(weights)};\n`);

  return loadEngine(join(directory, "no-dist"), path);
}
