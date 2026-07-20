import { writeFileSync } from "node:fs";
import { join } from "node:path";

import { labelId } from "../../../src/model/labels";
import { loadEngine } from "../../src/evaluate";
import { tempDir } from "./temp-dir";
import { createTestWeights } from "./test-weights";

/** Load test weights in an isolated directory owned by the test runner. */
export async function loadTestEngine(role = labelId.WEEKDAY_MON, fixedRequest = false) {
  const weights = createTestWeights(role, fixedRequest);
  const directory = tempDir("syntaxless-candidate-");
  const path = join(directory, "weights.gen.ts");

  writeFileSync(path, `export const weights = ${JSON.stringify(weights)};\n`);

  return loadEngine(join(directory, "no-dist"), path);
}
