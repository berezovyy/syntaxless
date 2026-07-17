import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";

import type { HumanEvaluationRecord } from "./human-evaluate.js";

const source = readFileSync(
  new URL("../../evaluation/compiler-contract-v2.json", import.meta.url),
  "utf8",
);

const manifest = JSON.parse(source) as {
  version: number;
  corrections: {
    id: string;
    recordSha256: string;
    expected: { supported: false; code: "non-unit-interval" };
  }[];
};

if (manifest.version !== 2) throw new Error("unsupported compiler correction manifest");
const corrections = new Map(manifest.corrections.map((row) => [row.recordSha256, row]));

export const COMPILER_CONTRACT = {
  version: 2,
  correctionsSha256: createHash("sha256").update(source).digest("hex"),
} as const;

/** Correct only exact reviewed historical targets, without rewriting fixture records or hashes. */
export function compileExpectation(
  record: HumanEvaluationRecord,
): HumanEvaluationRecord["expected"] {
  const digest = createHash("sha256").update(JSON.stringify(record)).digest("hex");
  const correction = corrections.get(digest);

  if (correction === undefined) return record.expected;

  if (
    correction.id !== record.id ||
    correction.expected.supported !== false ||
    correction.expected.code !== "non-unit-interval"
  ) {
    throw new Error("invalid compiler correction target");
  }

  return correction.expected;
}
