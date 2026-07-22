import { expect, test } from "bun:test";

import { parse } from "../../src/parser/parse.ts";
import type { ParseResult } from "../../src/parser/parse.ts";
import { featurizeRecord } from "../src/featurize.ts";
import { selectFloor } from "../src/freeze.ts";
import type { CorpusRecord } from "../src/generate.ts";

const rows: CorpusRecord[] = [
  { text: "required email", intent: { type: "email", required: true } },
  { text: "optional number", intent: { type: "number", required: false } },
  { text: "is this required?", intent: null },
].map((row, i) => ({
  ...row,
  id: `calibration-${i}`,
  family: row.intent ? "constraint" : "negative",
  template: row.intent?.type ?? "question",
  split: "validation",
  seed: 1,
  fingerprint: `calibration-${i}`,
  spans: [],
})) as CorpusRecord[];

async function predictions(): Promise<ParseResult[]> {
  return Promise.all(
    rows.map(async (row, i) => ({
      ...(await parse(row.text, { source: "rules" })),
      confidence: i === 0 ? 0.8 : 0.95,
    })),
  );
}

test(
  "confidence calibration preserves exact coverage " + "and cannot hide a confident false accept",
  async () => {
    const results = await predictions();
    const selected = await selectFloor(rows, results);

    expect(selected.floor).toBe(0.5);
    expect(selected.report.positiveExact).toBe(1);
    const unsafe: ParseResult[] = [...results];

    unsafe[2] = { ...results[0]!, confidence: 0.99 };
    await expect(selectFloor(rows, unsafe)).rejects.toThrow("quality gate");
  },
);

test("reserved rows cannot enter calibration or any training featurizer", async () => {
  const reserved = { ...rows[0]!, trainingEligible: false };

  await expect(selectFloor([reserved], [(await predictions())[0]!])).rejects.toThrow(
    "training-eligible validation",
  );

  expect(() => featurizeRecord(reserved, [])).toThrow("evaluation-only");

  await expect(
    selectFloor([{ ...rows[0]!, split: "heldout" }], [(await predictions())[0]!]),
  ).rejects.toThrow("validation rows only");
});
