import { expect, test } from "bun:test";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import type { AcceptedRealRecord } from "./accepted";
import { mixAcceptedIntoTrain, realCountForRt } from "./mix";

const sample = (id: string, text: string): AcceptedRealRecord => ({
  id,
  text,
  intent: { frequency: { unit: "day", interval: 1 } },
  spans: [{ start: 0, end: text.length, label: "RECUR", clauseStart: true }],
  agree: 3,
  exec: true,
  roundtrip: true,
  family: "interval",
});

test("realCountForRt is n_real such that n_real/(n_real+n_gen) ≈ target", () => {
  expect(realCountForRt(9, 0.1)).toBe(1);
  expect(realCountForRt(7, 0.3)).toBe(3);
});

test("mixAcceptedIntoTrain appends the RT share and tags source=llm", () => {
  const dir = mkdtempSync(join(tmpdir(), "cron-mix-"));
  const train = join(dir, "train.jsonl");

  const gen = Array.from({ length: 9 }, (_, index) =>
    JSON.stringify({
      id: `gen-${index}`,
      split: "train",
      family: "interval",
      fingerprint: `g${index}`,
      text: `gen ${index}`,
      intent: { frequency: { unit: "day", interval: 1 } },
      spans: [],
      source: "rule",
    }),
  );

  writeFileSync(train, `${gen.join("\n")}\n`);
  const accepted = [sample("real-a", "every day"), sample("real-b", "daily")];
  const mix = mixAcceptedIntoTrain(train, accepted, 0.1);

  expect(mix).toEqual({ nReal: 1, nGen: 9, rt: 0.1, fraction: "1/10" });
  const lines = readFileSync(train, "utf8").trim().split("\n");

  expect(lines).toHaveLength(10);
  const last = JSON.parse(lines[9]!) as { text: string; source: string };

  expect(last.text).toBe("every day");
  expect(last.source).toBe("llm");
});
