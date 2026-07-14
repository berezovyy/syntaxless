import { expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { isAcceptedReal, loadAcceptedReal } from "./accepted";

test("loadAcceptedReal keeps only agree:3 exec:true roundtrip:true rows", () => {
  const dir = mkdtempSync(join(tmpdir(), "cron-real-"));

  mkdirSync(dir, { recursive: true });

  writeFileSync(
    join(dir, "tatoeba.jsonl"),
    `${JSON.stringify({
      id: "ok",
      text: "every day",
      intent: { frequency: { unit: "day", interval: 1 } },
      spans: [{ start: 0, end: 5, label: "RECUR", clauseStart: true }],
      agree: 3,
      exec: true,
      roundtrip: true,
    })}\n${JSON.stringify({
      id: "bad",
      text: "nope",
      intent: null,
      spans: [],
      agree: 2,
      exec: true,
      roundtrip: true,
    })}\n`,
  );

  writeFileSync(join(dir, "rejected.jsonl"), `${JSON.stringify({ stage: "agree", text: "x" })}\n`);
  const loaded = loadAcceptedReal(dir);

  expect(loaded.map((row) => row.id)).toEqual(["ok"]);

  expect(isAcceptedReal({ agree: 3, exec: true, roundtrip: true, text: "a", spans: [] })).toBe(
    true,
  );

  expect(loadAcceptedReal(join(dir, "missing"))).toEqual([]);
});
