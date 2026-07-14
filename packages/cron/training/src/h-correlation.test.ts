import { expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { correlateEpoch1VsFinal, spearman } from "./h-correlation";

test("spearman is 1 for a monotone pair", () => {
  expect(spearman([1, 2, 3], [2, 4, 6])).toBe(1);
});

test("correlateEpoch1VsFinal reads A1/A2 history.h under training/cron/runs", () => {
  const report = correlateEpoch1VsFinal("training/cron/runs");
  const names = report.points.map((point) => point.run);

  expect(names).toContain("a1");
  expect(names).toContain("a2");
  expect(report.runs).toBeGreaterThanOrEqual(2);
  const a1 = report.points.find((point) => point.run === "a1");

  expect(a1?.epoch1).toBeGreaterThan(0);
  expect(a1?.final).toBeGreaterThan(0);
});

test("correlateEpoch1VsFinal reads history.h from report.json files", () => {
  const dir = mkdtempSync(join(tmpdir(), "cron-h-corr-"));

  mkdirSync(join(dir, "run-a"));
  mkdirSync(join(dir, "run-b"));

  writeFileSync(
    join(dir, "run-a", "report.json"),
    JSON.stringify({
      history: [
        { h: { ok: true, correct: 1, total: 10 } },
        { h: { ok: true, correct: 8, total: 10 } },
      ],
    }),
  );

  writeFileSync(
    join(dir, "run-b", "report.json"),
    JSON.stringify({
      history: [
        { h: { ok: true, correct: 2, total: 10 } },
        { h: { ok: true, correct: 9, total: 10 } },
      ],
    }),
  );

  const report = correlateEpoch1VsFinal(dir);

  expect(report.runs).toBe(2);
  expect(report.spearman).toBe(1);
  expect(report.points.map((point) => point.run)).toEqual(["run-a", "run-b"]);
});
