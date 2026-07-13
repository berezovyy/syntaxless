import { expect, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  appendLedgerRow,
  assertLedgerComplete,
  hashExperimentInputs,
  LEDGER_REQUIRED,
} from "./experiment";

test("hashExperimentInputs is stable for the shipped kinds/noise/splits/background/labels files", () => {
  const first = hashExperimentInputs();
  const second = hashExperimentInputs();

  expect(first).toBe(second);
  expect(first).toMatch(/^[0-9a-f]{64}$/);
});

test("finishing without EXEC/FA@cov/RT/stack is refused", () => {
  expect(LEDGER_REQUIRED).toEqual(["EXEC", "FA@cov", "RT", "stack"]);
  expect(() => assertLedgerComplete({ EXEC: "1" })).toThrow(/missing/);
  const dir = mkdtempSync(join(tmpdir(), "cron-experiment-"));

  try {
    const ledger = join(dir, "EXPERIMENTS.md");

    writeFileSync(ledger, "# ledger\n");

    expect(() =>
      appendLedgerRow(
        ledger,
        { name: "s4-refuse", change: "incomplete row" },
        {
          EXEC: "1",
          "FA@cov": "",
          RT: "0",
          stack: "tagger-typed",
        },
      ),
    ).toThrow(/refused/);

    expect(readFileSync(ledger, "utf8")).not.toContain("s4-refuse");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("appendLedgerRow writes EXEC/FA@cov/RT/stack", () => {
  const dir = mkdtempSync(join(tmpdir(), "cron-experiment-"));

  try {
    const ledger = join(dir, "EXPERIMENTS.md");

    writeFileSync(ledger, "# ledger\n");

    appendLedgerRow(
      ledger,
      { name: "s4-ok", change: "complete row" },
      {
        EXEC: "1/1",
        "FA@cov": "0@1",
        RT: "0",
        stack: "tagger-typed",
      },
    );

    const text = readFileSync(ledger, "utf8");

    expect(text).toContain("| s4-ok |");
    expect(text).toContain("| 1/1 |");
    expect(text).toContain("| 0@1 |");
    expect(text).toContain("| tagger-typed |");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
