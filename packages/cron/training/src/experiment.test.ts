import { expect, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  appendLedgerRow,
  assertLedgerComplete,
  hashExperimentInputs,
  LEDGER_REQUIRED,
  main,
} from "./experiment";
import type { StepRecord } from "./experiment";

function skipped(steps: readonly StepRecord[], prefix: string): boolean {
  const matched = steps.filter((step) => step.step.startsWith(prefix));

  return matched.length > 0 && matched.every((step) => step.skipped);
}

function invoked(steps: readonly StepRecord[], prefix: string): boolean {
  return steps.some(
    (step) =>
      step.step.startsWith(prefix) &&
      !step.skipped &&
      step.argv.some((arg) => arg.includes(prefix === "regen" ? "generate.ts" : `${prefix}.ts`)),
  );
}

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

test(
  "a second run with unchanged hashed inputs skips regen and featurize",
  () => {
    const dir = mkdtempSync(join(tmpdir(), "cron-experiment-run-"));
    const work = join(dir, "work");
    const ledger = join(dir, "EXPERIMENTS.md");
    const storage = join(dir, "runs");

    writeFileSync(ledger, "# ledger\n");

    const shared = [
      "--samples",
      "8",
      "--eval-samples",
      "4",
      "--epochs",
      "1",
      "--batch",
      "4",
      "--oracle-seeds",
      "0",
      "--kinds-sample",
      "0",
      "--device",
      "cpu",
      "--work",
      work,
      "--ledger",
      ledger,
      "--storage",
      storage,
    ];

    try {
      const first = main(["--name", "s4-skip-1", "--change", "tiny fixture first run", ...shared]);

      expect(first.skippedRegen).toBe(false);
      expect(invoked(first.steps, "regen")).toBe(true);
      expect(invoked(first.steps, "featurize")).toBe(true);
      expect(first.steps.some((step) => step.step === "oracle" && !step.skipped)).toBe(true);
      expect(first.steps.some((step) => step.step === "train" && !step.skipped)).toBe(true);
      expect(first.steps.some((step) => step.step === "export" && !step.skipped)).toBe(true);

      const second = main([
        "--name",
        "s4-skip-2",
        "--change",
        "tiny fixture second run",
        ...shared,
      ]);

      expect(second.inputHash).toBe(first.inputHash);
      expect(second.skippedRegen).toBe(true);
      expect(second.skippedFeaturize).toBe(true);
      expect(skipped(second.steps, "regen")).toBe(true);
      expect(skipped(second.steps, "featurize")).toBe(true);
      const recorded = JSON.parse(readFileSync(join(work, "steps.json"), "utf8")) as StepRecord[];

      expect(skipped(recorded, "regen")).toBe(true);
      expect(skipped(recorded, "featurize")).toBe(true);
      expect(second.steps.some((step) => step.step === "oracle" && !step.skipped)).toBe(true);

      const row = readFileSync(ledger, "utf8")
        .split("\n")
        .find((line) => line.includes("| s4-skip-1 |"));

      expect(row).toBeDefined();

      const cells = row!
        .split("|")
        .map((cell) => cell.trim())
        .filter((cell) => cell.length > 0);

      expect(cells[8]).toMatch(/^\d+\/\d+$/);
      expect(cells[9]!.length).toBeGreaterThan(0);
      expect(cells[10]!.length).toBeGreaterThan(0);
      expect(cells[11]).toBe("tagger-typed");
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  },
  { timeout: 180_000 },
);
