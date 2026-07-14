import { expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, unlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  appendLedgerRow,
  assertLedgerComplete,
  formatMeanRange,
  hashExperimentInputs,
  LEDGER_REQUIRED,
  main,
  metricsFromReports,
} from "./experiment";
import type { StepRecord } from "./experiment";
import { planKindRegen } from "./kind-hash";

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

test("formatMeanRange reports min-max when seeds disagree", () => {
  expect(formatMeanRange(["10/20", "10/20"])).toBe("10/20");
  expect(formatMeanRange(["8/20", "12/20"])).toBe("10.0/20 (8-12)");
});

test("hashExperimentInputs is stable for the shipped kinds/noise/splits/background/labels files", () => {
  const first = hashExperimentInputs();
  const second = hashExperimentInputs();

  expect(first).toBe(second);
  expect(first).toMatch(/^[0-9a-f]{64}$/);
});

function writeGoldAccepted(dir: string, count: number): string[] {
  const goldPath = join(import.meta.dir, "../data/gold/reserved-carriers.jsonl");

  const gold = readFileSync(goldPath, "utf8")
    .trim()
    .split("\n")
    .slice(0, count)
    .map(
      (line) =>
        JSON.parse(line) as {
          id: string;
          text: string;
          intent: unknown;
          spans: unknown;
          family: string;
        },
    );

  mkdirSync(dir, { recursive: true });

  writeFileSync(
    join(dir, "accepted.jsonl"),
    `${gold
      .map((row) =>
        JSON.stringify({
          id: row.id,
          text: row.text,
          intent: row.intent,
          spans: row.spans,
          family: row.family,
          agree: 3,
          exec: true,
          roundtrip: true,
        }),
      )
      .join("\n")}\n`,
  );

  return gold.map((row) => row.text);
}

function ledgerCells(ledger: string, name: string): string[] {
  const row = readFileSync(ledger, "utf8")
    .split("\n")
    .find((line) => line.includes(`| ${name} |`));

  expect(row).toBeDefined();

  return row!
    .split("|")
    .map((cell) => cell.trim())
    .filter((cell) => cell.length > 0);
}

test("A1/A2 EXPERIMENTS.md rows use execEqual EXEC and overlayOff.faAtCov", () => {
  const a1 = metricsFromReports(
    join(import.meta.dir, "../data/synth/experiments/a1/oracle.json"),
    join(import.meta.dir, "../data/synth/experiments/a1/eval-seed-20260913.json"),
    join(import.meta.dir, "../data/synth/experiments/a1/h-seed-20260913.json"),
    "tagger-typed",
    "0",
  );

  const a2 = metricsFromReports(
    join(import.meta.dir, "../data/synth/experiments/a2/oracle.json"),
    join(import.meta.dir, "../data/synth/experiments/a2/eval-seed-20260913.json"),
    join(import.meta.dir, "../data/synth/experiments/a2/h-seed-20260913.json"),
    "tagger-typed",
    "0",
  );

  expect(a1.O).toBe("360/360");
  expect(a1.EXEC).toMatch(/^\d+\/360$/);
  expect(a1.EXEC).not.toBe(a1.O);
  expect(a1["FA@cov"]).toMatch(/@/);
  expect(a1["FA@cov"]).not.toContain("0.4465370595382746");
  expect(a2.EXEC).not.toBe(a2.O);
  const ledger = readFileSync(join(import.meta.dir, "../../../../EXPERIMENTS.md"), "utf8");
  const a1Row = ledger.split("\n").find((line) => line.includes("| a1 |"));

  expect(a1Row).toContain(`| ${a1.EXEC} |`);
  expect(a1Row).toContain(`| ${a1["FA@cov"]} |`);
  expect(a1Row).not.toContain("| 360/360 | 360/360 |");
  const a2Row = ledger.split("\n").find((line) => line.includes("| a2 |"));

  expect(a2Row).toContain(`| ${a2.EXEC} |`);
  expect(a2Row).toContain(`| ${a2["FA@cov"]} |`);
});

test("metricsFromReports uses execEqual counts and FA@cov, not oracle O", () => {
  const dir = mkdtempSync(join(tmpdir(), "cron-metrics-"));

  try {
    const oracle = join(dir, "oracle.json");
    const evaluated = join(dir, "eval.json");
    const human = join(dir, "human.json");

    writeFileSync(oracle, `${JSON.stringify({ correct: 360, total: 360 })}\n`);

    writeFileSync(
      evaluated,
      `${JSON.stringify({ exec: { correct: 12, total: 360 }, faAtCov: "0.5@1" })}\n`,
    );

    writeFileSync(
      human,
      `${JSON.stringify({
        overlayOff: {
          correct: 735,
          total: 1646,
          faAtCov: "0.02@0.9",
          negativeFalseAccepts: 37,
          silentMeaningChanges: 70,
        },
      })}\n`,
    );

    const metrics = metricsFromReports(oracle, evaluated, human, "tagger-typed", "0");

    expect(metrics.O).toBe("360/360");
    expect(metrics.EXEC).toBe("12/360");
    expect(metrics.EXEC).not.toBe(metrics.O);
    expect(metrics["FA@cov"]).toBe("0.02@0.9");
    expect(metrics.H).toBe("735/1646");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("C1/C2 --rt without accepted real jsonl is refused", async () => {
  const dir = mkdtempSync(join(tmpdir(), "cron-c1-"));

  writeFileSync(join(dir, "EXPERIMENTS.md"), "# ledger\n");

  try {
    await expect(
      main([
        "--name",
        "c1",
        "--change",
        "RT 10% real mix",
        "--rt",
        "0.1",
        "--real",
        join(dir, "empty-real"),
        "--samples",
        "8",
        "--eval-samples",
        "4",
        "--epochs",
        "1",
        "--oracle-seeds",
        "0",
        "--work",
        join(dir, "work"),
        "--ledger",
        join(dir, "EXPERIMENTS.md"),
        "--storage",
        join(dir, "runs"),
      ]),
    ).rejects.toThrow(/C1\/C2 refused/);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
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
  async () => {
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
      const first = await main([
        "--name",
        "s4-skip-1",
        "--change",
        "tiny fixture first run",
        "--seeds",
        "1",
        ...shared,
      ]);

      expect(first.skippedRegen).toBe(false);
      expect(invoked(first.steps, "regen")).toBe(true);
      expect(invoked(first.steps, "featurize")).toBe(true);
      expect(first.steps.some((step) => step.step === "oracle" && !step.skipped)).toBe(true);
      const train = first.steps.find((step) => step.step.startsWith("train-seed-"));

      expect(train?.skipped).toBe(false);
      expect(train?.argv).toContain("--h-per-epoch");

      expect(
        first.steps.some((step) => step.step.startsWith("export-seed-") && !step.skipped),
      ).toBe(true);

      const report = JSON.parse(
        readFileSync(join(storage, "s4-skip-1", "report.json"), "utf8"),
      ) as {
        config?: { h_per_epoch?: boolean };
        history?: Array<{ epoch: number; h: unknown }>;
      };

      expect(report.config?.h_per_epoch).toBe(true);
      expect(report.history?.length).toBeGreaterThan(0);
      expect(report.history?.[0]?.h).toBeTruthy();

      const second = await main([
        "--name",
        "s4-skip-2",
        "--change",
        "tiny fixture second run",
        "--seeds",
        "1",
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

      expect(cells[4]).toMatch(/^\d+(?:\.\d+)?\/\d+/);
      expect(cells[8]).toMatch(/^\d+\/\d+$/);
      expect(cells[9]!.length).toBeGreaterThan(0);
      expect(cells[9]).not.toBe("n/a");
      expect(cells[10]!.length).toBeGreaterThan(0);
      expect(cells[11]).toBe("tagger-typed");
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  },
  { timeout: 180_000 },
);

test(
  "unchanged kind hashes skip regen; deleting one kind artifact re-renders only that kind",
  async () => {
    const dir = mkdtempSync(join(tmpdir(), "cron-kind-incr-"));
    const work = join(dir, "work");
    const ledger = join(dir, "EXPERIMENTS.md");
    const storage = join(dir, "runs");

    writeFileSync(ledger, "# ledger\n");

    const shared = [
      "--kinds",
      "cancel,incomplete",
      "--kind-cap",
      "12",
      "--seeds",
      "1",
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
      const first = await main([
        "--name",
        "s2-kinds-1",
        "--change",
        "incremental kinds",
        ...shared,
      ]);

      expect(first.renderedKinds.sort()).toEqual(["cancel", "incomplete"].sort());

      expect(first.steps.some((step) => step.step.startsWith("regen-kind:") && !step.skipped)).toBe(
        true,
      );

      const second = await main([
        "--name",
        "s2-kinds-2",
        "--change",
        "unchanged kind hashes",
        ...shared,
      ]);

      expect(second.skippedRegen).toBe(true);
      expect(second.renderedKinds).toEqual([]);
      expect(second.skippedKinds.sort()).toEqual(["cancel", "incomplete"].sort());

      expect(
        second.steps
          .filter((step) => step.step.startsWith("regen-kind:"))
          .every((step) => step.skipped),
      ).toBe(true);

      const plan = planKindRegen(work, ["cancel", "incomplete"]);
      const victim = plan.current[0]!;

      unlinkSync(victim.jsonl);

      const third = await main([
        "--name",
        "s2-kinds-3",
        "--change",
        "one kind hash miss",
        ...shared,
      ]);

      expect(third.renderedKinds).toEqual([victim.id]);
      expect(third.skippedKinds).toContain(plan.current[1]!.id);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  },
  { timeout: 180_000 },
);

test(
  "--seeds 3 performs three train/score passes and the ledger row is mean/range",
  async () => {
    const dir = mkdtempSync(join(tmpdir(), "cron-seeds-"));
    const work = join(dir, "work");
    const ledger = join(dir, "EXPERIMENTS.md");
    const storage = join(dir, "runs");

    writeFileSync(ledger, "# ledger\n");

    try {
      const result = await main([
        "--name",
        "s5-seeds",
        "--change",
        "three seeds",
        "--seeds",
        "3",
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
      ]);

      expect(result.seedRuns).toBe(3);

      const trains = result.steps.filter(
        (step) => step.step.startsWith("train-seed-") && !step.skipped,
      );

      expect(trains.length).toBe(3);

      const row = readFileSync(ledger, "utf8")
        .split("\n")
        .find((line) => line.includes("| s5-seeds |"));

      expect(row).toBeDefined();
      expect(row).toContain("seeds=3");
      expect(row).toMatch(/\d+(?:\.\d+)?\/\d+/);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  },
  { timeout: 180_000 },
);

test(
  "--rt 0.1 and --rt 0.3 mix accepted gold rows into train.jsonl and ledger RT",
  async () => {
    const dir = mkdtempSync(join(tmpdir(), "cron-rt-mix-"));
    const real = join(dir, "real");
    const texts = writeGoldAccepted(real, 4);
    const ledger = join(dir, "EXPERIMENTS.md");

    writeFileSync(ledger, "# ledger\n");

    try {
      const c1 = await main([
        "--name",
        "c1-rt",
        "--change",
        "RT 10% accepted mix",
        "--rt",
        "0.1",
        "--real",
        real,
        "--samples",
        "9",
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
        join(dir, "work-c1"),
        "--ledger",
        ledger,
        "--storage",
        join(dir, "runs-c1"),
      ]);

      const mixStep = c1.steps.find((step) => step.step === "mix-real");

      expect(mixStep?.skipped).toBe(false);

      const c1Train = readFileSync(join(dir, "work-c1", "train.jsonl"), "utf8")
        .trim()
        .split("\n")
        .map((line) => JSON.parse(line) as { text: string; source: string });

      const c1Real = c1Train.filter((row) => row.source === "llm");
      const c1Gen = c1Train.filter((row) => row.source !== "llm");

      expect(c1Gen).toHaveLength(9);
      expect(c1Real).toHaveLength(1);
      expect(c1Real[0]!.text).toBe(texts[0]!);

      const c1Mix = JSON.parse(readFileSync(join(dir, "work-c1", "mix.json"), "utf8")) as {
        nReal: number;
        nGen: number;
        fraction: string;
        rt: number;
      };

      expect(c1Mix).toEqual({ nReal: 1, nGen: 9, fraction: "1/10", rt: 0.1 });
      const c1Cells = ledgerCells(ledger, "c1-rt");

      expect(c1Cells[8]).toMatch(/^\d+\/\d+$/);
      expect(c1Cells[9]!.length).toBeGreaterThan(0);
      expect(c1Cells[10]).toBe("1/10");
      expect(c1Cells[11]).toBe("tagger-typed");

      const c2 = await main([
        "--name",
        "c2-rt",
        "--change",
        "RT 30% accepted mix",
        "--rt",
        "0.3",
        "--real",
        real,
        "--samples",
        "7",
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
        join(dir, "work-c2"),
        "--ledger",
        ledger,
        "--storage",
        join(dir, "runs-c2"),
      ]);

      expect(c2.steps.find((step) => step.step === "mix-real")?.skipped).toBe(false);

      const c2Train = readFileSync(join(dir, "work-c2", "train.jsonl"), "utf8")
        .trim()
        .split("\n")
        .map((line) => JSON.parse(line) as { text: string; source: string });

      const c2Real = c2Train.filter((row) => row.source === "llm");
      const c2Gen = c2Train.filter((row) => row.source !== "llm");

      expect(c2Gen).toHaveLength(7);
      expect(c2Real).toHaveLength(3);
      expect(c2Real.map((row) => row.text)).toEqual(texts.slice(0, 3));

      const c2Mix = JSON.parse(readFileSync(join(dir, "work-c2", "mix.json"), "utf8")) as {
        nReal: number;
        nGen: number;
        fraction: string;
        rt: number;
      };

      expect(c2Mix.nReal).toBe(3);
      expect(c2Mix.nGen).toBe(7);
      expect(c2Mix.fraction).toBe("3/10");
      const c2Cells = ledgerCells(ledger, "c2-rt");

      expect(c2Cells[8]).toMatch(/^\d+\/\d+$/);
      expect(c2Cells[9]!.length).toBeGreaterThan(0);
      expect(c2Cells[10]).toBe("3/10");
      expect(c2Cells[11]).toBe("tagger-typed");
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  },
  { timeout: 240_000 },
);

test(
  "experiment --stack seq2seq spawns seq2seq.py --b1 and writes a seq2seq ledger row",
  async () => {
    const dir = mkdtempSync(join(tmpdir(), "cron-b1-"));
    const work = join(dir, "work");
    const ledger = join(dir, "EXPERIMENTS.md");

    writeFileSync(ledger, "# ledger\n");

    try {
      const result = await main([
        "--name",
        "b1",
        "--change",
        "seq2seq B1 vs tagger",
        "--stack",
        "seq2seq",
        "--samples",
        "64",
        "--eval-samples",
        "16",
        "--epochs",
        "1",
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
        join(dir, "runs"),
      ]);

      const b1 = result.steps.find((step) => step.step === "seq2seq-b1");

      expect(b1?.skipped).toBe(false);
      expect(b1?.argv.some((arg) => arg.endsWith("seq2seq.py"))).toBe(true);
      expect(b1?.argv).toContain("--b1");

      const row = readFileSync(ledger, "utf8")
        .split("\n")
        .find((line) => line.includes("| b1 |"));

      expect(row).toBeDefined();
      expect(row).toContain("| seq2seq |");
      expect(row).toMatch(/\d+\/\d+/);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  },
  { timeout: 180_000 },
);
