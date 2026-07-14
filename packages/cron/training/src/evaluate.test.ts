import { describe, expect, test } from "bun:test";
import { readdirSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { compile } from "../../src/compiler/cron";
import { LABEL_COUNT, labelId, LABELS } from "../../src/model/labels";
import { tokenize } from "../../src/model/tokenizer";
import type { ParseResult } from "../../src/parser/parse";
import { loadTestEngine } from "../tests/helpers/candidate-engine";
import { tempDir } from "../tests/helpers/temp-dir";
import type { ContractReport, Engine, GoldRecord } from "./evaluate";
import {
  canonicalizeIntent,
  evaluateCorpus,
  lintGoldRecords,
  loadEngine,
  parseCorpus,
  riskCoverage,
} from "./evaluate";

describe("canonicalizeIntent", () => {
  test("normalizes semantic sets without mutating the input", () => {
    const intent = {
      frequency: { unit: "week" as const, interval: 1 },
      weekdays: ["friday" as const, "monday" as const],
      exceptions: {
        weekdays: ["sunday" as const, "tuesday" as const],
        dates: ["2026-12-02", "2026-01-01", "2026-12-02"],
      },
    };

    expect(canonicalizeIntent(intent)).toEqual({
      frequency: { unit: "week", interval: 1 },
      weekdays: ["monday", "friday"],
      exceptions: {
        weekdays: ["tuesday", "sunday"],
        dates: ["2026-01-01", "2026-12-02"],
      },
    });

    expect(intent.weekdays).toEqual(["friday", "monday"]);
  });
});

describe("parseCorpus does not patch gold semantics", () => {
  test("leaves 1:30 as 1:30 without inserting a meridiem", () => {
    const item: GoldRecord = {
      id: "clock",
      family: "time-of-day",
      text: "remind me 1:30 daily",
      spans: [
        { start: 0, end: 9, label: "O", clauseStart: false },
        { start: 10, end: 11, label: "HOUR_9", clauseStart: true },
        { start: 11, end: 12, label: "GLUE", clauseStart: false },
        { start: 12, end: 14, label: "MINUTE_30", clauseStart: false },
        { start: 15, end: 20, label: "RECUR", clauseStart: false },
      ],
      intent: {
        frequency: { unit: "day", interval: 1 },
        time: { hour: 13, minute: 30 },
      },
      expected: { supported: true, expression: "30 13 * * *" },
    };

    const [parsed] = parseCorpus(`${JSON.stringify(item)}\n`);

    expect(parsed?.text).toBe("remind me 1:30 daily");
    expect(parsed?.spans.some((span) => span.label.startsWith("MERIDIEM_"))).toBe(false);
    expect(parsed?.intent?.time).toEqual({ hour: 13, minute: 30 });

    for (const token of tokenize(item.text).filter((value) => value.kind !== 3)) {
      expect(item.spans.some((span) => token.start >= span.start && token.end <= span.end)).toBe(
        true,
      );
    }
  });

  test("does not rewrite a through date into an exclusive end", () => {
    const item: GoldRecord = {
      id: "bounds",
      family: "bounds",
      text: "from 2026-01-01 through 2026-01-31, every day",
      spans: [],
      intent: {
        frequency: { unit: "day", interval: 1 },
        bounds: { start: "2026-01-01", end: "2026-01-31" },
      },
      expected: { supported: false, expression: null },
    };

    const [parsed] = parseCorpus(`${JSON.stringify(item)}\n`);

    expect(parsed?.intent?.bounds).toEqual({
      start: "2026-01-01",
      end: "2026-01-31",
    });
  });
});

describe("lintGoldRecords", () => {
  test("rejects duplicate ids and overlapping spans", () => {
    const record: GoldRecord = {
      id: "same",
      family: "negative",
      text: "ordinary prose",
      spans: [{ start: 0, end: 8, label: "O", clauseStart: false }],
      intent: null,
      expected: { supported: false, expression: null },
    };

    expect(() => lintGoldRecords([record, record])).toThrow(/duplicate/);

    expect(() =>
      lintGoldRecords([
        {
          ...record,
          spans: [
            { start: 0, end: 8, label: "O", clauseStart: false },
            { start: 7, end: 13, label: "O", clauseStart: false },
          ],
        },
      ]),
    ).toThrow(/span/);
  });
});

describe("riskCoverage", () => {
  test("reports acceptance risk, silent changes, and negative false accepts", () => {
    const [point] = riskCoverage(
      [
        { accepted: true, correct: true, confidence: 0.9, negative: false },
        { accepted: true, correct: false, confidence: 0.8, negative: false },
        { accepted: true, correct: false, confidence: 0.7, negative: true },
        { accepted: false, correct: true, confidence: 0.1, negative: true },
      ],
      [0.5],
    );

    expect(point).toEqual({
      threshold: 0.5,
      total: 4,
      accepted: 3,
      acceptedWrong: 2,
      coverage: 0.75,
      risk: 2 / 3,
      silentMeaningChangeRate: 0.5,
      negativeFalseAcceptanceRate: 0.5,
    });
  });
});

describe("loadEngine candidate uses src labels", () => {
  test("a candidate is scored against typed LABELS without a dist build", async () => {
    const engine = await loadTestEngine();

    try {
      expect(engine.labelName(0)).toBe("O");
      expect(engine.labelName(5)).toBe("WEEKDAY_MON");
      expect(LABELS[5]).toBe("WEEKDAY_MON");
      expect(engine.labelName(LABEL_COUNT - 1)).toBe(LABELS[LABEL_COUNT - 1]!);
    } finally {
      await engine.dispose();
    }
  });

  test("failed candidate imports remove their scratch source tree", async () => {
    const directory = tempDir("syntaxless-invalid-candidate-");
    const path = join(directory, "weights.gen.ts");

    // The import in the copied source tree is deliberately invalid. Record that
    // exact tree rather than inspecting unrelated temporary directories.
    writeFileSync(
      path,
      `import { writeFileSync } from "node:fs";
       import { dirname, resolve } from "node:path";
       if (import.meta.dirname !== ${JSON.stringify(directory)}) {
         writeFileSync(${JSON.stringify(join(directory, "scratch.txt"))}, resolve(dirname(import.meta.path), "../.."));
         throw new Error("invalid candidate fixture");
       }
       export const weights = { q: "f", roleClasses: ${LABEL_COUNT} };\n`,
    );

    await expect(loadEngine(join(directory, "no-dist"), path)).rejects.toThrow(
      "invalid candidate fixture",
    );

    const scratch = await Bun.file(join(directory, "scratch.txt")).text();

    expect(scratch.startsWith(join(tmpdir(), "syntaxless-evaluate-"))).toBe(true);
    expect(() => readdirSync(scratch)).toThrow();
  });
});

test("evaluation keeps overlay-on and overlay-off scores separate and counts false accepts", async () => {
  const intent = { frequency: { unit: "hour" as const, interval: 1 } };
  const wrong = { frequency: { unit: "day" as const, interval: 1 } };

  const records: GoldRecord[] = [
    {
      id: "hourly",
      family: "interval",
      text: "hourly",
      speechAct: "request",
      spans: [{ start: 0, end: 6, label: "UNIT_HOUR", clauseStart: true, actionable: true }],
      intent,
      expected: { supported: true, expression: "0 * * * *" },
    },
    {
      id: "negative",
      family: "negative",
      text: "coffee",
      speechAct: "none",
      spans: [{ start: 0, end: 6, label: "O", clauseStart: false, actionable: false }],
      intent: null,
      expected: { supported: false, expression: null },
    },
  ];

  const corpus = join(tempDir("syntaxless-evaluation-"), "gold.jsonl");

  writeFileSync(corpus, records.map((row) => JSON.stringify(row)).join("\n"));

  const accepted: ParseResult = {
    ok: true,
    source: "model",
    intent,
    confidence: 0.9,
    diagnostics: [],
  };

  const rejected: ParseResult = {
    ok: false,
    source: "model",
    code: "low-confidence",
    reason: "fixture rejection",
    confidence: 0,
    diagnostics: [],
  };

  const engine: Engine = {
    artifactSha256: "a".repeat(64),
    compile,
    tokenize,
    inferTokens: (tokens) => ({
      labels: Uint8Array.from(tokens, (token) =>
        token.text === "hourly" ? labelId.UNIT_HOUR : labelId.O,
      ),
    }),
    labelName: (id) => LABELS[id]!,
    parse: async (text, options) =>
      (text === "hourly") !== (options?.overlay === "off") ? accepted : rejected,
    decodeRaw: (text, overlay) => {
      if (text !== "hourly") throw new RangeError("fixture negative");

      return { intent: overlay === "off" ? wrong : intent, confidence: 0.9 };
    },
    dispose: async () => {},
  };

  const result = await evaluateCorpus(engine, corpus, "unit");

  const counts = (reports: readonly ContractReport[]) =>
    reports.map(({ metric, correct, total }) => ({ metric, correct, total }));

  expect(counts(result.contracts)).toEqual([
    { metric: "tokenRoles", correct: 2, total: 2 },
    { metric: "rawDecode", correct: 2, total: 2 },
    { metric: "exactAst", correct: 2, total: 2 },
    { metric: "compiledExpression", correct: 2, total: 2 },
  ]);

  expect(counts(result.overlayOff.contracts)).toEqual([
    { metric: "rawDecode", correct: 1, total: 2 },
    { metric: "exactAst", correct: 0, total: 2 },
    { metric: "compiledExpression", correct: 0, total: 2 },
  ]);

  expect(result.exec).toEqual({ correct: 0, total: 2 });

  expect(
    result.overlayOff.selectivePrediction.find((point) => point.threshold === 0.5),
  ).toMatchObject({
    accepted: 1,
    acceptedWrong: 1,
    coverage: 0.5,
    negativeFalseAcceptanceRate: 1,
  });
});
