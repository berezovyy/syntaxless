import { describe, expect, test } from "bun:test";
import { writeFileSync } from "node:fs";
import { join } from "node:path";

import { compile } from "../../src/compiler/cron";
import { labelId, LABELS } from "../../src/model/labels";
import { tokenize } from "../../src/model/tokenizer";
import { weights } from "../../src/model/weights.gen";
import { normalize } from "../../src/parser/normalize";
import type { ParseResult } from "../../src/parser/parse";
import { tempDir } from "../tests/helpers/temp-dir";
import type { ContractReport, Engine, GoldRecord } from "./evaluate";
import { assertSameArtifact, evaluateCorpus, loadEngine, riskCoverage } from "./evaluate";

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
  test("calibration changes behavior and the identity reported by evaluation", async () => {
    const directory = tempDir("syntaxless-calibration-identity-");
    const permissivePath = join(directory, "permissive.ts");
    const strictPath = join(directory, "strict.ts");

    for (const [path, energyThreshold] of [
      [permissivePath, 1e9],
      [strictPath, -1e9],
    ] as const) {
      writeFileSync(
        path,
        `export const weights = ${JSON.stringify({ ...weights, energyThreshold })};\n`,
      );
    }

    const permissive = await loadEngine(join(directory, "no-dist"), permissivePath);
    const strict = await loadEngine(join(directory, "no-dist"), strictPath);
    const text = "every weekday at 9am";

    expect(permissive.predict(text).parse().ok).toBe(true);
    expect(strict.predict(text).parse().ok).toBe(false);
    expect(permissive.artifactSha256).not.toBe(strict.artifactSha256);
    expect(() => assertSameArtifact([permissive, strict])).toThrow("artifact mismatch");
  });
});

test("evaluation preserves original gold offsets, independent policies and false accepts", async () => {
  const intent = { frequency: { unit: "hour" as const, interval: 1 } };
  const wrong = { frequency: { unit: "day" as const, interval: 1 } };

  const records: GoldRecord[] = [
    {
      id: "hourly",
      family: "interval",
      text: "   HOURLY",
      speechAct: "request",
      spans: [{ start: 3, end: 9, label: "UNIT_HOUR", clauseStart: true, actionable: true }],
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

  const inferTokens: Engine["inferTokens"] = (tokens) => ({
    labels: Uint8Array.from(tokens, (token) => (token.start === 3 ? labelId.UNIT_HOUR : labelId.O)),
    boundaries: new Uint8Array(tokens.length),
    scores: new Float32Array(tokens.length),
    boundaryScores: new Float32Array(tokens.length),
  });

  const engine: Engine = {
    artifactSha256: "a".repeat(64),
    compile,
    tokenize,
    inferTokens,
    labelName: (id) => LABELS[id]!,
    predict: (text) => {
      const normalized = normalize(text);
      const tokens = tokenize(normalized);

      return {
        normalized,
        tokens,
        predictions: inferTokens(tokens),
        parse: (options) =>
          (normalized === "hourly") !== (options?.overlay === "off") ? accepted : rejected,
        decodeRaw: (overlay) => {
          if (normalized !== "hourly") throw new RangeError("fixture negative");

          return {
            intent: overlay === "off" ? wrong : intent,
            confidence: 0.9,
            evidence: {
              tokenMargin: 0.9,
              boundaryMargin: 0.9,
              scheduleKindAgreement: true,
              explicitScheduleEvidence: true,
              unconsumedScheduleLikeValues: 0,
              semanticConstraintViolations: 0,
              compilerConsistent: true,
            },
          };
        },
      };
    },
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
