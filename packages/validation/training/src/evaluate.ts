/** Intent-level metrics. Token accuracy is diagnostic, never a language gate. */
import { readFileSync, writeFileSync } from "node:fs";
import { isDeepStrictEqual } from "node:util";

import { interval } from "../../../shared/training/statistics.js";
import { compile } from "../../src/compiler/index.js";
import { tokenize } from "../../src/model/tokenizer.js";
import { resultFromLabeledTokens } from "../../src/parser/model.js";
import { parse } from "../../src/parser/parse.js";
import type { ParseResult } from "../../src/parser/parse.js";
import type { CorpusRecord } from "./generate.js";

export interface Counts {
  total: number;
  accepted: number;
  exact: number;
  wrong: number;
}

function counts(): Counts {
  return { total: 0, accepted: 0, exact: 0, wrong: 0 };
}

function rate(n: number, d: number): number {
  return d ? n / d : 0;
}

export async function evaluateRecords(
  records: readonly CorpusRecord[],
  predict: (row: CorpusRecord, index: number) => Promise<ParseResult>,
) {
  const positive = counts();
  const negative = counts();

  const families: Record<string, Counts> = {};

  const failures: Array<{
    text: string;
    expected: unknown;
    actual: unknown;
    family: string;
  }> = [];

  for (let i = 0; i < records.length; i++) {
    const row = records[i]!;
    const result = await predict(row, i);
    const aggregate = row.intent === null ? negative : positive;
    const family = `${row.intent === null ? "negative/" : "positive/"}${row.template}`;
    const familyCounts = (families[family] ??= counts());

    const exact =
      row.intent === null ? !result.ok : result.ok && isDeepStrictEqual(result.intent, row.intent);

    if (row.intent !== null) {
      const compiled = compile(row.intent);

      if (
        row.compileFailure
          ? compiled.supported || compiled.code !== row.compileFailure
          : !compiled.supported
      ) {
        throw new Error(`invalid compiler gold: ${row.id}`);
      }
    }

    for (const item of [aggregate, familyCounts]) {
      item.total++;
      item.accepted += Number(result.ok);
      item.exact += Number(exact);
      item.wrong += Number(result.ok && !exact);
    }

    if (!exact && failures.filter((entry) => entry.family === family).length < 3) {
      failures.push({
        text: row.text,
        expected: row.intent,
        actual: result.ok ? result.intent : { code: result.code, reason: result.reason },
        family,
      });
    }
  }

  const positiveExact = rate(positive.exact, positive.total);
  const wrongAll = rate(positive.wrong, positive.total);
  const wrongAccepted = rate(positive.wrong, positive.accepted);
  const negativeFalseAcceptance = rate(negative.accepted, negative.total);
  const supported = Object.entries(families).filter(([name]) => name.startsWith("positive/"));

  const passed =
    positive.total > 0 &&
    negative.total > 0 &&
    positiveExact >= 0.95 &&
    supported.every(([, c]) => rate(c.exact, c.total) >= 0.9) &&
    wrongAll <= 0.005 &&
    wrongAccepted <= 0.005 &&
    negativeFalseAcceptance <= 0.01;

  return {
    positive,
    negative,
    positiveExact,
    wrongAll,
    wrongAccepted,
    negativeFalseAcceptance,
    uncertainty95: {
      positiveExact: interval(positive.exact, positive.total),
      wrongAll: interval(positive.wrong, positive.total),
      wrongAccepted: interval(positive.wrong, positive.accepted),
      negativeFalseAcceptance: interval(negative.accepted, negative.total),
    },
    families,
    failures,
    syntheticGatePassed: passed,
    promotionEligible: false,
    evidence: "Generated development/synthetic evidence; reserved acceptance is separate.",
  };
}

export function readCorpus(path: string): CorpusRecord[] {
  return readFileSync(path, "utf8")
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line) as CorpusRecord);
}

export interface PredictionRow {
  index: number;
  labels: number[];
  scores: number[];
}

export async function evaluatePredictions(
  records: readonly CorpusRecord[],
  predictions: readonly PredictionRow[],
  floor = 0.5,
) {
  const byIndex = new Map(predictions.map((row) => [row.index, row]));

  if (byIndex.size !== records.length || predictions.length !== records.length) {
    throw new Error("prediction rows must exactly cover the evaluation set");
  }

  return evaluateRecords(records, async (row, i) => {
    const prediction = byIndex.get(i);

    if (!prediction) throw new Error(`missing prediction ${i}`);
    const tokens = tokenize(row.text);

    if (tokens.length !== prediction.labels.length || tokens.length !== prediction.scores.length) {
      throw new Error(`prediction token count mismatch: ${row.id}`);
    }

    const labeled = tokens.map((token, j) => ({
      token,
      role: prediction.labels[j]!,
      score: prediction.scores[j]!,
      boundary: false,
      boundaryScore: 1,
    }));

    return resultFromLabeledTokens(labeled, row.text, floor);
  });
}

if (import.meta.main) {
  const [corpus, split, artifactPath, reportPath, predictionsPath] = process.argv.slice(2);

  if (!corpus || !split || !reportPath) {
    throw new Error("usage: evaluate <corpus> <split> <artifact|-> <report> [predictions]");
  }

  const records = readCorpus(corpus).filter((row) => row.split === split);
  const weights = artifactPath && artifactPath !== "-" ? readFileSync(artifactPath) : undefined;

  const report = predictionsPath
    ? await evaluatePredictions(
        records,
        readFileSync(predictionsPath, "utf8")
          .split("\n")
          .filter(Boolean)
          .map((line) => JSON.parse(line)),
      )
    : await evaluateRecords(records, async (row) =>
        parse(row.text, { source: "model", ...(weights ? { weights } : {}) }),
      );

  writeFileSync(reportPath, JSON.stringify(report, null, 2) + "\n");

  console.log(
    JSON.stringify({
      positiveExact: report.positiveExact,
      wrongAccepted: report.wrongAccepted,
      negativeFalseAcceptance: report.negativeFalseAcceptance,
      syntheticGatePassed: report.syntheticGatePassed,
    }),
  );
}
