import { readFileSync, writeFileSync } from "node:fs";
import { isDeepStrictEqual } from "node:util";

import { interval } from "../../../validation/training/src/evaluate.js";
import { LABELS } from "../../src/labels.js";
import { tokenize } from "../../src/model/tokenizer.js";
import { parse } from "../../src/parse.js";
import type { ParseResult } from "../../src/parse.js";
import { prepare } from "../../src/prepare.js";
import { resolveRoles } from "../../src/resolve.js";
import { readRows } from "./corpus.js";
import type { Row } from "./corpus.js";

export const FAMILIES = [
  "eq",
  "ne",
  "lt",
  "le",
  "gt",
  "ge",
  "in",
  "range",
  "null",
  "and",
  "or",
  "exclusion",
  "not",
  "group",
] as const;

export async function evaluate(
  rows: readonly Row[],
  predict: (row: Row, index: number) => Promise<ParseResult>,
) {
  const positive = { total: 0, accepted: 0, exact: 0, wrong: 0 };
  const negative = { total: 0, accepted: 0 };

  const families: Record<
    string,
    { total: number; exact: number; wrong: number; accepted: number }
  > = {};

  const failures: unknown[] = [];

  for (const [i, row] of rows.entries()) {
    const result = await predict(row, i);
    const family = (row.intent ? "positive/" : "negative/") + row.family;

    const count = (families[family] ??= { total: 0, exact: 0, wrong: 0, accepted: 0 });

    count.total++;

    const exact = row.intent
      ? result.ok && isDeepStrictEqual(result.intent, row.intent)
      : !result.ok;

    count.exact += Number(exact);
    count.accepted += Number(result.ok);
    count.wrong += Number(result.ok && !exact);

    if (row.intent) {
      positive.total++;
      positive.accepted += Number(result.ok);
      positive.exact += Number(exact);
      positive.wrong += Number(result.ok && !exact);
    } else {
      negative.total++;
      negative.accepted += Number(result.ok);
    }

    if (!exact && failures.filter((f: any) => f.family === family).length < 3) {
      failures.push({ family, text: row.text, expected: row.intent, actual: result });
    }
  }

  const positiveExact = positive.exact / positive.total;
  const wrongAll = positive.wrong / positive.total;
  const wrongAccepted = positive.wrong / Math.max(1, positive.accepted);
  const negativeFalseAcceptance = negative.accepted / negative.total;

  const syntheticGatePassed =
    positiveExact >= 0.95 &&
    wrongAll <= 0.005 &&
    wrongAccepted <= 0.005 &&
    negativeFalseAcceptance <= 0.01 &&
    FAMILIES.every((f) => {
      const c = families["positive/" + f];

      return c && c.exact / c.total >= 0.9;
    });

  return {
    positive,
    negative,
    positiveExact,
    wrongAll,
    wrongAccepted,
    negativeFalseAcceptance,
    families,
    failures,
    syntheticGatePassed,
    uncertainty95: {
      positiveExact: interval(positive.exact, positive.total),
      wrongAll: interval(positive.wrong, positive.total),
      wrongAccepted: interval(positive.wrong, positive.accepted),
      negativeFalseAcceptance: interval(negative.accepted, negative.total),
    },
    promotionEligible: false,
  };
}

if (import.meta.main) {
  const [corpus, split, artifact, out, predictionPath] = process.argv.slice(2);

  const rows = readRows(corpus!).filter((r) => r.split === split);
  const weights = artifact === "-" ? undefined : readFileSync(artifact!);

  const predictions = predictionPath
    ? readFileSync(predictionPath, "utf8")
        .trim()
        .split("\n")
        .map((l) => JSON.parse(l))
        .sort((a, b) => a.index - b.index)
    : undefined;

  const report = await evaluate(rows, async (row, i) => {
    if (weights) return parse(row.text, { ...row.context, weights });

    const prepared = prepare(row.text, row.context);
    const tokens = tokenize(prepared.text);
    const p = predictions?.[i];

    if (!p || p.index !== i) throw new Error("prediction sequence mismatch");

    const labels = prepared.lexemes.map((l) => {
      const index = tokens.findIndex((t) => t.start === l.start && t.end === l.end);
      const role = LABELS[p.labels[index]];

      if (!role) throw new Error("unknown predicted role");

      return { role, score: p.scores[index] };
    });

    try {
      const decoded = resolveRoles(prepared, labels, row.context);

      return decoded.confidence >= 0.5
        ? { ok: true, ...decoded, source: "model" }
        : { ok: false, code: "low-confidence", reason: "below floor", source: "model" };
    } catch (error) {
      return {
        ok: false,
        code: "invalid-input",
        reason: String(error),
        source: "model",
      };
    }
  });

  writeFileSync(out!, JSON.stringify(report, null, 2) + "\n");

  console.log(
    JSON.stringify({
      positiveExact: report.positiveExact,
      wrongAccepted: report.wrongAccepted,
      negativeFalseAcceptance: report.negativeFalseAcceptance,
      syntheticGatePassed: report.syntheticGatePassed,
    }),
  );
}
