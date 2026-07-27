import { expect, test } from "bun:test";

import { interval } from "../../../shared/training/statistics.ts";
import { parse } from "../../src/index.ts";
import type { ParseResult } from "../../src/parser/parse.ts";
import { evaluateRecords } from "../src/evaluate.ts";
import type { CorpusRecord } from "../src/generate.ts";

test(
  "intent metrics count refusals, wrong accepted bounds, " + "and false acceptance separately",
  async () => {
    const rows: CorpusRecord[] = [
      {
        text: "required number more than 10",
        intent: {
          type: "number",
          required: true,
          minimum: { value: 10, inclusive: false },
        },
      },
      { text: "optional email", intent: { type: "email", required: false } },
      { text: "required boolean", intent: { type: "boolean", required: true } },
      { text: "what does required email mean?", intent: null },
      { text: "send an email", intent: null },
    ].map((row, i) => ({
      ...row,
      id: `metric-${i}`,
      family: row.intent ? "constraint" : "negative",
      split: "heldout",
      seed: 1,
      template: row.intent?.type ?? "question",
      fingerprint: `group-${i}`,
      spans: [],
    })) as CorpusRecord[];

    const correct = await evaluateRecords(rows, (row) => parse(row.text, { source: "rules" }));

    expect(correct.positive).toEqual({ total: 3, accepted: 3, exact: 3, wrong: 0 });
    expect(correct.negative.accepted).toBe(0);

    const broken = await evaluateRecords(rows, async (row, index): Promise<ParseResult> => {
      if (index === 1) {
        return {
          ok: false,
          code: "low-confidence",
          reason: "refused",
          confidence: 0,
          diagnostics: [],
          source: "model",
        };
      }

      if (index === 0 || index === 3) {
        return {
          ok: true,
          intent: {
            type: "number",
            required: true,
            minimum: { value: 10, inclusive: true },
          },
          confidence: 1,
          diagnostics: [],
          source: "model",
        };
      }

      return parse(row.text, { source: "rules" });
    });

    expect(broken.positive).toEqual({ total: 3, accepted: 2, exact: 1, wrong: 1 });
    expect(broken.negative).toEqual({ total: 2, accepted: 1, exact: 1, wrong: 1 });
    expect(broken.wrongAll).toBe(1 / 3);
    expect(broken.wrongAccepted).toBe(1 / 2);
    expect(broken.positiveExact).toBe(1 / 3);
    expect(broken.syntheticGatePassed).toBe(false);
    expect(broken.promotionEligible).toBe(false);
    expect(interval(0, 600)[1]).toBeGreaterThan(0);
  },
);
