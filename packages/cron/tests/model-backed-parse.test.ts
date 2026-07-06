/**
 * Model-backed parse tests (Phase 18).
 *
 * The parity rows draw on the Phase 04 phrase table. The rule path is the
 * baseline: every phrase must parse to its table intent. The model path must
 * either agree exactly or fail explicitly (`low-confidence` / `model-error`);
 * it never returns a partial or guessed intent. The bundled Phase 15 weights
 * are small, so the parity assertion is stated against the model's explicit
 * contract rather than weakened: a mismatch that still returns `ok: true` is a
 * failure.
 *
 * The low-confidence and unknown-role rows exercise the pure mapping in
 * `resultFromLabeledTokens`, which is where the band and role validation live.
 */
import { describe, expect, test } from "bun:test";

import { parse } from "../src/index.ts";
import type { ParseResult, ScheduleIntent } from "../src/index.ts";
import { LABEL_COUNT, labelId } from "../src/model/labels.ts";
import { tokenize } from "../src/model/tokenizer.ts";
import type { LabeledToken } from "../src/parser/decode.ts";
import {
  classifyConfidence,
  CONFIDENCE_THRESHOLDS,
  labelTokens,
  resultFromLabeledTokens,
} from "../src/parser/index.ts";

const WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday"] as const;

const phraseCases: Array<{ phrase: string; intent: ScheduleIntent }> = [
  { phrase: "every day", intent: { frequency: { unit: "day", interval: 1 } } },
  {
    phrase: "every weekday",
    intent: { frequency: { unit: "week", interval: 1 }, weekdays: [...WEEKDAYS] },
  },
  {
    phrase: "every Monday",
    intent: { frequency: { unit: "week", interval: 1 }, weekdays: ["monday"] },
  },
  {
    phrase: "every Monday and Friday",
    intent: {
      frequency: { unit: "week", interval: 1 },
      weekdays: ["monday", "friday"],
    },
  },
  { phrase: "every hour", intent: { frequency: { unit: "hour", interval: 1 } } },
  {
    phrase: "every 30 minutes",
    intent: { frequency: { unit: "minute", interval: 30 } },
  },
  {
    phrase: "at 9",
    intent: { frequency: { unit: "day", interval: 1 }, time: { hour: 9, minute: 0 } },
  },
  {
    phrase: "at 9:30",
    intent: { frequency: { unit: "day", interval: 1 }, time: { hour: 9, minute: 30 } },
  },
  {
    phrase: "at 9pm",
    intent: { frequency: { unit: "day", interval: 1 }, time: { hour: 21, minute: 0 } },
  },
  {
    phrase: "at nine",
    intent: { frequency: { unit: "day", interval: 1 }, time: { hour: 9, minute: 0 } },
  },
  {
    phrase: "at twelve am",
    intent: { frequency: { unit: "day", interval: 1 }, time: { hour: 0, minute: 0 } },
  },
  {
    phrase: "every other Friday",
    intent: { frequency: { unit: "week", interval: 2 }, weekdays: ["friday"] },
  },
  {
    phrase: "Monday through Friday",
    intent: { frequency: { unit: "week", interval: 1 }, weekdays: [...WEEKDAYS] },
  },
  {
    phrase: "the 15th",
    intent: { frequency: { unit: "month", interval: 1 }, dayOfMonth: 15 },
  },
  {
    phrase: "first monday",
    intent: {
      frequency: { unit: "month", interval: 1 },
      ordinalWeekday: { ordinal: "first", weekday: "monday" },
    },
  },
];

describe("model-backed parse parity with the Phase 04 table", () => {
  test.each(phraseCases)("$phrase", async ({ phrase, intent }) => {
    const rules = await parse(phrase, { source: "rules" });

    expect(rules.ok).toBe(true);
    if (!rules.ok) return;
    expect(rules.intent).toEqual(intent);
    expect(rules.source).toBe("rules");
    expect(rules.confidence).toBe(1);
    expect(rules.diagnostics).toEqual([]);

    const model = await parse(phrase);

    expect(model.source).toBe("model");

    if (model.ok) {
      // A success must be the exact rule intent, never a near miss.
      expect(model.intent).toEqual(rules.intent);
      expect(model.confidence).toBeGreaterThanOrEqual(CONFIDENCE_THRESHOLDS.medium);
    } else {
      // A model that cannot understand a phrase fails explicitly and guesses
      // nothing: the failure carries no intent and a specific failure code.
      expect(["low-confidence", "model-error"]).toContain(model.code);
      expect(model).not.toHaveProperty("intent");
    }
  });
});

describe("exact lexical schedule and rejection surfaces", () => {
  test.each(["MON", "mon", "mOn", "the first day of the week", "on the first day of the week"])(
    "%s resolves to Monday after normalization",
    async (phrase) => {
      const result = await parse(phrase);

      expect(result.ok).toBe(true);
      if (!result.ok) return;

      expect(result.intent).toEqual({
        frequency: { unit: "week", interval: 1 },
        weekdays: ["monday"],
      });
    },
  );

  test("a bare exact clock does not depend on model confidence", async () => {
    const result = await parse("at 9");

    expect(result.ok).toBe(true);
    if (!result.ok) return;

    expect(result.intent).toEqual({
      frequency: { unit: "day", interval: 1 },
      time: { hour: 9, minute: 0 },
    });
  });

  test("an exact ordinal weekday remains monthly", async () => {
    const result = await parse("first Monday");

    expect(result.ok).toBe(true);
    if (!result.ok) return;

    expect(result.intent).toEqual({
      frequency: { unit: "month", interval: 1 },
      ordinalWeekday: { ordinal: "first", weekday: "monday" },
    });
  });

  test("a missing space cannot silently erase an explicit weekday group", async () => {
    const result = await parse("every1 weekday at 9");

    expect(result.ok).toBe(true);
    if (!result.ok) return;

    expect(result.intent).toEqual({
      frequency: { unit: "week", interval: 1 },
      weekdays: [...WEEKDAYS],
      time: { hour: 9, minute: 0 },
    });
  });

  test.each([
    ["starting 2026-05-26, every day", { start: "2026-05-26" }],
    ["ending 2027-01-20, every day", { end: "2027-01-20" }],
    ["from 2026-02-14 through 2027-12-28, every day", { start: "2026-02-14", end: "2027-12-29" }],
    [
      "between March 2, 2026 and April 4, 2026 every day",
      { start: "2026-03-02", end: "2026-04-04" },
    ],
  ] as const)("preserves explicit bounds in %s", async (phrase, bounds) => {
    const result = await parse(phrase);

    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.intent.bounds).toEqual(bounds);
  });

  test("an invalid explicit bound fails instead of becoming an unbounded schedule", async () => {
    const result = await parse("every day starting 2026-02-30");

    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.code).toBe("model-error");
  });

  test.each([
    "I did not mean every hour",
    "Maybe Tuesday or Wednesday",
    "How does an hourly schedule work?",
    "At half past",
    "The last Friday of",
    "Weekdays except",
    "Monthly on the",
    "Nine people joined on Tuesday",
    "The deadline moved from Monday to Friday",
    "This report is produced every month by another system",
    "Forget what I said about Friday",
    "Should this happen every week?",
    "Who changed it to Tuesday?",
    "Why did this run at midnight?",
    "The meeting happened Monday at nine",
    "I finished the report last Friday",
    "Alex arrived at half past eight",
    "The office closes on weekends",
    "I used to call every Sunday",
  ])("%s is explicitly rejected", async (phrase) => {
    const result = await parse(phrase);

    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result).not.toHaveProperty("intent");
  });
});

describe("confidence bands", () => {
  const labeledWith = (text: string, roles: number[], score: number): LabeledToken[] => {
    const tokens = tokenize(text).filter((token) => token.kind !== 3);

    return tokens.map((token, index) => ({
      token,
      role: roles[index]!,
      boundary: index === 0,
      score,
    }));
  };

  test("classifies high, medium, and low pure scores", () => {
    expect(classifyConfidence(1)).toBe("high");
    expect(classifyConfidence(CONFIDENCE_THRESHOLDS.high)).toBe("high");
    expect(classifyConfidence(0.7)).toBe("medium");
    expect(classifyConfidence(CONFIDENCE_THRESHOLDS.medium)).toBe("medium");
    expect(classifyConfidence(0.2)).toBe("low");
  });

  test("a low band clears the intent and carries a warning", () => {
    // every 30 minutes -> frequency minute/30; the INTERVAL token is content.
    const labeled = labeledWith(
      "every 30 minutes",
      [labelId.RECUR, labelId.INTERVAL, labelId.GLUE],
      0.01,
    );

    const result: ParseResult = resultFromLabeledTokens(labeled);

    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.code).toBe("low-confidence");
    expect(result).not.toHaveProperty("intent");
    expect(result.confidence).toBeLessThan(CONFIDENCE_THRESHOLDS.medium);
    expect(result.diagnostics.some((note) => note.severity === "warning")).toBe(true);
  });

  test("a medium band keeps the intent and attaches a suggestion", () => {
    const labeled = labeledWith(
      "every 30 minutes",
      [labelId.RECUR, labelId.INTERVAL, labelId.GLUE],
      0.7,
    );

    const result = resultFromLabeledTokens(labeled);

    expect(result.ok).toBe(true);

    if (result.ok) {
      expect(result.intent).toEqual({ frequency: { unit: "minute", interval: 30 } });
      expect(result.diagnostics.some((note) => note.severity === "suggestion")).toBe(true);
    }
  });

  test("a high band is clean", () => {
    const labeled = labeledWith(
      "every 30 minutes",
      [labelId.RECUR, labelId.INTERVAL, labelId.GLUE],
      0.99,
    );

    const result = resultFromLabeledTokens(labeled);

    expect(result.ok).toBe(true);
    if (result.ok) expect(result.diagnostics).toEqual([]);
  });
});

describe("unknown role ids", () => {
  test("an out-of-range role yields model-error with no intent", () => {
    const labeled = labelTokens(tokenize("every day"));

    const corrupt = labeled.map((item, index) =>
      index === 0 ? { ...item, role: LABEL_COUNT } : item,
    );

    const result = resultFromLabeledTokens(corrupt);

    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.code).toBe("model-error");
    expect(result).not.toHaveProperty("intent");
    expect(result.diagnostics.some((note) => note.severity === "error")).toBe(true);
  });

  test("an unrecognized schedule is a model-error, not a guess", () => {
    const labeled = labelTokens(tokenize("blah blah"));
    const result = resultFromLabeledTokens(labeled);

    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.code).toBe("model-error");
    expect(result).not.toHaveProperty("intent");
  });
});
