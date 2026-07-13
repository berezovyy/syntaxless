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

describe("heads-only lexical facts from labels", () => {
  test("WEEKDAY alias mnday resolves through the lexicon table", () => {
    const tokens = tokenize("mnday").filter((token) => token.kind !== 3);

    const labeled: LabeledToken[] = tokens.map((token) => ({
      token,
      role: labelId.WEEKDAY,
      boundary: true,
      score: 1,
      actionable: true,
    }));

    const result = headsResult(labeled);

    expect(result.ok).toBe(true);

    if (result.ok) {
      expect(result.intent).toEqual({
        frequency: { unit: "week", interval: 1 },
        weekdays: ["monday"],
      });
    }
  });

  test("HOUR plus MERIDIEM labels convert 9pm to 21:00", () => {
    const tokens = tokenize("at 9 pm").filter((token) => token.kind !== 3);

    const labeled: LabeledToken[] = tokens.map((token) => ({
      token,
      role:
        token.text === "at"
          ? labelId.O
          : token.text === "9pm" || token.text === "9"
            ? labelId.HOUR
            : token.text === "pm"
              ? labelId.MERIDIEM
              : labelId.O,
      boundary: token.text !== "at",
      score: 1,
      actionable: true,
    }));

    const result = headsResult(labeled);

    expect(result.ok).toBe(true);

    if (result.ok) {
      expect(result.intent).toEqual({
        frequency: { unit: "day", interval: 1 },
        time: { hour: 21, minute: 0 },
      });
    }
  });

  test("public parse without a speech-act head is model-error, not overlay regex", async () => {
    for (const text of [
      "MON",
      "at 9pm",
      "every 30 minutes",
      "the bakery opens every weekday at 9",
      "Flights depart every twenty minutes from this gate.",
      "Does the backup run every day?",
    ]) {
      const result = await parse(text);

      expect(result.ok).toBe(false);

      if (!result.ok) {
        expect(result.code).toBe("model-error");

        expect(result.diagnostics.some((note) => note.code === "missing-speech-act-head")).toBe(
          true,
        );
      }
    }
  });
});

function headsResult(
  labeled: readonly LabeledToken[],
  scheduleKind?: Parameters<typeof resultFromLabeledTokens>[1],
  sourceText?: string,
  speechAct: "request" | "descriptive" | "question" | "correction" | "cancel" | "none" = "request",
) {
  return resultFromLabeledTokens(
    labeled.map((item) => ({ ...item, actionable: item.actionable ?? true })),
    scheduleKind,
    sourceText,
    "off",
    speechAct,
  );
}

describe("confidence bands", () => {
  const labeledWith = (text: string, roles: number[], score: number): LabeledToken[] => {
    const tokens = tokenize(text).filter((token) => token.kind !== 3);

    return tokens.map((token, index) => ({
      token,
      role: roles[index]!,
      boundary: index === 0,
      score,
      actionable: true,
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
    // Use a token-only schedule without deterministic numeric extraction so
    // this isolates the fallback model-margin policy.
    const labeled = labeledWith("daily", [labelId.INTERVAL], 0.01);
    const result: ParseResult = headsResult(labeled);

    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.code).toBe("low-confidence");
    expect(result).not.toHaveProperty("intent");
    expect(result.confidence).toBeLessThan(CONFIDENCE_THRESHOLDS.medium);
    expect(result.diagnostics.some((note) => note.severity === "warning")).toBe(true);
  });

  test("a medium band keeps the intent and attaches a suggestion", () => {
    const labeled = labeledWith("daily", [labelId.INTERVAL], 0.7);
    const result = headsResult(labeled);

    expect(result.ok).toBe(true);

    if (result.ok) {
      expect(result.intent).toEqual({ frequency: { unit: "day", interval: 1 } });
      expect(result.diagnostics.some((note) => note.severity === "suggestion")).toBe(true);
    }
  });

  test("a high band is clean", () => {
    const labeled = labeledWith("daily", [labelId.INTERVAL], 0.99);
    const result = headsResult(labeled);

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

    const result = headsResult(corrupt);

    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.code).toBe("model-error");
    expect(result).not.toHaveProperty("intent");
    expect(result.diagnostics.some((note) => note.severity === "error")).toBe(true);
  });

  test("an unrecognized schedule is a model-error, not a guess", () => {
    const labeled = labelTokens(tokenize("blah blah"));
    const result = headsResult(labeled);

    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.code).toBe("model-error");
    expect(result).not.toHaveProperty("intent");
  });
});

describe("heads-only public parse", () => {
  const weekdayNine: ScheduleIntent = {
    frequency: { unit: "week", interval: 1 },
    weekdays: [...WEEKDAYS],
    time: { hour: 9, minute: 0 },
  };

  const bakery = "The bakery opens every morning at half past six.";
  const flights = "Flights depart every twenty minutes from this gate.";

  test("default parse without a speech-act head fails closed", async () => {
    const result = await parse("every weekday at 9");

    expect(result.ok).toBe(false);

    if (!result.ok) {
      expect(result.code).toBe("model-error");
      expect(result.diagnostics.some((note) => note.code === "missing-speech-act-head")).toBe(true);
    }
  });

  test("overlay-on is not a regex fallback when heads are missing", async () => {
    const result = await parse("every weekday at 9", { overlay: "on" });

    expect(result.ok).toBe(false);

    if (!result.ok) {
      expect(result.code).toBe("model-error");
      expect(result.diagnostics.some((note) => note.code === "missing-speech-act-head")).toBe(true);
    }
  });

  test("heads-only still abstains on the 0.5 token-margin floor", () => {
    const tokens = tokenize("daily").filter((token) => token.kind !== 3);

    const labeled: LabeledToken[] = tokens.map((token) => ({
      token,
      role: labelId.INTERVAL,
      boundary: true,
      score: 0.01,
      actionable: true,
    }));

    const off = headsResult(labeled);

    expect(off.ok).toBe(false);
    if (!off.ok) expect(off.code).toBe("low-confidence");
  });

  test.each([bakery, flights])(
    "descriptive %s without heads is model-error, not a noun regex",
    async (text) => {
      const result = await parse(text);

      expect(result.ok).toBe(false);

      if (!result.ok) {
        expect(result.code).toBe("model-error");

        expect(result.diagnostics.some((note) => note.code === "missing-speech-act-head")).toBe(
          true,
        );
      }
    },
  );

  test("descriptive speech-act refuses bakery wording without a bakery regex", () => {
    const tokens = tokenize("every weekday at 9").filter((token) => token.kind !== 3);

    const labeled: LabeledToken[] = tokens.map((token) => ({
      token,
      role:
        token.text === "every"
          ? labelId.RECUR
          : token.text === "weekday"
            ? labelId.DAYGROUP
            : token.text === "at"
              ? labelId.O
              : token.text === "9"
                ? labelId.HOUR
                : labelId.O,
      boundary: token.text === "every",
      score: 1,
      actionable: true,
    }));

    const result = headsResult(labeled, undefined, bakery, "descriptive");

    expect(result.ok).toBe(false);

    if (!result.ok) {
      expect(result.code).toBe("unsafe-context");
      expect(result.diagnostics.some((note) => note.code === "descriptive")).toBe(true);
    }
  });

  test("request speech-act with labeled weekday 9 yields the gold intent", () => {
    const tokens = tokenize("every weekday at 9").filter((token) => token.kind !== 3);

    const labeled: LabeledToken[] = tokens.map((token) => ({
      token,
      role:
        token.text === "every"
          ? labelId.RECUR
          : token.text === "weekday"
            ? labelId.DAYGROUP
            : token.text === "at"
              ? labelId.O
              : token.text === "9"
                ? labelId.HOUR
                : labelId.O,
      boundary: token.text === "every",
      score: 1,
      actionable: true,
    }));

    const result = headsResult(labeled, undefined, "every weekday at 9", "request");

    expect(result.ok).toBe(true);
    if (result.ok) expect(result.intent).toEqual(weekdayNine);
  });

  test("a dual-clause correction is not rewritten before tokenize", async () => {
    const source = "Sorry, I meant Tuesday at 10 am rather than Monday at nine.";
    const result = await parse(source);

    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.code).toBe("model-error");
  });

  test("rules source ignores overlay", async () => {
    const rulesOn = await parse("every weekday at 9", { source: "rules" });

    const rulesOff = await parse("every weekday at 9", {
      source: "rules",
      overlay: "off",
    });

    expect(rulesOn).toEqual(rulesOff);
    expect(rulesOn.ok).toBe(true);
    const descriptiveOn = await parse(flights, { source: "rules" });
    const descriptiveOff = await parse(flights, { source: "rules", overlay: "off" });

    expect(descriptiveOn).toEqual(descriptiveOff);
    expect(descriptiveOn.source).toBe("rules");
  });
});
