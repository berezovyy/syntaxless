import { describe, expect, test } from "bun:test";

import type { RenderedExample } from "../../training/src/renderer.js";
import { Sentence } from "../../training/src/sentence.js";
import { labelId } from "../model/labels.js";
import type { Label } from "../model/labels.js";
import type { SpeechAct } from "../model/speech-acts.js";
import { tokenize } from "../model/tokenizer.js";
import type { ScheduleIntent } from "../types/intent.js";
import { decodeIntent } from "./decode.js";
import type { LabeledToken } from "./decode.js";
import { resultFromLabeledTokens } from "./model.js";

function actionable(labeled: readonly LabeledToken[]): LabeledToken[] {
  return labeled.map((item) => ({
    ...item,
    actionable: item.actionable ?? true,
  }));
}

function heads(
  labeled: readonly LabeledToken[],
  scheduleKind?: Parameters<typeof decodeIntent>[1],
  sourceText?: string,
  speechAct: SpeechAct = "request",
) {
  return decodeIntent(actionable(labeled), scheduleKind, sourceText, "off", speechAct);
}

/** Turn rendered spans into perfect labels, mirroring Phase 13 alignment. */
function labelTokens(example: RenderedExample, score = 1): LabeledToken[] {
  const tokens = tokenize(example.text);
  const labeled: LabeledToken[] = [];
  let spanIndex = 0;

  for (const token of tokens) {
    if (token.kind === 3) continue;

    while (spanIndex < example.spans.length && example.spans[spanIndex]!.end <= token.start) {
      spanIndex++;
    }

    const span = example.spans[spanIndex];

    if (span === undefined || token.start < span.start || token.end > span.end) {
      throw new Error(`unaligned token: ${JSON.stringify(token.text)}`);
    }

    labeled.push({
      token,
      role: labelId[span.label],
      boundary: span.clauseStart && token.start === span.start,
      score,
    });
  }

  return labeled;
}

describe("range and interval rules", () => {
  test("labeled weekday range uses the connector and both endpoints", () => {
    const tokens = tokenize("wednesday through sunday").filter((token) => token.kind !== 3);

    const labeled: LabeledToken[] = tokens.map((token) => ({
      token,
      role:
        token.text === "through"
          ? labelId.BOUND_BETWEEN
          : token.text === "wednesday"
            ? labelId.WEEKDAY_WED
            : token.text === "sunday"
              ? labelId.WEEKDAY_SUN
              : labelId.O,
      boundary: token.text === "wednesday",
      score: 1,
    }));

    expect(heads(labeled).intent).toEqual({
      frequency: { unit: "week", interval: 1 },
      weekdays: ["wednesday", "thursday", "friday", "saturday", "sunday"],
    });
  });

  test("labeled ordinal weekday is monthly", () => {
    const tokens = tokenize("third friday monthly").filter((token) => token.kind !== 3);

    const labeled: LabeledToken[] = tokens.map((token) => ({
      token,
      role:
        token.text === "third"
          ? labelId.ORDINAL_3
          : token.text === "friday"
            ? labelId.WEEKDAY_FRI
            : token.text === "monthly"
              ? labelId.UNIT_MONTH
              : labelId.O,
      boundary: token.text === "third",
      score: 1,
    }));

    expect(heads(labeled).intent).toEqual({
      frequency: { unit: "month", interval: 1 },
      ordinalWeekday: { ordinal: "third", weekday: "friday" },
    });
  });

  test("explicit context numbers do not become intervals, and unknown digits still refuse", () => {
    const tokens = tokenize("daily for account 17").filter((token) => token.kind !== 3);

    const labeled: LabeledToken[] = tokens.map((token) => ({
      token,
      role:
        token.text === "daily"
          ? labelId.UNIT_DAY
          : token.kind === 1
            ? labelId.CONTEXT_NUMBER
            : labelId.O,
      boundary: token.text === "daily",
      score: 1,
      actionable: token.text === "daily",
    }));

    expect(heads(labeled).intent).toEqual({ frequency: { unit: "day", interval: 1 } });

    expect(() =>
      heads(
        labeled.map((item) =>
          item.role === labelId.CONTEXT_NUMBER ? { ...item, role: labelId.O } : item,
        ),
      ),
    ).toThrow(/uncounted quantity/);

    expect(
      heads(
        labeled.map((item) =>
          item.role === labelId.CONTEXT_NUMBER ? { ...item, score: 0.2 } : item,
        ),
      ).confidence,
    ).toBe(0.2);
  });
});

describe("heads-only decode", () => {
  test("at 9pm is 21:00 from HOUR and MERIDIEM labels", () => {
    const intent: ScheduleIntent = {
      frequency: { unit: "day", interval: 1 },
      time: { hour: 21, minute: 0 },
    };

    const tokens = tokenize("at 9 pm").filter((token) => token.kind !== 3);

    const labeled: LabeledToken[] = tokens.map((token) => ({
      token,
      role:
        token.text === "at"
          ? labelId.O
          : token.text === "9"
            ? labelId.HOUR_21
            : token.text === "pm"
              ? labelId.MERIDIEM_PM
              : labelId.O,
      boundary: token.text === "9",
      score: 1,
      actionable: true,
    }));

    expect(heads(labeled).intent).toEqual(intent);
  });
});

test("at <minute> past <hour> is a relative clock, not an hour after at", () => {
  const sentence = new Sentence();

  sentence.clause();
  sentence.add("at", "GLUE");
  sentence.add("1", "MINUTE_LIT");
  sentence.add("minute past", "GLUE");
  sentence.add("twelve", "HOUR_0");
  sentence.add("am", "MERIDIEM_AM");

  const decoded = heads(
    labelTokens({
      text: sentence.text,
      spans: sentence.spans,
      template: "minute-past",
      style: "minute-past",
      speechAct: "request",
    }),
  );

  expect(decoded.intent).toEqual({
    frequency: { unit: "day", interval: 1 },
    time: { hour: 0, minute: 1 },
  });
});

test("a labeled through between ISO dates is an inclusive end bound", () => {
  const sentence = new Sentence();

  sentence.clause();
  sentence.add("2026-02-14", "DOM_LIT");
  sentence.add("through", "BOUND_END_INCL");
  sentence.add("2027-12-28", "DOM_LIT");
  sentence.add(",", "GLUE", "");
  sentence.add("every", "RECUR");
  sentence.add("day", "UNIT_DAY");

  const decoded = heads(
    labelTokens({
      text: sentence.text,
      spans: sentence.spans,
      template: "fixture",
      style: "fixture",
      speechAct: "request",
    }),
  );

  expect(decoded.intent).toEqual({
    frequency: { unit: "day", interval: 1 },
    bounds: { start: "2026-02-14", end: "2027-12-29" },
  });
});

test("COUNT cannot silently default an unresolved interval or recover token digits", () => {
  const sentence = new Sentence();

  sentence.clause();
  sentence.add("every", "RECUR");
  sentence.add("5", "COUNT");
  sentence.add("minutes", "UNIT_MIN");

  expect(() =>
    heads(
      labelTokens({
        text: sentence.text,
        spans: sentence.spans,
        template: "count",
        style: "count",
        speechAct: "request",
      }),
    ),
  ).toThrow(/uncounted quantity/);
});

const invalidSchedules: Array<{
  name: string;
  parts: Array<[string, Label]>;
  reason: RegExp;
}> = [
  {
    name: "malformed hour cannot vanish",
    parts: [
      ["later", "HOUR_LIT"],
      ["daily", "UNIT_DAY"],
    ],
    reason: /invalid HOUR_LIT literal/,
  },
  {
    name: "start bound requires an operand",
    parts: [
      ["daily", "UNIT_DAY"],
      ["from", "BOUND_START"],
    ],
    reason: /incomplete schedule operator/,
  },
  {
    name: "weekday range connector requires an endpoint",
    parts: [
      ["monday", "WEEKDAY_MON"],
      ["through", "BOUND_BETWEEN"],
    ],
    reason: /incomplete weekday range/,
  },
];

test.each(invalidSchedules)("selected schedule refuses: $name", ({ parts, reason }) => {
  const sentence = new Sentence();

  sentence.clause();
  for (const [text, label] of parts) sentence.add(text, label);

  const labeled = labelTokens({
    text: sentence.text,
    spans: sentence.spans,
    template: "invalid-value",
    style: "fixture",
    speechAct: "request",
  });

  for (const input of [labeled, actionable(labeled)]) {
    expect(() => decodeIntent(input, undefined, sentence.text, "off", "request")).toThrow(reason);
    const result = resultFromLabeledTokens(input, undefined, sentence.text, "off", "request");

    expect(result.ok).toBe(false);

    if (!result.ok) {
      expect(result.code).toBe("model-error");
      expect(result.confidence).toBe(0);
      expect(result.reason).toMatch(reason);
    }
  }
});

test("a malformed value in a discarded correction clause cannot block the replacement", () => {
  const labeled: LabeledToken[] = tokenize("every many minutes; instead friday")
    .filter((token) => token.kind !== 3)
    .map((token) => ({
      token,
      role:
        token.text === "every"
          ? labelId.RECUR
          : token.text === "many"
            ? labelId.INTERVAL_LIT
            : token.text === "minutes"
              ? labelId.UNIT_MIN
              : token.text === "friday"
                ? labelId.WEEKDAY_FRI
                : labelId.O,
      boundary: token.text === "every" || token.text === "friday",
      score: 1,
      actionable: token.text === "friday",
    }));

  expect(decodeIntent(labeled, undefined, undefined, "off", "correction").intent).toEqual({
    frequency: { unit: "week", interval: 1 },
    weekdays: ["friday"],
  });
});

const conflictingFacts: Array<{
  name: string;
  splitIndex: number;
  parts: Array<[string, Label]>;
}> = [
  {
    name: "hour",
    splitIndex: 3,
    parts: [
      ["daily", "UNIT_DAY"],
      ["nine", "HOUR_9"],
      ["and", "GLUE"],
      ["ten", "HOUR_10"],
    ],
  },
  {
    name: "unit",
    splitIndex: 2,
    parts: [
      ["daily", "UNIT_DAY"],
      ["and", "GLUE"],
      ["weekly", "UNIT_WEEK"],
    ],
  },
];

test.each(conflictingFacts)(
  "conflicting selected $name facts refuse regardless of clause boundaries",
  ({ name, parts, splitIndex }) => {
    for (const split of [false, true]) {
      const sentence = new Sentence();

      sentence.clause();

      for (let index = 0; index < parts.length; index++) {
        if (split && index === splitIndex) sentence.clause();
        const [text, label] = parts[index]!;

        sentence.add(text, label);
      }

      const labeled = labelTokens({
        text: sentence.text,
        spans: sentence.spans,
        template: "conflicting-facts",
        style: "fixture",
        speechAct: "request",
      });

      for (const input of [labeled, actionable(labeled)]) {
        expect(() => decodeIntent(input, undefined, sentence.text, "off", "request")).toThrow(
          `conflicting ${name}`,
        );

        const result = resultFromLabeledTokens(input, undefined, sentence.text, "off", "request");

        expect(result.ok).toBe(false);

        if (!result.ok) {
          expect(result.code).toBe("model-error");
          expect(result.confidence).toBe(0);
          expect(result.reason).toContain(`conflicting ${name}`);
        }
      }
    }
  },
);

test("agreeing scalar facts, compact clocks and date bounds can repeat across clauses", () => {
  const sentence = new Sentence();

  for (const compact of [false, true]) {
    sentence.clause();
    sentence.add("daily", "UNIT_DAY");

    if (compact) {
      sentence.add("0930", "HOUR_LIT");
    } else {
      sentence.add("nine", "HOUR_9");
      sentence.add("half", "MINUTE_30");
    }

    sentence.add("am", "MERIDIEM_AM");
    sentence.add("from", "BOUND_START");
    sentence.add("2026-05-08", "DOM_LIT");
  }

  const decoded = heads(
    labelTokens({
      text: sentence.text,
      spans: sentence.spans,
      template: "agreeing-facts",
      style: "fixture",
      speechAct: "request",
    }),
  );

  expect(decoded.intent).toEqual({
    frequency: { unit: "day", interval: 1 },
    time: { hour: 9, minute: 30 },
    bounds: { start: "2026-05-08" },
  });
});
