/**
 * Decoder oracle: rendered phrase + perfect spans -> original intent.
 *
 * Every row is drawn from the Phase 03 representability table. The phrase is
 * produced by the Phase 07 renderer, its recorded spans are turned into perfect
 * `LabeledToken`s exactly the way Phase 13 featurization does, and
 * `decodeIntent` must rebuild the table intent byte for byte. A label the
 * decoder mishandles fails the row.
 */
import { describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";

import { renderIntent } from "../../training/src/renderer.js";
import type { RenderedExample } from "../../training/src/renderer.js";
import type { Rng } from "../../training/src/rng.js";
import { Sentence } from "../../training/src/sentence.js";
import type { Span } from "../../training/src/sentence.js";
import { LABEL_COUNT, labelId } from "../model/labels.js";
import { speechActFromId } from "../model/speech-acts.js";
import type { SpeechAct } from "../model/speech-acts.js";
import { tokenize } from "../model/tokenizer.js";
import type { ScheduleIntent, Weekday } from "../types/intent.js";
import { decodeIntent, UnknownRoleError, UnsafeScheduleContextError } from "./decode.js";
import type { LabeledToken } from "./decode.js";
import { WEEKDAY_SEQUENCE } from "./lexicon.js";
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

/** Deterministic rng that never emits carrier prose, so phrases stay canonical. */
const noCarrier: Rng = {
  next: () => 0,
  int: () => 0,
  pick: (items) => items[0]!,
  weighted: (items) => items[0]![0],
  bool: () => false,
};

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

/** Concatenate rendered pieces into one multi-clause phrase, offsetting spans. */
function concat(pieces: readonly RenderedExample[]): RenderedExample {
  let text = "";
  const spans: Span[] = [];

  for (const piece of pieces) {
    if (text.length > 0) text += " ";
    const offset = text.length;

    text += piece.text;

    for (const span of piece.spans) {
      spans.push({ ...span, start: span.start + offset, end: span.end + offset });
    }
  }

  return {
    text,
    spans,
    template: pieces.map((piece) => piece.template).join("+"),
    style: "combined",
    speechAct: "request",
  };
}

function isContiguousRun(days: readonly Weekday[]): boolean {
  if (days.length < 2) return false;
  const start = WEEKDAY_SEQUENCE.indexOf(days[0]!);

  return days.every((day, index) => WEEKDAY_SEQUENCE[(start + index) % 7] === day);
}

function selectorPiece(intent: ScheduleIntent): RenderedExample | undefined {
  if (intent.ordinalWeekday !== undefined) {
    return renderIntent(intent, "ordinal-weekday", noCarrier, "the-ordinal-of");
  }

  if (intent.dayOfMonth !== undefined) {
    return renderIntent(intent, "day-of-month", noCarrier, "on-the");
  }

  if (intent.weekdays !== undefined) {
    const contiguous = isContiguousRun(intent.weekdays);

    return renderIntent(
      intent,
      contiguous ? "weekday-range" : "weekday-list",
      noCarrier,
      contiguous ? "through" : "on-days",
    );
  }

  return undefined;
}

/** Render every intent field the decoder is expected to recover. */
function renderPhrase(intent: ScheduleIntent): RenderedExample {
  const pieces: RenderedExample[] = [];

  if (intent.bounds !== undefined) {
    pieces.push(renderIntent(intent, "bounds", noCarrier, "from-to"));
  } else {
    if (intent.frequency.interval > 1) {
      pieces.push(renderIntent(intent, "interval", noCarrier, "every"));
    }

    const selector = selectorPiece(intent);

    if (selector !== undefined) {
      pieces.push(selector);
    } else if (intent.frequency.interval <= 1) {
      pieces.push(renderIntent(intent, "interval", noCarrier, "every"));
    }
  }

  if (intent.time !== undefined && intent.time.hour !== undefined) {
    pieces.push(renderIntent(intent, "time-of-day", noCarrier, "digits"));
  }

  return concat(pieces);
}

const weekdays = (days: Weekday[], time?: { hour: number; minute: number }): ScheduleIntent => ({
  frequency: { unit: "week", interval: 1 },
  weekdays: days,
  ...(time === undefined ? {} : { time }),
});

/**
 * Phase 03 representability table. The `every-2-hours-at-15` row is omitted: its
 * `time` has no `hour`, which the time renderer cannot produce.
 */
const oracle: Array<{ name: string; intent: ScheduleIntent }> = [
  { name: "every-minute", intent: { frequency: { unit: "minute", interval: 1 } } },
  { name: "every-30-minutes", intent: { frequency: { unit: "minute", interval: 30 } } },
  { name: "every-hour", intent: { frequency: { unit: "hour", interval: 1 } } },
  { name: "every-day", intent: { frequency: { unit: "day", interval: 1 } } },
  {
    name: "every-day-at-9",
    intent: { frequency: { unit: "day", interval: 1 }, time: { hour: 9, minute: 0 } },
  },
  { name: "every-3-days", intent: { frequency: { unit: "day", interval: 3 } } },
  { name: "monday", intent: weekdays(["monday"], { hour: 9, minute: 0 }) },
  {
    name: "monday-friday",
    intent: weekdays(["monday", "tuesday", "wednesday", "thursday", "friday"], {
      hour: 9,
      minute: 0,
    }),
  },
  { name: "monday-and-friday", intent: weekdays(["monday", "friday"], { hour: 9, minute: 0 }) },
  {
    name: "friday-sunday",
    intent: weekdays(["friday", "saturday", "sunday"], { hour: 9, minute: 0 }),
  },
  { name: "every-week", intent: { frequency: { unit: "week", interval: 1 } } },
  { name: "every-other-week", intent: { frequency: { unit: "week", interval: 2 } } },
  {
    name: "first-of-month",
    intent: {
      frequency: { unit: "month", interval: 1 },
      dayOfMonth: 1,
      time: { hour: 9, minute: 0 },
    },
  },
  {
    name: "month-day-15",
    intent: {
      frequency: { unit: "month", interval: 1 },
      dayOfMonth: 15,
      time: { hour: 9, minute: 0 },
    },
  },
  { name: "every-month", intent: { frequency: { unit: "month", interval: 1 } } },
  {
    name: "every-2-months",
    intent: { frequency: { unit: "month", interval: 2 }, dayOfMonth: 1 },
  },
  {
    name: "first-monday",
    intent: {
      frequency: { unit: "month", interval: 1 },
      ordinalWeekday: { ordinal: "first", weekday: "monday" },
    },
  },
  {
    name: "last-friday",
    intent: {
      frequency: { unit: "month", interval: 1 },
      ordinalWeekday: { ordinal: "last", weekday: "friday" },
    },
  },
  {
    name: "bounded",
    intent: {
      frequency: { unit: "day", interval: 1 },
      bounds: { start: "2026-01-01" },
    },
  },
];

describe("perfect-label oracle over the Phase 03 table", () => {
  test.each(oracle)("$name decodes back to its intent", ({ intent }) => {
    const example = renderPhrase(intent);

    expect(example.text.length).toBeGreaterThan(0);
    const decoded = heads(labelTokens(example));

    expect(decoded.intent).toEqual(intent);
  });
});

describe("range and interval rules", () => {
  test("monday through friday expands to an ordered weekday list", () => {
    const intent = weekdays(["monday", "tuesday", "wednesday", "thursday", "friday"]);
    const example = renderIntent(intent, "weekday-range", noCarrier, "through");

    expect(example.text).toBe("monday through friday");
    expect(heads(labelTokens(example)).intent).toEqual(intent);
  });

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

  test("all-O labels do not recover a schedule from raw English", () => {
    const tokens = tokenize("remind me to the 24th day, monthly").filter(
      (token) => token.kind !== 3,
    );

    const labeled: LabeledToken[] = tokens.map((token) => ({
      token,
      role: labelId.O,
      boundary: false,
      score: 1,
    }));

    expect(() => heads(labeled)).toThrow(/no-actionable-span|no recognizable schedule/);
  });

  test("every other friday multiplies the interval by two", () => {
    const tokens = tokenize("every other friday").filter((token) => token.kind !== 3);
    const roles = [labelId.RECUR, labelId.INTERVAL_2, labelId.WEEKDAY_FRI];

    const labeled: LabeledToken[] = tokens.map((token, index) => ({
      token,
      role: roles[index]!,
      boundary: index === 0,
      score: 1,
    }));

    expect(heads(labeled).intent).toEqual({
      frequency: { unit: "week", interval: 2 },
      weekdays: ["friday"],
    });
  });

  test("INTERVAL plus unit GLUE keeps the number out of the clock", () => {
    const tokens = tokenize("every 43 minutes").filter((token) => token.kind !== 3);

    const labeled: LabeledToken[] = tokens.map((token) => ({
      token,
      role:
        token.text === "every"
          ? labelId.RECUR
          : token.text === "43"
            ? labelId.INTERVAL_LIT
            : token.text === "minutes"
              ? labelId.UNIT_MIN
              : labelId.O,
      boundary: token.text === "every",
      score: 1,
    }));

    expect(heads(labeled).intent).toEqual({
      frequency: { unit: "minute", interval: 43 },
    });
  });

  test("labeled EXCEPT keeps the exception weekday", () => {
    const tokens = tokenize("daily except friday").filter((token) => token.kind !== 3);

    const labeled: LabeledToken[] = tokens.map((token) => ({
      token,
      role:
        token.text === "daily"
          ? labelId.UNIT_DAY
          : token.text === "except"
            ? labelId.EXCEPT
            : token.text === "friday"
              ? labelId.WEEKDAY_FRI
              : labelId.O,
      boundary: token.text === "daily",
      score: 1,
    }));

    expect(heads(labeled).intent).toEqual({
      frequency: { unit: "day", interval: 1 },
      exceptions: { weekdays: ["friday"] },
    });
  });

  test("weekday selectors constrain a mistaken daily unit", () => {
    const weekday = labelTokens(
      renderIntent(
        { frequency: { unit: "week", interval: 1 }, weekdays: ["monday"] },
        "weekday-list",
        noCarrier,
        "on-days",
      ),
    ).map((item) => (item.role === labelId.GLUE ? item : { ...item, role: labelId.WEEKDAY_MON }));

    const daily = tokenize("daily").filter((token) => token.kind !== 3)[0]!;

    expect(
      heads([{ token: daily, role: labelId.UNIT_DAY, boundary: true, score: 1 }, ...weekday]).intent
        .frequency.unit,
    ).toBe("week");

    const monthly = labelTokens(
      renderIntent(
        { frequency: { unit: "month", interval: 1 }, dayOfMonth: 15 },
        "day-of-month",
        noCarrier,
        "on-the",
      ),
    );

    expect(heads(monthly).intent.frequency.unit).toBe("month");
  });
});

describe("confidence", () => {
  test("is the weakest margin among content tokens", () => {
    const intent: ScheduleIntent = { frequency: { unit: "day", interval: 1 } };
    const token = tokenize("daily")[0]!;

    const labeled: LabeledToken[] = [
      { token, role: labelId.UNIT_DAY, boundary: true, score: 0.05 },
    ];

    const decoded = heads(labeled);

    expect(decoded.intent).toEqual(intent);
    expect(decoded.confidence).toBeCloseTo(0.05);
  });

  test("does not treat an uncalibrated kind margin as sequence confidence", () => {
    const intent: ScheduleIntent = { frequency: { unit: "day", interval: 3 } };

    const decoded = heads(labelTokens(renderPhrase(intent)), {
      kind: "interval",
      score: 0.72,
    });

    expect(decoded.intent).toEqual(intent);
    expect(decoded.confidence).toBe(1);
  });

  test("confidence equals tokenMargin, never 1 from a raw-string hit", () => {
    const labeled = labelTokens(
      renderPhrase({
        frequency: { unit: "week", interval: 1 },
        weekdays: ["monday", "tuesday", "wednesday", "thursday", "friday"],
        time: { hour: 9, minute: 0 },
      }),
      0.72,
    );

    const decoded = heads(labeled);

    expect(decoded.evidence.tokenMargin).toBeCloseTo(0.72);
    expect(decoded.confidence).toBe(decoded.evidence.tokenMargin);
    expect(decoded.confidence).not.toBe(1);
  });
});

describe("global schedule-kind safety", () => {
  test("token labels win over a contradicting global kind", () => {
    const labeled = tokenize("kindly execute daily")
      .filter((token) => token.kind !== 3)
      .map((token) => ({
        token,
        role: token.text === "daily" ? labelId.UNIT_DAY : labelId.O,
        boundary: token.text === "daily",
        score: 1,
      }));

    expect(heads(labeled, { kind: "weekly", score: 0.9 }).intent).toEqual({
      frequency: { unit: "day", interval: 1 },
    });
  });

  test("a confident none kind does not veto labeled INTERVAL", () => {
    const labeled = tokenize("kindly execute daily")
      .filter((token) => token.kind !== 3)
      .map((token) => ({
        token,
        role: token.text === "daily" ? labelId.UNIT_DAY : labelId.O,
        boundary: token.text === "daily",
        score: 1,
      }));

    expect(heads(labeled, { kind: "none", score: 0.9 }).intent).toEqual({
      frequency: { unit: "day", interval: 1 },
    });
  });

  test("an uncertain global kind does not override explicit token evidence", () => {
    const intent: ScheduleIntent = { frequency: { unit: "day", interval: 3 } };

    const decoded = heads(labelTokens(renderPhrase(intent)), {
      kind: "weekly",
      score: 0.2,
    });

    expect(decoded.intent).toEqual(intent);
  });
});

describe("decodeIntent rejects corrupt input", () => {
  const base = labelTokens(renderPhrase({ frequency: { unit: "day", interval: 1 } }));

  test("a role id at LABEL_COUNT is an unknown role", () => {
    expect(() => heads([{ ...base[0]!, role: LABEL_COUNT }])).toThrow(UnknownRoleError);
  });

  test("a negative role id is an unknown role", () => {
    expect(() => heads([{ ...base[0]!, role: -1 }])).toThrow(UnknownRoleError);
  });

  test("a non-integer role id is an unknown role", () => {
    expect(() => heads([{ ...base[0]!, role: 1.5 }])).toThrow(UnknownRoleError);
  });

  test("an out-of-bounds day of month is rejected before compile", () => {
    const sentence = new Sentence();

    sentence.clause();
    sentence.add("32", "DOM_LIT");
    sentence.add("monthly", "UNIT_MONTH");

    expect(() =>
      heads(
        labelTokens({
          text: sentence.text,
          spans: sentence.spans,
          template: "dom-32",
          style: "dom-32",
          speechAct: "request",
        }),
      ),
    ).toThrow(RangeError);
  });

  test("heads-only still throws UnknownRoleError on an out-of-range role", () => {
    expect(() => heads([{ ...base[0]!, role: LABEL_COUNT, actionable: true }])).toThrow(
      UnknownRoleError,
    );
  });
});

describe("heads-only decode", () => {
  const descriptive = "This is a descriptive wrap of a timetable sentence.";

  const weekdayNine = weekdays(["monday", "tuesday", "wednesday", "thursday", "friday"], {
    hour: 9,
    minute: 0,
  });

  test("missing speech-act is a model error, never a regex fallback", () => {
    const labeled = labelTokens(renderPhrase(weekdayNine));

    expect(() => decodeIntent(labeled, undefined, descriptive)).toThrow(
      /heads-only requires a speech-act/,
    );
  });

  test("request still skips the discourse regex catalog", () => {
    const labeled = labelTokens(renderPhrase(weekdayNine)).map((item) => ({
      ...item,
      actionable: true,
    }));

    const decoded = decodeIntent(labeled, undefined, descriptive, "on", "request");

    expect(decoded.intent).toEqual(weekdayNine);
  });

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

describe("overlay-off speech-act heads", () => {
  const weekdayNine = weekdays(["monday", "tuesday", "wednesday", "thursday", "friday"], {
    hour: 9,
    minute: 0,
  });

  function actionableSchedule(): LabeledToken[] {
    return labelTokens(renderPhrase(weekdayNine)).map((item) => ({
      ...item,
      actionable: true,
    }));
  }

  test("descriptive speech-act refuses without using a noun regex", () => {
    const labeled = actionableSchedule();

    expect(() =>
      decodeIntent(labeled, undefined, "the shop opens every weekday at 9", "off", "descriptive"),
    ).toThrow(UnsafeScheduleContextError);

    const result = resultFromLabeledTokens(
      labeled,
      undefined,
      "the shop opens every weekday at 9",
      "off",
      "descriptive",
    );

    expect(result.ok).toBe(false);

    if (!result.ok) {
      expect(result.code).toBe("unsafe-context");
      expect(result.diagnostics.some((note) => note.code === "descriptive")).toBe(true);
    }
  });

  test("request speech-act with actionable tokens yields the weekday 09:00 intent", () => {
    const decoded = decodeIntent(
      actionableSchedule(),
      undefined,
      "every weekday at 9",
      "off",
      "request",
    );

    expect(decoded.intent).toEqual(weekdayNine);
  });

  test("correction keeps only Friday actionable tokens", () => {
    const thursday = renderIntent(
      { frequency: { unit: "week", interval: 1 }, weekdays: ["thursday"] },
      "terse",
      noCarrier,
      "bare",
    );

    const friday = renderIntent(
      { frequency: { unit: "week", interval: 1 }, weekdays: ["friday"] },
      "terse",
      noCarrier,
      "bare",
    );

    const combined = concat([thursday, friday]);

    const labeled = labelTokens(combined).map((item) => ({
      ...item,
      actionable: combined.text.slice(item.token.start, item.token.end) === friday.text,
    }));

    const decoded = decodeIntent(labeled, undefined, combined.text, "off", "correction");

    expect(decoded.intent).toEqual({
      frequency: { unit: "week", interval: 1 },
      weekdays: ["friday"],
    });
  });

  test.each(["question", "none"] as const)("overlay-off %s speech-act is unsafe-context", (act) => {
    const result = resultFromLabeledTokens(
      actionableSchedule(),
      undefined,
      "every weekday at 9",
      "off",
      act,
    );

    expect(result.ok).toBe(false);

    if (!result.ok) {
      expect(result.code).toBe("unsafe-context");
      expect(result.diagnostics.some((note) => note.code === act)).toBe(true);
    }
  });

  test("overlay-off request with no actionable tokens rejects", () => {
    const labeled = labelTokens(renderPhrase(weekdayNine)).map((item) => ({
      ...item,
      actionable: false,
    }));

    const result = resultFromLabeledTokens(
      labeled,
      undefined,
      "every weekday at 9",
      "off",
      "request",
    );

    expect(result.ok).toBe(false);

    if (!result.ok) {
      expect(result.code).toBe("unsafe-context");
      expect(result.diagnostics.some((note) => note.code === "no-actionable-span")).toBe(true);
    }
  });

  test("unknown speech-act id is never treated as request", () => {
    expect(() => speechActFromId(99)).toThrow(RangeError);
  });
});

test("a labeled ISO date without a bound keyword is a start bound", () => {
  const sentence = new Sentence();

  sentence.clause();
  sentence.add("2026-05-26", "DOM_LIT");
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
    bounds: { start: "2026-05-26" },
  });
});

test("a labeled weekday description with a digit round-trips", () => {
  const sentence = new Sentence();

  sentence.clause();
  sentence.add("every", "RECUR");
  sentence.add("day 6 of the week", "WEEKDAY_SAT");

  const decoded = heads(
    labelTokens({
      text: sentence.text,
      spans: sentence.spans,
      template: "description",
      style: "description",
      speechAct: "request",
    }),
  );

  expect(decoded.intent).toEqual({
    frequency: { unit: "week", interval: 1 },
    weekdays: ["saturday"],
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

test("structural GLUE and RECUR follow an actionable clause even when their bits are false", () => {
  const sentence = new Sentence();

  sentence.clause();
  sentence.add("every", "RECUR", " ", false);
  sentence.add("5", "INTERVAL_5", " ", true);
  sentence.add("minutes", "UNIT_MIN", " ", true);

  const labeled = labelTokens({
    text: sentence.text,
    spans: sentence.spans,
    template: "every-n",
    style: "every-n",
    speechAct: "request",
  }).map((item) => {
    const span = sentence.spans.find(
      (value) => item.token.start >= value.start && item.token.end <= value.end,
    );

    return span?.actionable === undefined ? item : { ...item, actionable: span.actionable };
  });

  expect(decodeIntent(labeled, undefined, sentence.text, "off", "request").intent).toEqual({
    frequency: { unit: "minute", interval: 5 },
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

test("WEEKDAY_MON on any surface is monday; HOUR_LIT reads digits", () => {
  const monday = new Sentence();

  monday.clause();
  monday.add("wrd", "WEEKDAY_MON");

  expect(
    heads(
      labelTokens({
        text: monday.text,
        spans: monday.spans,
        template: "typed",
        style: "typed",
        speechAct: "request",
      }),
    ).intent,
  ).toEqual({
    frequency: { unit: "week", interval: 1 },
    weekdays: ["monday"],
  });

  const clock = new Sentence();

  clock.clause();
  clock.add("14", "HOUR_LIT");
  clock.add("daily", "UNIT_DAY");

  expect(
    heads(
      labelTokens({
        text: clock.text,
        spans: clock.spans,
        template: "typed",
        style: "typed",
        speechAct: "request",
      }),
    ).intent,
  ).toEqual({
    frequency: { unit: "day", interval: 1 },
    time: { hour: 14, minute: 0 },
  });
});

test("COUNT does not read token digits; interval is INTERVAL_* / INTERVAL_LIT", () => {
  const sentence = new Sentence();

  sentence.clause();
  sentence.add("every", "RECUR");
  sentence.add("5", "COUNT");
  sentence.add("minutes", "UNIT_MIN");

  expect(
    heads(
      labelTokens({
        text: sentence.text,
        spans: sentence.spans,
        template: "count",
        style: "count",
        speechAct: "request",
      }),
    ).intent,
  ).toEqual({ frequency: { unit: "minute", interval: 1 } });
});

test("decode.ts has no token-string alias tables", () => {
  const source = readFileSync(new URL("./decode.ts", import.meta.url), "utf8");

  for (const name of [
    "weekdayOf",
    "ORDINAL_ALIASES",
    "BOUND_ALIASES",
    "NAMED_HOURS",
    "MINUTE_IDIOMS",
    "FREQUENCY_IDIOMS",
    "ADVERB_FREQUENCIES",
    "UNIT_ALIASES",
    "uniqueSingleEditAlias",
  ]) {
    expect(source.includes(name)).toBe(false);
  }
});
