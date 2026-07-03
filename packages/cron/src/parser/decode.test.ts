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

import { renderIntent } from "../../training/src/renderer.js";
import type { RenderedExample } from "../../training/src/renderer.js";
import type { Rng } from "../../training/src/rng.js";
import type { Span } from "../../training/src/sentence.js";
import { LABEL_COUNT, labelId } from "../model/labels.js";
import { tokenize } from "../model/tokenizer.js";
import type { ScheduleIntent, Weekday } from "../types/intent.js";
import { decodeIntent, UnknownRoleError } from "./decode.js";
import type { LabeledToken } from "./decode.js";
import { WEEKDAY_SEQUENCE } from "./lexicon.js";

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
    const decoded = decodeIntent(labelTokens(example));

    expect(decoded.intent).toEqual(intent);
  });
});

describe("range and interval rules", () => {
  test("monday through friday expands to an ordered weekday list", () => {
    const intent = weekdays(["monday", "tuesday", "wednesday", "thursday", "friday"]);
    const example = renderIntent(intent, "weekday-range", noCarrier, "through");

    expect(example.text).toBe("monday through friday");
    const labeled = labelTokens(example);

    expect(decodeIntent(labeled).intent).toEqual(intent);
  });

  test("every other friday multiplies the interval by two", () => {
    const tokens = tokenize("every other friday").filter((token) => token.kind !== 3);
    const roles = [labelId.RECUR, labelId.GLUE, labelId.WEEKDAY];

    const labeled: LabeledToken[] = tokens.map((token, index) => ({
      token,
      role: roles[index]!,
      boundary: index === 0,
      score: 1,
    }));

    expect(decodeIntent(labeled).intent).toEqual({
      frequency: { unit: "week", interval: 2 },
      weekdays: ["friday"],
    });
  });
});

describe("confidence", () => {
  test("is the weakest margin among content tokens", () => {
    const intent: ScheduleIntent = { frequency: { unit: "day", interval: 3 } };

    const labeled = labelTokens(renderPhrase(intent)).map((item) =>
      item.role === labelId.INTERVAL ? { ...item, score: 0.05 } : item,
    );

    const decoded = decodeIntent(labeled);

    expect(decoded.intent).toEqual(intent);
    expect(decoded.confidence).toBeCloseTo(0.05);
  });
});

describe("decodeIntent rejects corrupt input", () => {
  const base = labelTokens(renderPhrase({ frequency: { unit: "day", interval: 1 } }));

  test("a role id at LABEL_COUNT is an unknown role", () => {
    expect(() => decodeIntent([{ ...base[0]!, role: LABEL_COUNT }])).toThrow(UnknownRoleError);
  });

  test("a negative role id is an unknown role", () => {
    expect(() => decodeIntent([{ ...base[0]!, role: -1 }])).toThrow(UnknownRoleError);
  });

  test("a non-integer role id is an unknown role", () => {
    expect(() => decodeIntent([{ ...base[0]!, role: 1.5 }])).toThrow(UnknownRoleError);
  });

  test("an out-of-bounds day of month is rejected before compile", () => {
    const example = renderPhrase({
      frequency: { unit: "month", interval: 1 },
      dayOfMonth: 32,
    });

    expect(() => decodeIntent(labelTokens(example))).toThrow(RangeError);
  });
});
