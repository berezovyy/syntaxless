/**
 * Deterministic lexicon aligner: the label oracle.
 *
 * Text comes from an untrusted model; labels come only from the verified
 * intent and the runtime lexicon. No model output influences a label. Given a
 * schedule that the compiler already accepted, this module walks the shared
 * tokenizer's tokens and assigns each one a training role:
 *
 * - structural words (`RECUR`, `GLUE`, `BOUND`, `EXCEPT`) keep their lexicon
 *   label; unit words are treated as `GLUE`, matching the Phase 07 renderer;
 * - a token that supplies a required intent value gets that value's role;
 * - carrier words are `O`.
 *
 * A schedule-looking word absent from the lexicon is rejected `unknown-lexeme`
 * rather than guessed. A span layout that cannot cover whole tokens is rejected
 * `unaligned`. Value agreement with the intent is checked by `verify.ts`; this
 * module only labels and reports what it saw.
 */
import {
  dayGroupLabel,
  hourLabel,
  intervalLabel,
  isStructural,
  meridiemLabel,
  minuteLabel,
  ordinalLabel,
  unitLabel,
  weekdayLabel,
} from "../../../src/model/labels.js";
import type { ContentLabel, Label } from "../../../src/model/labels.js";
import { tokenize } from "../../../src/model/tokenizer.js";
import type { RawToken } from "../../../src/model/tokenizer.js";
import {
  ADVERB_FREQUENCIES,
  LOOKUP,
  NUMBER_WORDS,
  ORDINAL_ALIASES,
  UNIT_ALIASES,
  WEEKDAY_ALIASES,
  WEEKDAY_GROUPS,
  WEEKDAY_PLURAL_ALIASES,
  WORDS,
} from "../../../src/parser/lexicon.js";
import type { FrequencyUnit, ScheduleIntent, Weekday } from "../../../src/types/intent.js";

/** A labeled span. `value` is present only for content roles. */
export interface AlignedSpan {
  start: number;
  end: number;
  label: Label;
  clauseStart: boolean;
  value?: string | number;
}

/** A content span flattened for the value cross-check. */
export interface AlignedValue {
  label: ContentLabel;
  value: string | number;
  start: number;
  end: number;
}

export type AlignFailureReason = "unknown-lexeme" | "unaligned";

export type AlignResult =
  | { ok: true; spans: AlignedSpan[]; slots: AlignedValue[] }
  | { ok: false; reason: AlignFailureReason; detail: string };

/**
 * One required reading of the intent. `covers` lists the weekdays a
 * `DAYGROUP` surface stands for, so the cross-check can accept "weekends" for
 * a Saturday+Sunday intent.
 */
export interface ExpectedValue {
  label: ContentLabel;
  value: string | number;
  required: boolean;
  covers?: readonly Weekday[];
}

const WEEKDAY_SURFACES = new Map<string, Weekday>();

for (const [word, weekday] of Object.entries(WEEKDAY_ALIASES)) {
  WEEKDAY_SURFACES.set(word, weekday);
}

for (const [word, weekday] of Object.entries(WEEKDAY_PLURAL_ALIASES)) {
  WEEKDAY_SURFACES.set(word, weekday);
}

const UNIT_SURFACES = new Map<string, FrequencyUnit>();

for (const [word, unit] of Object.entries(UNIT_ALIASES)) {
  UNIT_SURFACES.set(word, unit);
}

for (const [word, unit] of Object.entries(ADVERB_FREQUENCIES)) {
  UNIT_SURFACES.set(word, unit);
}

const MERIDIEM_WORDS: ReadonlySet<string> = new Set([WORDS.meridiem.am, WORDS.meridiem.pm]);

const SUFFIX_WORDS: ReadonlySet<string> = new Set(Object.values(WORDS.suffix));

/**
 * Morphology that suggests a token is trying to be a schedule word: adverbs of
 * frequency (`weekly`, `fortnightly`) and unit-like nouns (`yearly`,
 * `holidays`). Carrier prose in the generator never ends in one of these, so
 * the heuristic is safe and a hit is a data point for the coverage report.
 */
const SCHEDULE_SUFFIXES: readonly string[] = [
  "ly",
  "day",
  "days",
  "week",
  "weeks",
  "month",
  "months",
  "hour",
  "hours",
  "minute",
  "minutes",
  "year",
  "years",
];

const SCHEDULE_WORDS: ReadonlySet<string> = new Set([
  "noon",
  "midnight",
  "o'clock",
  "quarter",
  "half",
]);

function looksScheduleWord(word: string): boolean {
  if (SCHEDULE_WORDS.has(word)) return true;

  return SCHEDULE_SUFFIXES.some((suffix) => word.length > suffix.length && word.endsWith(suffix));
}

function wordsForNumber(value: number): string[] {
  return Object.entries(NUMBER_WORDS)
    .filter(([, number]) => number === value)
    .map(([word]) => word);
}

function hour12(hour: number): number {
  return hour % 12 === 0 ? 12 : hour % 12;
}

/** Every value the intent must be able to read back out of the text. */
export function expectedValues(intent: ScheduleIntent): ExpectedValue[] {
  const expected: ExpectedValue[] = [];

  const weekdays = new Set<Weekday>(intent.weekdays ?? []);

  if (intent.ordinalWeekday !== undefined) {
    weekdays.add(intent.ordinalWeekday.weekday);
  }

  for (const weekday of intent.exceptions?.weekdays ?? []) {
    weekdays.add(weekday);
  }

  for (const weekday of weekdays) {
    expected.push({ label: weekdayLabel(weekday), value: weekday, required: true });
  }

  for (const [word, group] of Object.entries(WEEKDAY_GROUPS)) {
    if (group.length > 1 && group.every((day) => weekdays.has(day))) {
      expected.push({
        label: word.includes("weekend") ? dayGroupLabel("weekend") : dayGroupLabel("weekday"),
        value: word,
        required: false,
        covers: group,
      });
    }
  }

  expected.push({
    label: intervalLabel(intent.frequency.interval),
    value: intent.frequency.interval,
    required: intent.frequency.interval > 1,
  });

  if (intent.ordinalWeekday !== undefined) {
    expected.push({
      label: ordinalLabel(intent.ordinalWeekday.ordinal),
      value: intent.ordinalWeekday.ordinal,
      required: true,
    });
  }

  if (intent.dayOfMonth !== undefined) {
    expected.push({ label: "DOM_LIT", value: intent.dayOfMonth, required: true });
  }

  if (intent.time !== undefined) {
    const { hour, minute } = intent.time;

    expected.push({ label: hourLabel(hour), value: hour, required: true });

    expected.push({
      label: minuteLabel(minute),
      value: minute,
      required: minute !== 0,
    });

    expected.push({
      label: meridiemLabel(hour >= 12 ? "pm" : "am"),
      value: hour >= 12 ? "pm" : "am",
      required: false,
    });
  }

  if (intent.bounds?.start !== undefined) {
    expected.push({ label: "DOM_LIT", value: intent.bounds.start, required: true });
  }

  if (intent.bounds?.end !== undefined) {
    expected.push({ label: "DOM_LIT", value: intent.bounds.end, required: true });
  }

  return expected;
}

interface NumericObligation {
  label: ContentLabel;
  value: number;
  digits: ReadonlySet<string>;
  words: ReadonlySet<string>;
  satisfied: boolean;
}

function numericObligations(intent: ScheduleIntent): NumericObligation[] {
  const obligations: NumericObligation[] = [];
  const interval = intent.frequency.interval;

  obligations.push({
    label: intervalLabel(interval),
    value: interval,
    digits: new Set([String(interval)]),
    words: new Set(wordsForNumber(interval)),
    satisfied: false,
  });

  if (intent.time !== undefined) {
    const { hour, minute } = intent.time;
    const twelve = hour12(hour);

    obligations.push({
      label: hourLabel(hour),
      value: hour,
      digits: new Set([String(hour), String(twelve)]),
      words: new Set([...wordsForNumber(hour), ...wordsForNumber(twelve)]),
      satisfied: false,
    });

    obligations.push({
      label: minuteLabel(minute),
      value: minute,
      digits: new Set([String(minute), String(minute).padStart(2, "0")]),
      words: new Set(wordsForNumber(minute)),
      satisfied: false,
    });
  }

  if (intent.dayOfMonth !== undefined) {
    obligations.push({
      label: "DOM_LIT",
      value: intent.dayOfMonth,
      digits: new Set([String(intent.dayOfMonth)]),
      words: new Set(wordsForNumber(intent.dayOfMonth)),
      satisfied: false,
    });
  }

  return obligations;
}

class AlignError extends Error {
  readonly reason: AlignFailureReason;
  constructor(reason: AlignFailureReason, detail: string) {
    super(detail);
    this.reason = reason;
  }
}

function makeSpan(label: Label, start: number, end: number, value?: string | number): AlignedSpan {
  return value === undefined
    ? { start, end, label, clauseStart: false }
    : { start, end, label, clauseStart: false, value };
}

/**
 * Align `text` to `intent`. Pure and deterministic: every label is derived
 * from the intent or the lexicon, never from the text's own structure.
 */
export function alignText(text: string, intent: ScheduleIntent): AlignResult {
  try {
    const tokens = tokenize(text);
    const assigned = new Map<number, { label: Label; value?: string | number }>();
    const numeric = numericObligations(intent);

    const matchNumeric = (
      surface: string,
      kind: "digit" | "word",
    ): { label: ContentLabel; value: number } | null => {
      let fallback: NumericObligation | undefined;

      for (const obligation of numeric) {
        const has =
          kind === "digit" ? obligation.digits.has(surface) : obligation.words.has(surface);

        if (!has) continue;

        if (!obligation.satisfied) {
          obligation.satisfied = true;

          return { label: obligation.label, value: obligation.value };
        }

        fallback ??= obligation;
      }

      return fallback === undefined ? null : { label: fallback.label, value: fallback.value };
    };

    // Bounds are ISO dates: locate each exact string, then cover its tokens.
    const boundRanges: Array<{ start: number; end: number; value: string }> = [];

    for (const date of [intent.bounds?.start, intent.bounds?.end]) {
      if (date === undefined) continue;
      const index = text.indexOf(date);

      if (index >= 0) boundRanges.push({ start: index, end: index + date.length, value: date });
    }

    // Daypart qualifiers ("in the morning") are multi-token meridiem phrases.
    const phraseTokens = Object.entries(WORDS.qualifier).map(([daypart, phrase]) => ({
      words: phrase.split(" "),
      meridiem: (daypart === "morning" ? "am" : "pm") as "am" | "pm",
    }));

    const wordIndices = tokens
      .map((token, index) => (token.kind === 0 ? index : -1))
      .filter((index) => index >= 0);

    for (let cursor = 0; cursor < wordIndices.length; cursor++) {
      for (const phrase of phraseTokens) {
        if (cursor + phrase.words.length > wordIndices.length) continue;

        const matches = phrase.words.every(
          (word, offset) => tokens[wordIndices[cursor + offset]!]!.text.toLowerCase() === word,
        );

        if (!matches) continue;

        for (let offset = 0; offset < phrase.words.length; offset++) {
          assigned.set(wordIndices[cursor + offset]!, {
            label: meridiemLabel(phrase.meridiem),
            value: phrase.meridiem,
          });
        }

        cursor += phrase.words.length - 1;
        break;
      }
    }

    for (let index = 0; index < tokens.length; index++) {
      const token = tokens[index]!;

      if (token.kind === 3) continue;
      if (assigned.has(index)) continue;

      const bound = boundRanges.find(
        (range) => token.start >= range.start && token.end <= range.end,
      );

      if (bound !== undefined) {
        assigned.set(index, { label: "DOM_LIT", value: bound.value });
        continue;
      }

      const surface = token.text.toLowerCase();

      if (token.kind === 2) {
        const entry = LOOKUP.get(surface);

        assigned.set(index, entry ? { label: entry.label } : { label: "O" });
        continue;
      }

      if (token.kind === 1) {
        const match = matchNumeric(token.text, "digit");

        assigned.set(index, match ?? { label: "O" });
        continue;
      }

      // A number word: a required numeric slot if one matches, else the
      // lexicon's own `COUNT` reading so the cross-check can reject it.
      const number = NUMBER_WORDS[surface];

      if (number !== undefined) {
        const match = matchNumeric(surface, "word");

        assigned.set(index, match ?? { label: "COUNT", value: number });
        continue;
      }

      const weekday = WEEKDAY_SURFACES.get(surface);

      if (weekday !== undefined) {
        assigned.set(index, { label: weekdayLabel(weekday), value: weekday });
        continue;
      }

      const group = WEEKDAY_GROUPS[surface];

      if (group !== undefined) {
        assigned.set(index, {
          label: surface.includes("weekend") ? dayGroupLabel("weekend") : dayGroupLabel("weekday"),
          value: surface,
        });

        continue;
      }

      const ordinal = ORDINAL_ALIASES[surface];

      if (ordinal !== undefined) {
        assigned.set(index, { label: ordinalLabel(ordinal), value: ordinal });
        continue;
      }

      if (MERIDIEM_WORDS.has(surface)) {
        assigned.set(index, {
          label: meridiemLabel(surface === "pm" ? "pm" : "am"),
          value: surface,
        });

        continue;
      }

      const unit = UNIT_SURFACES.get(surface);

      if (unit !== undefined) {
        assigned.set(index, { label: unitLabel(unit), value: unit });
        continue;
      }

      if (SUFFIX_WORDS.has(surface)) {
        assigned.set(index, { label: "GLUE" });
        continue;
      }

      const entry = LOOKUP.get(surface);

      if (entry !== undefined && isStructural(entry.label)) {
        assigned.set(index, { label: entry.label });
        continue;
      }

      if (looksScheduleWord(surface)) {
        throw new AlignError("unknown-lexeme", `schedule word not in lexicon: ${surface}`);
      }

      assigned.set(index, { label: "O" });
    }

    const spans: AlignedSpan[] = [];
    let current: AlignedSpan | undefined;

    for (let index = 0; index < tokens.length; index++) {
      const token = tokens[index]!;

      if (token.kind === 3) continue;
      const reading = assigned.get(index);

      if (reading === undefined) continue;

      if (
        current !== undefined &&
        current.end === token.start &&
        current.label === reading.label &&
        current.value === reading.value
      ) {
        current.end = token.end;
        continue;
      }

      current = makeSpan(reading.label, token.start, token.end, reading.value);
      spans.push(current);
    }

    validateCoverage(tokens, spans);

    let opened = false;

    for (const span of spans) {
      if (!opened && span.label !== "O" && !isStructural(span.label)) {
        span.clauseStart = true;
        opened = true;
      }
    }

    const slots: AlignedValue[] = [];

    for (const span of spans) {
      if (span.value === undefined) continue;
      if (isStructural(span.label) || span.label === "O") continue;

      slots.push({
        label: span.label as ContentLabel,
        value: span.value,
        start: span.start,
        end: span.end,
      });
    }

    return { ok: true, spans, slots };
  } catch (error) {
    if (error instanceof AlignError) {
      return { ok: false, reason: error.reason, detail: error.message };
    }

    throw error;
  }
}

/** Whole-token coverage: ordered, non-overlapping, every token covered once. */
function validateCoverage(tokens: readonly RawToken[], spans: readonly AlignedSpan[]): void {
  const boundaries = new Set<number>();

  for (const token of tokens) {
    boundaries.add(token.start);
    boundaries.add(token.end);
  }

  let cursor = 0;

  for (const span of spans) {
    if (span.start < cursor) {
      throw new AlignError("unaligned", `overlapping span at ${span.start}`);
    }

    if (!boundaries.has(span.start) || !boundaries.has(span.end)) {
      throw new AlignError("unaligned", `span straddles a token: ${span.start}-${span.end}`);
    }

    cursor = span.end;
  }

  for (const token of tokens) {
    if (token.kind === 3) continue;
    const covered = spans.some((span) => token.start >= span.start && token.end <= span.end);

    if (!covered) {
      throw new AlignError("unaligned", `uncovered token: ${token.text}`);
    }
  }
}
