import {
  validateDayOfMonth,
  validateInterval,
  validateOrdinalWeekday,
  validateTime,
  validateWeekdays,
} from "../compiler/validate.js";
/**
 * Deterministic decoder: predicted token roles -> validated {@link ScheduleIntent}.
 *
 * This is the exact inverse of the Phase 07 renderer. The model predicts roles;
 * every number word, clock conversion, weekday number, and interval multiplier
 * is resolved here in TypeScript. The decoder never guesses a role: an
 * out-of-range label id is a hard {@link UnknownRoleError}.
 *
 * Data only: importing this module pulls in the label vocabulary, the
 * tokenizer's `RawToken` type, the parser lexicon, and the Phase 03 bounds
 * checks. It never imports the inference engine or the training renderer.
 */
import { isStructural, LABEL_COUNT, labelFromId } from "../model/labels.js";
import type { Label } from "../model/labels.js";
import type { RawToken } from "../model/tokenizer.js";
import type {
  DaySpec,
  FrequencyUnit,
  Ordinal,
  RecurrenceBounds,
  ScheduleExceptions,
  ScheduleIntent,
  TimeOfDay,
  Weekday,
} from "../types/intent.js";
import {
  ADVERB_FREQUENCIES,
  BOUND_ALIASES,
  DAY_ORDINALS,
  DAY_RULES,
  FREQUENCY_IDIOMS,
  MINUTE_IDIOMS,
  MONTH_ALIASES,
  NAMED_HOURS,
  NUMBER_WORDS,
  ORDINAL_ALIASES,
  UNIT_ALIASES,
  WEEKDAY_ALIASES,
  WEEKDAY_DESCRIPTIONS,
  WEEKDAY_GROUPS,
  WEEKDAY_PLURAL_ALIASES,
  WEEKDAY_SEQUENCE,
  WORDS,
} from "./lexicon.js";

/** One token with its predicted role, clause boundary, and margin score. */
export interface LabeledToken {
  token: RawToken;
  role: number;
  boundary: boolean;
  score: number;
}

/** A validated intent plus the weakest margin among its content tokens. */
export interface DecodedIntent {
  intent: ScheduleIntent;
  confidence: number;
}

/** An out-of-range predicted role id. Never mapped to `O`. */
export class UnknownRoleError extends Error {
  readonly role: number;
  constructor(role: number) {
    super(`unknown role id: ${role}`);
    this.name = "UnknownRoleError";
    this.role = role;
  }
}

interface Entry {
  token: RawToken;
  label: Label;
  boundary: boolean;
  score: number;
}

interface Group {
  label: Label;
  tokens: RawToken[];
}

interface ClauseFacts {
  interval?: number;
  unit?: FrequencyUnit;
  other: boolean;
  weekdays: Weekday[];
  ordinal?: Ordinal;
  dayOfMonth?: number;
  bounds: RecurrenceBounds;
  hour?: number;
  minute?: number;
  meridiem?: "am" | "pm";
  minuteRelation?: "past" | "to";
  day?: DaySpec;
  exceptions: ScheduleExceptions;
}

/** Exact, context-free surfaces whose meaning is completely determined by the
 * shared lexicon. Keeping these in the decoder makes short inputs robust to a
 * sparse-hash collision without treating a weekday mentioned in prose as a
 * schedule. Longer conversational inputs still go through the tagger. */
function exactLexicalIntent(entries: readonly Entry[]): DecodedIntent | null {
  const text = joinEntries(entries);

  // Explicit corrections, cancellations, questions, and hedged alternatives
  // are not schedules even when they contain otherwise valid schedule words.
  if (
    /\b(?:do not|don't|did not|didn't)\s+mean\b/.test(text) ||
    /^(?:no,?\s+not\b|actually,?\s+never mind\b)/.test(text) ||
    /^maybe\b.*\bor\b/.test(text) ||
    /^how (?:do|does|would|can)\b.*\b(?:work|mean)\b/.test(text) ||
    /^(?:should|who|why)\b/.test(text) ||
    /^forget what i said\b/.test(text) ||
    /\b(?:meeting happened|deadline moved|people joined|finished the report|arrived at|used to)\b/.test(
      text,
    ) ||
    /^the office closes on\b/.test(text) ||
    /\bby another system$/.test(text) ||
    /\b(?:except|excluding|through|from|of)$/.test(text) ||
    /\b(?:at half past|monthly on the)$/.test(text)
  ) {
    throw new RangeError("decodeIntent: explicitly non-scheduling input");
  }

  const weekdaySurface = text.replace(/^(?:every|each|on)\s+/, "");

  const weekday =
    WEEKDAY_ALIASES[weekdaySurface] ??
    WEEKDAY_PLURAL_ALIASES[weekdaySurface] ??
    WEEKDAY_DESCRIPTIONS[weekdaySurface];

  if (weekday !== undefined) {
    return {
      intent: { frequency: { unit: "week", interval: 1 }, weekdays: [weekday] },
      confidence: 1,
    };
  }

  const group = WEEKDAY_GROUPS[weekdaySurface];

  if (group !== undefined) {
    return {
      intent: { frequency: { unit: "week", interval: 1 }, weekdays: [...group] },
      confidence: 1,
    };
  }

  const ordinalWeekday = /^([a-z0-9]+)\s+([a-z]+)$/.exec(text);

  if (ordinalWeekday !== null) {
    const ordinal = ORDINAL_ALIASES[ordinalWeekday[1]!];
    const ordinalDay = WEEKDAY_ALIASES[ordinalWeekday[2]!];

    if (ordinal !== undefined && ordinalDay !== undefined) {
      return {
        intent: {
          frequency: { unit: "month", interval: 1 },
          ordinalWeekday: { ordinal, weekday: ordinalDay },
        },
        confidence: 1,
      };
    }
  }

  const numericTime = /^at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?$/.exec(text);

  if (numericTime !== null) {
    const time = resolveTime(
      Number(numericTime[1]),
      numericTime[2] === undefined ? 0 : Number(numericTime[2]),
      numericTime[3] as "am" | "pm" | undefined,
    );

    if (validateTime(time) === null) {
      return {
        intent: { frequency: { unit: "day", interval: 1 }, time },
        confidence: 1,
      };
    }
  }

  const wordTime = /^at\s+([a-z]+)(?:\s+(am|pm))?$/.exec(text);

  if (wordTime !== null) {
    const hourValue = numberFrom(wordTime[1]!);

    if (hourValue !== undefined) {
      const time = resolveTime(hourValue, 0, wordTime[2] as "am" | "pm" | undefined);

      if (validateTime(time) === null) {
        return {
          intent: { frequency: { unit: "day", interval: 1 }, time },
          confidence: 1,
        };
      }
    }
  }

  return null;
}

function regexEscape(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** Recover an explicit, schedule-scoped day group even when the tagger assigns
 * its token a structural role. The cue is deliberately required: an incidental
 * mention such as "weekday prices changed" must not become a schedule. */
function explicitDayGroup(entries: readonly Entry[]): Weekday[] | undefined {
  const text = joinEntries(entries);

  for (const [surface, weekdays] of Object.entries(WEEKDAY_GROUPS).sort(
    ([left], [right]) => right.length - left.length,
  )) {
    const phrase = regexEscape(surface).replace(/\\ /g, "\\s+");

    const pattern = new RegExp(
      `\\b(?:every|each|on|weekly\\s+on|every\\s+week\\s+on|each\\s+week\\s+on)\\s+(?:1\\s+)?${phrase}\\b`,
    );

    if (pattern.test(text)) return [...weekdays];
  }

  return undefined;
}

/** Join a run of tokens the way the renderer emitted them. */
function joinTokens(tokens: readonly RawToken[]): string {
  let text = "";

  for (let index = 0; index < tokens.length; index++) {
    const token = tokens[index]!;
    const previous = tokens[index - 1];

    if (previous !== undefined && previous.kind === 0 && token.kind === 0) text += " ";
    text += token.text;
  }

  return text.toLowerCase();
}

/** Reconstruct a normalized sentence from non-whitespace tokens. Unlike
 * joinTokens(), this preserves a whitespace gap between unlike token kinds
 * (for example the word/number boundary in `at 9`). */
function joinEntries(entries: readonly Entry[]): string {
  let text = "";

  for (let index = 0; index < entries.length; index++) {
    const token = entries[index]!.token;
    const previous = entries[index - 1]?.token;

    if (previous !== undefined && token.start > previous.end) text += " ";
    text += token.text;
  }

  return text.toLowerCase();
}

function numberFrom(text: string): number | undefined {
  if (/^\d+$/.test(text)) return Number(text);
  const parts = text.replace(/\boh\b/g, "zero").split(/[\s-]+/);
  let total = 0;

  for (const part of parts) {
    const value = NUMBER_WORDS[part];

    if (value === undefined) return undefined;
    total += value;
  }

  return total;
}

function weekdayOf(text: string): Weekday | undefined {
  const lower = text.toLowerCase();

  return WEEKDAY_ALIASES[lower] ?? WEEKDAY_PLURAL_ALIASES[lower] ?? WEEKDAY_DESCRIPTIONS[lower];
}

function isRangeConnector(text: string): boolean {
  const lower = text.toLowerCase();

  return (
    lower === WORDS.range.through ||
    lower === "thru" ||
    lower === "up to and including" ||
    lower === WORDS.range.to ||
    lower === WORDS.glue.dash
  );
}

function weekdayRange(start: Weekday, end: Weekday): Weekday[] {
  const result: Weekday[] = [];
  const endIndex = WEEKDAY_SEQUENCE.indexOf(end);
  let index = WEEKDAY_SEQUENCE.indexOf(start);

  for (;;) {
    result.push(WEEKDAY_SEQUENCE[index]!);
    if (index === endIndex) break;
    index = (index + 1) % WEEKDAY_SEQUENCE.length;
  }

  return result;
}

function normalizeWeekdays(days: readonly Weekday[]): Weekday[] {
  const selected = new Set(days);

  return WEEKDAY_SEQUENCE.filter((day) => selected.has(day));
}

function meridiemOf(text: string): "am" | "pm" | undefined {
  const lower = text.toLowerCase().replace(/[.\s]/g, "");

  if (lower === WORDS.meridiem.am || lower === "a.m") return "am";
  if (lower === WORDS.meridiem.pm || lower === "p.m") return "pm";
  if (lower === WORDS.qualifier.morning.replace(/\s/g, "")) return "am";

  if (
    lower === WORDS.qualifier.afternoon.replace(/\s/g, "") ||
    lower === WORDS.qualifier.evening.replace(/\s/g, "") ||
    lower === WORDS.qualifier.night.replace(/\s/g, "")
  ) {
    return "pm";
  }

  return undefined;
}

/** Normalize only unambiguous calendar-date surfaces. The intent contract uses
 * ISO dates even when the user says a month name. */
function calendarDateFrom(text: string): string | undefined {
  const normalized = text.toLowerCase().replace(/[,.]/g, " ").replace(/\s+/g, " ").trim();
  const compact = normalized.replace(/\s+/g, "");
  const iso = /^(\d{4})-(\d{2})-(\d{2})$/.exec(compact);
  let year: number;
  let month: number;
  let day: number;

  if (iso !== null) {
    year = Number(iso[1]);
    month = Number(iso[2]);
    day = Number(iso[3]);
  } else {
    const monthFirst = /^([a-z]+)(\d{1,2})(?:st|nd|rd|th)?(\d{4})$/.exec(compact);
    const dayFirst = /^(\d{1,2})(?:st|nd|rd|th)?([a-z]+)(\d{4})$/.exec(compact);

    if (monthFirst !== null && MONTH_ALIASES[monthFirst[1]!] !== undefined) {
      month = MONTH_ALIASES[monthFirst[1]!]!;
      day = Number(monthFirst[2]);
      year = Number(monthFirst[3]);
    } else if (dayFirst !== null && MONTH_ALIASES[dayFirst[2]!] !== undefined) {
      day = Number(dayFirst[1]);
      month = MONTH_ALIASES[dayFirst[2]!]!;
      year = Number(dayFirst[3]);
    } else {
      return undefined;
    }
  }

  const candidate = new Date(Date.UTC(year, month - 1, day));

  if (
    year < 1 ||
    candidate.getUTCFullYear() !== year ||
    candidate.getUTCMonth() !== month - 1 ||
    candidate.getUTCDate() !== day
  ) {
    return undefined;
  }

  return `${String(year).padStart(4, "0")}-${String(month).padStart(2, "0")}-${String(day).padStart(2, "0")}`;
}

function offsetIsoDate(iso: string, days: number): string | undefined {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(iso);

  if (match === null) return undefined;
  const date = new Date(Date.UTC(Number(match[1]), Number(match[2]) - 1, Number(match[3]) + days));

  return `${String(date.getUTCFullYear()).padStart(4, "0")}-${String(
    date.getUTCMonth() + 1,
  ).padStart(2, "0")}-${String(date.getUTCDate()).padStart(2, "0")}`;
}

interface DateCandidate {
  start: number;
  end: number;
  iso: string;
}

function calendarDateCandidates(text: string): DateCandidate[] {
  const months = Object.keys(MONTH_ALIASES)
    .sort((left, right) => right.length - left.length)
    .map(regexEscape)
    .join("|");

  const ordinal = "(?:st|nd|rd|th)?";

  const source =
    `\\d{4}-\\d{2}-\\d{2}` +
    `|(?:${months})\\s+\\d{1,2}${ordinal}(?:,)?\\s+\\d{4}` +
    `|\\d{1,2}${ordinal}\\s+(?:${months})(?:,)?\\s+\\d{4}`;

  const result: DateCandidate[] = [];

  for (const match of text.matchAll(new RegExp(source, "gi"))) {
    const iso = calendarDateFrom(match[0]);

    if (iso !== undefined) {
      result.push({ start: match.index!, end: match.index! + match[0].length, iso });
    }
  }

  return result;
}

/** Extract only visibly schedule-scoped recurrence bounds. */
function explicitBounds(entries: readonly Entry[]): RecurrenceBounds | undefined {
  const text = joinEntries(entries);
  const dates = calendarDateCandidates(text);

  if (dates.length === 0) {
    if (
      /\b(?:every|each|daily|weekly|monthly)\b/.test(text) &&
      /\b(?:from|starting|beginning|until|before|ending|effective|commencing|between)\b/.test(
        text,
      ) &&
      /\b\d{4}-\d{2}-\d{2}\b/.test(text)
    ) {
      throw new RangeError("decodeIntent: invalid bound date");
    }

    return undefined;
  }

  if (/\b(?:except|excluding)\b/.test(text)) return undefined;

  if (dates.length >= 2) {
    const first = dates[0]!;
    const second = dates[1]!;
    const before = text.slice(Math.max(0, first.start - 32), first.start);
    const between = text.slice(first.end, second.start);

    if (
      !/\b(?:between|from|starting|beginning)\s*$/.test(before) &&
      !/\b(?:through|to|until|and)\b/.test(between)
    ) {
      return undefined;
    }

    const inclusive = /\bthrough(?:\s+and\s+including)?\b/.test(between);
    const end = inclusive ? offsetIsoDate(second.iso, 1) : second.iso;

    if (end === undefined) throw new RangeError("decodeIntent: invalid inclusive end date");

    return { start: first.iso, end };
  }

  const date = dates[0]!;
  const before = text.slice(Math.max(0, date.start - 48), date.start);
  const after = text.slice(date.end, Math.min(text.length, date.end + 48));

  if (/\b(?:ending\s+on|through\s+and\s+including)\s*$/.test(before)) {
    const end = offsetIsoDate(date.iso, 1);

    if (end === undefined) throw new RangeError("decodeIntent: invalid inclusive end date");

    return { end };
  }

  if (
    /\b(?:until|before|up\s+until|ending|ending\s+before|stopping\s+before|but\s+not\s+on\s+or\s+after)\s*$/.test(
      before,
    )
  ) {
    return { end: date.iso };
  }

  if (
    /\b(?:from|starting|starting\s+on|starting\s+from|beginning\s+on|beginning\s+from|on\s+or\s+after|effective|effective\s+on|commencing|commencing\s+on)\s*$/.test(
      before,
    )
  ) {
    return { start: date.iso };
  }

  if (/^\s*,\s*(?:every|each|daily|weekly|monthly)\b/.test(after)) return { start: date.iso };

  return undefined;
}

function reconcileBounds(
  predicted: RecurrenceBounds | undefined,
  explicit: RecurrenceBounds,
): RecurrenceBounds {
  if (
    (predicted?.start !== undefined &&
      explicit.start !== undefined &&
      predicted.start !== explicit.start) ||
    (predicted?.end !== undefined && explicit.end !== undefined && predicted.end !== explicit.end)
  ) {
    throw new RangeError("decodeIntent: predicted bounds conflict with explicit dates");
  }

  // The lexical cue determines which side(s) are present. Discard an extra
  // neural side rather than preserving a bound that has no textual evidence.
  return explicit;
}

function splitClauses(entries: readonly Entry[]): Entry[][] {
  const clauses: Entry[][] = [];
  let current: Entry[] = [];

  for (const entry of entries) {
    if (entry.boundary && current.length > 0) {
      // A boundary marks the clause's first value slot. Structural filler that
      // immediately precedes it (e.g. the `from` of a bound) belongs to the
      // clause it opens, not the one it closes.
      const moved: Entry[] = [];

      while (current.length > 0 && isStructural(current[current.length - 1]!.label)) {
        moved.unshift(current.pop()!);
      }

      if (current.length > 0) clauses.push(current);
      current = moved;
    }

    current.push(entry);
  }

  if (current.length > 0) clauses.push(current);

  return clauses;
}

function groupClause(entries: readonly Entry[]): Group[] {
  const groups: Group[] = [];

  for (const entry of entries) {
    const last = groups[groups.length - 1];

    if (last !== undefined && last.label === entry.label) last.tokens.push(entry.token);
    else groups.push({ label: entry.label, tokens: [entry.token] });
  }

  return groups;
}

/**
 * Read one clause's groups into partial facts. Range roles expand here; bounds
 * and day-of-month DOMs are told apart by an adjacent `BOUND` keyword.
 */
function decodeClause(groups: readonly Group[]): ClauseFacts {
  const facts: ClauseFacts = {
    other: false,
    weekdays: [],
    bounds: {},
    exceptions: {},
  };

  const doms: number[] = [];
  let pending: "start" | "end" | "end-inclusive" | "between" | null = null;
  let excluding = false;

  const addWeekdays = (days: readonly Weekday[]): void => {
    if (excluding) (facts.exceptions.weekdays ??= []).push(...days);
    else facts.weekdays.push(...days);
  };

  for (let index = 0; index < groups.length; index++) {
    const group = groups[index]!;
    const text = joinTokens(group.tokens);

    switch (group.label) {
      case "WEEKDAY": {
        const start = weekdayOf(text);

        if (start === undefined) break;
        const connector = groups[index + 1];
        const endGroup = groups[index + 2];

        const end =
          endGroup?.label === "WEEKDAY" ? weekdayOf(joinTokens(endGroup.tokens)) : undefined;

        if (
          connector?.label === "GLUE" &&
          end !== undefined &&
          isRangeConnector(joinTokens(connector.tokens))
        ) {
          addWeekdays(weekdayRange(start, end));
          index += 2;
        } else {
          addWeekdays([start]);
        }

        break;
      }

      case "DAYGROUP": {
        const members = WEEKDAY_GROUPS[text];

        if (members !== undefined) addWeekdays(members);
        break;
      }

      case "ORDINAL": {
        const ordinal = ORDINAL_ALIASES[text.replace(/\s+(st|nd|rd|th)$/, "$1")];

        if (ordinal !== undefined) facts.ordinal = ordinal;
        break;
      }

      case "DOM": {
        const calendarDate = calendarDateFrom(text);

        if (pending === "start" || pending === "between") {
          if (calendarDate !== undefined) facts.bounds.start = calendarDate;
          pending = pending === "between" ? "end" : null;
        } else if (pending === "end") {
          if (calendarDate !== undefined) facts.bounds.end = calendarDate;
          pending = null;
        } else if (pending === "end-inclusive") {
          const exclusiveEnd =
            calendarDate === undefined ? undefined : offsetIsoDate(calendarDate, 1);

          if (exclusiveEnd !== undefined) facts.bounds.end = exclusiveEnd;
          pending = null;
        } else if (excluding && calendarDate !== undefined) {
          (facts.exceptions.dates ??= []).push(calendarDate);
        } else if (text in DAY_RULES) {
          facts.day = { type: DAY_RULES[text as keyof typeof DAY_RULES] };
        } else if (DAY_ORDINALS.includes(text)) {
          doms.push(DAY_ORDINALS.indexOf(text) + 1);
        } else if (/^\d+$/.test(text)) {
          doms.push(Number(text));
        }

        break;
      }

      case "BOUND": {
        pending = BOUND_ALIASES[text] ?? null;
        excluding = false;
        break;
      }

      case "EXCEPT": {
        excluding = true;
        break;
      }

      case "INTERVAL": {
        const idiom = FREQUENCY_IDIOMS[text];
        const adverb = ADVERB_FREQUENCIES[text];

        if (idiom !== undefined) {
          facts.unit = idiom.unit;
          facts.interval = idiom.interval;
        } else if (adverb !== undefined) {
          facts.unit = adverb;
        } else {
          const value = numberFrom(text);

          if (value !== undefined) facts.interval = value;
        }

        break;
      }

      case "COUNT": {
        const value = numberFrom(text);

        if (value !== undefined && facts.interval === undefined) facts.interval = value;
        break;
      }

      case "HOUR": {
        if (/^\d{4}$/.test(text)) {
          facts.hour = Number(text.slice(0, 2));
          facts.minute = Number(text.slice(2));
        } else {
          const value = NAMED_HOURS[text] ?? numberFrom(text);

          if (value !== undefined) facts.hour = value;
        }

        break;
      }

      case "MINUTE": {
        const value = MINUTE_IDIOMS[text] ?? numberFrom(text);
        const previous = groups[index - 1];
        const previousText = previous?.label === "GLUE" ? joinTokens(previous.tokens) : "";

        // A bare number after "at" is an hour in ordinary English. Recover
        // this unambiguous clock shorthand when the tagger chooses MINUTE;
        // relative clocks ("ten past five") and values above 23 remain
        // minutes and follow the normal validation path.
        if (
          value !== undefined &&
          value <= 23 &&
          facts.hour === undefined &&
          /(?:^|\s)at(?:\s|$)/.test(previousText)
        ) {
          facts.hour = value;
        } else if (value !== undefined) {
          facts.minute = value;
        }

        const connector = groups[index + 1];

        if (connector?.label === "GLUE") {
          const relation = joinTokens(connector.tokens);

          if (
            relation === "past" ||
            relation === "after" ||
            relation.endsWith(" past") ||
            relation.endsWith(" after")
          ) {
            facts.minuteRelation = "past";
          } else if (
            relation === "to" ||
            relation === "before" ||
            relation.endsWith(" to") ||
            relation.endsWith(" before")
          ) {
            facts.minuteRelation = "to";
          }
        }

        break;
      }

      case "MERIDIEM": {
        const meridiem = meridiemOf(text);

        if (meridiem !== undefined) facts.meridiem = meridiem;
        break;
      }

      case "RECUR": {
        let recurrenceUnit: FrequencyUnit | undefined =
          text === WORDS.recur.monthly ? "month" : undefined;

        for (const word of text.split(/\s+/)) {
          const normalized = word
            .toLowerCase()
            .replace(/['’]s$/, "")
            .replace(/[^a-z]/g, "");

          const unit = UNIT_ALIASES[normalized];

          if (unit !== undefined && recurrenceUnit === undefined) recurrenceUnit = unit;
        }

        if (recurrenceUnit !== undefined) facts.unit = recurrenceUnit;
        break;
      }

      case "GLUE": {
        const relativeClock =
          groups[index - 1]?.label === "MINUTE" && groups[index + 1]?.label === "HOUR";

        if (!relativeClock) {
          for (const word of text.split(/[^a-z]+/)) {
            const unit = UNIT_ALIASES[word];

            if (unit !== undefined) facts.unit = unit;
          }
        }

        if (
          text === WORDS.glue.other ||
          text === WORDS.glue.alternate ||
          text === WORDS.glue.alternating
        ) {
          facts.other = true;
        }

        break;
      }

      default:
        break;
    }
  }

  if (doms.length > 0) facts.dayOfMonth = doms[doms.length - 1]!;

  return facts;
}

function resolveTime(
  hour: number | undefined,
  minute: number | undefined,
  meridiem: "am" | "pm" | undefined,
): TimeOfDay {
  let resolved = hour ?? 0;

  if (meridiem === "am") {
    if (resolved < 1 || resolved > 12) throw new RangeError(`invalid 12-hour time: ${resolved}`);
    resolved = resolved === 12 ? 0 : resolved;
  } else if (meridiem === "pm") {
    if (resolved < 1 || resolved > 12) throw new RangeError(`invalid 12-hour time: ${resolved}`);
    resolved = resolved === 12 ? 12 : resolved + 12;
  }

  return { hour: resolved, minute: minute ?? 0 };
}

/**
 * Turn labeled tokens into a validated intent. Throws {@link UnknownRoleError}
 * for an out-of-range role id and `RangeError` for an unrecognized or
 * out-of-bounds schedule.
 */
export function decodeIntent(labeled: readonly LabeledToken[]): DecodedIntent {
  const entries: Entry[] = [];

  for (const item of labeled) {
    if (!Number.isInteger(item.role) || item.role < 0 || item.role >= LABEL_COUNT) {
      throw new UnknownRoleError(item.role);
    }

    if (item.token.kind === 3) continue;

    entries.push({
      token: item.token,
      label: labelFromId(item.role),
      boundary: item.boundary,
      score: item.score,
    });
  }

  if (entries.length === 0) throw new RangeError("decodeIntent: no tokens");

  const lexical = exactLexicalIntent(entries);

  if (lexical !== null) return lexical;

  let interval: number | undefined;
  let unit: FrequencyUnit | undefined;
  let other = false;
  let ordinal: Ordinal | undefined;
  let dayOfMonth: number | undefined;
  let hour: number | undefined;
  let minute: number | undefined;
  let meridiem: "am" | "pm" | undefined;
  let minuteRelation: "past" | "to" | undefined;
  let day: DaySpec | undefined;
  const weekdays: Weekday[] = [];
  const bounds: RecurrenceBounds = {};
  const exceptions: ScheduleExceptions = {};

  for (const clause of splitClauses(entries)) {
    const facts = decodeClause(groupClause(clause));

    interval ??= facts.interval;
    unit ??= facts.unit;
    other ||= facts.other;
    weekdays.push(...facts.weekdays);
    ordinal ??= facts.ordinal;
    dayOfMonth ??= facts.dayOfMonth;

    if (bounds.start === undefined && facts.bounds.start !== undefined) {
      bounds.start = facts.bounds.start;
    }

    if (bounds.end === undefined && facts.bounds.end !== undefined) bounds.end = facts.bounds.end;
    hour ??= facts.hour;
    minute ??= facts.minute;
    meridiem ??= facts.meridiem;
    minuteRelation ??= facts.minuteRelation;
    day ??= facts.day;

    if (facts.exceptions.weekdays !== undefined) {
      (exceptions.weekdays ??= []).push(...facts.exceptions.weekdays);
    }

    if (facts.exceptions.dates !== undefined) {
      (exceptions.dates ??= []).push(...facts.exceptions.dates);
    }
  }

  let resolvedUnit = unit;

  if (resolvedUnit === undefined) {
    if (ordinal !== undefined || day !== undefined) {
      resolvedUnit = "month";
    } else if (dayOfMonth !== undefined) {
      resolvedUnit = "month";
    } else if (weekdays.length > 0) {
      resolvedUnit = "week";
    } else if (bounds.start !== undefined || bounds.end !== undefined) {
      resolvedUnit = "day";
    } else if (hour !== undefined || minute !== undefined || meridiem !== undefined) {
      resolvedUnit = "day";
    }
  }

  if (resolvedUnit === undefined) throw new RangeError("decodeIntent: no recognizable schedule");

  const resolvedInterval = other ? 2 : (interval ?? 1);

  const intent: ScheduleIntent = {
    frequency: { unit: resolvedUnit, interval: resolvedInterval },
  };

  if (ordinal !== undefined) {
    const weekday = normalizeWeekdays(weekdays)[0];

    if (weekday === undefined) throw new RangeError("decodeIntent: ordinal without a weekday");
    intent.ordinalWeekday = { ordinal, weekday };
  } else if (weekdays.length > 0) {
    intent.weekdays = normalizeWeekdays(weekdays);
  }

  if (dayOfMonth !== undefined) intent.dayOfMonth = dayOfMonth;
  if (day !== undefined) intent.day = day;
  if (bounds.start !== undefined || bounds.end !== undefined) intent.bounds = bounds;

  if (exceptions.weekdays !== undefined) {
    exceptions.weekdays = normalizeWeekdays(exceptions.weekdays);
  }

  if (exceptions.weekdays !== undefined || exceptions.dates !== undefined) {
    intent.exceptions = exceptions;
  }

  if (hour !== undefined || minute !== undefined || meridiem !== undefined) {
    intent.time = resolveTime(hour, minute, meridiem);

    if (minuteRelation === "to") {
      const total = (intent.time.hour * 60 - intent.time.minute + 1440) % 1440;

      intent.time = { hour: Math.floor(total / 60), minute: total % 60 };
    }
  }

  const dayGroupEvidence = explicitDayGroup(entries);

  if (dayGroupEvidence !== undefined) {
    // An explicit, separate daily recurrence is genuinely contradictory. A
    // spurious daily unit inferred from the word "weekday" is safe to repair.
    const text = joinEntries(entries);

    if (/\b(?:every|each)\s+day\b/.test(text)) {
      throw new RangeError("decodeIntent: weekdays conflict with daily frequency");
    }

    intent.frequency.unit = "week";
    intent.weekdays = normalizeWeekdays([...(intent.weekdays ?? []), ...dayGroupEvidence]);
  }

  const boundEvidence = explicitBounds(entries);

  if (boundEvidence !== undefined) intent.bounds = reconcileBounds(intent.bounds, boundEvidence);

  const failure =
    validateInterval(intent.frequency.unit, intent.frequency.interval) ??
    validateTime(intent.time) ??
    validateWeekdays(intent.weekdays) ??
    validateDayOfMonth(intent.dayOfMonth) ??
    validateOrdinalWeekday(intent.ordinalWeekday);

  if (failure !== null) throw new RangeError(failure.reason);

  let confidence = 1;

  for (const entry of entries) {
    if (entry.label === "O" || isStructural(entry.label)) continue;
    confidence = Math.min(confidence, entry.score);
  }

  return { intent, confidence };
}
