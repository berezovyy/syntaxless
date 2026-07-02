import type {
  FrequencyUnit,
  OrdinalWeekday,
  ScheduleIntent,
  TimeOfDay,
  Weekday,
} from "../types/intent.js";
import {
  ADVERB_FREQUENCIES,
  NUMBER_WORDS,
  ORDINAL_ALIASES,
  UNIT_ALIASES,
  WEEKDAY_ALIASES,
  WEEKDAY_GROUPS,
  WEEKDAY_SEQUENCE,
} from "./lexicon.js";
import { parseWithModel } from "./model.js";
import type { ModelParserOptions } from "./model.js";
import { normalize } from "./normalize.js";

export type ParseFailureCode =
  | "empty"
  | "unrecognized"
  | "unrecognized-time"
  | "unrecognized-weekday"
  | "unrecognized-interval"
  | "conflicting-selectors"
  | "low-confidence"
  | "model-error";

/** Which understanding step produced a {@link ParseResult}. */
export type ParseSource = "rules" | "model";

/** A machine-readable note attached to a parse. Never carries an intent fact. */
export interface Diagnostic {
  code: string;
  severity: "error" | "warning" | "suggestion";
  message: string;
}

/**
 * Additive extension of the Phase 04 result: the original `ok`/`intent`/
 * `reason`/`code` members are unchanged, and every variant now carries the
 * confidence, diagnostics, and the source that produced it.
 */
export type ParseResult =
  | {
      ok: true;
      intent: ScheduleIntent;
      confidence: number;
      diagnostics: Diagnostic[];
      source: ParseSource;
    }
  | {
      ok: false;
      reason: string;
      code: ParseFailureCode;
      confidence: number;
      diagnostics: Diagnostic[];
      source: ParseSource;
    };

interface Frequency {
  unit: FrequencyUnit;
  interval: number;
}

function fail(
  code: ParseFailureCode,
  reason: string,
  source: ParseSource = "rules",
  diagnostics: Diagnostic[] = [],
): ParseResult {
  return { ok: false, reason, code, confidence: 0, diagnostics, source };
}

function addWeekdayRange(target: Weekday[], start: Weekday, end: Weekday): void {
  const startIndex = WEEKDAY_SEQUENCE.indexOf(start);
  const endIndex = WEEKDAY_SEQUENCE.indexOf(end);
  let index = startIndex;

  for (;;) {
    target.push(WEEKDAY_SEQUENCE[index]!);
    if (index === endIndex) return;
    index = (index + 1) % WEEKDAY_SEQUENCE.length;
  }
}

function normalizeWeekdayList(days: Weekday[]): Weekday[] {
  const selected = new Set(days);

  return WEEKDAY_SEQUENCE.filter((day) => selected.has(day));
}

function resolveMeridiem(
  hour: number,
  minute: number,
  meridiem: string | undefined,
): TimeOfDay | "invalid" {
  if (meridiem === "pm") {
    if (hour < 1 || hour > 12) return "invalid";
    hour = hour === 12 ? 12 : hour + 12;
  } else if (meridiem === "am") {
    if (hour < 1 || hour > 12) return "invalid";
    hour = hour === 12 ? 0 : hour;
  } else if (hour < 0 || hour > 23) {
    return "invalid";
  }

  if (minute < 0 || minute > 59) return "invalid";

  return { hour, minute };
}

function parseTime(text: string): TimeOfDay | "invalid" | null {
  const numeric = /\bat\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b/.exec(text);

  if (numeric) {
    const hour = Number(numeric[1]!);
    const minute = numeric[2] === undefined ? 0 : Number(numeric[2]);

    return resolveMeridiem(hour, minute, numeric[3]);
  }

  const word = /\bat\s+([a-z]+)(?:\s+(am|pm))?\b/.exec(text);

  if (word) {
    const value = NUMBER_WORDS[word[1]!];

    if (value === undefined) return "invalid";

    return resolveMeridiem(value, 0, word[2]);
  }

  if (/\bat\b/.test(text)) return "invalid";

  return null;
}

/**
 * v0 rule-based parser. It deliberately recognizes a small, closed grammar and fails
 * explicitly on anything else. It is the baseline the model path must match and the
 * fallback reachable through `parse(text, { source: "rules" })`.
 */
async function parseWithRules(text: string): Promise<ParseResult> {
  const normalized = normalize(text);

  if (normalized.length === 0) {
    return fail("empty", "empty input");
  }

  const time = parseTime(normalized);

  if (time === "invalid") {
    return fail("unrecognized-time", `unrecognized time in: ${normalized}`);
  }

  const tokens = normalized.split(/[\s,]+/).filter((token) => token.length > 0);
  const used = new Array<boolean>(tokens.length).fill(false);

  let ordinalWeekday: OrdinalWeekday | null = null;

  for (let i = 0; i < tokens.length - 1; i += 1) {
    const ordinal = ORDINAL_ALIASES[tokens[i]!];
    const weekday = WEEKDAY_ALIASES[tokens[i + 1]!];

    if (ordinal !== undefined && weekday !== undefined) {
      ordinalWeekday = { ordinal, weekday };
      used[i] = true;
      used[i + 1] = true;
      break;
    }
  }

  let dayOfMonth: number | null = null;

  for (let i = 0; i < tokens.length; i += 1) {
    if (used[i]) continue;
    const token = tokens[i]!;

    const ordinalDay = /^(\d{1,2})(?:st|nd|rd|th)$/.exec(token);

    if (ordinalDay) {
      dayOfMonth = Number(ordinalDay[1]!);
      used[i] = true;
      continue;
    }

    if (token === "day") {
      const next = tokens[i + 1];

      if (next !== undefined && /^\d{1,2}$/.test(next)) {
        dayOfMonth = Number(next);
        used[i] = true;
        used[i + 1] = true;
      }
    }
  }

  if (dayOfMonth !== null && (dayOfMonth < 1 || dayOfMonth > 31)) {
    return fail("unrecognized", `invalid day of month: ${dayOfMonth}`);
  }

  const foundWeekdays: Weekday[] = [];
  let unknownDayLike: string | null = null;

  for (let i = 0; i < tokens.length; i += 1) {
    if (used[i]) continue;
    const token = tokens[i]!;

    if (token.includes("-")) {
      const parts = token.split("-");
      const start = parts[0] === undefined ? undefined : WEEKDAY_ALIASES[parts[0]];
      const end = parts[1] === undefined ? undefined : WEEKDAY_ALIASES[parts[1]];

      if (parts.length === 2 && start !== undefined && end !== undefined) {
        addWeekdayRange(foundWeekdays, start, end);
        used[i] = true;
        continue;
      }
    }

    const group = WEEKDAY_GROUPS[token];

    if (group !== undefined) {
      foundWeekdays.push(...group);
      used[i] = true;
      continue;
    }

    const weekday = WEEKDAY_ALIASES[token];

    if (weekday !== undefined) {
      const connector = tokens[i + 1];
      const endToken = tokens[i + 2];

      if (
        (connector === "through" || connector === "to" || connector === "-") &&
        endToken !== undefined &&
        WEEKDAY_ALIASES[endToken] !== undefined
      ) {
        addWeekdayRange(foundWeekdays, weekday, WEEKDAY_ALIASES[endToken]!);
        used[i] = true;
        used[i + 1] = true;
        used[i + 2] = true;
        continue;
      }

      foundWeekdays.push(weekday);
      used[i] = true;
      continue;
    }

    if (
      /^[a-z]+$/.test(token) &&
      (token.endsWith("day") || token.endsWith("days")) &&
      UNIT_ALIASES[token] === undefined
    ) {
      unknownDayLike = token;
    }
  }

  if (unknownDayLike !== null) {
    return fail("unrecognized-weekday", `unknown weekday: ${unknownDayLike}`);
  }

  const frequencies: Frequency[] = [];

  for (let i = 0; i < tokens.length; i += 1) {
    const token = tokens[i]!;

    const adverbUnit = ADVERB_FREQUENCIES[token];

    if (adverbUnit !== undefined) {
      frequencies.push({ unit: adverbUnit, interval: 1 });
      continue;
    }

    if (token !== "every" && token !== "each") continue;

    const next = tokens[i + 1];

    if (next === undefined) {
      return fail("unrecognized-interval", "missing interval after 'every'");
    }

    const unit = UNIT_ALIASES[next];

    if (unit !== undefined) {
      frequencies.push({ unit, interval: 1 });
      i += 1;
      continue;
    }

    const numeric = NUMBER_WORDS[next] ?? (/^\d{1,3}$/.test(next) ? Number(next) : undefined);

    if (numeric !== undefined) {
      const intervalUnit = tokens[i + 2] === undefined ? undefined : UNIT_ALIASES[tokens[i + 2]!];

      if (intervalUnit === undefined) {
        return fail("unrecognized-interval", `missing unit after '${next}'`);
      }

      frequencies.push({ unit: intervalUnit, interval: numeric });
      i += 2;
      continue;
    }

    if (next === "other") {
      const after = tokens[i + 2];

      if (after !== undefined && UNIT_ALIASES[after] !== undefined) {
        frequencies.push({ unit: UNIT_ALIASES[after]!, interval: 2 });
        i += 2;
        continue;
      }

      if (after !== undefined && WEEKDAY_ALIASES[after] !== undefined) {
        frequencies.push({ unit: "week", interval: 2 });
        i += 2;
        continue;
      }

      return fail("unrecognized-interval", "missing unit after 'every other'");
    }

    if (WEEKDAY_ALIASES[next] !== undefined || WEEKDAY_GROUPS[next] !== undefined) {
      continue;
    }

    return fail("unrecognized-interval", `unknown interval: ${next}`);
  }

  let frequency: Frequency | null = null;

  for (const candidate of frequencies) {
    if (frequency === null) {
      frequency = candidate;
      continue;
    }

    if (frequency.unit !== candidate.unit) {
      return fail("conflicting-selectors", "multiple frequency units were specified");
    }
  }

  if (foundWeekdays.length > 0) {
    if (frequency === null) {
      frequency = { unit: "week", interval: 1 };
    } else if (frequency.unit !== "week") {
      return fail("conflicting-selectors", "weekdays conflict with the selected frequency");
    }
  }

  if (dayOfMonth !== null || ordinalWeekday !== null) {
    if (frequency === null) {
      frequency = { unit: "month", interval: 1 };
    } else if (frequency.unit !== "month") {
      return fail("conflicting-selectors", "day-of-month conflicts with the selected frequency");
    }
  }

  if (foundWeekdays.length > 0 && dayOfMonth !== null) {
    return fail("conflicting-selectors", "weekdays and dayOfMonth cannot be combined");
  }

  if (frequency === null && time !== null) {
    frequency = { unit: "day", interval: 1 };
  }

  const recognized =
    time !== null ||
    foundWeekdays.length > 0 ||
    dayOfMonth !== null ||
    ordinalWeekday !== null ||
    frequencies.length > 0;

  if (frequency === null || !recognized) {
    return fail("unrecognized", `unrecognized schedule: ${normalized}`);
  }

  const intent: ScheduleIntent = { frequency };

  if (time !== null) intent.time = time;
  if (foundWeekdays.length > 0) intent.weekdays = normalizeWeekdayList(foundWeekdays);
  if (dayOfMonth !== null) intent.dayOfMonth = dayOfMonth;
  if (ordinalWeekday !== null) intent.ordinalWeekday = ordinalWeekday;

  return { ok: true, intent, confidence: 1, diagnostics: [], source: "rules" };
}

/**
 * Public async parse. The model path is the default; `{ source: "rules" }`
 * reaches the deterministic Phase 04 baseline. Both paths return the same
 * additive {@link ParseResult}, so `cron()` never needs to know which ran.
 */
export async function parse(text: string, options?: ModelParserOptions): Promise<ParseResult> {
  if (options?.source === "rules") return parseWithRules(text);

  return parseWithModel(text, options);
}
