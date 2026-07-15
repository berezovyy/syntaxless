import { WEEKDAY_NUMBER } from "../compiler/cron.js";
import type { ScheduleIntent } from "../types/intent.js";
import { validate } from "../validate.js";
import {
  civilEpoch,
  dateText,
  DAY,
  FIXED_CALENDAR,
  fromEpoch,
  instantValue,
  MAX_DATE,
  MIN_DATE,
  RecurrenceError,
  sameLocal,
} from "./calendar.js";
import type { LocalDateTime, RecurrenceCalendar, RecurrenceFailureCode } from "./calendar.js";

export interface RecurrencePolicies {
  gap?: "skip" | "earlier" | "later" | "reject";
  fold?: "earlier" | "later" | "both" | "reject";
  monthEnd?: "skip" | "constrain" | "reject";
}

export interface RecurrenceOptions {
  timezone: string;
  reference: string | number;
  /** Phase only, not an implicit start bound. Defaults to the reference instant. */
  anchor?: string | number;
  count?: number;
  /** Exclusive absolute horizon. At least count or before is required. */
  before?: string | number;
  policies?: RecurrencePolicies;
  calendar?: RecurrenceCalendar;
  /** Bounded work as well as bounded output. Maximum 100000, default 10000. */
  maxSteps?: number;
}

export interface Occurrence {
  instant: string;
  local: LocalDateTime;
  requestedLocal: LocalDateTime;
}

export type RecurrenceResult =
  | {
      ok: true;
      occurrences: Occurrence[];
      stop: "count" | "horizon" | "intent-end" | "range-end";
      examinedSteps: number;
      policies: Required<RecurrencePolicies>;
    }
  | { ok: false; code: RecurrenceFailureCode; reason: string };

function keys(value: unknown, allowed: readonly string[]): value is Record<string, unknown> {
  return (
    value !== null &&
    typeof value === "object" &&
    !Array.isArray(value) &&
    Reflect.ownKeys(value).every((key) => typeof key === "string" && allowed.includes(key))
  );
}

function integer(value: unknown, max: number): value is number {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 1 && value <= max;
}

function context(options: RecurrenceOptions) {
  if (
    !keys(options, [
      "timezone",
      "reference",
      "anchor",
      "count",
      "before",
      "policies",
      "calendar",
      "maxSteps",
    ]) ||
    typeof options.timezone !== "string" ||
    !options.timezone ||
    (options.count === undefined && options.before === undefined) ||
    (options.count !== undefined && !integer(options.count, 1000)) ||
    (options.maxSteps !== undefined && !integer(options.maxSteps, 100000))
  ) {
    throw new RecurrenceError(
      "invalid-context",
      "provide a timezone, reference, and count from 1 to 1000 or an exclusive horizon",
    );
  }

  const reference = instantValue(options.reference);
  const anchor = options.anchor === undefined ? reference : instantValue(options.anchor);
  const before = options.before === undefined ? MAX_DATE : instantValue(options.before);

  if (before <= reference) {
    throw new RecurrenceError("invalid-context", "the horizon must follow the reference instant");
  }

  const p = options.policies === undefined ? {} : options.policies;

  if (
    !keys(p, ["gap", "fold", "monthEnd"]) ||
    (p.gap !== undefined &&
      (typeof p.gap !== "string" || !["skip", "earlier", "later", "reject"].includes(p.gap))) ||
    (p.fold !== undefined &&
      (typeof p.fold !== "string" || !["earlier", "later", "both", "reject"].includes(p.fold))) ||
    (p.monthEnd !== undefined &&
      (typeof p.monthEnd !== "string" || !["skip", "constrain", "reject"].includes(p.monthEnd)))
  ) {
    throw new RecurrenceError(
      "invalid-context",
      "provide supported gap, fold and month-end policies",
    );
  }

  const calendar = options.calendar === undefined ? FIXED_CALENDAR : options.calendar;

  if (
    !calendar ||
    typeof calendar.toLocal !== "function" ||
    typeof calendar.resolve !== "function"
  ) {
    throw new RecurrenceError("invalid-context", "provide a recurrence calendar adapter");
  }

  return {
    reference,
    anchor,
    before,
    calendar,
    policies: {
      gap: p.gap ?? "skip",
      fold: p.fold ?? "earlier",
      monthEnd: p.monthEnd ?? "skip",
    } as Required<RecurrencePolicies>,
  };
}

function localValue(calendar: RecurrenceCalendar, epoch: number, timezone: string): LocalDateTime {
  const pseudo = civilEpoch(calendar.toLocal(epoch, timezone));

  if (!Number.isSafeInteger(epoch) || Math.abs(pseudo - epoch) >= DAY) {
    throw new RecurrenceError(
      "invalid-calendar",
      "calendar offsets must be strictly below 24 hours",
    );
  }

  return fromEpoch(pseudo);
}

function resolveValues(calendar: RecurrenceCalendar, local: LocalDateTime, timezone: string) {
  const pseudo = civilEpoch(local);
  const earlier = calendar.resolve(local, timezone, "earlier");
  const later = calendar.resolve(local, timezone, "later");

  if (
    ![earlier, later].every(
      (value) => Number.isSafeInteger(value) && Math.abs(value - pseudo) < DAY,
    ) ||
    earlier > later
  ) {
    throw new RecurrenceError(
      "invalid-calendar",
      "calendar resolutions must be ordered instants within 24 hours of the civil fields",
    );
  }

  const a = localValue(calendar, earlier, timezone);
  const b = localValue(calendar, later, timezone);

  const aMatches = sameLocal(a, local);
  const bMatches = sameLocal(b, local);

  if (aMatches !== bMatches || (earlier === later && !aMatches)) {
    throw new RecurrenceError(
      "invalid-calendar",
      "calendar resolutions must consistently identify an ordinary time, gap or fold",
    );
  }

  return { earlier, later, gap: !aMatches, fold: aMatches && earlier !== later };
}

const dayStart = (epoch: number): number => Math.floor(epoch / DAY) * DAY;

const monday = (epoch: number): number =>
  dayStart(epoch) - ((new Date(epoch).getUTCDay() + 6) % 7) * DAY;

interface Candidate {
  local: LocalDateTime;
  missingDay?: boolean;
}

/** Enumerate civil phases directly, rather than scanning from the original anchor. */
function* candidates(
  intent: ScheduleIntent,
  anchor: LocalDateTime,
  start: number,
): Generator<Candidate> {
  const { unit, interval } = intent.frequency;
  const clock = (intent.time?.hour ?? 0) * 3_600_000 + (intent.time?.minute ?? 0) * 60_000;
  const anchorEpoch = civilEpoch(anchor);

  if (unit === "month") {
    const base = (anchor.year - 1) * 12 + anchor.month - 1;
    const first = fromEpoch(start);
    let period = Math.floor(((first.year - 1) * 12 + first.month - 1 - base) / interval);

    while (true) {
      const monthIndex = base + period * interval;

      period++;
      if (monthIndex < 0) continue;
      if (monthIndex >= 9999 * 12) return;

      const year = Math.floor(monthIndex / 12) + 1;
      const month = (monthIndex % 12) + 1;

      const firstEpoch = civilEpoch({
        year,
        month,
        day: 1,
        hour: 0,
        minute: 0,
        second: 0,
        millisecond: 0,
      });

      const next = new Date(firstEpoch);

      next.setUTCMonth(next.getUTCMonth() + 1);
      const lastDay = (next.getTime() - firstEpoch) / DAY;
      let selected = intent.dayOfMonth ?? anchor.day;

      if (intent.ordinalWeekday !== undefined) {
        const { ordinal, weekday } = intent.ordinalWeekday;
        const target = WEEKDAY_NUMBER[weekday];

        selected =
          ordinal === "last"
            ? lastDay - ((new Date(firstEpoch + (lastDay - 1) * DAY).getUTCDay() - target + 7) % 7)
            : 1 +
              ((target - new Date(firstEpoch).getUTCDay() + 7) % 7) +
              ["first", "second", "third", "fourth"].indexOf(ordinal) * 7;
      } else if (intent.day !== undefined) {
        selected = lastDay;

        if (intent.day.type === "last_business_day") {
          const weekday = new Date(firstEpoch + (lastDay - 1) * DAY).getUTCDay();

          if (weekday === 0) selected -= 2;
          if (weekday === 6) selected -= 1;
        }
      }

      yield {
        local: fromEpoch(firstEpoch + (Math.min(selected, lastDay) - 1) * DAY + clock),
        ...(selected > lastDay ? { missingDay: true } : {}),
      };
    }
  }

  if (unit === "week") {
    const base = monday(anchorEpoch);
    const step = interval * 7 * DAY;

    const weekdays =
      intent.weekdays === undefined
        ? [(new Date(anchorEpoch).getUTCDay() + 6) % 7]
        : intent.weekdays.map((day) => (WEEKDAY_NUMBER[day] + 6) % 7).sort((a, b) => a - b);

    let period = Math.floor((start - base) / step);

    while (true) {
      const week = base + period * step;

      period++;
      if (week + 7 * DAY < MIN_DATE) continue;
      if (week >= MAX_DATE) return;

      for (const weekday of weekdays) {
        const value = week + weekday * DAY + clock;

        if (value >= MIN_DATE && value < MAX_DATE) yield { local: fromEpoch(value) };
      }
    }
  }

  const scale = unit === "minute" ? 60_000 : unit === "hour" ? 3_600_000 : DAY;

  const base =
    unit === "minute"
      ? Math.floor(anchorEpoch / 60_000) * 60_000
      : unit === "hour"
        ? Math.floor(anchorEpoch / 3_600_000) * 3_600_000 + (intent.time?.minute ?? 0) * 60_000
        : dayStart(anchorEpoch) + clock;

  const step = interval * scale;
  let period = Math.ceil((Math.max(MIN_DATE, start) - base) / step);

  while (true) {
    const value = base + period * step;

    period++;
    if (value < MIN_DATE) continue;
    if (value >= MAX_DATE) return;
    yield { local: fromEpoch(value) };
  }
}

interface Stored {
  epoch: number;
  occurrence: Occurrence;
}

/** Pure preview. The reference is exclusive, and no timers or tasks are created. */
export function recurrences(intent: ScheduleIntent, options: RecurrenceOptions): RecurrenceResult {
  validate(intent);

  try {
    const { reference, anchor, before, calendar, policies } = context(options);

    localValue(calendar, reference, options.timezone);
    const anchorLocal = localValue(calendar, anchor, options.timezone);

    const startBound = intent.bounds?.start;
    const endBound = intent.bounds?.end;

    const start = Math.max(
      MIN_DATE,
      reference - DAY,
      startBound === undefined ? MIN_DATE : new Date(`${startBound}T00:00:00Z`).getTime() - 2 * DAY,
    );

    const end = Math.min(
      MAX_DATE,
      endBound === undefined ? MAX_DATE : new Date(`${endBound}T00:00:00Z`).getTime() + 2 * DAY,
    );

    const excludedDates = new Set(intent.exceptions?.dates);
    const excludedDays = new Set(intent.exceptions?.weekdays?.map((day) => WEEKDAY_NUMBER[day]));
    const best: Stored[] = [];
    let examinedSteps = 0;
    let pending: { epoch: number; error: RecurrenceError } | undefined;

    const eligible = (epoch: number): LocalDateTime | undefined => {
      if (epoch <= reference || epoch >= before || epoch < MIN_DATE || epoch >= MAX_DATE) {
        return undefined;
      }

      const local = localValue(calendar, epoch, options.timezone);
      const date = dateText(local);

      if (
        (startBound !== undefined && date < startBound) ||
        (endBound !== undefined && date >= endBound) ||
        excludedDates.has(date) ||
        excludedDays.has(new Date(civilEpoch(local)).getUTCDay())
      ) {
        return undefined;
      }

      return local;
    };

    const finish = (stop: "count" | "horizon" | "intent-end" | "range-end"): RecurrenceResult => {
      const cutoff = stop === "count" ? best[best.length - 1]!.epoch : before;

      if (pending !== undefined && pending.epoch <= cutoff) throw pending.error;

      return {
        ok: true,
        occurrences: best.map((item) => item.occurrence),
        stop,
        examinedSteps,
        policies,
      };
    };

    for (const candidate of candidates(intent, anchorLocal, start)) {
      const pseudo = civilEpoch(candidate.local);

      if (
        options.count !== undefined &&
        best.length >= options.count &&
        pseudo - DAY >= best[best.length - 1]!.epoch
      ) {
        return finish("count");
      }

      if (pseudo - DAY >= before) return finish("horizon");
      if (pseudo >= end) return finish("intent-end");

      if (++examinedSteps > (options.maxSteps ?? 10000)) {
        throw new RecurrenceError("work-limit", "the preview exceeded its candidate budget");
      }

      if (candidate.missingDay && policies.monthEnd === "skip") continue;
      const resolved = resolveValues(calendar, candidate.local, options.timezone);

      const code =
        candidate.missingDay && policies.monthEnd === "reject"
          ? "month-end"
          : resolved.gap && policies.gap === "reject"
            ? "nonexistent-time"
            : resolved.fold && policies.fold === "reject"
              ? "repeated-time"
              : undefined;

      if (code !== undefined) {
        for (const epoch of [resolved.earlier, resolved.later]) {
          if (eligible(epoch) !== undefined && (pending === undefined || epoch < pending.epoch)) {
            pending = {
              epoch,
              error: new RecurrenceError(code, `the requested recurrence encountered ${code}`),
            };
          }
        }

        continue;
      }

      if (resolved.gap && policies.gap === "skip") continue;

      const epochs =
        resolved.fold && policies.fold === "both"
          ? [resolved.earlier, resolved.later]
          : [
              resolved.gap
                ? policies.gap === "earlier"
                  ? resolved.earlier
                  : resolved.later
                : policies.fold === "later"
                  ? resolved.later
                  : resolved.earlier,
            ];

      for (const epoch of epochs) {
        const local = eligible(epoch);

        if (local === undefined) continue;

        let low = 0;
        let high = best.length;

        while (low < high) {
          const middle = (low + high) >>> 1;

          if (best[middle]!.epoch < epoch) low = middle + 1;
          else high = middle;
        }

        if (best[low]?.epoch === epoch) continue;

        if (options.count !== undefined && best.length >= options.count && low >= options.count) {
          continue;
        }

        best.splice(low, 0, {
          epoch,
          occurrence: {
            instant: new Date(epoch).toISOString(),
            local,
            requestedLocal: { ...candidate.local },
          },
        });

        if (options.count !== undefined && best.length > options.count) best.pop();

        if (best.length > 1000) {
          throw new RecurrenceError(
            "output-limit",
            "a preview cannot contain more than 1000 occurrences",
          );
        }
      }
    }

    if (options.count !== undefined && best.length >= options.count) return finish("count");

    return finish(endBound === undefined ? "range-end" : "intent-end");
  } catch (error) {
    if (error instanceof RecurrenceError) {
      return { ok: false, code: error.code, reason: error.message };
    }

    if (error instanceof RangeError) {
      return { ok: false, code: "invalid-calendar", reason: error.message };
    }

    throw error;
  }
}
