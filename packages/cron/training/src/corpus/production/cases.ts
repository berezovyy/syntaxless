import { validateInterval } from "../../../../src/compiler/validate";
/** Explicit semantic grids. Surface forms and gold ASTs are constructed together,
 * never recovered from model predictions or evaluation examples. */
import {
  hourLabel,
  intervalLabel,
  type Label,
  minuteLabel,
  ordinalLabel,
  unitLabel,
  weekdayLabel,
} from "../../../../src/model/labels";
import type {
  FrequencyUnit,
  Ordinal,
  ScheduleIntent,
  TimeOfDay,
  Weekday,
} from "../../../../src/types/intent";
import type { CaseSet, Fragment, SemanticCase } from "./constructions";

const fragment = (text: string, label: Label, separator?: string): Fragment => ({
  text,
  label,
  ...(separator === undefined ? {} : { separator }),
});

const glue = (text: string): Fragment => fragment(text, "GLUE");

const DAYS: readonly Weekday[] = [
  "monday",
  "tuesday",
  "wednesday",
  "thursday",
  "friday",
  "saturday",
  "sunday",
];

const UNITS: readonly FrequencyUnit[] = ["minute", "hour", "day", "week", "month"];
const ORDINALS: readonly Ordinal[] = ["first", "second", "third", "fourth", "last"];

const WORDS: readonly string[] = [
  "zero",
  "one",
  "two",
  "three",
  "four",
  "five",
  "six",
  "seven",
  "eight",
  "nine",
  "ten",
  "eleven",
  "twelve",
] as const;

export const INTERVAL_VALUES = [
  1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 17, 20, 23, 25, 29, 30, 31, 45, 59, 60, 90, 120,
] as const;

/** Runtime-admitted semantic domain, not a relabeling of rejected requests. */
function intervalValues(unit: FrequencyUnit): readonly number[] {
  return INTERVAL_VALUES.filter((value) => validateInterval(unit, value) === null);
}

export function intervalDomainExclusions() {
  return UNITS.flatMap((unit) =>
    INTERVAL_VALUES.flatMap((interval) => {
      const failure = validateInterval(unit, interval);

      return failure
        ? [
            {
              unit,
              interval,
              reason: failure.reason,
              action: "excluded-from-positive-and-inner-schedule-gold-not-relabeled-as-negative",
            },
          ]
        : [];
    }),
  );
}

const SHORT_UNITS: Record<FrequencyUnit, string> = {
  minute: "min",
  hour: "hr",
  day: "day",
  week: "wk",
  month: "mo",
};

export const CLOCK_STYLES = ["24h", "12h", "natural"] as const;
export type ClockStyle = (typeof CLOCK_STYLES)[number];

export function clockFragments(time: TimeOfDay, style: ClockStyle): Fragment[] {
  const { hour, minute } = time;
  const meridiem = fragment(hour < 12 ? "am" : "pm", hour < 12 ? "MERIDIEM_AM" : "MERIDIEM_PM");
  const shownHour = hour % 12 || 12;

  if (style === "natural") {
    if (minute === 0 && (hour === 0 || hour === 12)) {
      return [fragment(hour === 0 ? "midnight" : "noon", hourLabel(hour))];
    }

    if (minute === 0) return [fragment(WORDS[shownHour]!, hourLabel(hour)), meridiem];

    if (minute === 15 || minute === 30) {
      return [
        fragment(minute === 15 ? "quarter" : "half", minuteLabel(minute)),
        glue("past"),
        fragment(WORDS[shownHour]!, hourLabel(hour)),
        meridiem,
      ];
    }

    if (minute === 45) {
      const next = (hour + 1) % 24;

      return [
        fragment("quarter", "MINUTE_45"),
        glue("to"),
        fragment(WORDS[next % 12 || 12]!, hourLabel(hour)),
        fragment(next < 12 ? "am" : "pm", next < 12 ? "MERIDIEM_AM" : "MERIDIEM_PM"),
      ];
    }

    // Non-quarter word minutes cannot be represented by MINUTE_LIT.
    throw new Error(`Unrepresentable natural clock: ${hour}:${minute}`);
  }

  const parts = [
    fragment(
      String(style === "24h" ? hour : shownHour).padStart(style === "24h" ? 2 : 1, "0"),
      hourLabel(hour),
    ),
    fragment(":", "GLUE", ""),
    fragment(String(minute).padStart(2, "0"), minuteLabel(minute), ""),
  ];

  if (style === "12h") parts.push(meridiem);

  return parts;
}

function styles(time: TimeOfDay): readonly ClockStyle[] {
  return time.minute % 15 === 0 ? CLOCK_STYLES : ["24h", "12h"];
}

function* allClocks(): Generator<TimeOfDay> {
  for (let hour = 0; hour < 24; hour++) {
    for (let minute = 0; minute < 60; minute++) yield { hour, minute };
  }
}

/** Every selector sees every minute and all 24 hours, without a fixed selector/time correlation. */
function* selectorClocks(offset: number): Generator<TimeOfDay> {
  for (let minute = 0; minute < 60; minute++) yield { hour: (minute + offset) % 24, minute };
}

function weekdays(days: readonly Weekday[], abbreviated: boolean): Fragment[] {
  return days.flatMap((day, index) => [
    ...(index ? [glue("and")] : []),
    fragment(abbreviated ? day.slice(0, 3) : day, weekdayLabel(day)),
  ]);
}

function suffix(day: number): string {
  return day % 100 >= 11 && day % 100 <= 13
    ? "th"
    : ({ 1: "st", 2: "nd", 3: "rd" }[day % 10] ?? "th");
}

function* intervals(): Generator<SemanticCase> {
  for (const unit of UNITS) {
    for (const n of intervalValues(unit)) {
      for (const words of n >= 2 && n <= 12 ? [false, true] : [false]) {
        for (const abbreviated of [false, true]) {
          // There is no value-one closed role; spelling 'one' as INTERVAL_LIT would silently decode incorrectly.
          const slots = {
            n: [fragment(words ? WORDS[n]! : String(n), intervalLabel(n))],
            unit: [
              fragment(
                abbreviated ? SHORT_UNITS[unit] : `${unit}${n === 1 ? "" : "s"}`,
                unitLabel(unit),
              ),
            ],
          };

          yield {
            id: `${unit}-${n}-${words ? "word" : "digit"}-${abbreviated ? "short" : "full"}`,
            semanticFamily: `${unit}-interval`,
            family: "interval",
            intent: { frequency: { unit, interval: n } },
            slots,
            tags: [
              ...(words ? ["number-words-representable"] : []),
              ...(abbreviated ? ["unit-abbreviation"] : []),
            ],
          };
        }
      }
    }
  }
}

function* daily(): Generator<SemanticCase> {
  for (const time of allClocks()) {
    for (const style of styles(time)) {
      yield {
        id: `${time.hour}-${time.minute}-${style}`,
        semanticFamily: "daily-clock",
        family: "time-of-day",
        intent: { frequency: { unit: "day", interval: 1 }, time },
        slots: { clock: clockFragments(time, style) },
        tags: [
          style === "24h" ? "24-hour-clock" : style === "12h" ? "12-hour-clock" : "natural-clock",
        ],
      };
    }
  }
}

function* intervalClocks(): Generator<SemanticCase> {
  for (const semantic of intervals()) {
    // An absolute clock on a multi-day/week/month recurrence is meaningful.
    // Do not manufacture an anchoring policy for minute/hour-plus-clock input.
    if (semantic.intent.frequency.unit === "minute" || semantic.intent.frequency.unit === "hour") {
      continue;
    }

    for (const time of selectorClocks(semantic.intent.frequency.interval)) {
      for (const style of styles(time)) {
        yield {
          ...semantic,
          id: `${semantic.id}-${time.hour}-${time.minute}-${style}`,
          intent: { ...semantic.intent, time },
          slots: { ...semantic.slots, clock: clockFragments(time, style) },
          tags: [...semantic.tags, "interval-and-clock-where-valid"],
        };
      }
    }
  }
}

interface Selector {
  id: string;
  semanticFamily: string;
  family: SemanticCase["family"];
  intent: ScheduleIntent;
  full: Fragment[];
  short: Fragment[];
}

function* weeklySelectors(): Generator<Selector> {
  const make = (
    id: string,
    days: Weekday[],
    family: SemanticCase["family"],
    semanticFamily: string,
  ): Selector => ({
    id,
    semanticFamily,
    family,
    intent: { frequency: { unit: "week", interval: 1 }, weekdays: days },
    full: weekdays(days, false),
    short: weekdays(days, true),
  });

  for (const day of DAYS) yield make(day, [day], "weekday-list", "single-weekday");

  for (let a = 0; a < 7; a++) {
    for (let b = a + 1; b < 7; b++) {
      yield make(`${a}-${b}`, [DAYS[a]!, DAYS[b]!], "weekday-list", "weekday-list");
    }
  }

  for (let a = 0; a < 7; a++) {
    for (let length = 2; length <= 7; length++) {
      const b = (a + length - 1) % 7;

      const range = make(
        `range-${a}-${b}`,
        Array.from({ length }, (_, offset) => DAYS[(a + offset) % 7]!),
        "weekday-range",
        "weekday-range",
      );

      range.full = [
        fragment(DAYS[a]!, weekdayLabel(DAYS[a]!)),
        fragment("through", "BOUND_END_INCL"),
        fragment(DAYS[b]!, weekdayLabel(DAYS[b]!)),
      ];

      range.short = [
        fragment(DAYS[a]!.slice(0, 3), weekdayLabel(DAYS[a]!)),
        fragment("through", "BOUND_END_INCL"),
        fragment(DAYS[b]!.slice(0, 3), weekdayLabel(DAYS[b]!)),
      ];

      yield range;
    }
  }

  yield make("mon-wed-fri", ["monday", "wednesday", "friday"], "weekday-list", "weekday-list");

  for (const [id, days, label] of [
    ["weekdays", DAYS.slice(0, 5), "DAYGROUP_WEEKDAY"],
    ["weekends", DAYS.slice(5), "DAYGROUP_WEEKEND"],
  ] as const) {
    yield {
      ...make(id, [...days], "weekday-list", "day-group"),
      full: [fragment(id, label)],
      short: [fragment(id, label)],
    };
  }
}

function* monthlySelectors(): Generator<Selector> {
  for (let dayOfMonth = 1; dayOfMonth <= 31; dayOfMonth++) {
    const parts = [
      fragment(String(dayOfMonth), "DOM_LIT"),
      fragment(suffix(dayOfMonth), "GLUE", ""),
      glue("of every"),
      fragment("month", "UNIT_MONTH"),
    ];

    yield {
      id: `dom-${dayOfMonth}`,
      semanticFamily: "day-of-month",
      family: "day-of-month",
      intent: { frequency: { unit: "month", interval: 1 }, dayOfMonth },
      full: parts,
      short: parts,
    };
  }

  for (const ordinal of ORDINALS) {
    for (const weekday of DAYS) {
      const parts = [
        fragment(ordinal, ordinalLabel(ordinal)),
        fragment(weekday, weekdayLabel(weekday)),
        glue("of every"),
        fragment("month", "UNIT_MONTH"),
      ];

      yield {
        id: `${ordinal}-${weekday}`,
        semanticFamily: "ordinal-weekday",
        family: "ordinal-weekday",
        intent: {
          frequency: { unit: "month", interval: 1 },
          ordinalWeekday: { ordinal, weekday },
        },
        full: parts,
        short: parts.map((part) =>
          part.label.startsWith("WEEKDAY_") ? { ...part, text: weekday.slice(0, 3) } : part,
        ),
      };
    }
  }

  for (const type of ["last_day_of_month", "last_business_day"] as const) {
    const parts = [
      fragment("last", "ORDINAL_LAST"),
      fragment(
        type === "last_day_of_month" ? "day" : "business day",
        type === "last_day_of_month" ? "GLUE" : "DAYGROUP_WEEKDAY",
      ),
      glue("of every"),
      fragment("month", "UNIT_MONTH"),
    ];

    yield {
      id: type,
      semanticFamily: type === "last_day_of_month" ? "last-day-of-month" : "last-business-day",
      family: "day-of-month",
      intent: { frequency: { unit: "month", interval: 1 }, day: { type } },
      full: parts,
      short: parts,
    };
  }
}

function* selectors(which: "weekly" | "monthly", timed = true): Generator<SemanticCase> {
  let offset = 0;

  for (const selector of which === "weekly" ? weeklySelectors() : monthlySelectors()) {
    for (const abbreviated of [false, true]) {
      const selected = abbreviated ? selector.short : selector.full;

      // Some selectors have no short variant: avoid source duplication before publication.
      if (abbreviated && JSON.stringify(selected) === JSON.stringify(selector.full)) continue;

      if (!timed) {
        yield {
          id: `${selector.id}-${abbreviated ? "short" : "full"}`,
          semanticFamily: selector.semanticFamily,
          family: selector.family,
          intent: selector.intent,
          slots: { selector: selected },
          tags: ["untimed-selector", ...(abbreviated ? ["weekday-abbreviation"] : [])],
        };

        continue;
      }

      const interaction =
        selector.semanticFamily === "single-weekday"
          ? "weekday-and-clock"
          : selector.semanticFamily === "day-of-month"
            ? "monthly-date-and-clock"
            : ["last-day-of-month", "last-business-day"].includes(selector.semanticFamily)
              ? "day-rule-and-clock"
              : `${selector.semanticFamily}-and-clock`;

      for (const time of selectorClocks(offset)) {
        for (const style of styles(time)) {
          yield {
            id: `${selector.id}-${abbreviated ? "short" : "full"}-${time.hour}-${time.minute}-${style}`,
            semanticFamily: selector.semanticFamily,
            family: selector.family,
            intent: { ...selector.intent, time },
            slots: { selector: selected, clock: clockFragments(time, style) },
            tags: [interaction, ...(abbreviated ? ["weekday-abbreviation"] : [])],
          };
        }
      }
    }

    offset++;
  }
}

const WINDOWS = [
  {
    start: "2027-01-31",
    end: "2027-02-03",
    inclusive: "2027-02-02",
    tag: "month-boundary",
  },
  {
    start: "2027-12-31",
    end: "2028-01-03",
    inclusive: "2028-01-02",
    tag: "year-boundary",
  },
  { start: "2028-02-28", end: "2028-03-01", inclusive: "2028-02-29", tag: "leap-day" },
  {
    start: "2028-06-01",
    end: "2028-06-30",
    inclusive: "2028-06-29",
    tag: "ordinary-window",
  },
] as const;

function* bounds(): Generator<SemanticCase> {
  for (const window of WINDOWS) {
    for (const kind of [
      "start-bound",
      "exclusive-end-bound",
      "inclusive-end-bound",
      "date-window",
    ] as const) {
      for (const time of selectorClocks(3)) {
        for (const style of styles(time)) {
          const from = [fragment("starting on", "BOUND_START"), fragment(window.start, "DOM_LIT")];
          const until = [fragment("until", "BOUND_END_EXCL"), fragment(window.end, "DOM_LIT")];

          const through = [
            fragment("through", "BOUND_END_INCL"),
            fragment(window.inclusive, "DOM_LIT"),
          ];

          const parts =
            kind === "start-bound"
              ? from
              : kind === "exclusive-end-bound"
                ? until
                : kind === "inclusive-end-bound"
                  ? through
                  : [...from, ...until];

          yield {
            id: `${kind}-${window.start}-${time.hour}-${time.minute}-${style}`,
            semanticFamily: kind,
            family: "bounds",
            intent: {
              frequency: { unit: "day", interval: 1 },
              time,
              bounds: {
                ...(kind === "start-bound" || kind === "date-window"
                  ? { start: window.start }
                  : {}),
                ...(kind !== "start-bound" ? { end: window.end } : {}),
              },
            },
            slots: { bounds: parts, clock: clockFragments(time, style) },
            tags: [
              window.tag,
              "bounds-and-clock",
              ...(kind === "inclusive-end-bound" ? ["inclusive-to-exclusive-end"] : []),
            ],
          };
        }
      }
    }
  }
}

function* exceptions(): Generator<SemanticCase> {
  for (const day of DAYS) {
    for (const selected of [false, true]) {
      for (const time of selectorClocks(DAYS.indexOf(day))) {
        const selectedDays = DAYS.slice(0, 5);

        // Do not author redundant exceptions outside the selected weekdays.
        if (selected && !selectedDays.includes(day)) continue;

        yield {
          id: `${selected ? "weekday" : "daily"}-except-${day}-${time.hour}-${time.minute}`,
          semanticFamily: selected ? "selector-with-exception" : "weekday-exception",
          family: "exception",
          intent: {
            frequency: { unit: selected ? "week" : "day", interval: 1 },
            time,
            ...(selected ? { weekdays: selectedDays } : {}),
            exceptions: { weekdays: [day] },
          },
          slots: {
            base: [
              fragment(selected ? "weekdays" : "daily", selected ? "DAYGROUP_WEEKDAY" : "UNIT_DAY"),
            ],
            excluded: weekdays([day], false),
            clock: clockFragments(time, "24h"),
          },
          tags: ["exception-and-clock", ...(selected ? ["selector-and-exception"] : [])],
        };
      }
    }
  }

  for (const window of WINDOWS) {
    for (const time of selectorClocks(11)) {
      yield {
        id: `date-${window.inclusive}-${time.hour}-${time.minute}`,
        semanticFamily: "date-exception",
        family: "exception",
        intent: {
          frequency: { unit: "day", interval: 1 },
          time,
          exceptions: { dates: [window.inclusive] },
        },
        slots: {
          base: [fragment("daily", "UNIT_DAY")],
          excluded: [fragment(window.inclusive, "DOM_LIT")],
          clock: clockFragments(time, "24h"),
        },
        tags: ["exception-and-clock", window.tag],
      };
    }
  }
}

function* corrections(): Generator<SemanticCase> {
  for (const unit of UNITS) {
    for (const n of intervalValues(unit)) {
      const old = n === 7 ? 11 : 7;

      const schedule = (value: number): Fragment[] => [
        fragment("every", "RECUR"),
        fragment(String(value), intervalLabel(value)),
        fragment(`${unit}${value === 1 ? "" : "s"}`, unitLabel(unit)),
      ];

      yield {
        id: `${unit}-${old}-to-${n}`,
        semanticFamily: "correction-interval",
        family: "interval",
        intent: { frequency: { unit, interval: n } },
        discarded: { intent: { frequency: { unit, interval: old } }, slot: "old" },
        slots: { old: schedule(old), next: schedule(n) },
        tags: ["old-and-new-schedule-with-actionability"],
      };
    }
  }

  for (const time of allClocks()) {
    const old = { hour: (time.hour + 5) % 24, minute: (time.minute + 17) % 60 };

    const schedule = (t: TimeOfDay): Fragment[] => [
      fragment("daily", "UNIT_DAY"),
      glue("at"),
      ...clockFragments(t, "24h"),
    ];

    yield {
      id: `clock-${time.hour}-${time.minute}`,
      semanticFamily: "correction-clock",
      family: "time-of-day",
      intent: { frequency: { unit: "day", interval: 1 }, time },
      discarded: {
        intent: { frequency: { unit: "day", interval: 1 }, time: old },
        slot: "old",
      },
      slots: { old: schedule(old), next: schedule(time) },
      tags: ["old-and-new-schedule-with-actionability"],
    };
  }

  for (const old of DAYS) {
    for (const next of DAYS) {
      if (old === next) continue;

      yield {
        id: `${old}-to-${next}`,
        semanticFamily: "correction-weekday",
        family: "weekday-list",
        intent: { frequency: { unit: "week", interval: 1 }, weekdays: [next] },
        discarded: {
          intent: { frequency: { unit: "week", interval: 1 }, weekdays: [old] },
          slot: "old",
        },
        slots: {
          old: [fragment("every", "RECUR"), ...weekdays([old], false)],
          next: [fragment("every", "RECUR"), ...weekdays([next], false)],
        },
        tags: ["old-and-new-schedule-with-actionability"],
      };
    }
  }
}

export function semanticCases(set: CaseSet): Generator<SemanticCase> {
  switch (set) {
    case "interval":
      return intervals();
    case "interval-clock":
      return intervalClocks();
    case "daily":
      return daily();
    case "weekly":
      return selectors("weekly");
    case "weekly-untimed":
      return selectors("weekly", false);
    case "monthly":
      return selectors("monthly");
    case "monthly-untimed":
      return selectors("monthly", false);
    case "bounds":
      return bounds();
    case "exceptions":
      return exceptions();
    case "correction":
      return corrections();
  }
}
