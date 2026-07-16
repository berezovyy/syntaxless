/**
 * Typed-value label vocabulary. The label *is* the value for every closed
 * class; only `*_LIT` values come from token digits or ISO dates.
 *
 * Never reorder `LABELS`. There are no reserved slots: every id is in
 * `[0, LABEL_COUNT)`. An out-of-range predicted id is a hard error.
 */
import type { FrequencyUnit, Ordinal, Weekday } from "../types/intent.js";

export const WEEKDAY_LABEL_CODES = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"] as const;

export const WEEKDAYS: readonly Weekday[] = [
  "monday",
  "tuesday",
  "wednesday",
  "thursday",
  "friday",
  "saturday",
  "sunday",
];

export const LABELS = [
  "O",
  "RECUR",
  "GLUE",
  "EDGE",
  "COUNT",
  "WEEKDAY_MON",
  "WEEKDAY_TUE",
  "WEEKDAY_WED",
  "WEEKDAY_THU",
  "WEEKDAY_FRI",
  "WEEKDAY_SAT",
  "WEEKDAY_SUN",
  "DAYGROUP_WEEKDAY",
  "DAYGROUP_WEEKEND",
  "DAYGROUP_DAY",
  "HOUR_0",
  "HOUR_1",
  "HOUR_2",
  "HOUR_3",
  "HOUR_4",
  "HOUR_5",
  "HOUR_6",
  "HOUR_7",
  "HOUR_8",
  "HOUR_9",
  "HOUR_10",
  "HOUR_11",
  "HOUR_12",
  "HOUR_13",
  "HOUR_14",
  "HOUR_15",
  "HOUR_16",
  "HOUR_17",
  "HOUR_18",
  "HOUR_19",
  "HOUR_20",
  "HOUR_21",
  "HOUR_22",
  "HOUR_23",
  "HOUR_LIT",
  "MINUTE_0",
  "MINUTE_15",
  "MINUTE_30",
  "MINUTE_45",
  "MINUTE_LIT",
  "MERIDIEM_AM",
  "MERIDIEM_PM",
  "MERIDIEM_NONE",
  "ORDINAL_1",
  "ORDINAL_2",
  "ORDINAL_3",
  "ORDINAL_4",
  "ORDINAL_5",
  "ORDINAL_LAST",
  "BOUND_START",
  "BOUND_END_EXCL",
  "BOUND_END_INCL",
  "BOUND_BETWEEN",
  "EXCEPT",
  "UNIT_MIN",
  "UNIT_HOUR",
  "UNIT_DAY",
  "UNIT_WEEK",
  "UNIT_MONTH",
  "INTERVAL_2",
  "INTERVAL_3",
  "INTERVAL_4",
  "INTERVAL_5",
  "INTERVAL_6",
  "INTERVAL_7",
  "INTERVAL_8",
  "INTERVAL_9",
  "INTERVAL_10",
  "INTERVAL_11",
  "INTERVAL_12",
  "INTERVAL_LIT",
  "DOM_LIT",
  // An explicitly irrelevant number, not an unknown/missing schedule operand.
  "CONTEXT_NUMBER",
] as const;

export type Label = (typeof LABELS)[number];

export const LABEL_COUNT = LABELS.length;

/** Exact typed vocabularies: shipped 77-class artifacts and the appended context role. */
export function isTypedVocabulary(labels: readonly string[], classes: number): boolean {
  return (
    (classes === LABEL_COUNT || classes === 77) &&
    labels.length === classes &&
    labels.every((label, id) => label === LABELS[id])
  );
}

export const labelId = Object.fromEntries(LABELS.map((name, id) => [name, id])) as Record<
  Label,
  number
>;

/** A predicted role is just a numeric label id. */
export type Role = number;

export const STRUCTURAL_LABELS = [
  "RECUR",
  "GLUE",
  "EDGE",
  "BOUND_START",
  "BOUND_END_EXCL",
  "BOUND_END_INCL",
  "BOUND_BETWEEN",
  "EXCEPT",
] as const;

export type StructuralLabel = (typeof STRUCTURAL_LABELS)[number];

export const CONTENT_LABELS = LABELS.filter(
  (label) => isScheduleLabel(label) && !(STRUCTURAL_LABELS as readonly string[]).includes(label),
) as readonly Label[];

export type ContentLabel = (typeof CONTENT_LABELS)[number];

const STRUCTURAL_SET: ReadonlySet<string> = new Set(STRUCTURAL_LABELS);

/** A role belonging to a mentioned schedule, including its structural tokens. */
export function isScheduleLabel(label: string): boolean {
  return label !== "O" && label !== "CONTEXT_NUMBER";
}

/** True when `label` is literal filler rather than a value slot. */
export function isStructural(label: Label): boolean {
  return STRUCTURAL_SET.has(label);
}

/** True when the value comes from the token's digits or ISO date. */
export function isLiteralLabel(label: Label): boolean {
  return label.endsWith("_LIT");
}

/**
 * Map a predicted role id back to a label. Out-of-range ids throw; the caller
 * must never clamp or substitute `"O"` for a corrupt prediction.
 */
export function labelFromId(id: number): Label {
  const label = LABELS[id];

  if (label === undefined) throw new RangeError(`label id out of range: ${id}`);

  return label;
}

export type BoundKind = "start" | "end-excl" | "end-incl" | "between";
export type DayGroupKind = "weekday" | "weekend" | "day";
export type MeridiemKind = "am" | "pm" | "none";

export type LabelValue =
  Weekday | FrequencyUnit | Ordinal | BoundKind | DayGroupKind | MeridiemKind | number;

const WEEKDAY_BY_CODE: Record<(typeof WEEKDAY_LABEL_CODES)[number], Weekday> = {
  MON: "monday",
  TUE: "tuesday",
  WED: "wednesday",
  THU: "thursday",
  FRI: "friday",
  SAT: "saturday",
  SUN: "sunday",
};

const WEEKDAY_TO_LABEL: Record<Weekday, Label> = {
  monday: "WEEKDAY_MON",
  tuesday: "WEEKDAY_TUE",
  wednesday: "WEEKDAY_WED",
  thursday: "WEEKDAY_THU",
  friday: "WEEKDAY_FRI",
  saturday: "WEEKDAY_SAT",
  sunday: "WEEKDAY_SUN",
};

const UNIT_TO_LABEL: Record<FrequencyUnit, Label> = {
  minute: "UNIT_MIN",
  hour: "UNIT_HOUR",
  day: "UNIT_DAY",
  week: "UNIT_WEEK",
  month: "UNIT_MONTH",
};

const ORDINAL_TO_LABEL: Record<Ordinal, Label> = {
  first: "ORDINAL_1",
  second: "ORDINAL_2",
  third: "ORDINAL_3",
  fourth: "ORDINAL_4",
  last: "ORDINAL_LAST",
};

const BOUND_TO_LABEL: Record<BoundKind, Label> = {
  start: "BOUND_START",
  "end-excl": "BOUND_END_EXCL",
  "end-incl": "BOUND_END_INCL",
  between: "BOUND_BETWEEN",
};

const DAYGROUP_TO_LABEL: Record<DayGroupKind, Label> = {
  weekday: "DAYGROUP_WEEKDAY",
  weekend: "DAYGROUP_WEEKEND",
  day: "DAYGROUP_DAY",
};

const CLOSED_VALUES: Partial<Record<Label, LabelValue>> = {
  WEEKDAY_MON: "monday",
  WEEKDAY_TUE: "tuesday",
  WEEKDAY_WED: "wednesday",
  WEEKDAY_THU: "thursday",
  WEEKDAY_FRI: "friday",
  WEEKDAY_SAT: "saturday",
  WEEKDAY_SUN: "sunday",
  DAYGROUP_WEEKDAY: "weekday",
  DAYGROUP_WEEKEND: "weekend",
  DAYGROUP_DAY: "day",
  MERIDIEM_AM: "am",
  MERIDIEM_PM: "pm",
  MERIDIEM_NONE: "none",
  ORDINAL_1: "first",
  ORDINAL_2: "second",
  ORDINAL_3: "third",
  ORDINAL_4: "fourth",
  ORDINAL_5: 5,
  ORDINAL_LAST: "last",
  BOUND_START: "start",
  BOUND_END_EXCL: "end-excl",
  BOUND_END_INCL: "end-incl",
  BOUND_BETWEEN: "between",
  UNIT_MIN: "minute",
  UNIT_HOUR: "hour",
  UNIT_DAY: "day",
  UNIT_WEEK: "week",
  UNIT_MONTH: "month",
};

for (let hour = 0; hour <= 23; hour++) CLOSED_VALUES[`HOUR_${hour}` as Label] = hour;
for (const minute of [0, 15, 30, 45] as const) CLOSED_VALUES[`MINUTE_${minute}` as Label] = minute;

for (let interval = 2; interval <= 12; interval++) {
  CLOSED_VALUES[`INTERVAL_${interval}` as Label] = interval;
}

/**
 * Closed-class value for `label`. `*_LIT`, `O`, `RECUR`, `GLUE`, `EDGE`,
 * `COUNT`, and `EXCEPT` have no closed value. Literal values come from token
 * digits or an ISO date.
 */
export function valueOf(label: Label): LabelValue | undefined {
  return CLOSED_VALUES[label];
}

export function weekdayLabel(weekday: Weekday): Label {
  return WEEKDAY_TO_LABEL[weekday];
}

export function weekdayFromCode(code: (typeof WEEKDAY_LABEL_CODES)[number]): Weekday {
  return WEEKDAY_BY_CODE[code];
}

export function hourLabel(hour24: number): Label {
  if (!Number.isInteger(hour24) || hour24 < 0 || hour24 > 23) {
    throw new RangeError(`hour out of range: ${hour24}`);
  }

  return `HOUR_${hour24}` as Label;
}

export function minuteLabel(minute: number): Label {
  if (!Number.isInteger(minute) || minute < 0 || minute > 59) {
    throw new RangeError(`minute out of range: ${minute}`);
  }

  if (minute === 0 || minute === 15 || minute === 30 || minute === 45) {
    return `MINUTE_${minute}` as Label;
  }

  return "MINUTE_LIT";
}

export function intervalLabel(interval: number): Label {
  if (!Number.isInteger(interval) || interval < 1) {
    throw new RangeError(`interval out of range: ${interval}`);
  }

  if (interval >= 2 && interval <= 12) return `INTERVAL_${interval}` as Label;

  return "INTERVAL_LIT";
}

export function unitLabel(unit: FrequencyUnit): Label {
  return UNIT_TO_LABEL[unit];
}

export function ordinalLabel(ordinal: Ordinal): Label {
  return ORDINAL_TO_LABEL[ordinal];
}

export function meridiemLabel(meridiem: MeridiemKind): Label {
  if (meridiem === "am") return "MERIDIEM_AM";
  if (meridiem === "pm") return "MERIDIEM_PM";

  return "MERIDIEM_NONE";
}

export function boundLabel(kind: BoundKind): Label {
  return BOUND_TO_LABEL[kind];
}

export function dayGroupLabel(kind: DayGroupKind): Label {
  return DAYGROUP_TO_LABEL[kind];
}
