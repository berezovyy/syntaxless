export const LABELS = [
  "O",
  "DAY",
  "WEEK",
  "MONTH",
  "YEAR",
  "CALENDAR",
  "ELAPSED",
  "UNIT",
  "GLUE",
  "UNSUPPORTED",
  "ROLLING",
  "PERIODS",
  "ONE",
] as const;

export type Role = (typeof LABELS)[number];
export const LABEL_COUNT = LABELS.length;
