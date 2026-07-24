export const LABELS = [
  "O",
  "RETRIES",
  "INITIAL",
  "CAP",
  "FACTOR",
  "UNIT",
  "CONSTANT",
  "EXPONENTIAL",
  "DOUBLE",
  "TRIPLE",
  "JITTER_FULL",
  "JITTER_EQUAL",
  "JITTER_PERCENT",
  "PERCENT",
  "GLUE",
  "UNSUPPORTED",
  "JITTER_NONE",
] as const;

export type Role = (typeof LABELS)[number];
export const LABEL_COUNT = LABELS.length;
