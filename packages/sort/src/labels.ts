export const LABELS = [
  "O",
  "FIELD",
  "ASC",
  "DESC",
  "NULLS",
  "FIRST",
  "LAST",
  "NEXT",
  "GLUE",
  "UNSUPPORTED",
] as const;

export type Role = (typeof LABELS)[number];
export const LABEL_COUNT = LABELS.length;
