export const LABELS = [
  "O",
  "FIELD",
  "VALUE",
  "EQ",
  "NE",
  "LT",
  "LE",
  "GT",
  "GE",
  "IN",
  "BETWEEN",
  "IS_NULL",
  "IS_NOT_NULL",
  "AND",
  "OR",
  "NOT",
  "EXCLUDE",
  "LPAREN",
  "RPAREN",
  "GLUE",
  "UNSUPPORTED",
] as const;

export type Role = (typeof LABELS)[number];
export const LABEL_COUNT = LABELS.length;
