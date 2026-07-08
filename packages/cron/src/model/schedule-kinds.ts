/** Coarse utterance-level schedule classes predicted alongside token roles. */
export const SCHEDULE_KINDS = [
  "none",
  "interval",
  "daily",
  "weekly",
  "monthly",
  "ordinal",
  "bounded",
  "exception",
  "ambiguous",
] as const;

export type ScheduleKind = (typeof SCHEDULE_KINDS)[number];
