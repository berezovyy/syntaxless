/** Coarse utterance-level schedule shape predicted alongside token roles.
 * This is not speech-act; see `speech-acts.ts`. */
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
