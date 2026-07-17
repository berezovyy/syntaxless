import type { ScheduleIntent, TimeOfDay } from "../../src/types/intent.js";

/** Clock renderers cannot render an hourly minute anchor as a daily clock. */
export function requireClock(intent: ScheduleIntent): TimeOfDay {
  const time = intent.time;

  if (time === undefined || time.hour === undefined) {
    throw new RangeError("clock rendering requires both hour and minute");
  }

  return time;
}
