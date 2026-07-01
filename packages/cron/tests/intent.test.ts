import { expect, test } from "bun:test";

import type { ScheduleIntent } from "../src/index.ts";

test("a weekday intent satisfies the public type", () => {
  const intent = {
    frequency: { unit: "week", interval: 1 },
    weekdays: ["monday", "tuesday", "wednesday", "thursday", "friday"],
    time: { hour: 9, minute: 0 },
  } satisfies ScheduleIntent;

  expect(intent.frequency.unit).toBe("week");
});
