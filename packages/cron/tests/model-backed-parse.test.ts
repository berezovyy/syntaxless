import { expect, test } from "bun:test";

import { cron } from "../src/index.ts";
import type { ScheduleIntent } from "../src/index.ts";

const weekdayMorning: ScheduleIntent = {
  frequency: { unit: "week", interval: 1 },
  weekdays: ["monday", "tuesday", "wednesday", "thursday", "friday"],
  time: { hour: 9, minute: 0 },
};

test("cron uses the bundled model result without inventing or changing fields", async () => {
  expect(await cron("every weekday at 9am")).toEqual({
    supported: true,
    expression: "0 9 * * 1-5",
    intent: weekdayMorning,
  });
});
