import { expect, test } from "bun:test";

import { execEqual, execRisk } from "./exec-score";

test("EXEC is 1 when compiled cron matches, 0 when it does not", () => {
  const daily = { frequency: { unit: "day" as const, interval: 1 } };

  const weekly = {
    frequency: { unit: "week" as const, interval: 1 },
    weekdays: ["monday" as const],
  };

  expect(execEqual(daily, daily)).toBe(true);
  expect(execRisk(daily, daily)).toBe(0);
  expect(execEqual(daily, weekly)).toBe(false);
  expect(execRisk(daily, weekly)).toBe(1);
  expect(execEqual(null, null)).toBe(true);
  expect(execEqual(daily, null)).toBe(false);
});
