import { expect, test } from "bun:test";

import { cron, parse } from "../src/index.ts";
import type { ScheduleIntent } from "../src/index.ts";

const weekdayMorning: ScheduleIntent = {
  frequency: { unit: "week", interval: 1 },
  weekdays: ["monday", "tuesday", "wednesday", "thursday", "friday"],
  time: { hour: 9, minute: 0 },
};

test.each([undefined, "on", "off"] as const)(
  "the bundled production model parses an explicit schedule (overlay=%s)",
  async (overlay) => {
    const result = await parse("every weekday at 9am", overlay === undefined ? {} : { overlay });

    expect(result).toMatchObject({
      ok: true,
      source: "model",
      intent: weekdayMorning,
      diagnostics: [],
    });
  },
);

test("cron uses the bundled model result without inventing or changing fields", async () => {
  expect(await cron("every weekday at 9am")).toEqual({
    supported: true,
    expression: "0 9 * * 1-5",
    intent: weekdayMorning,
  });
});

test("the explicit rules baseline remains independently selectable", async () => {
  expect(await parse("every Monday", { source: "rules" })).toEqual({
    ok: true,
    source: "rules",
    confidence: 1,
    diagnostics: [],
    intent: { frequency: { unit: "week", interval: 1 }, weekdays: ["monday"] },
  });
});
