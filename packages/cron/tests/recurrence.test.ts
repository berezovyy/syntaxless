import { expect, test } from "bun:test";

import { Temporal } from "@js-temporal/polyfill";

import { IntentError, recurrences } from "../src/index.ts";
import type { RecurrenceOptions, ScheduleIntent } from "../src/index.ts";
import { createTemporalCalendar } from "../src/temporal.ts";

const calendar = createTemporalCalendar(Temporal);
const daily: ScheduleIntent = { frequency: { unit: "day", interval: 1 } };
const utc = { timezone: "UTC", reference: "2026-01-01T00:00:00Z", count: 2 };

function times(intent: ScheduleIntent, options: RecurrenceOptions): string[] {
  const result = recurrences(intent, options);

  if (!result.ok) throw new Error(JSON.stringify(result));

  return result.occurrences.map((item) => item.instant);
}

test("civil interval phases use an explicit anchor and an exclusive reference", () => {
  expect(
    times(
      { frequency: { unit: "minute", interval: 7 } },
      { ...utc, reference: "2026-01-01T00:05:00Z", anchor: "2026-01-01T00:00:00Z" },
    ),
  ).toEqual(["2026-01-01T00:07:00.000Z", "2026-01-01T00:14:00.000Z"]);

  expect(
    times(
      { frequency: { unit: "hour", interval: 5 }, time: { minute: 15 } },
      { ...utc, reference: "2026-01-01T00:05:00Z", anchor: utc.reference },
    ),
  ).toEqual(["2026-01-01T00:15:00.000Z", "2026-01-01T05:15:00.000Z"]);

  expect(
    times(
      { frequency: { unit: "day", interval: 2 } },
      { ...utc, reference: "2028-02-27T12:00:00Z" },
    ),
  ).toEqual(["2028-02-29T00:00:00.000Z", "2028-03-02T00:00:00.000Z"]);

  expect(
    times(
      { frequency: { unit: "week", interval: 2 }, weekdays: ["friday", "monday"] },
      { ...utc, reference: "2026-02-26T12:00:00Z", anchor: "2026-02-25T12:00:00Z" },
    ),
  ).toEqual(["2026-02-27T00:00:00.000Z", "2026-03-09T00:00:00.000Z"]);

  expect(
    times(
      { frequency: { unit: "week", interval: 1 } },
      { ...utc, reference: "2026-02-26T12:00:00Z" },
    ),
  ).toEqual(["2026-03-05T00:00:00.000Z", "2026-03-12T00:00:00.000Z"]);

  expect(
    times(
      { frequency: { unit: "month", interval: 2 } },
      { ...utc, reference: "2026-01-15T12:00:00Z" },
    ),
  ).toEqual(["2026-03-15T00:00:00.000Z", "2026-05-15T00:00:00.000Z"]);
});

test("bounds and exceptions filter actual local dates, without changing inputs", () => {
  const intent: ScheduleIntent = {
    ...daily,
    time: { hour: 23, minute: 30 },
    bounds: { start: "2026-01-02", end: "2026-01-06" },
    exceptions: { dates: ["2026-01-03"], weekdays: ["sunday"] },
  };

  const options = { ...utc, count: 10, timezone: "+02:00" };
  const before = JSON.stringify({ intent, options });
  const result = recurrences(intent, options);

  expect(result.ok).toBe(true);

  if (result.ok) {
    expect(result.stop).toBe("intent-end");

    expect(result.occurrences.map((item) => item.instant)).toEqual([
      "2026-01-02T21:30:00.000Z",
      "2026-01-05T21:30:00.000Z",
    ]);
  }

  expect(JSON.stringify({ intent, options })).toBe(before);

  expect(
    times(daily, {
      timezone: utc.timezone,
      reference: utc.reference,
      before: "2026-01-03T00:00:00Z",
    }),
  ).toEqual(["2026-01-02T00:00:00.000Z"]);

  expect(
    times(daily, {
      ...utc,
      timezone: "+05:45",
      reference: "0099-12-31T12:00:00+05:45",
    }),
  ).toEqual(["0099-12-31T18:15:00.000Z", "0100-01-01T18:15:00.000Z"]);
});

test("monthly day selectors and month-end policies preserve the selected rule", () => {
  const monthly: ScheduleIntent = {
    frequency: { unit: "month", interval: 1 },
    dayOfMonth: 31,
  };

  const options = { ...utc, reference: "2028-01-31T12:00:00Z" };

  expect(times(monthly, options)).toEqual(["2028-03-31T00:00:00.000Z", "2028-05-31T00:00:00.000Z"]);

  expect(times(monthly, { ...options, policies: { monthEnd: "constrain" } })).toEqual([
    "2028-02-29T00:00:00.000Z",
    "2028-03-31T00:00:00.000Z",
  ]);

  expect(recurrences(monthly, { ...options, policies: { monthEnd: "reject" } })).toMatchObject({
    ok: false,
    code: "month-end",
  });

  expect(times(monthly, { ...utc, count: 1, policies: { monthEnd: "reject" } })).toEqual([
    "2026-01-31T00:00:00.000Z",
  ]);

  expect(
    times(
      {
        frequency: { unit: "month", interval: 1 },
        ordinalWeekday: { ordinal: "fourth", weekday: "monday" },
      },
      utc,
    ),
  ).toEqual(["2026-01-26T00:00:00.000Z", "2026-02-23T00:00:00.000Z"]);

  expect(
    times(
      {
        frequency: { unit: "month", interval: 1 },
        ordinalWeekday: { ordinal: "last", weekday: "friday" },
      },
      utc,
    ),
  ).toEqual(["2026-01-30T00:00:00.000Z", "2026-02-27T00:00:00.000Z"]);

  expect(
    times({ frequency: { unit: "month", interval: 1 }, day: { type: "last_business_day" } }, utc),
  ).toEqual(["2026-01-30T00:00:00.000Z", "2026-02-27T00:00:00.000Z"]);

  expect(
    times({ frequency: { unit: "month", interval: 1 }, day: { type: "last_day_of_month" } }, utc),
  ).toEqual(["2026-01-31T00:00:00.000Z", "2026-02-28T00:00:00.000Z"]);
});

test("Temporal gaps skip or shift explicitly, including half-hour transitions", () => {
  const intent = { ...daily, time: { hour: 2, minute: 30 } };

  const options = {
    ...utc,
    calendar,
    timezone: "America/New_York",
    reference: "2026-03-07T12:00:00Z",
  };

  expect(times(intent, options)).toEqual(["2026-03-09T06:30:00.000Z", "2026-03-10T06:30:00.000Z"]);

  expect(times(intent, { ...options, policies: { gap: "earlier" } })[0]).toBe(
    "2026-03-08T06:30:00.000Z",
  );

  const shifted = recurrences(intent, { ...options, policies: { gap: "later" } });

  expect(shifted.ok).toBe(true);

  if (shifted.ok) {
    expect(shifted.occurrences[0]!.instant).toBe("2026-03-08T07:30:00.000Z");
    expect(shifted.occurrences[0]!.requestedLocal.hour).toBe(2);
    expect(shifted.occurrences[0]!.local.hour).toBe(3);
  }

  expect(recurrences(intent, { ...options, policies: { gap: "reject" } })).toMatchObject({
    ok: false,
    code: "nonexistent-time",
  });

  expect(
    times(
      { ...daily, time: { hour: 2, minute: 15 } },
      {
        ...options,
        timezone: "Australia/Lord_Howe",
        reference: "2026-10-03T12:00:00Z",
        count: 1,
        policies: { gap: "later" },
      },
    ),
  ).toEqual(["2026-10-03T15:45:00.000Z"]);
});

test("fold choices remain chronological even across multiple minute candidates", () => {
  const intent = { ...daily, time: { hour: 1, minute: 30 } };

  const options = {
    ...utc,
    calendar,
    timezone: "America/New_York",
    reference: "2026-10-31T12:00:00Z",
  };

  expect(times(intent, options)[0]).toBe("2026-11-01T05:30:00.000Z");

  expect(times(intent, { ...options, policies: { fold: "later" } })[0]).toBe(
    "2026-11-01T06:30:00.000Z",
  );

  expect(times(intent, { ...options, policies: { fold: "both" } })).toEqual([
    "2026-11-01T05:30:00.000Z",
    "2026-11-01T06:30:00.000Z",
  ]);

  const pair = recurrences(intent, { ...options, policies: { fold: "both" } });

  if (!pair.ok) throw new Error("expected the fold pair");
  expect(pair.occurrences[0]!.requestedLocal).not.toBe(pair.occurrences[1]!.requestedLocal);
  const buffer = calendar.toLocal(Date.parse(options.reference), options.timezone);

  const reusable = {
    ...calendar,
    toLocal: (epoch: number, timezone: string) =>
      Object.assign(buffer, calendar.toLocal(epoch, timezone)),
  };

  const borrowed = recurrences(intent, {
    ...options,
    calendar: reusable,
    policies: { fold: "both" },
  });

  if (!borrowed.ok) throw new Error("expected the fold pair with a reusable conversion buffer");

  expect(
    borrowed.occurrences.map((item) => [
      item.local.month,
      item.local.day,
      item.local.hour,
      item.local.minute,
    ]),
  ).toEqual([
    [11, 1, 1, 30],
    [11, 1, 1, 30],
  ]);

  expect(recurrences(intent, { ...options, policies: { fold: "reject" } })).toMatchObject({
    ok: false,
    code: "repeated-time",
  });

  expect(
    times(intent, {
      ...options,
      reference: "2026-11-01T05:45:00Z",
      policies: { fold: "both" },
      count: 1,
    }),
  ).toEqual(["2026-11-01T06:30:00.000Z"]);

  expect(
    times(
      { frequency: { unit: "minute", interval: 15 } },
      {
        ...options,
        anchor: "2026-11-01T00:00:00-04:00",
        reference: "2026-11-01T05:40:00Z",
        policies: { fold: "both" },
        count: 4,
      },
    ),
  ).toEqual([
    "2026-11-01T05:45:00.000Z",
    "2026-11-01T06:00:00.000Z",
    "2026-11-01T06:15:00.000Z",
    "2026-11-01T06:30:00.000Z",
  ]);
});

test("a shifted civil day is deduplicated and exclusions use the actual date", () => {
  const intent = { ...daily, time: { hour: 9, minute: 0 } };

  const options: RecurrenceOptions = {
    calendar,
    timezone: "Pacific/Apia",
    reference: "2011-12-29T20:00:00Z",
    count: 2,
    policies: { gap: "later" },
  };

  expect(times(intent, options)).toEqual(["2011-12-30T19:00:00.000Z", "2011-12-31T19:00:00.000Z"]);

  expect(times({ ...intent, exceptions: { dates: ["2011-12-31"] } }, options)).toEqual([
    "2011-12-31T19:00:00.000Z",
    "2012-01-01T19:00:00.000Z",
  ]);
});

test("historical timezone offsets preserve seconds rather than rounding to minutes", () => {
  expect(
    times(
      { ...daily, time: { hour: 9, minute: 0 } },
      { ...utc, calendar, timezone: "Europe/Paris", reference: "1900-01-01T00:00:00Z" },
    ),
  ).toEqual(["1900-01-01T08:50:39.000Z", "1900-01-02T08:50:39.000Z"]);
});

test("jumped UTC candidates agree with independent minute scans across a leap month", () => {
  const anchor = Date.parse("2028-02-01T00:00:00Z");
  const reference = Date.parse("2028-02-02T08:15:00Z");
  const before = Date.parse("2028-03-03T00:00:00Z");

  const intents: ScheduleIntent[] = [
    { frequency: { unit: "day", interval: 3 }, time: { hour: 8, minute: 20 } },
    {
      frequency: { unit: "week", interval: 2 },
      weekdays: ["monday", "saturday"],
      time: { hour: 9, minute: 10 },
    },
    ...["first", "second", "third", "fourth", "last"].map(
      (ordinal) =>
        ({
          frequency: { unit: "month", interval: 1 },
          ordinalWeekday: { ordinal, weekday: "monday" },
          time: { hour: 4, minute: 0 },
        }) as ScheduleIntent,
    ),
    {
      frequency: { unit: "month", interval: 1 },
      dayOfMonth: 29,
      time: { hour: 2, minute: 0 },
    },
  ];

  for (const intent of intents) {
    const expected: string[] = [];

    for (let value = reference + 60000; value < before; value += 60000) {
      const date = new Date(value);

      if (
        date.getUTCHours() !== intent.time!.hour ||
        date.getUTCMinutes() !== intent.time!.minute
      ) {
        continue;
      }

      if (intent.frequency.unit === "day" && Math.floor((value - anchor) / 86400000) % 3 !== 0) {
        continue;
      }

      if (intent.frequency.unit === "week") {
        const elapsedDays = Math.floor((value - anchor) / 86400000);

        if (Math.floor((elapsedDays + 1) / 7) % 2 !== 0 || ![1, 6].includes(date.getUTCDay())) {
          continue;
        }
      }

      if (intent.dayOfMonth !== undefined && date.getUTCDate() !== intent.dayOfMonth) continue;

      if (intent.ordinalWeekday !== undefined) {
        if (date.getUTCDay() !== 1) continue;
        const ordinal = intent.ordinalWeekday.ordinal;

        if (ordinal === "last") {
          if (new Date(value + 7 * 86400000).getUTCMonth() === date.getUTCMonth()) continue;
        } else if (
          Math.floor((date.getUTCDate() - 1) / 7) !==
          ["first", "second", "third", "fourth"].indexOf(ordinal)
        ) {
          continue;
        }
      }

      expected.push(date.toISOString());
    }

    expect(times(intent, { timezone: "UTC", reference, anchor, before })).toEqual(expected);
  }
});

test(
  "invalid contexts and unattainable counts " + "fail within explicit work and output budgets",
  () => {
    for (const options of [
      { timezone: "UTC", reference: utc.reference },
      { ...utc, count: 0 },
      { ...utc, count: 1001 },
      { ...utc, maxSteps: 100001 },
      { ...utc, reference: "2026-02-30T00:00:00Z" },
      { ...utc, reference: "2026-01-01T00:00:00" },
      { ...utc, before: utc.reference },
      { ...utc, policies: { gap: "guess" } },
      { ...utc, policies: null },
      { ...utc, calendar: null },
    ]) {
      expect(recurrences(daily, options as RecurrenceOptions)).toMatchObject({
        ok: false,
        code: "invalid-context",
      });
    }

    expect(recurrences(daily, { ...utc, timezone: "America/New_York" })).toMatchObject({
      ok: false,
      code: "missing-calendar",
    });

    expect(recurrences(daily, { ...utc, calendar, timezone: "Invalid/Zone" })).toMatchObject({
      ok: false,
      code: "invalid-calendar",
    });

    expect(
      recurrences(
        {
          ...daily,
          exceptions: {
            weekdays: [
              "monday",
              "tuesday",
              "wednesday",
              "thursday",
              "friday",
              "saturday",
              "sunday",
            ],
          },
        },
        { ...utc, maxSteps: 20 },
      ),
    ).toMatchObject({ ok: false, code: "work-limit" });

    expect(
      recurrences(
        { frequency: { unit: "minute", interval: 1 } },
        { timezone: "UTC", reference: utc.reference, before: "2026-01-02T00:00:00Z" },
      ),
    ).toMatchObject({ ok: false, code: "output-limit" });

    expect(() => recurrences({ ...daily, dayOfMonth: 2 }, utc)).toThrow(IntentError);

    expect(recurrences(daily, { ...utc, reference: "9999-12-31T00:00:00Z" })).toMatchObject({
      ok: true,
      occurrences: [],
      stop: "range-end",
    });

    expect(
      recurrences(
        {
          frequency: { unit: "month", interval: Number.MAX_SAFE_INTEGER },
          dayOfMonth: 1,
        },
        utc,
      ),
    ).toMatchObject({ ok: true, occurrences: [], stop: "range-end" });
  },
);
