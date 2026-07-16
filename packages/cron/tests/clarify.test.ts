import { expect, test } from "bun:test";

import { clarify, compile, IntentError, resolveClarification } from "../src/index.js";
import type { ClarificationAnswer, ScheduleIntent } from "../src/index.js";

test(
  "missing cron days produce typed questions and " + "validated answers preserve existing facts",
  () => {
    const cases: [ScheduleIntent, ClarificationAnswer, string][] = [
      [
        { frequency: { unit: "week", interval: 1 }, time: { hour: 9, minute: 25 } },
        { weekdays: ["monday", "friday"] },
        "25 9 * * 1,5",
      ],
      [
        { frequency: { unit: "month", interval: 1 }, time: { hour: 23, minute: 45 } },
        { dayOfMonth: 31 },
        "45 23 31 * *",
      ],
    ];

    for (const [intent, answer, expression] of cases) {
      const original = JSON.stringify(intent);
      const answerBefore = JSON.stringify(answer);

      const clarification = clarify(intent)!;

      expect(clarification.code).toBe("missing-anchor");
      expect(clarification.field).toBe("weekdays" in answer ? "weekdays" : "dayOfMonth");
      const resolved = resolveClarification(intent, answer);

      expect(compile(resolved)).toEqual({ supported: true, expression });
      expect(resolved.frequency).toEqual(intent.frequency);
      expect(resolved.time).toEqual(intent.time);
      expect(JSON.stringify(intent)).toBe(original);
      expect(JSON.stringify(answer)).toBe(answerBefore);
      resolved.frequency.interval = 2;
      resolved.time!.minute = 0;
      if (resolved.weekdays) resolved.weekdays[0] = "sunday";
      expect(JSON.stringify(intent)).toBe(original);
      expect(JSON.stringify(answer)).toBe(answerBefore);
    }

    const weekly = clarify(cases[0]![0]);

    if (weekly?.field !== "weekdays") throw new Error("weekly clarification missing");

    expect(weekly.allowed).toEqual([
      "monday",
      "tuesday",
      "wednesday",
      "thursday",
      "friday",
      "saturday",
      "sunday",
    ]);

    weekly.allowed.length = 0;
    const next = clarify(cases[0]![0]);

    expect(next?.field === "weekdays" && next.allowed.length).toBe(7);
    const monthly = clarify(cases[1]![0]);

    expect(monthly?.field === "dayOfMonth" && [monthly.minimum, monthly.maximum]).toEqual([1, 31]);
  },
);

test(
  "clarification refuses invalid or unrelated " + "answers and doesn't invent alternative meanings",
  () => {
    const weekly: ScheduleIntent = { frequency: { unit: "week", interval: 1 } };

    for (const answer of [
      null,
      [],
      {},
      { dayOfMonth: 3 },
      { weekdays: ["monday"], time: { hour: 21, minute: 0 } },
      { weekdays: [] },
      { weekdays: ["monday", "monday"] },
      { weekdays: ["holiday"] },
      { weekdays: [() => "monday"] },
      Object.create({ weekdays: ["monday"] }),
      { weekdays: ["monday"], [Symbol("extra")]: true },
    ]) {
      expect(() => resolveClarification(weekly, answer as ClarificationAnswer)).toThrow(
        IntentError,
      );
    }

    const monthly: ScheduleIntent = { frequency: { unit: "month", interval: 1 } };

    for (const dayOfMonth of [0, 32, 1.5, NaN, "3"]) {
      expect(() => resolveClarification(monthly, { dayOfMonth } as ClarificationAnswer)).toThrow(
        IntentError,
      );
    }

    expect(compile(resolveClarification(monthly, { dayOfMonth: 1 }))).toEqual({
      supported: true,
      expression: "0 0 1 * *",
    });

    const settled: ScheduleIntent = {
      ...weekly,
      weekdays: ["monday"],
      time: { hour: 9, minute: 0 },
    };

    expect(clarify(settled)).toBeNull();
    expect(() => resolveClarification(settled, { weekdays: ["friday"] })).toThrow(IntentError);

    for (const intent of [
      { frequency: { unit: "day", interval: 1 }, time: { hour: 9, minute: 0 } },
      { frequency: { unit: "week", interval: 2 } },
      { ...weekly, bounds: { start: "2026-01-01" } },
      { ...monthly, day: { type: "last_day_of_month" } },
      { ...monthly, ordinalWeekday: { ordinal: "first", weekday: "monday" } },
      { ...weekly, exceptions: { weekdays: ["sunday"] } },
    ] as ScheduleIntent[]) {
      const before = JSON.stringify(intent);

      expect(clarify(intent)).toBeNull();
      expect(JSON.stringify(intent)).toBe(before);
    }

    expect(() =>
      clarify({ frequency: { unit: "year", interval: 1 } } as unknown as ScheduleIntent),
    ).toThrow(IntentError);
  },
);
