import { expect, test } from "bun:test";

import type { ScheduleIntent, Weekday } from "../../src/index";
import { createRng } from "../src/rng";
import type { Family } from "../src/sampler";
import { assertValid, FAMILY_WEIGHTS, sampleIntent } from "../src/sampler";

const FAMILIES = Object.keys(FAMILY_WEIGHTS) as Family[];

const WEEKDAY_ORDER: readonly Weekday[] = [
  "monday",
  "tuesday",
  "wednesday",
  "thursday",
  "friday",
  "saturday",
  "sunday",
];

function weightedEntries(): Array<[Family, number]> {
  return Object.entries(FAMILY_WEIGHTS) as Array<[Family, number]>;
}

function sequence(seed: number, length: number): string {
  const rng = createRng(seed);
  const entries = weightedEntries();
  const samples = [];

  for (let i = 0; i < length; i++) {
    samples.push(sampleIntent(rng, rng.weighted(entries)));
  }

  return JSON.stringify(samples);
}

function isContiguous(weekdays: readonly Weekday[]): boolean {
  const sorted = weekdays.map((day) => WEEKDAY_ORDER.indexOf(day)).sort((a, b) => a - b);

  return sorted.every((value, index) => index === 0 || value === sorted[index - 1]! + 1);
}

test("the same seed produces an identical 1000-sample sequence", () => {
  expect(sequence(1234, 1000)).toBe(sequence(1234, 1000));
});

test("different seeds produce different sequences", () => {
  expect(sequence(1, 1000)).not.toBe(sequence(2, 1000));
});

test("every family can be forced and yields its expected top-level field", () => {
  for (const family of FAMILIES) {
    const sample = sampleIntent(createRng(7), family);

    expect(sample.family).toBe(family);

    if (family === "negative") {
      expect(sample.intent).toBeNull();
      continue;
    }

    const intent = sample.intent;

    expect(intent).not.toBeNull();
    if (intent === null) throw new Error("unreachable");
    assertValid(intent);

    switch (family) {
      case "interval":
        expect(intent.frequency.interval).toBeGreaterThanOrEqual(2);
        break;
      case "weekday-list":
        expect(intent.weekdays?.length).toBeGreaterThanOrEqual(2);
        expect(intent.weekdays?.length).toBeLessThanOrEqual(4);
        break;
      case "weekday-range":
        expect(intent.weekdays?.length).toBeGreaterThanOrEqual(2);
        expect(intent.weekdays?.length).toBeLessThanOrEqual(5);
        expect(isContiguous(intent.weekdays!)).toBe(true);
        break;
      case "ordinal-weekday":
        expect(intent.ordinalWeekday).toBeDefined();
        break;
      case "day-of-month":
        expect(intent.dayOfMonth).toBeGreaterThanOrEqual(1);
        expect(intent.dayOfMonth).toBeLessThanOrEqual(28);
        break;
      case "time-of-day":
        expect(intent.time).toBeDefined();
        expect([0, 15, 30, 45]).toContain(intent.time!.minute);
        break;
      case "bounds":
        expect(intent.bounds).toBeDefined();
        break;
      case "exception":
        expect(intent.exceptions?.weekdays?.length).toBeGreaterThanOrEqual(1);
        break;
      case "terse":
        expect(intent.weekdays?.length).toBeGreaterThanOrEqual(1);
        break;
    }
  }
});

test("10k weighted draws hit every family and pass assertValid", () => {
  const rng = createRng(99);
  const entries = weightedEntries();
  const seen = new Set<Family>();

  for (let i = 0; i < 10_000; i++) {
    const family = rng.weighted(entries);

    seen.add(family);
    const sample = sampleIntent(rng, family);

    if (sample.intent === null) {
      expect(family).toBe("negative");
    } else {
      assertValid(sample.intent);
    }
  }

  expect([...seen].sort()).toEqual([...FAMILIES].sort());
});

test("the negative family never produces a schedule", () => {
  const rng = createRng(5);

  for (let i = 0; i < 50; i++) {
    const sample = sampleIntent(rng, "negative");

    expect(sample.intent).toBeNull();
    expect(sample.variant).toBe("prose");
  }
});

test("assertValid rejects out-of-shape intents", () => {
  const empty: ScheduleIntent = {
    frequency: { unit: "week", interval: 1 },
    weekdays: [],
  };

  expect(() => assertValid(empty)).toThrow();

  const duplicate: ScheduleIntent = {
    frequency: { unit: "week", interval: 1 },
    weekdays: ["monday", "monday"],
  };

  expect(() => assertValid(duplicate)).toThrow();

  const badHour: ScheduleIntent = {
    frequency: { unit: "day", interval: 1 },
    time: { hour: 24, minute: 0 },
  };

  expect(() => assertValid(badHour)).toThrow();
});
