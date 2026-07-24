import { expect, test } from "bun:test";

import { compile } from "../src/compile.js";
import { explain } from "../src/explain.js";
import type { Role } from "../src/labels.js";
import { prepare } from "../src/prepare.js";
import { resolveRoles } from "../src/resolve.js";

function resolve(text: string, roles: Role[]) {
  return resolveRoles(
    prepare(text),
    roles.map((role) => ({ role, score: 1 })),
  ).intent;
}

test(
  "reference preserves article counts and exact calendar windows, " +
    "with explicit context required only for preview",
  () => {
    const intent = resolve("keep daily backups for a week and monthly backups for a year", [
      "GLUE",
      "DAY",
      "GLUE",
      "GLUE",
      "CALENDAR",
      "UNIT",
      "GLUE",
      "MONTH",
      "GLUE",
      "GLUE",
      "CALENDAR",
      "UNIT",
    ]);

    expect(intent).toEqual({
      tiers: [
        { bucket: "day", window: { kind: "calendar", unit: "day", count: 7 } },
        { bucket: "month", window: { kind: "calendar", unit: "month", count: 12 } },
      ],
    });

    expect(explain(intent)).toContain("including the current period");

    expect(compile(intent, {} as Parameters<typeof compile>[1])).toMatchObject({
      supported: false,
      code: "missing-context",
    });

    expect(() =>
      resolve("keep one backup per day for half", [
        "GLUE",
        "ONE",
        "GLUE",
        "GLUE",
        "DAY",
        "GLUE",
        "UNSUPPORTED",
      ]),
    ).toThrow("non-actionable");
  },
);

test(
  "exact elapsed units, calendar clipping and representative " +
    "limits preserve facts rather than guessing",
  () => {
    expect(
      resolve("keep one backup per day for the last 1.001 seconds", [
        "GLUE",
        "ONE",
        "GLUE",
        "GLUE",
        "DAY",
        "GLUE",
        "GLUE",
        "ROLLING",
        "ELAPSED",
        "UNIT",
      ]),
    ).toEqual({
      tiers: [{ bucket: "day", window: { kind: "elapsed", milliseconds: 1001 } }],
    });

    expect(
      resolve("keep daily backups for 12 hours", [
        "GLUE",
        "DAY",
        "GLUE",
        "GLUE",
        "CALENDAR",
        "UNIT",
      ]),
    ).toEqual({
      tiers: [{ bucket: "day", window: { kind: "elapsed", milliseconds: 43200000 } }],
    });

    expect(() =>
      resolve("keep daily backups for one calendar hour", [
        "GLUE",
        "DAY",
        "GLUE",
        "GLUE",
        "CALENDAR",
        "PERIODS",
        "UNIT",
      ]),
    ).toThrow("conflict");

    expect(
      resolve("keep daily backups for one calendar month", [
        "GLUE",
        "DAY",
        "GLUE",
        "GLUE",
        "CALENDAR",
        "PERIODS",
        "UNIT",
      ]),
    ).toEqual({
      tiers: [{ bucket: "day", window: { kind: "calendar", unit: "month", count: 1 } }],
    });

    expect(() =>
      resolve("keep two backups per day for a week", [
        "GLUE",
        "ONE",
        "GLUE",
        "GLUE",
        "DAY",
        "GLUE",
        "CALENDAR",
        "UNIT",
      ]),
    ).toThrow("one newest");

    expect(() =>
      resolve("keep daily backups for the last 0.0001 ms", [
        "GLUE",
        "DAY",
        "GLUE",
        "GLUE",
        "GLUE",
        "ROLLING",
        "ELAPSED",
        "UNIT",
      ]),
    ).toThrow("whole units");

    expect(() =>
      resolve("keep daily backups for the last 2 months", [
        "GLUE",
        "DAY",
        "GLUE",
        "GLUE",
        "GLUE",
        "ROLLING",
        "ELAPSED",
        "UNIT",
      ]),
    ).toThrow("not elapsed");

    expect(() =>
      resolve("keep daily backups for the last 2 days", [
        "GLUE",
        "DAY",
        "GLUE",
        "GLUE",
        "GLUE",
        "ROLLING",
        "CALENDAR",
        "UNIT",
      ]),
    ).toThrow("conflict");

    expect(() =>
      resolve("keep daily backups for one week", ["GLUE", "O", "GLUE", "GLUE", "CALENDAR", "UNIT"]),
    ).toThrow("preserve");

    expect(() =>
      resolve("keep daily backups for one week and daily backups for two weeks", [
        "GLUE",
        "DAY",
        "GLUE",
        "GLUE",
        "CALENDAR",
        "UNIT",
        "GLUE",
        "DAY",
        "GLUE",
        "GLUE",
        "CALENDAR",
        "UNIT",
      ]),
    ).toThrow("at most one tier");
  },
);
