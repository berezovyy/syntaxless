import { expect, test } from "bun:test";

import { compile } from "../src/compile.js";
import type { Backup, CompileOptions, RetentionIntent } from "../src/types.js";

const stamp = (date: string) => Date.parse(date);
const backup = (id: string, date: string): Backup => ({ id, timestamp: stamp(date) });

const reference: RetentionIntent = {
  tiers: [
    { bucket: "day", window: { kind: "calendar", unit: "day", count: 7 } },
    { bucket: "month", window: { kind: "calendar", unit: "month", count: 12 } },
  ],
};

function preview(
  intent: RetentionIntent,
  backups: readonly Backup[],
  date = "2024-03-15T12:00:00Z",
  timeZone = "UTC",
) {
  const result = compile(intent, { backups, timeZone, referenceInstant: stamp(date) });

  if (!result.supported) throw Error(result.reason);

  return { keep: result.keep, remove: result.remove };
}

test(
  "reference uses current-inclusive calendar buckets, " +
    "keeps overlap provenance and does not fill missing buckets",
  () => {
    const backups = [
      backup("today-early", "2024-03-15T01:00:00Z"),
      backup("today-late", "2024-03-15T02:00:00Z"),
      backup("lower-day", "2024-03-09T00:00:00Z"),
      backup("prior-day", "2024-03-08T23:59:59Z"),
      backup("february", "2024-02-20T00:00:00Z"),
      backup("lower-month", "2023-04-01T00:00:00Z"),
      backup("old", "2023-03-31T23:59:59Z"),
    ];

    const original = structuredClone(backups);

    expect(preview(reference, backups)).toEqual({
      keep: [
        { id: "today-late", tierIndexes: [0, 1] },
        { id: "lower-day", tierIndexes: [0] },
        { id: "february", tierIndexes: [1] },
        { id: "lower-month", tierIndexes: [1] },
      ],
      remove: ["today-early", "prior-day", "old"],
    });

    expect(backups).toEqual(original);
    expect(preview(reference, [])).toEqual({ keep: [], remove: [] });

    expect(
      preview(
        {
          tiers: [{ bucket: "day", window: { kind: "calendar", unit: "day", count: 1 } }],
        },
        [backup("leap", "2024-02-29T01:00:00Z"), backup("prior", "2024-02-28T23:00:00Z")],
        "2024-02-29T12:00:00Z",
      ),
    ).toEqual({ keep: [{ id: "leap", tierIndexes: [0] }], remove: ["prior"] });

    expect(
      preview(
        {
          tiers: [{ bucket: "week", window: { kind: "calendar", unit: "week", count: 1 } }],
        },
        [backup("sunday", "2024-03-03T23:59:59Z"), backup("monday", "2024-03-04T00:00:00Z")],
        "2024-03-04T12:00:00Z",
      ),
    ).toEqual({ keep: [{ id: "monday", tierIndexes: [0] }], remove: ["sunday"] });

    expect(
      preview(
        {
          tiers: [{ bucket: "year", window: { kind: "calendar", unit: "year", count: 1 } }],
        },
        [backup("december", "2024-12-31T23:59:59Z"), backup("january", "2025-01-01T00:00:00Z")],
        "2025-01-01T12:00:00Z",
      ),
    ).toEqual({ keep: [{ id: "january", tierIndexes: [0] }], remove: ["december"] });
  },
);

test(
  "calendar dates differ from elapsed windows across DST, " +
    "exact cutoffs and repeated hours use instants",
  () => {
    const backups = [
      backup("march9", "2024-03-09T05:00:00Z"),
      backup("march10", "2024-03-10T05:00:00Z"),
      backup("march11", "2024-03-11T04:00:00Z"),
    ];

    const calendar: RetentionIntent = {
      tiers: [{ bucket: "day", window: { kind: "calendar", unit: "day", count: 2 } }],
    };

    const elapsed: RetentionIntent = {
      tiers: [{ bucket: "day", window: { kind: "elapsed", milliseconds: 48 * 3600000 } }],
    };

    expect(preview(calendar, backups, "2024-03-11T04:30:00Z", "America/New_York")).toEqual({
      keep: [
        { id: "march10", tierIndexes: [0] },
        { id: "march11", tierIndexes: [0] },
      ],
      remove: ["march9"],
    });

    expect(preview(elapsed, backups, "2024-03-11T04:30:00Z", "America/New_York").remove).toEqual(
      [],
    );

    const cutoff = stamp("2024-03-15T12:00:00Z") - 3600000;

    expect(
      compile(
        {
          tiers: [{ bucket: "day", window: { kind: "elapsed", milliseconds: 3600000 } }],
        },
        {
          timeZone: "UTC",
          referenceInstant: stamp("2024-03-15T12:00:00Z"),
          backups: [
            { id: "before", timestamp: cutoff - 1 },
            { id: "at", timestamp: cutoff },
          ],
        },
      ),
    ).toEqual({
      supported: true,
      keep: [{ id: "at", tierIndexes: [0] }],
      remove: ["before"],
    });

    expect(
      preview(
        calendar,
        [
          backup("first-0130", "2024-11-03T05:30:00Z"),
          backup("second-0130", "2024-11-03T06:30:00Z"),
          backup("next-day", "2024-11-04T05:00:00Z"),
        ],
        "2024-11-04T05:30:00Z",
        "America/New_York",
      ),
    ).toEqual({
      keep: [
        { id: "second-0130", tierIndexes: [0] },
        { id: "next-day", tierIndexes: [0] },
      ],
      remove: ["first-0130"],
    });
  },
);

test(
  "ties use Unicode scalar order and invalid context, " +
    "policies or metadata refuse the complete preview",
  () => {
    const same = stamp("2024-03-15T01:00:00Z");

    expect(
      compile(reference, {
        timeZone: "UTC",
        referenceInstant: same,
        backups: [
          { id: "😀", timestamp: same },
          { id: "\uE000", timestamp: same },
        ],
      }),
    ).toEqual({
      supported: true,
      keep: [{ id: "\uE000", tierIndexes: [0, 1] }],
      remove: ["😀"],
    });

    expect(compile(reference, {} as CompileOptions)).toMatchObject({
      supported: false,
      code: "missing-context",
    });

    expect(
      compile(reference, {
        backups: [],
        timeZone: "Missing/Timezone",
        referenceInstant: same,
      }),
    ).toMatchObject({ supported: false, code: "invalid-context" });

    expect(
      compile(reference, {
        backups: [{ id: "future", timestamp: same + 1 }],
        timeZone: "UTC",
        referenceInstant: same,
      }),
    ).toMatchObject({ supported: false, code: "invalid-backups" });

    for (const backups of [
      [
        { id: "a", timestamp: same },
        { id: "a", timestamp: same },
      ],
      [{ id: "bad\0", timestamp: same }],
      [{ id: "bad", timestamp: 0.5 }],
    ]) {
      expect(
        compile(reference, { backups, timeZone: "UTC", referenceInstant: same }),
      ).toMatchObject({
        supported: false,
        code: "invalid-backups",
      });
    }

    for (const intent of [
      { tiers: [] },
      { tiers: [reference.tiers[0]!, reference.tiers[0]!] },
      { tiers: [{ bucket: "day", window: { kind: "calendar", unit: "day", count: 0 } }] },
    ]) {
      expect(
        compile(intent as RetentionIntent, {
          backups: [],
          timeZone: "UTC",
          referenceInstant: same,
        }).supported,
      ).toBe(false);
    }
  },
);
