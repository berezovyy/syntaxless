/** Evaluation-only writer with independent UTC selection and hand-checked zoned targets. */
import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";

import type {
  Backup,
  Bucket,
  CompileOptions,
  Preview,
  RetentionIntent,
  Tier,
} from "../src/types.js";
import type { Row } from "../training/src/corpus.js";
import { BACKUPS, REFERENCE, REFERENCE_PREVIEW } from "./fixtures.js";

export interface Acceptance extends Row {
  critical: boolean;
  compileOptions: CompileOptions;
  compileTarget: ({ supported: true } & Preview) | { supported: false; code: string } | null;
}

const hash = (b: string | Uint8Array) => createHash("sha256").update(b).digest("hex");

const families = [
  "daily",
  "weekly",
  "monthly",
  "yearly",
  "mixed-tiers",
  "calendar-conversion",
  "elapsed",
  "word-numbers",
  "calendar-months",
  "clock-duration",
];

const bucketNames: Record<Bucket, string> = {
  day: "daily",
  week: "weekly",
  month: "monthly",
  year: "yearly",
};

function spell(value: number): string {
  const small = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
  ];

  if (value < 20) return small[value]!;

  const tens = [
    "",
    "",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
  ];

  return tens[Math.floor(value / 10)]! + (value % 10 ? "-" + small[value % 10] : "");
}

const rank = (timestamp: number, bucket: Bucket): number => {
  const d = new Date(timestamp);

  if (bucket === "day") return Math.floor(timestamp / 86400000);
  if (bucket === "week") return Math.floor((Math.floor(timestamp / 86400000) + 3) / 7);
  if (bucket === "month") return d.getUTCFullYear() * 12 + d.getUTCMonth();

  return d.getUTCFullYear();
};

function utcTarget(intent: RetentionIntent): Acceptance["compileTarget"] {
  const selected = new Map<string, number[]>();

  for (const [i, tier] of intent.tiers.entries()) {
    const candidates = BACKUPS.filter((b) =>
      tier.window.kind === "elapsed"
        ? REFERENCE - b.timestamp <= tier.window.milliseconds
        : rank(REFERENCE, tier.window.unit) - rank(b.timestamp, tier.window.unit) <
          tier.window.count,
    );

    // Sort independently, then choose the first candidate in each calendar bucket.
    const sorted = [...candidates].sort(
      (a, b) => b.timestamp - a.timestamp || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0),
    );

    const buckets = new Set<number>();

    for (const backup of sorted) {
      const key = rank(backup.timestamp, tier.bucket);

      if (buckets.has(key)) continue;
      buckets.add(key);
      selected.set(backup.id, [...(selected.get(backup.id) ?? []), i]);
    }
  }

  return {
    supported: true,
    keep: BACKUPS.filter((b) => selected.has(b.id)).map((b) => ({
      id: b.id,
      tierIndexes: selected.get(b.id)!,
    })),
    remove: BACKUPS.filter((b) => !selected.has(b.id)).map((b) => b.id),
  };
}

export function author(revision = 1): Acceptance[] {
  if (![1, 2, 3, 4, 5, 6].includes(revision)) throw Error("unsupported authoring revision");
  const rows: Acceptance[] = [];

  const normal: CompileOptions = {
    backups: BACKUPS,
    timeZone: "UTC",
    referenceInstant: REFERENCE,
  };

  function add(
    text: string,
    intent: RetentionIntent | null,
    family: string,
    critical = false,
    options = normal,
    target?: Acceptance["compileTarget"],
  ) {
    rows.push({
      text,
      intent,
      family,
      critical,
      context: {},
      roles: [],
      split: "heldout",
      fingerprint: hash(text),
      trainingEligible: false,
      compileOptions: options,
      compileTarget: target ?? (intent ? utcTarget(intent) : null),
    });
  }

  const prefixes =
    revision === 1
      ? [
          "For our backup storage,",
          "Please configure backups to",
          "I would like to",
          "In the backup settings,",
          "For recovery copies,",
          "Set the retention policy to",
        ]
      : revision === 2
        ? [
            "For the recovery policy,",
            "I need the service to",
            "Please arrange the application to",
            "The backup service should",
            "Within our storage policy,",
            "I want the caller to",
          ]
        : revision === 3
          ? [
              "In the recovery settings,",
              "We would like the application to",
              "Please configure the service to",
              "For backup operations,",
              "The caller needs to",
              "Set the backup options to",
            ]
          : revision === 4
            ? [
                "For our archive policy,",
                "Please set backup retention to",
                "We want the service to",
                "In the storage settings,",
                "For stored snapshots,",
                "The application must",
              ]
            : revision === 5
              ? [
                  "For routine backups,",
                  "We need the application to",
                  "Please update retention to",
                  "Within the archive settings,",
                  "For the service backups,",
                  "The storage service must",
                ]
              : [
                  "For our backup configuration,",
                  "Please configure the archive to",
                  "We want the application to",
                  "Within the service settings,",
                  "For stored backup copies,",
                  "The backup service should",
                ];

  for (const family of families.filter((family) => revision >= 4 || family !== "clock-duration")) {
    for (let i = 0; i < 75; i++) {
      const n = 1 + i + (family !== "word-numbers" ? (revision - 1) * 100 : 0);
      let text = `${prefixes[i % prefixes.length]} retain `;
      const tiers: Tier[] = [];

      const calendar = (bucket: Bucket, unit: Bucket, count: number, words = false) => {
        const normalized =
          unit === "week" && bucket !== "week"
            ? { unit: "day" as const, count: count * 7 }
            : unit === "year" && bucket !== "year"
              ? { unit: "month" as const, count: count * 12 }
              : { unit, count };

        tiers.push({ bucket, window: { kind: "calendar", ...normalized } });

        const amount = words ? spell(count) : count;
        const plural = count === 1 ? "" : "s";

        return `${bucketNames[bucket]} backup copies for ${amount} ${unit}${plural}`;
      };

      if (["daily", "weekly", "monthly", "yearly"].includes(family)) {
        const bucket = (
          { daily: "day", weekly: "week", monthly: "month", yearly: "year" } as const
        )[family as "daily" | "weekly" | "monthly" | "yearly"];

        text += calendar(bucket, bucket, n);
      } else if (family === "mixed-tiers") {
        text += calendar("day", "week", n) + " and " + calendar("month", "year", n);
      } else if (family === "calendar-conversion") {
        text += i % 2 ? calendar("week", "day", n) : calendar("month", "year", n);
      } else if (family === "elapsed") {
        const bucket = (["day", "week", "month", "year"] as const)[i % 4]!;
        const hours = i + 1 + (revision - 1) * 120;

        tiers.push({
          bucket,
          window: {
            kind: "elapsed",
            milliseconds: hours * 3600000,
          },
        });

        text += `one newest backup per ${bucket} within the past ${hours} hours`;
      } else if (family === "clock-duration") {
        const bucket = (["day", "week", "month", "year"] as const)[i % 4]!;

        const unit = [
          { name: "milliseconds", scale: 1 },
          { name: "seconds", scale: 1000 },
          { name: "minutes", scale: 60000 },
          { name: "hours", scale: 3600000 },
        ][i % 4]!;

        const count = i + 37;

        tiers.push({
          bucket,
          window: { kind: "elapsed", milliseconds: count * unit.scale },
        });

        text += `${bucketNames[bucket]} backup copies for ${count} ${unit.name}`;
      } else if (family === "word-numbers") {
        text += calendar("day", "day", n, true);
      } else {
        text += calendar(i % 2 ? "day" : "week", i % 3 ? "month" : "year", n);
      }

      add(text + ".", { tiers }, family);
    }
  }

  for (let i = 0; i < 630; i++) {
    const n =
      (revision === 6
        ? 4500
        : revision === 5
          ? 3600
          : revision === 4
            ? 2700
            : revision === 3
              ? 1800
              : revision === 2
                ? 900
                : 13) + i;

    const core = `keep daily backups for ${n} days`;
    const mode = i % 9;

    const variants = [
      revision === 6
        ? `Can we ${core}?`
        : revision === 5
          ? `Is it possible to ${core}?`
          : revision === 4
            ? `May the application ${core}?`
            : revision === 3
              ? `Could the caller ${core}?`
              : revision === 2
                ? `Can the service ${core}?`
                : `Could we ${core}?`,
      revision === 6
        ? `The archived settings formerly specified to ${core}.`
        : revision === 5
          ? `The saved configuration formerly recorded a request to ${core}.`
          : revision === 4
            ? `The documentation previously specified to ${core}.`
            : revision === 3
              ? `The earlier settings recorded a request to ${core}.`
              : revision === 2
                ? `The archived policy specified to ${core}.`
                : `The earlier configuration said to ${core}.`,
      revision === 6
        ? `Maybe the application could ${core}.`
        : revision === 5
          ? `Perhaps the caller could ${core}.`
          : revision === 4
            ? `Maybe we should ${core}.`
            : revision === 3
              ? `Probably ${core}.`
              : revision === 2
                ? `Perhaps ${core}.`
                : `Maybe ${core}.`,
      `Keep daily backups for ${n} fortnights.`,
      `${core} and daily backups for ${n + 1} days.`,
      `Keep monthly backups for zero months, instead of ${n} months.`,
      `Keep two backups per day for ${n} days.`,
      `Keep daily backups for 0.00${n} calendar months.`,
      `Keep monthly backups for the last ${n} months.`,
    ];

    add(
      variants[mode]!,
      null,
      [
        "question",
        "description",
        "uncertainty",
        "unknown-unit",
        "duplicate-tier",
        "zero-window",
        "representative-count",
        "fractional-period",
        "elapsed-calendar-unit",
      ][mode]!,
    );
  }

  if (revision < 4) {
    const reference: RetentionIntent = {
      tiers: [
        { bucket: "day", window: { kind: "calendar", unit: "day", count: 7 } },
        { bucket: "month", window: { kind: "calendar", unit: "month", count: 12 } },
      ],
    };

    add(
      "keep daily backups for a week and monthly backups for a year",
      reference,
      "mixed-tiers",
      true,
      normal,
      { supported: true, ...REFERENCE_PREVIEW },
    );

    const spring: readonly Backup[] = [
      { id: "march9", timestamp: Date.parse("2024-03-09T05:00:00Z") },
      { id: "march10", timestamp: Date.parse("2024-03-10T05:00:00Z") },
      { id: "march11", timestamp: Date.parse("2024-03-11T04:00:00Z") },
    ];

    const springOptions = {
      backups: spring,
      timeZone: "America/New_York",
      referenceInstant: Date.parse("2024-03-11T04:30:00Z"),
    };

    add(
      "keep daily backups for two days",
      {
        tiers: [{ bucket: "day", window: { kind: "calendar", unit: "day", count: 2 } }],
      },
      "daily",
      true,
      springOptions,
      {
        supported: true,
        keep: [
          { id: "march10", tierIndexes: [0] },
          { id: "march11", tierIndexes: [0] },
        ],
        remove: ["march9"],
      },
    );

    add(
      "keep one backup per day for the last 48 hours",
      {
        tiers: [{ bucket: "day", window: { kind: "elapsed", milliseconds: 172800000 } }],
      },
      "elapsed",
      true,
      springOptions,
      {
        supported: true,
        keep: spring.map((b) => ({ id: b.id, tierIndexes: [0] })),
        remove: [],
      },
    );

    const fall = {
      timeZone: "America/New_York",
      referenceInstant: Date.parse("2024-11-04T05:30:00Z"),
      backups: [
        { id: "first0130", timestamp: Date.parse("2024-11-03T05:30:00Z") },
        { id: "second0130", timestamp: Date.parse("2024-11-03T06:30:00Z") },
        { id: "nov4", timestamp: Date.parse("2024-11-04T05:00:00Z") },
      ],
    };

    add(
      "retain daily snapshots for 2 calendar days",
      {
        tiers: [{ bucket: "day", window: { kind: "calendar", unit: "day", count: 2 } }],
      },
      "daily",
      true,
      fall,
      {
        supported: true,
        keep: [
          { id: "second0130", tierIndexes: [0] },
          { id: "nov4", tierIndexes: [0] },
        ],
        remove: ["first0130"],
      },
    );

    add(
      "keep weekly backups for one week",
      {
        tiers: [{ bucket: "week", window: { kind: "calendar", unit: "week", count: 1 } }],
      },
      "weekly",
      true,
      {
        timeZone: "UTC",
        referenceInstant: Date.parse("2024-03-04T12:00:00Z"),
        backups: [
          { id: "sunday", timestamp: Date.parse("2024-03-03T23:59:59Z") },
          { id: "monday", timestamp: Date.parse("2024-03-04T00:00:00Z") },
        ],
      },
      {
        supported: true,
        keep: [{ id: "monday", tierIndexes: [0] }],
        remove: ["sunday"],
      },
    );

    add(
      "retain yearly backups for a year",
      {
        tiers: [{ bucket: "year", window: { kind: "calendar", unit: "year", count: 1 } }],
      },
      "yearly",
      true,
      {
        timeZone: "UTC",
        referenceInstant: Date.parse("2025-01-01T12:00:00Z"),
        backups: [
          { id: "dec31", timestamp: Date.parse("2024-12-31T23:59:59Z") },
          { id: "jan1", timestamp: Date.parse("2025-01-01T00:00:00Z") },
        ],
      },
      { supported: true, keep: [{ id: "jan1", tierIndexes: [0] }], remove: ["dec31"] },
    );

    add(
      "keep daily backups for one calendar month",
      {
        tiers: [{ bucket: "day", window: { kind: "calendar", unit: "month", count: 1 } }],
      },
      "calendar-months",
      true,
      {
        timeZone: "UTC",
        referenceInstant: Date.parse("2024-02-29T12:00:00Z"),
        backups: [
          { id: "jan31", timestamp: Date.parse("2024-01-31T23:59:59Z") },
          { id: "feb1", timestamp: Date.parse("2024-02-01T00:00:00Z") },
          { id: "feb29", timestamp: Date.parse("2024-02-29T00:00:00Z") },
        ],
      },
      {
        supported: true,
        keep: [
          { id: "feb1", tierIndexes: [0] },
          { id: "feb29", tierIndexes: [0] },
        ],
        remove: ["jan31"],
      },
    );

    add(
      "keep daily snapshots for the last 0.001 seconds",
      { tiers: [{ bucket: "day", window: { kind: "elapsed", milliseconds: 1 } }] },
      "elapsed",
      true,
      {
        timeZone: "UTC",
        referenceInstant: 1000,
        backups: [
          { id: "before", timestamp: 998 },
          { id: "at", timestamp: 999 },
        ],
      },
      { supported: true, keep: [{ id: "at", tierIndexes: [0] }], remove: ["before"] },
    );

    add(
      "keep monthly backups for one month",
      {
        tiers: [{ bucket: "month", window: { kind: "calendar", unit: "month", count: 1 } }],
      },
      "monthly",
      true,
      {
        timeZone: "UTC",
        referenceInstant: 1000,
        backups: [{ id: "future", timestamp: 1001 }],
      },
      { supported: false, code: "invalid-backups" },
    );

    add(
      "keep yearly snapshots for two years",
      {
        tiers: [{ bucket: "year", window: { kind: "calendar", unit: "year", count: 2 } }],
      },
      "yearly",
      true,
      {} as CompileOptions,
      { supported: false, code: "missing-context" },
    );

    for (const [text, family] of [
      ["keep two backups per day for a week", "representative-count"],
      ["keep daily backups for zero days", "zero-window"],
      ["keep daily backups for the last two months", "elapsed-calendar-unit"],
      ["keep monthly backups for one month and monthly backups for two months", "duplicate-tier"],
      ["keep daily backups for 0.0001 milliseconds", "fractional-period"],
      ["keep daily backups for one week?", "question"],
    ]) {
      add(text!, null, family!, true);
    }

    if (revision === 2) {
      const wording: Record<string, string> = {
        "keep daily backups for a week and monthly backups for a year":
          "retain daily backup copies for seven days and monthly backup copies for twelve months",
        "keep daily backups for two days":
          "preserve one latest backup per day for two calendar days",
        "keep one backup per day for the last 48 hours":
          "retain daily backups within the past two days",
        "retain daily snapshots for 2 calendar days": "keep daily backup copies for 2 days",
        "keep weekly backups for one week": "keep one weekly backup for a week",
        "retain yearly backups for a year": "keep one backup per year for one calendar year",
        "keep daily backups for one calendar month":
          "preserve daily backup copies for one calendar month",
        "keep daily snapshots for the last 0.001 seconds":
          "retain daily backups over an elapsed window of 1 millisecond",
        "keep monthly backups for one month": "retain monthly backup copies for 1 calendar month",
        "keep yearly snapshots for two years": "preserve copies per year for 2 years",
        "keep two backups per day for a week":
          "retain two newest backup copies per day for seven days",
        "keep daily backups for zero days": "retain daily copies for zero calendar days",
        "keep daily backups for the last two months":
          "retain daily backups within the past two months",
        "keep monthly backups for one month and monthly backups for two months":
          "retain monthly copies for one month and monthly copies for three months",
        "keep daily backups for 0.0001 milliseconds":
          "retain daily backups over a rolling window of 0.0002 ms",
        "keep daily backups for one week?": "may we retain daily backups for seven days?",
      };

      for (const row of rows) {
        if (row.critical && wording[row.text]) {
          row.text = wording[row.text]!;
          row.fingerprint = hash(row.text);
        }
      }
    }

    if (revision === 3) {
      const wording: Record<string, string> = {
        "keep daily backups for a week and monthly backups for a year":
          "keep daily snapshots for 7 calendar days plus monthly snapshots for 12 calendar months",
        "keep daily backups for two days": "keep one daily backup for 2 days",
        "keep one backup per day for the last 48 hours":
          "preserve one daily backup within the past 48 hours",
        "retain daily snapshots for 2 calendar days": "preserve daily copies for two calendar days",
        "keep weekly backups for one week": "retain one snapshot per week for one week",
        "retain yearly backups for a year": "retain one annual backup for one year",
        "keep daily backups for one calendar month": "retain one daily snapshot for one month",
        "keep daily snapshots for the last 0.001 seconds":
          "preserve daily snapshots over a rolling window of 0.001 seconds",
        "keep monthly backups for one month": "keep one monthly backup for one calendar month",
        "keep yearly snapshots for two years": "retain one annual snapshot for two calendar years",
        "keep two backups per day for a week": "preserve 2 daily backup copies for a week",
        "keep daily backups for zero days": "preserve daily backup copies for 0 days",
        "keep daily backups for the last two months":
          "preserve daily snapshots over an elapsed window of two months",
        "keep monthly backups for one month and monthly backups for two months":
          "preserve monthly backups for one month plus monthly backups for four months",
        "keep daily backups for 0.0001 milliseconds":
          "preserve daily copies over the last 0.0003 milliseconds",
        "keep daily backups for one week?":
          "could the application keep daily snapshots for 7 days?",
      };

      for (const row of rows) {
        if (row.critical && wording[row.text]) {
          row.text = wording[row.text]!;
          row.fingerprint = hash(row.text);
        }
      }
    }
  }

  if (revision === 4) {
    const reference: RetentionIntent = {
      tiers: [
        { bucket: "day", window: { kind: "calendar", unit: "day", count: 7 } },
        { bucket: "month", window: { kind: "calendar", unit: "month", count: 12 } },
      ],
    };

    add(
      "preserve daily backup copies for seven days and monthly backup copies for twelve months",
      reference,
      "mixed-tiers",
      true,
      normal,
      { supported: true, ...REFERENCE_PREVIEW },
    );

    const spring = {
      timeZone: "America/New_York",
      referenceInstant: Date.parse("2024-03-11T04:30:00Z"),
      backups: [
        { id: "march9", timestamp: Date.parse("2024-03-09T05:00:00Z") },
        { id: "march10", timestamp: Date.parse("2024-03-10T05:00:00Z") },
        { id: "march11", timestamp: Date.parse("2024-03-11T04:00:00Z") },
      ],
    };

    add(
      "retain copies per day for one calendar day",
      {
        tiers: [{ bucket: "day", window: { kind: "calendar", unit: "day", count: 1 } }],
      },
      "daily",
      true,
      spring,
      {
        supported: true,
        keep: [{ id: "march11", tierIndexes: [0] }],
        remove: ["march9", "march10"],
      },
    );

    add(
      "preserve copies per day within the past 47 hours",
      {
        tiers: [{ bucket: "day", window: { kind: "elapsed", milliseconds: 169200000 } }],
      },
      "elapsed",
      true,
      spring,
      {
        supported: true,
        keep: [
          { id: "march10", tierIndexes: [0] },
          { id: "march11", tierIndexes: [0] },
        ],
        remove: ["march9"],
      },
    );

    const fall = {
      timeZone: "America/New_York",
      referenceInstant: Date.parse("2024-11-04T05:30:00Z"),
      backups: [
        { id: "first0130", timestamp: Date.parse("2024-11-03T05:30:00Z") },
        { id: "second0130", timestamp: Date.parse("2024-11-03T06:30:00Z") },
        { id: "nov4", timestamp: Date.parse("2024-11-04T05:00:00Z") },
      ],
    };

    add(
      "keep snapshots per day for two days",
      {
        tiers: [{ bucket: "day", window: { kind: "calendar", unit: "day", count: 2 } }],
      },
      "daily",
      true,
      fall,
      {
        supported: true,
        keep: [
          { id: "second0130", tierIndexes: [0] },
          { id: "nov4", tierIndexes: [0] },
        ],
        remove: ["first0130"],
      },
    );

    add(
      "keep one weekly backup for twelve hours",
      {
        tiers: [{ bucket: "week", window: { kind: "elapsed", milliseconds: 43200000 } }],
      },
      "clock-duration",
      true,
      normal,
      {
        supported: true,
        keep: [{ id: "current-late", tierIndexes: [0] }],
        remove: ["current-early", "march9", "march8", "february", "april2023", "march2023"],
      },
    );

    add(
      "retain daily snapshots for 1.001 seconds",
      { tiers: [{ bucket: "day", window: { kind: "elapsed", milliseconds: 1001 } }] },
      "clock-duration",
      true,
      {
        timeZone: "UTC",
        referenceInstant: 100000,
        backups: [
          { id: "before", timestamp: 98998 },
          { id: "at", timestamp: 98999 },
        ],
      },
      { supported: true, keep: [{ id: "at", tierIndexes: [0] }], remove: ["before"] },
    );

    add(
      "retain daily backup copies for one month",
      {
        tiers: [{ bucket: "day", window: { kind: "calendar", unit: "month", count: 1 } }],
      },
      "calendar-months",
      true,
      {
        timeZone: "UTC",
        referenceInstant: Date.parse("2024-02-29T12:00:00Z"),
        backups: [
          { id: "jan31", timestamp: Date.parse("2024-01-31T23:59:59Z") },
          { id: "feb1", timestamp: Date.parse("2024-02-01T00:00:00Z") },
          { id: "feb29", timestamp: Date.parse("2024-02-29T00:00:00Z") },
        ],
      },
      {
        supported: true,
        keep: [
          { id: "feb1", tierIndexes: [0] },
          { id: "feb29", tierIndexes: [0] },
        ],
        remove: ["jan31"],
      },
    );

    add(
      "keep weekly snapshots for two calendar weeks",
      {
        tiers: [{ bucket: "week", window: { kind: "calendar", unit: "week", count: 2 } }],
      },
      "weekly",
      true,
      {
        timeZone: "UTC",
        referenceInstant: Date.parse("2024-03-04T12:00:00Z"),
        backups: [
          { id: "sunday", timestamp: Date.parse("2024-03-03T23:59:59Z") },
          { id: "monday", timestamp: Date.parse("2024-03-04T00:00:00Z") },
        ],
      },
      {
        supported: true,
        keep: [
          { id: "sunday", tierIndexes: [0] },
          { id: "monday", tierIndexes: [0] },
        ],
        remove: [],
      },
    );

    add(
      "preserve annual backup copies for one calendar year",
      {
        tiers: [{ bucket: "year", window: { kind: "calendar", unit: "year", count: 1 } }],
      },
      "yearly",
      true,
      {
        timeZone: "UTC",
        referenceInstant: Date.parse("2025-01-01T12:00:00Z"),
        backups: [
          { id: "dec31", timestamp: Date.parse("2024-12-31T23:59:59Z") },
          { id: "jan1", timestamp: Date.parse("2025-01-01T00:00:00Z") },
        ],
      },
      { supported: true, keep: [{ id: "jan1", tierIndexes: [0] }], remove: ["dec31"] },
    );

    add(
      "retain annual copies for three years",
      {
        tiers: [{ bucket: "year", window: { kind: "calendar", unit: "year", count: 3 } }],
      },
      "yearly",
      true,
      {} as CompileOptions,
      { supported: false, code: "missing-context" },
    );

    add(
      "retain monthly copies for a month",
      {
        tiers: [{ bucket: "month", window: { kind: "calendar", unit: "month", count: 1 } }],
      },
      "monthly",
      true,
      {
        timeZone: "UTC",
        referenceInstant: 2000,
        backups: [{ id: "future", timestamp: 2001 }],
      },
      { supported: false, code: "invalid-backups" },
    );

    for (const [text, family] of [
      ["retain daily backups for one calendar hour", "calendar-clock-conflict"],
      ["keep daily backups for 0 milliseconds", "zero-window"],
      ["retain two snapshots per week for a year", "representative-count"],
      ["retain daily copies within the past three months", "elapsed-calendar-unit"],
      ["retain daily backups for one week and daily backups for two weeks", "duplicate-tier"],
      ["keep copies per day for 0.0004 milliseconds", "fractional-period"],
      ["may the service retain daily backups for four weeks?", "question"],
    ]) {
      add(text!, null, family!, true);
    }
  }

  if (revision >= 5) {
    const reference: RetentionIntent = {
      tiers: [
        { bucket: "day", window: { kind: "calendar", unit: "day", count: 14 } },
        { bucket: "month", window: { kind: "calendar", unit: "month", count: 24 } },
      ],
    };

    add(
      "retain daily snapshots for two weeks plus monthly snapshots for two years",
      reference,
      "mixed-tiers",
      true,
      normal,
      {
        supported: true,
        keep: [
          { id: "current-late", tierIndexes: [0, 1] },
          { id: "march9", tierIndexes: [0] },
          { id: "march8", tierIndexes: [0] },
          { id: "february", tierIndexes: [1] },
          { id: "april2023", tierIndexes: [1] },
          { id: "march2023", tierIndexes: [1] },
        ],
        remove: ["current-early"],
      },
    );

    const spring = {
      timeZone: "America/New_York",
      referenceInstant: Date.parse("2024-03-11T04:30:00Z"),
      backups: [
        { id: "march9", timestamp: Date.parse("2024-03-09T05:00:00Z") },
        { id: "march10", timestamp: Date.parse("2024-03-10T05:00:00Z") },
        { id: "march11", timestamp: Date.parse("2024-03-11T04:00:00Z") },
      ],
    };

    add(
      "preserve daily backup copies for three calendar days",
      {
        tiers: [{ bucket: "day", window: { kind: "calendar", unit: "day", count: 3 } }],
      },
      "daily",
      true,
      spring,
      {
        supported: true,
        keep: spring.backups.map((b) => ({ id: b.id, tierIndexes: [0] })),
        remove: [],
      },
    );

    add(
      "retain backup copies per day for the last 46 hours",
      {
        tiers: [{ bucket: "day", window: { kind: "elapsed", milliseconds: 165600000 } }],
      },
      "elapsed",
      true,
      spring,
      {
        supported: true,
        keep: [
          { id: "march10", tierIndexes: [0] },
          { id: "march11", tierIndexes: [0] },
        ],
        remove: ["march9"],
      },
    );

    const fall = {
      timeZone: "America/New_York",
      referenceInstant: Date.parse("2024-11-04T05:30:00Z"),
      backups: [
        { id: "first0130", timestamp: Date.parse("2024-11-03T05:30:00Z") },
        { id: "second0130", timestamp: Date.parse("2024-11-03T06:30:00Z") },
        { id: "nov4", timestamp: Date.parse("2024-11-04T05:00:00Z") },
      ],
    };

    add(
      "retain one daily snapshot for two days",
      {
        tiers: [{ bucket: "day", window: { kind: "calendar", unit: "day", count: 2 } }],
      },
      "daily",
      true,
      fall,
      {
        supported: true,
        keep: [
          { id: "second0130", tierIndexes: [0] },
          { id: "nov4", tierIndexes: [0] },
        ],
        remove: ["first0130"],
      },
    );

    add(
      "preserve one weekly snapshot for 15 hours",
      {
        tiers: [{ bucket: "week", window: { kind: "elapsed", milliseconds: 54000000 } }],
      },
      "clock-duration",
      true,
      normal,
      {
        supported: true,
        keep: [{ id: "current-late", tierIndexes: [0] }],
        remove: ["current-early", "march9", "march8", "february", "april2023", "march2023"],
      },
    );

    add(
      "keep daily backup copies for 0.25 hours",
      { tiers: [{ bucket: "day", window: { kind: "elapsed", milliseconds: 900000 } }] },
      "clock-duration",
      true,
      {
        timeZone: "UTC",
        referenceInstant: REFERENCE,
        backups: [
          { id: "before", timestamp: REFERENCE - 900001 },
          { id: "at", timestamp: REFERENCE - 900000 },
        ],
      },
      { supported: true, keep: [{ id: "at", tierIndexes: [0] }], remove: ["before"] },
    );

    add(
      "preserve daily backup copies for two calendar months",
      {
        tiers: [{ bucket: "day", window: { kind: "calendar", unit: "month", count: 2 } }],
      },
      "calendar-months",
      true,
      {
        timeZone: "UTC",
        referenceInstant: Date.parse("2024-02-29T12:00:00Z"),
        backups: [
          { id: "dec31", timestamp: Date.parse("2023-12-31T23:59:59Z") },
          { id: "jan31", timestamp: Date.parse("2024-01-31T23:59:59Z") },
          { id: "feb29", timestamp: Date.parse("2024-02-29T00:00:00Z") },
        ],
      },
      {
        supported: true,
        keep: [
          { id: "jan31", tierIndexes: [0] },
          { id: "feb29", tierIndexes: [0] },
        ],
        remove: ["dec31"],
      },
    );

    add(
      "preserve weekly backup copies for 1 calendar week",
      {
        tiers: [{ bucket: "week", window: { kind: "calendar", unit: "week", count: 1 } }],
      },
      "weekly",
      true,
      {
        timeZone: "UTC",
        referenceInstant: Date.parse("2024-03-04T12:00:00Z"),
        backups: [
          { id: "sunday", timestamp: Date.parse("2024-03-03T23:59:59Z") },
          { id: "monday", timestamp: Date.parse("2024-03-04T00:00:00Z") },
        ],
      },
      {
        supported: true,
        keep: [{ id: "monday", tierIndexes: [0] }],
        remove: ["sunday"],
      },
    );

    add(
      "keep annual snapshots for two calendar years",
      {
        tiers: [{ bucket: "year", window: { kind: "calendar", unit: "year", count: 2 } }],
      },
      "yearly",
      true,
      {
        timeZone: "UTC",
        referenceInstant: Date.parse("2025-01-01T12:00:00Z"),
        backups: [
          { id: "dec31", timestamp: Date.parse("2024-12-31T23:59:59Z") },
          { id: "jan1", timestamp: Date.parse("2025-01-01T00:00:00Z") },
        ],
      },
      {
        supported: true,
        keep: [
          { id: "dec31", tierIndexes: [0] },
          { id: "jan1", tierIndexes: [0] },
        ],
        remove: [],
      },
    );

    add(
      "keep one annual backup for four years",
      {
        tiers: [{ bucket: "year", window: { kind: "calendar", unit: "year", count: 4 } }],
      },
      "yearly",
      true,
      {} as CompileOptions,
      { supported: false, code: "missing-context" },
    );

    add(
      "preserve one monthly backup for two months",
      {
        tiers: [{ bucket: "month", window: { kind: "calendar", unit: "month", count: 2 } }],
      },
      "monthly",
      true,
      {
        timeZone: "UTC",
        referenceInstant: 3000,
        backups: [{ id: "future", timestamp: 3001 }],
      },
      { supported: false, code: "invalid-backups" },
    );

    for (const [text, family] of [
      ["preserve daily backups for two calendar seconds", "calendar-clock-conflict"],
      ["retain daily copies for 0 seconds", "zero-window"],
      ["preserve two backup copies per month for one year", "representative-count"],
      ["preserve daily copies within the past four months", "elapsed-calendar-unit"],
      [
        "preserve weekly backups for one week plus weekly backups for three weeks",
        "duplicate-tier",
      ],
      ["retain backups per day for 0.0005 ms", "fractional-period"],
      ["can we preserve daily snapshots for five weeks?", "question"],
    ]) {
      add(text!, null, family!, true);
    }
  }

  if (revision === 6) {
    for (const row of rows.filter((row) => row.critical)) {
      row.text = "For the stored backup policy, " + row.text;
    }
  }

  return rows;
}

if (import.meta.main) {
  const [scopePath, out, revision] = process.argv.slice(2);

  if (!scopePath || !out) throw Error("usage: author <scope-freeze.json> <new-reserved.jsonl>");

  const bytes = readFileSync(scopePath);
  const scope = JSON.parse(bytes.toString());
  const rows = author(Number(revision ?? 1));
  const body = rows.map((r) => JSON.stringify(r)).join("\n") + "\n";

  writeFileSync(out, body, { flag: "wx" });

  writeFileSync(
    out + ".meta.json",
    JSON.stringify(
      {
        sha256: hash(body),
        candidateSha256: scope.artifactFileSha256,
        scopeFreezeSha256: hash(bytes),
        createdAt: new Date().toISOString(),
        trainingEligible: false,
        rows: rows.length,
        authorship:
          "AI-authored by the same implementation assistant through an evaluation-only writer",
        independenceLimits:
          "No production parser/compiler or training renderer imports. " +
          "Independent UTC selection and hand-checked timezone witnesses. " +
          "Bounded constructions and numeric variation are not independent human evidence " +
          "or population accuracy.",
      },
      null,
      2,
    ) + "\n",
    { flag: "wx" },
  );

  console.log({ rows: rows.length, sha256: hash(body) });
}
