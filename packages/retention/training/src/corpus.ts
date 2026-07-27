/** Compact constructions, exact supervision and structural split grouping. */
import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { isDeepStrictEqual } from "node:util";

import { writeJsonl } from "../../../shared/training/jsonl.js";
import { reservedHashes } from "../../../shared/training/reserved.js";
import type { Role } from "../../src/labels.js";
import { prepare } from "../../src/prepare.js";
import { resolveRoles } from "../../src/resolve.js";
import type { Bucket, RetentionIntent, Tier } from "../../src/types.js";

export const FAMILIES = [
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
] as const;

export interface Row {
  text: string;
  context: Record<string, never>;
  intent: RetentionIntent | null;
  roles: readonly Role[];
  family: string;
  split: "train" | "validation" | "heldout";
  fingerprint: string;
  trainingEligible?: boolean;
}

export const hash = (bytes: string | Uint8Array) =>
  createHash("sha256").update(bytes).digest("hex");

export const readRows = (path: string): Row[] =>
  readFileSync(path, "utf8")
    .trim()
    .split("\n")
    .map((line) => JSON.parse(line));

const bucketRoles: Record<Bucket, Role> = {
  day: "DAY",
  week: "WEEK",
  month: "MONTH",
  year: "YEAR",
};

const adjectives: Record<Bucket, string> = {
  day: "daily",
  week: "weekly",
  month: "monthly",
  year: "yearly",
};

const names = [
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
];

function* sample(count: number, seed: number, excluded = new Set<string>()): Generator<Row> {
  if (!Number.isSafeInteger(count) || count < 1 || count > 100000) {
    throw Error("corpus rows must be in [1,100000]");
  }

  let state = seed;

  const random = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;

    return state / 2 ** 32;
  };

  const pick = <T>(values: readonly T[]): T => values[Math.floor(random() * values.length)]!;
  const seen = new Set<string>();
  let emitted = 0;

  for (let attempt = 0; emitted < count; attempt++) {
    if (attempt > count * 100) throw Error("corpus deduplication exhausted");

    const family = pick(FAMILIES);
    const positive = random() < 0.67;

    let text = "";
    let roles: Role[] = [];

    const add = (piece: string, role: Role = "GLUE") => {
      if (!piece) return;
      text += (text ? " " : "") + piece;

      roles.push(
        ...prepare(piece).lexemes.map((l) => {
          if (l.numeric !== undefined) {
            return role === "CALENDAR" || role === "ELAPSED" || role === "ONE" ? role : "ONE";
          }

          if (l.unit) return bucketRoles[l.unit as Bucket] === role ? role : "UNIT";
          if (l.bucket) return bucketRoles[l.bucket];
          if (l.rolling) return "ROLLING";
          if (l.calendar) return "PERIODS";

          return role === "O" || role === "UNSUPPORTED" ? role : "GLUE";
        }),
      );
    };

    let prefix = pick([
      "",
      "please",
      "for the backup policy,",
      "I want to",
      "I need to",
      "the service should",
      "configure retention to",
      "for our storage settings,",
      "in the application configuration,",
      "for the saved backup policy,",
    ]);

    if (random() < 0.35) {
      const owner = pick(["I", "we", "the application", "the service", "the caller"]);

      const modal =
        owner === "I" || owner === "we"
          ? pick(["would like to", "want to", "need to"])
          : pick(["should", "must", "needs to", "is required to"]);

      prefix = `${owner} ${modal}`;
    }

    if (random() < 0.2) {
      const courtesy = pick(["", "please"]);
      const verb = pick(["configure", "set", "update", "arrange", "adjust"]);

      const target = pick([
        "the application",
        "the caller",
        "the service",
        "the archive",
        "the backup policy",
      ]);

      prefix = `${courtesy} ${verb} ${target} to`.trim();
    }

    add(prefix, "O");

    const prefixLength = text.length;
    const prefixRoles = roles.length;

    add(pick(["keep", "retain", "preserve"]));

    const selected: Bucket[] =
      family === "mixed-tiers"
        ? pick([
            ["day", "month"],
            ["week", "year"],
            ["day", "week", "month"],
            ["day", "week", "month", "year"],
          ] as const).slice()
        : [
            family === "daily"
              ? "day"
              : family === "weekly"
                ? "week"
                : family === "monthly"
                  ? "month"
                  : family === "yearly"
                    ? "year"
                    : family === "calendar-months"
                      ? pick(["day", "week"] as const)
                      : pick(["day", "week", "month", "year"] as const),
          ];

    const tiers: Tier[] = [];

    for (const [i, bucket] of selected.entries()) {
      if (i) add(pick(["and", ", and", "; retain", "plus"]));

      if (random() < 0.35) {
        if (random() < 0.65) add(pick(["one", "1"]), "ONE");

        add(
          pick([
            "newest backup per",
            "backup per",
            "latest backup per",
            "backup copy per",
            "copies per",
            "snapshots per",
            "backups per",
          ]),
        );

        add(bucket, bucketRoles[bucket]);
      } else {
        const representative = random() < 0.4;

        if (representative) add(pick(["one", "1"]), "ONE");

        add(
          bucket === "year" ? pick(["yearly", "annual"]) : adjectives[bucket],
          bucketRoles[bucket],
        );

        add(
          representative
            ? pick(["backup", "snapshot", "backup copy"])
            : pick(["backups", "snapshots", "backup copies"]),
        );
      }

      const amount = pick([1, 2, 3, 4, 5, 6, 7, 8, 10, 12]);
      const rolling = family === "elapsed" || (family === "mixed-tiers" && random() < 0.2);
      const clock = family === "clock-duration";

      if (rolling || clock) {
        add(
          clock
            ? pick(["for", "over", "covering"])
            : pick([
                "for the last",
                "within the past",
                "over a rolling window of",
                "over an elapsed window of",
              ]),
        );

        const unit = pick(
          [
            { name: "milliseconds", scale: 1 },
            { name: "seconds", scale: 1000 },
            { name: "minutes", scale: 60000 },
            { name: "hours", scale: 3600000 },
            { name: "days", scale: 86400000 },
            { name: "weeks", scale: 604800000 },
          ].filter((unit) => !clock || unit.scale < 86400000),
        );

        const value =
          unit.scale === 1
            ? String(amount)
            : pick([String(amount), "0.001", "0.1", "0.25", "1.001"]);

        add(value, clock ? "CALENDAR" : "ELAPSED");
        add(unit.name, "UNIT");

        tiers.push({
          bucket,
          window: {
            kind: "elapsed",
            milliseconds: Math.round(Number(value) * unit.scale),
          },
        });
      } else {
        let unit: Bucket = family === "calendar-months" ? pick(["month", "year"] as const) : bucket;

        if (family === "calendar-conversion") {
          unit =
            bucket === "day"
              ? "week"
              : bucket === "month"
                ? "year"
                : bucket === "week"
                  ? "day"
                  : "month";
        }

        add(pick(["for", "over", "covering"]));

        add(
          family === "word-numbers"
            ? names[amount]!
            : pick([String(amount), names[amount]!, ...(amount === 1 ? ["a"] : [])]),
          "CALENDAR",
        );

        // An article is a numeric fact only beside its supported unit.
        if (roles.at(-1) === "GLUE" && text.endsWith(" a")) roles[roles.length - 1] = "CALENDAR";
        const calendarWord = !text.endsWith(" a") && random() < 0.25;

        if (calendarWord) add("calendar", "PERIODS");
        add(unit + (amount === 1 ? "" : "s"), "UNIT");

        const normalized =
          unit === "week" && bucket !== "week"
            ? { unit: "day" as const, count: amount * 7 }
            : unit === "year" && bucket !== "year"
              ? { unit: "month" as const, count: amount * 12 }
              : { unit, count: amount };

        tiers.push({ bucket, window: { kind: "calendar", ...normalized } });
      }
    }

    // Keep the base construction and all negative variants in one split.
    const fingerprint = hash(JSON.stringify([prepare(text).lexemes.map((l) => l.text), roles]));

    const bucket = parseInt(fingerprint.slice(0, 8), 16) % 10;
    const split = bucket < 8 ? "train" : bucket === 8 ? "validation" : "heldout";

    let gold: RetentionIntent | null = { tiers };
    let name: string = family;

    if (!positive) {
      const mode = pick([0, 1, 1, 2, 3, 4, 5, 6, 7, 8]);

      name = [
        "question",
        "description",
        "uncertainty",
        "unknown-unit",
        "duplicate-tier",
        "zero-window",
        "representative-count",
        "fractional-period",
        "elapsed-calendar-unit",
      ][mode]!;

      if (mode === 0) {
        add("?", "UNSUPPORTED");
      } else if (mode === 1 || mode === 2) {
        const body = mode === 1 ? text.slice(prefixLength).trimStart() : text;
        const originalRoles = mode === 1 ? roles.slice(prefixRoles) : roles;

        text = "";
        roles = [];

        if (mode === 1) {
          add(
            pick([
              "the previous configuration",
              "the old policy",
              "the saved settings",
              "the documentation",
            ]),
            "O",
          );

          if (random() < 0.7) {
            add(pick(["previously", "earlier", "formerly", "historically"]), "UNSUPPORTED");
          }

          add(pick(["said", "specified", "recorded", "described", "used"]), "UNSUPPORTED");
          add("to");
        } else {
          add(pick(["perhaps", "maybe", "probably"]), "UNSUPPORTED");
        }

        text += " " + body;
        roles.push(...originalRoles);
      } else if (mode === 4) {
        const bucket = selected[0]!;

        add("and");
        add(adjectives[bucket], bucketRoles[bucket]);
        add("backups for");
        add("3", "CALENDAR");
        add(bucket + "s", "UNIT");
      } else if (mode === 3) {
        add("using an unknown period unit", "UNSUPPORTED");
      } else if (mode === 5) {
        add("for zero retention periods", "UNSUPPORTED");
      } else if (mode === 6) {
        add("and keep");
        add("2", "ONE");
        add("representatives per bucket");
      } else if (mode === 7) {
        add("including a fraction of a calendar period", "UNSUPPORTED");
      } else {
        add("treat calendar months as fixed elapsed time", "UNSUPPORTED");
      }

      gold = null;
    }

    const prepared = prepare(text);
    let decoded: RetentionIntent | null = null;

    try {
      decoded = resolveRoles(
        prepared,
        roles.map((role) => ({ role, score: 1 })),
      ).intent;
    } catch {}

    if (!isDeepStrictEqual(decoded, gold)) {
      throw Error(
        `invalid supervision: ${text}: ${JSON.stringify(decoded)} vs ${JSON.stringify(gold)}`,
      );
    }

    if (excluded.has(hash(text.trim().toLowerCase())) || seen.has(text)) continue;
    seen.add(text);

    yield {
      text,
      context: {},
      intent: gold,
      roles,
      family: name,
      split,
      fingerprint,
    };

    emitted++;
  }
}

export function generate(count = 8000, seed = 20261012): Row[] {
  return [...sample(count, seed)];
}

export function writeCorpus(path: string, count = 8000): Row[] {
  const rows: Row[] = [];

  const excluded = reservedHashes(
    resolve(import.meta.dir, "../../evaluation"),
    resolve(import.meta.dir, "../../evaluation/consumed"),
  );

  function* records(): Generator<Row> {
    for (const row of sample(count, 20261012, excluded)) {
      rows.push(row);
      yield row;
    }
  }

  const written = writeJsonl(path, records());

  writeFileSync(
    path + ".manifest.json",
    JSON.stringify(
      {
        rows: rows.length,
        sha256: written.sha256,
        generatorSha256: hash(readFileSync(import.meta.filename)),
        seed: 20261012,
        trainingEligible: true,
        excludedReservedTexts: excluded.size,
      },
      null,
      2,
    ) + "\n",
    { flag: "wx" },
  );

  return rows;
}
