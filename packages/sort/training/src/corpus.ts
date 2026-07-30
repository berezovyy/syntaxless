/** Small, seeded construction corpus; exact caller facts never come from the model. */
import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { isDeepStrictEqual } from "node:util";

import { writeJsonl } from "../../../shared/training/jsonl.js";
import { reservedHashes } from "../../../shared/training/reserved.js";
import type { Role } from "../../src/labels.js";
import { prepare } from "../../src/prepare.js";
import { resolveRoles } from "../../src/resolve.js";
import type { Context, Direction, Field, SortIntent, SortKey } from "../../src/types.js";

export const FAMILIES = [
  "explicit-direction",
  "ordering-alias",
  "priority",
  "null-first",
  "null-last",
  "null-update",
  "caller-default",
] as const;

export interface Row {
  text: string;
  context: Context;
  intent: SortIntent | null;
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

function* sample(count: number, seed: number, excluded = new Set<string>()): Generator<Row> {
  if (!Number.isSafeInteger(count) || count < 1 || count > 100000) {
    throw new Error("corpus rows must be in [1,100000]");
  }

  let state = seed;

  const random = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;

    return state / 2 ** 32;
  };

  const pick = <T>(xs: readonly T[]): T => xs[Math.floor(random() * xs.length)]!;
  const seen = new Set<string>();
  let emitted = 0;

  for (let attempt = 0; emitted < count; attempt++) {
    if (attempt > count * 100) throw new Error("construction corpus exhausted");

    const context: Context = {
      defaultNulls: pick(["first", "last"]),
      tieBreaker: `key${attempt}`,
      fields: [
        {
          id: `metric${attempt}`,
          type: "number",
          column: "amount",
          aliases: [pick(["price", "prices", "score", "amount", "balance"])],
          orderingAliases: [
            {
              text: pick(["cheapest", "lowest cost", "least expensive"]),
              direction: "asc",
            },
            { text: pick(["newest", "latest", "most recent"]), direction: "desc" },
          ],
        },
        {
          id: `time${attempt}`,
          type: "number",
          column: "created",
          aliases: ["created", "timestamp"],
          orderingAliases: [
            { text: "latest entry", direction: "desc" },
            { text: "oldest entry", direction: "asc" },
          ],
        },
        {
          id: `name${attempt}`,
          type: "string",
          column: "label",
          aliases: [pick(["name", "label", "title", "category"])],
          orderingAliases: [
            { text: "alphabetical", direction: "asc" },
            { text: "reverse alphabetical", direction: "desc" },
          ],
        },
        {
          id: `flag${attempt}`,
          type: "boolean",
          column: "flag",
          aliases: [pick(["enabled", "verified", "approved", "flag"])],
        },
        { id: `key${attempt}`, type: "number", column: "id" },
      ],
    };

    let text = "";
    let roles: Role[] = [];

    const add = (piece: string, role: Role) => {
      if (!piece) return;
      text += (text ? " " : "") + piece;

      roles.push(
        ...prepare(piece, context).lexemes.map((l) =>
          l.field ? "FIELD" : l.direction ? (l.direction === "asc" ? "ASC" : "DESC") : role,
        ),
      );
    };

    const object = pick([
      "records",
      "entries",
      "items",
      "rows",
      "result",
      "results",
      "output",
      "list",
    ]);

    const prefix = pick([
      "",
      "sort by",
      "order by",
      "arrange by",
      `${pick(["please sort", "please arrange", "order", "arrange"])} the ${object} by`,
      `${pick(["show", "display", "return", "list"])} the ${object} ` +
        `${pick(["sorted", "ordered", "arranged"])} by`,
      `for my ${object}, ${pick(["sort", "order", "arrange"])} by`,
      `I ${pick(["want", "need"])} the ${object} ${pick(["sorted", "ordered"])} by`,
    ]);

    add(prefix, "O");
    const keys: SortKey[] = [];
    const fields = context.fields.slice(0, -1);

    function order(field: Field, alias: boolean, placement?: "first" | "last"): void {
      const direction: Direction = pick(["asc", "desc"]);

      if (alias && field.orderingAliases) {
        const entry = field.orderingAliases.find((a) => a.direction === direction)!;

        add(entry.text, "FIELD");
        add(pick(["", "first"]), "GLUE");
      } else {
        add(pick([field.id, ...field.aliases!]), "FIELD");

        add(
          pick(
            direction === "asc"
              ? [
                  "ascending",
                  "in ascending order",
                  "low to high",
                  "lowest to highest",
                  "increasing",
                ]
              : [
                  "descending",
                  "in descending order",
                  "high to low",
                  "highest to lowest",
                  "decreasing",
                ],
          ),
          direction === "asc" ? "ASC" : "DESC",
        );
      }

      if (placement) {
        add(pick(["", "with", "placing"]), "GLUE");
        add(pick(["nulls", "missing values", "absent values", "null values"]), "NULLS");

        add(
          pick(
            placement === "first"
              ? ["first", "at the start", "at the beginning"]
              : ["last", "at the end", "at the bottom"],
          ),
          placement === "first" ? "FIRST" : "LAST",
        );
      }

      keys.push({
        field: field.id,
        direction,
        nulls: placement ?? context.defaultNulls!,
        nullsSource: placement ? "explicit" : "caller-default",
      });
    }

    const positive = random() < 0.67;
    const family = pick(FAMILIES);

    let gold: SortIntent | null;

    if (family === "priority" || family === "null-update") {
      const a = pick(fields);
      const b = pick(fields.filter((f) => f.id !== a.id));

      order(a, random() < 0.5);
      add(pick([", then", "then", "followed by", "and then", ", followed by", ", next"]), "NEXT");
      order(b, random() < 0.5);

      if (random() < 0.25) {
        add(pick(["then", ", then", "followed by"]), "NEXT");

        order(
          fields.find((f) => f.id !== a.id && f.id !== b.id)!,
          false,
        );
      }

      if (family === "null-update") {
        const update = pick(keys);
        const placement = pick(["first", "last"] as const);
        const f = fields.find((f) => f.id === update.field)!;

        add(pick([",", "then", ", then"]), "NEXT");
        add(pick(["missing", "null", "absent"]), "NULLS");
        add(pick([f.id, ...f.aliases!]), "FIELD");
        add(placement, placement === "first" ? "FIRST" : "LAST");
        update.nulls = placement;
        update.nullsSource = "explicit";
      }
    } else {
      order(
        pick(family === "ordering-alias" ? fields.filter((f) => f.orderingAliases) : fields),
        family === "ordering-alias",
        family === "null-first" ? "first" : family === "null-last" ? "last" : undefined,
      );
    }

    gold = { keys, tieBreaker: context.tieBreaker! };

    // Assign the base construction before negative wording or context changes.
    const fingerprint = hash(
      JSON.stringify([prepare(text, context).lexemes.map((l) => l.text), roles]),
    );

    const bucket = parseInt(fingerprint.slice(0, 8), 16) % 10;
    const split = bucket < 8 ? "train" : bucket === 8 ? "validation" : "heldout";
    let name: string = family;

    if (!positive) {
      const mode = Math.floor(random() * 8);

      name = [
        "question",
        "historical",
        "uncertain",
        "unknown-field",
        "contradiction",
        "missing-policy",
        "unsupported",
        "incomplete",
      ][mode]!;

      if (mode === 0) {
        add("?", "UNSUPPORTED");
      } else if (mode === 1 || mode === 2) {
        if (random() < 0.5) {
          add(
            pick(
              mode === 1
                ? ["was requested previously", "had been used", "was specified in the report"]
                : ["perhaps", "maybe", "probably"],
            ),
            "UNSUPPORTED",
          );
        } else {
          const originalText = text.slice(prefix ? prefix.length + 1 : 0);

          const originalRoles = roles.slice(
            prepare(prefix || "neutral", context).lexemes.length * Number(Boolean(prefix)),
          );

          text = "";
          roles = [];

          if (mode === 1) {
            add(
              pick([
                "the previous list",
                "the saved ordering",
                "the old report",
                "an earlier request",
              ]),
              "O",
            );

            add(pick(["used", "had", "specified", "was", "were"]), "UNSUPPORTED");
            add(pick(["sorted by", "ordered by", "arranged by", ""]), "O");
          } else {
            add(pick(["maybe", "perhaps", "probably", "possibly"]), "UNSUPPORTED");
            add(pick(["sort by", "please arrange by", "order by"]), "O");
          }

          text += (text ? " " : "") + originalText;
          roles.push(...originalRoles);
        }
      } else if (mode === 3) {
        add("then", "NEXT");
        add("unlisted", "FIELD");
        add("ascending", "ASC");
      } else if (mode === 4) {
        add(
          keys.at(-1)!.direction === "asc" ? "descending" : "ascending",
          keys.at(-1)!.direction === "asc" ? "DESC" : "ASC",
        );
      } else if (mode === 5) {
        delete context.defaultNulls;

        keys.forEach((k) => {
          k.nullsSource = "caller-default";
        });

        if (keys.some((k) => k.nullsSource === "explicit") || roles.includes("NULLS")) {
          add("without a policy", "UNSUPPORTED");
        }
      } else if (mode === 6) {
        add(
          pick([
            "by distance from my location",
            "with locale collation",
            "shuffle randomly",
            "by a custom formula",
            "using regex",
          ]),
          "UNSUPPORTED",
        );
      } else {
        add("then", "NEXT");
        add(context.fields.at(-1)!.id, "FIELD");
      }

      gold = null;
    }

    const prepared = prepare(text, context);
    let decoded: SortIntent | null = null;

    try {
      decoded = resolveRoles(
        prepared,
        roles.map((role) => ({ role, score: 1 })),
        context,
      ).intent;
    } catch {}

    if (!isDeepStrictEqual(decoded, gold)) {
      throw new Error(`invalid supervision: ${text}: ${JSON.stringify(decoded)}`);
    }

    if (excluded.has(hash(text.trim().toLowerCase())) || seen.has(text)) continue;
    seen.add(text);
    yield { text, context, intent: gold, roles, family: name, split, fingerprint };
    emitted++;
  }
}

export function generate(count = 8000, seed = 20261011): Row[] {
  return [...sample(count, seed)];
}

export function writeCorpus(path: string, count = 8000): Row[] {
  const rows: Row[] = [];

  const excluded = reservedHashes(
    resolve(import.meta.dir, "../../evaluation/reserved"),
    resolve(import.meta.dir, "../../evaluation/consumed"),
  );

  function* records(): Generator<Row> {
    for (const row of sample(count, 20261011, excluded)) {
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
        seed: 20261011,
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
