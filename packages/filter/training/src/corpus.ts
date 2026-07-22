/** Compact, seeded examples with caller field names separated from language roles. */
import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { isDeepStrictEqual } from "node:util";

import { writeJsonl } from "../../../shared/training/jsonl.js";
import { reservedHashes } from "../../../shared/training/reserved.js";
import type { Role } from "../../src/labels.js";
import { prepare } from "../../src/prepare.js";
import { resolveRoles } from "../../src/resolve.js";
import type { Context, FilterIntent, Scalar } from "../../src/types.js";

export interface Row {
  text: string;
  context: Context;
  intent: FilterIntent | null;
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

function* sample(
  count = 8000,
  seed = 20261011,
  excluded: ReadonlySet<string> = new Set(),
): Generator<Row> {
  if (!Number.isSafeInteger(count) || count < 1 || count > 100000) {
    throw new Error("corpus rows must be in [1,100000]");
  }

  let state = seed;

  const random = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;

    return state / 2 ** 32;
  };

  const pick = <T>(items: readonly T[]): T => items[Math.floor(random() * items.length)]!;
  const integer = (max: number) => Math.floor(random() * max);
  let emitted = 0;
  const seen = new Set<string>();

  for (let attempt = 0; emitted < count; attempt++) {
    if (attempt > count * 100) throw new Error("corpus deduplication exhausted");

    const context: Context = {
      fields: [
        {
          id: `metric${attempt}`,
          aliases: [pick(["spending", "balance", "credits", "score", "amount"])],
          type: "number",
          column: "amount",
        },
        {
          id: `category${attempt}`,
          aliases: [pick(["status", "region", "label", "plan", "category"])],
          type: "string",
          column: "label",
        },
        {
          id: `flag${attempt}`,
          aliases: [pick(["trial", "enabled", "verified", "flag", "approved"])],
          type: "boolean",
          column: "flag",
        },
      ],
    };

    let text = "";
    let roles: Role[] = [];

    const add = (piece: string, role: Role): void => {
      if (!piece) return;
      const prepared = prepare(piece, context);

      text += (text ? " " : "") + piece;

      roles.push(
        ...prepared.lexemes.map((lexeme) =>
          lexeme.field !== undefined
            ? ("FIELD" as const)
            : lexeme.value !== undefined
              ? ("VALUE" as const)
              : role,
        ),
      );
    };

    let family = "";

    function atom(shape: number): FilterIntent {
      const field = context.fields[integer(3)]!;
      const alias = pick([field.id, ...field.aliases!]);

      const literal = (): Scalar =>
        field.type === "number"
          ? integer(2001) - 1000 + pick([0, 0.25, 0.5])
          : field.type === "boolean"
            ? random() < 0.5
            : pick([
                "active",
                "pending",
                "trial",
                "Berlin",
                "A&B",
                "O'Reilly",
                "",
                "😀",
                `item-${integer(1000)}`,
              ]);

      const v = literal();

      add(alias, "FIELD");

      if (shape === 7) {
        const isNull = random() < 0.5;

        add(
          pick(
            isNull
              ? ["is null", "is missing", "is absent"]
              : ["is not null", "is present", "is defined"],
          ),
          isNull ? "IS_NULL" : "IS_NOT_NULL",
        );

        return { kind: "null", field: field.id, isNull };
      }

      if (shape === 6) {
        add(pick(["in", "is one of", "belongs to"]), "IN");
        const values = [v, literal(), literal()];

        for (const [i, value] of values.entries()) {
          if (i) add(",", "GLUE");
          add(JSON.stringify(value), "VALUE");
        }

        return { kind: "in", field: field.id, values };
      }

      if (shape === 8 && field.type === "number") {
        const lower = v as number;
        const upper = lower + 1 + integer(100);

        const style = pick(["between", "from", "in the range"]);

        add(style, "BETWEEN");
        add(String(lower), "VALUE");
        add(style === "between" ? "and" : "to", "GLUE");
        add(String(upper), "VALUE");

        return {
          kind: "range",
          field: field.id,
          lower: { value: lower, inclusive: true },
          upper: { value: upper, inclusive: true },
        };
      }

      const op =
        field.type === "boolean"
          ? pick(["eq", "ne"] as const)
          : pick(["eq", "ne", "lt", "le", "gt", "ge"] as const);

      const operator: Record<typeof op, readonly string[]> = {
        eq: ["equals", "is equal to", "is"],
        ne: ["does not equal", "is not equal to", "is not"],
        lt: ["under", "below", "less than"],
        le: ["at most", "no more than", "less than or equal to"],
        gt: ["over", "above", "greater than"],
        ge: ["at least", "no less than", "greater than or equal to"],
      };

      add(pick(operator[op]), op.toUpperCase() as Role);
      add(JSON.stringify(v), "VALUE");

      return { kind: "compare", field: field.id, op, value: v };
    }

    if (random() < 0.5) {
      add(
        pick([
          "show",
          "find",
          "select",
          "keep records where",
          "filter by",
          "only include records where",
          "for the query:",
          "for this set:",
          "for these entries:",
          "the query is:",
          "our filter should be:",
          "use this predicate:",
          "select the matching rows with",
          "for the current records:",
          "find the matching entries where",
          "filter all entries by",
          "select the results with",
          "show the matching records where",
          "for a query on this data:",
          "show the records meeting",
          "find entries that satisfy",
          "select all rows satisfying",
          "return only records matching",
          "filter entries that meet",
        ]),
        "O",
      );
    }

    const shape = integer(14);

    if (shape === 12) add("not", "NOT");
    if (shape === 13) add("(", "LPAREN");
    let intent = atom(shape);

    family = intent.kind === "compare" ? intent.op : intent.kind;

    if (shape === 12) {
      intent = { kind: "not", child: intent };
      family = "not";
    } else if (shape === 13) {
      add("or", "OR");
      const alternative = atom(integer(9));

      add(")", "RPAREN");
      add("and", "AND");

      intent = {
        kind: "and",
        children: [{ kind: "or", children: [intent, alternative] }, atom(integer(9))],
      };

      family = "group";
    } else if (shape === 9 || shape === 10) {
      const connective = shape === 9 ? "and" : "or";

      add(
        pick(connective === "and" ? ["and", "also", "", ","] : ["or", "alternatively"]),
        connective.toUpperCase() as Role,
      );

      const second = atom(integer(9));

      intent = { kind: connective, children: [intent, second] };
      family = connective;
    } else if (shape === 11) {
      const children = [intent];

      if (random() < 0.5) {
        add(pick(["and", "", ","]), "AND");
        children.push(atom(integer(9)));
      }

      if (random() < 0.5) add(",", "O");
      add(pick(["excluding", "except", "but not"]), "EXCLUDE");
      const second = atom(integer(9));

      intent = { kind: "and", children: [...children, { kind: "not", child: second }] };
      family = "exclusion";
    }

    const canonical = prepare(text, context);

    const fingerprint = hash(
      JSON.stringify(canonical.lexemes.map((lexeme, i) => [lexeme.text, roles[i]])),
    );

    const bucket = parseInt(fingerprint.slice(0, 8), 16) % 10;
    const split = bucket < 8 ? "train" : bucket === 8 ? "validation" : "heldout";
    let gold: FilterIntent | null = intent;

    if (random() < 0.35) {
      const mode = integer(14);
      const base = text;
      const original = roles;

      text = "";
      roles = [];

      const markers = [
        ["should we use", "UNSUPPORTED"],
        ["describes", "UNSUPPORTED"],
        ["do not apply", "UNSUPPORTED"],
        ["previously we used", "UNSUPPORTED"],
        ["maybe", "UNSUPPORTED"],
        ["explain", "UNSUPPORTED"],
        ["someone suggested", "UNSUPPORTED"],
      ] as const;

      if (mode < 7) {
        if (mode === 1) {
          add(
            pick(["the guide", "the reference", "our notes", "the documentation", "the help page"]),
            "O",
          );

          add(pick(["describes", "mentions", "discusses", "explains", "shows"]), "UNSUPPORTED");
        } else if (mode === 3) {
          add(
            pick([
              "the earlier configuration",
              "the previous settings",
              "the old report",
              "the former rule",
            ]),
            "O",
          );

          add(pick(["used", "had", "specified"]), "UNSUPPORTED");
        } else if (mode !== 0 || random() < 0.5) {
          add(markers[mode]![0], markers[mode]![1]);
        }

        text += (text ? " " : "") + base;
        roles.push(...original);
        if (mode === 0 || mode === 5) add("?", "UNSUPPORTED");
      } else if (mode < 10) {
        text = base;
        roles = [...original];

        add(
          pick(
            mode === 7
              ? ["approximately", "roughly", "around"]
              : mode === 8
                ? ["matching a pattern", "using regex"]
                : ["bytes", "seconds", "percent"],
          ),
          "UNSUPPORTED",
        );
      } else if (mode === 10) {
        add(context.fields[0]!.id, "FIELD");
        add("over", "GT");
      } else if (mode === 11) {
        add("unlisted", "FIELD");
        add("over", "GT");
        add("100", "VALUE");
      } else if (mode === 12) {
        add(context.fields[0]!.id, "FIELD");
        add("between", "BETWEEN");
        add("101", "VALUE");
        add("and", "GLUE");
        add("100", "VALUE");
      } else {
        atom(0);
        add("and", "AND");
        atom(0);
        add("or", "OR");
        atom(0);
      }

      gold = null;

      family = [
        "question",
        "description",
        "negation",
        "historical",
        "ambiguity",
        "question",
        "quotation",
        "approximate",
        "unsupported-pattern",
        "unsupported-unit",
        "incomplete",
        "unknown-field",
        "inverted",
        "mixed-grouping",
      ][mode]!;
    }

    const normalized = prepare(text, context);

    const result = (() => {
      try {
        return resolveRoles(
          normalized,
          roles.map((role) => ({ role, score: 1 })),
          context,
        ).intent;
      } catch {
        return null;
      }
    })();

    if (!isDeepStrictEqual(result, gold)) {
      throw new Error(`invalid supervision: ${text}: ${JSON.stringify(result)}`);
    }

    if (excluded.has(hash(text.trim().toLowerCase())) || seen.has(text)) continue;
    seen.add(text);
    yield { text, context, intent: gold, roles, family, split, fingerprint };
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
