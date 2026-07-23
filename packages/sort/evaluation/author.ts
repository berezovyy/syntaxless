import { createHash } from "node:crypto";
/** Evaluation-only writer. No production parser, compiler, model or training imports. */
import { readFileSync, writeFileSync } from "node:fs";

import type { Context, SortIntent, SortKey } from "../src/types.js";
import type { Row } from "../training/src/corpus.js";
import { WITNESSES } from "./fixtures.js";

export interface Acceptance extends Row {
  critical: boolean;
  compileTarget: readonly number[] | null;
}

const families = [
  "explicit-direction",
  "ordering-alias",
  "priority",
  "null-first",
  "null-last",
  "null-update",
  "caller-default",
];

const hash = (bytes: string | Uint8Array) => createHash("sha256").update(bytes).digest("hex");

function target(intent: SortIntent, context: Context): number[] {
  const order = [...WITNESSES];

  order.sort((left, right) => {
    for (const key of intent.keys) {
      const column = context.fields.find((f) => f.id === key.field)!.column as keyof typeof left;

      const a = left[column];
      const b = right[column];

      if (a === null || b === null) {
        if (a === b) continue;

        return (a === null ? 1 : -1) * (key.nulls === "first" ? -1 : 1);
      }

      // UTF-8 BINARY and Unicode scalar order agree; avoid UTF-16's surrogate order.
      const charsA = typeof a === "string" ? Array.from(a, (c) => c.codePointAt(0)!) : [Number(a)];
      const charsB = typeof b === "string" ? Array.from(b, (c) => c.codePointAt(0)!) : [Number(b)];
      let c = 0;

      for (let i = 0; i < Math.min(charsA.length, charsB.length); i++) {
        if (charsA[i] !== charsB[i]) {
          c = charsA[i]! < charsB[i]! ? -1 : 1;
          break;
        }
      }

      c ||= charsA.length < charsB.length ? -1 : charsA.length > charsB.length ? 1 : 0;
      if (c) return c * (key.direction === "asc" ? 1 : -1);
    }

    return left.id - right.id;
  });

  return order.map((r) => r.id);
}

export function author(revision = 1): Acceptance[] {
  if (![1, 2, 3].includes(revision)) throw new Error("unsupported authoring revision");
  const rows: Acceptance[] = [];

  function context(n: number): Context {
    return {
      defaultNulls: "last",
      tieBreaker: `record${n}`,
      fields: [
        {
          id: `cost${n}`,
          type: "number",
          column: "amount",
          aliases: [`prices${n}`],
          orderingAliases: [{ text: `cheapest${n}`, direction: "asc" }],
        },
        {
          id: `time${n}`,
          type: "number",
          column: "created",
          orderingAliases: [{ text: `newest${n}`, direction: "desc" }],
        },
        {
          id: `title${n}`,
          type: "string",
          column: "label",
          orderingAliases: [{ text: `alphabetical${n}`, direction: "asc" }],
        },
        { id: `active${n}`, type: "boolean", column: "flag" },
        { id: `record${n}`, type: "number", column: "id" },
      ],
    };
  }

  function key(
    field: string,
    direction: "asc" | "desc",
    nulls: "first" | "last" = "last",
    explicit = false,
  ): SortKey {
    return {
      field,
      direction,
      nulls,
      nullsSource: explicit ? "explicit" : "caller-default",
    };
  }

  function add(
    text: string,
    ctx: Context,
    keys: SortKey[] | null,
    family: string,
    critical = false,
    expected?: number[],
  ) {
    const intent = keys ? { keys, tieBreaker: ctx.tieBreaker! } : null;

    rows.push({
      text,
      context: ctx,
      intent,
      roles: [],
      family,
      critical,
      compileTarget: expected ?? (intent ? target(intent, ctx) : null),
      split: "heldout",
      fingerprint: hash(text),
      trainingEligible: false,
    });
  }

  const preambles =
    revision === 1
      ? [
          "Please arrange the result by",
          "For my list, sort by",
          "I want the records ordered by",
          "Show the entries sorted by",
          "Return the results ordered by",
          "Arrange the records by",
        ]
      : revision === 2
        ? [
            "For the output, please order by",
            "Arrange my entries by",
            "I need the list sorted by",
            "Please show results ordered by",
            "Display the records sorted by",
            "Return my records ordered by",
          ]
        : [
            "For my rows, sort by",
            "Please arrange the entries by",
            "Show my records ordered by",
            "I need my entries sorted by",
            "Return the list arranged by",
            "Display my records sorted by",
          ];

  for (const [f, family] of families.entries()) {
    for (let i = 0; i < 78; i++) {
      const n = (revision - 1) * 3000 + f * 100 + i;
      const ctx = context(n);
      const prefix = preambles[i % preambles.length]!;
      const asc = i % 2 === 0;
      const dir = asc ? "asc" : "desc";
      const word = asc ? "ascending" : "descending";

      const field = i % 3 === 0 ? `cost${n}` : i % 3 === 1 ? `title${n}` : `active${n}`;

      let phrase = `${field} ${word}`;
      let keys = [key(field, dir)];

      if (family === "ordering-alias") {
        phrase = i % 2 === 0 ? `cheapest${n} first` : `newest${n} first`;
        keys = [key(i % 2 === 0 ? `cost${n}` : `time${n}`, i % 2 === 0 ? "asc" : "desc")];
      } else if (family === "priority") {
        phrase = `newest${n} first, then cheapest${n}, then title${n} ascending`;
        keys = [key(`time${n}`, "desc"), key(`cost${n}`, "asc"), key(`title${n}`, "asc")];
      } else if (family === "null-first" || family === "null-last") {
        const nulls = family === "null-first" ? "first" : "last";

        phrase += ` with null values ${nulls}`;
        keys = [key(field, dir, nulls, true)];
      } else if (family === "null-update") {
        const nulls = asc ? "first" : "last";

        phrase = `newest${n} first, then cheapest${n}, missing prices${n} ${nulls}`;
        keys = [key(`time${n}`, "desc"), key(`cost${n}`, "asc", nulls, true)];
      } else if (family === "caller-default") {
        ctx.defaultNulls = asc ? "first" : "last";
        keys = [key(field, dir, ctx.defaultNulls)];
      }

      add(`${prefix} ${phrase}.`, ctx, keys, family);
    }
  }

  for (let i = 0; i < 630; i++) {
    const n = 1000 + (revision - 1) * 4000 + i;
    const ctx = context(n);
    const phrase = `cost${n} descending`;

    const mode = i % 7;

    const text = [
      `Can I sort by ${phrase}?`,
      revision === 1
        ? `The report said rows were sorted by ${phrase}.`
        : `The saved query specified ${phrase}.`,
      revision === 1 ? `Maybe arrange by ${phrase}.` : `Perhaps order the records by ${phrase}.`,
      `Sort by ${phrase} then absentField${n} ascending.`,
      `Order cost${n} ascending and descending.`,
      `Sort by ${phrase} using local collation.`,
      `Sort by ${phrase}, then title${n}.`,
    ][mode]!;

    add(
      text,
      ctx,
      null,
      [
        "question",
        "description",
        "uncertainty",
        "unknown-field",
        "contradiction",
        "unsupported",
        "incomplete",
      ][mode]!,
    );
  }

  const ctx: Context = {
    defaultNulls: "last",
    tieBreaker: "id",
    fields: [
      {
        id: "created",
        type: "number",
        column: "created",
        orderingAliases: [{ text: "newest", direction: "desc" }],
      },
      {
        id: "price",
        type: "number",
        column: "amount",
        aliases: ["prices"],
        orderingAliases: [{ text: "cheapest", direction: "asc" }],
      },
      { id: "id", type: "number", column: "id" },
    ],
  };

  add(
    "newest first, then cheapest, missing prices last",
    ctx,
    [key("created", "desc"), key("price", "asc", "last", true)],
    "null-update",
    true,
    [8, 1, 6, 3, 2, 4, 7, 5],
  );

  add(
    "price ascending nulls first",
    ctx,
    [key("price", "asc", "first", true)],
    "null-first",
    true,
    [3, 8, 2, 4, 5, 7, 1, 6],
  );

  add("price ascending and descending", ctx, null, "contradiction", true);

  add(
    "newest then cheapest then missing prices first then missing prices last",
    ctx,
    null,
    "contradiction",
    true,
  );

  add("newest first?", ctx, null, "question", true);
  add("newest then cheapest then nulls first", ctx, null, "ambiguous-null", true);
  add("order by distance ascending", ctx, null, "unknown-field", true);
  const missing = { ...ctx };

  delete missing.defaultNulls;
  add("please order by price descending", missing, null, "missing-policy", true);
  add("price", ctx, null, "missing-direction", true);

  if (revision === 2) {
    for (const row of rows.filter((row) => row.critical)) row.text = "For the output, " + row.text;
  }

  if (revision === 3) {
    for (const row of rows.filter((row) => row.critical)) row.text = "For my records, " + row.text;
  }

  return rows;
}

if (import.meta.main) {
  const [scopePath, out, revisionText] = process.argv.slice(2);

  if (!scopePath || !out) throw new Error("usage: author <scope-freeze.json> <output.jsonl>");

  const scopeBytes = readFileSync(scopePath);
  const scope = JSON.parse(scopeBytes.toString());
  const rows = author(Number(revisionText ?? 1));

  const body = rows.map((r) => JSON.stringify(r)).join("\n") + "\n";

  writeFileSync(out, body, { flag: "wx" });

  writeFileSync(
    out + ".meta.json",
    JSON.stringify(
      {
        sha256: hash(body),
        candidateSha256: scope.artifactFileSha256,
        scopeFreezeSha256: hash(scopeBytes),
        createdAt: new Date().toISOString(),
        trainingEligible: false,
        authorship:
          "AI-authored by the same implementation assistant " +
          "through a separate evaluation-only writer",
        independenceLimits:
          "No production/training imports. " +
          "Independent paired order targets and hand-checked critical targets. " +
          "Caller identifiers and bounded constructions vary; " +
          "no independent human author or population accuracy claim.",
        rows: rows.length,
      },
      null,
      2,
    ) + "\n",
    { flag: "wx" },
  );

  console.log({ rows: rows.length, sha256: hash(body) });
}
