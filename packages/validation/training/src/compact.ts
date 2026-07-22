/** Bounded, seeded corpus. Paired statements and hard negatives share a split. */
import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import { writeJsonl } from "../../../shared/training/jsonl.js";
import { reservedHashes } from "../../../shared/training/reserved.js";
import { compile } from "../../src/compiler/index.js";
import { TYPE_LABELS } from "../../src/model/labels.js";
import { normalize } from "../../src/parser/normalize.js";
import type { FieldIntent, FieldType } from "../../src/types/intent.js";
import { structuralPattern } from "./fingerprint.js";
import type { CorpusRecord, Split } from "./generate.js";
import { renderField } from "./renderer.js";
import type { RenderedExample, RenderOrder, SpokenComparison } from "./renderer.js";
import { createRng } from "./rng.js";
import { Sentence } from "./sentence.js";

const TYPES: readonly FieldType[] = [
  "string",
  "email",
  "url",
  "uuid",
  "number",
  "integer",
  "boolean",
];

const ORDERS: readonly RenderOrder[] = [
  "presence-type-glue-comparison",
  "type-presence-comparison",
  "comparison-presence-type",
  "presence-type-comparison",
  "type-comparison-presence",
  "type-copula-presence-comparison",
];

const PREFIXES = [
  "",
  "please use",
  "make this field",
  "the field must be",
  "set the field to",
  "for this field use",
  "validate as",
  "a",
  "the",
  "use",
  "please validate as",
  "the input must be",
  "for this value use",
  "for this input",
  "validate this field as",
  "this field must be",
  "the value should be",
  "set this entry to",
  "the parameter must be",
  "our field needs",
  "for this value:",
  "for the input:",
  "for this field: please use",
  "for this form: for this field:",
  "validation rule:",
  "the requirement is:",
];

const SUFFIXES = [
  "",
  "",
  "",
  "for the input",
  "for this value",
  "in this form",
  "for this entry",
  "for this field",
];

function sha(text: string): string {
  return createHash("sha256").update(text).digest("hex");
}

function prepend(example: RenderedExample, prefix: string): RenderedExample {
  if (!prefix) return example;
  const shift = prefix.length + 1;

  return {
    ...example,
    text: `${prefix} ${example.text}`,
    spans: [
      { start: 0, end: prefix.length, label: "O", clauseStart: false },
      ...example.spans.map((span) => ({
        ...span,
        start: span.start + shift,
        end: span.end + shift,
      })),
    ],
  };
}

function decorate(example: RenderedExample, prefix: string, suffix: string): RenderedExample {
  const result = prepend(example, prefix);

  if (!suffix) return result;

  return {
    ...result,
    text: `${result.text} ${suffix}`,
    spans: [
      ...result.spans,
      {
        start: result.text.length + 1,
        end: result.text.length + 1 + suffix.length,
        label: "O",
        clauseStart: false,
      },
    ],
  };
}

function* sampleCompact(
  count = 8_000,
  seed = 20261011,
  excluded: ReadonlySet<string> = new Set(),
): Generator<CorpusRecord> {
  if (!Number.isInteger(count) || count < 1 || count > 100_000) {
    throw new RangeError("corpus count must be in [1, 100000]");
  }

  const rng = createRng(seed);
  let emitted = 0;
  const meanings = new Map<string, string>();
  const negativeTarget = Math.round(count * 0.35);
  let negatives = 0;

  for (let attempt = 0; emitted < count; attempt++) {
    if (attempt > count * 100) {
      throw new Error("cannot produce enough unique rows within the bounded search");
    }

    const type = rng.pick(TYPES);
    const intent: FieldIntent = { type, required: rng.bool() };
    const shape = type === "boolean" ? 0 : rng.int(0, 7);
    const numeric = type === "number" || type === "integer";

    // Short word numbers need actual supervised examples; uniform wide ranges
    // otherwise leave words such as "eleven" almost absent from the corpus.
    const min = numeric
      ? (rng.bool() ? rng.int(-20, 30) : rng.int(-1000, 1000)) +
        (rng.bool(0.2) ? rng.pick([0.25, 0.5, 0.75]) : 0)
      : rng.bool(0.1)
        ? 0
        : rng.bool()
          ? rng.int(1, 30)
          : rng.int(1, 2000);

    const max = min + (rng.bool() ? rng.int(1, 30) : rng.int(1, 2000));

    if (numeric) {
      if ([1, 3, 5, 6, 7].includes(shape)) {
        intent.minimum = { value: min, inclusive: shape !== 3 && rng.bool() };
      }

      if ([2, 4, 5, 6, 7].includes(shape)) {
        intent.maximum = {
          value: shape === 6 ? min : max,
          inclusive: shape !== 4 && rng.bool(),
        };
      }

      if (shape === 6) {
        intent.minimum!.inclusive = true;
        intent.maximum!.inclusive = true;
      }
    } else {
      if ([1, 3, 5, 6, 7].includes(shape)) intent.minLength = min;
      if ([2, 4, 5, 6, 7].includes(shape)) intent.maxLength = shape === 6 ? min : max;
    }

    if (shape === 7 && numeric) {
      intent.minimum!.inclusive = true;
      intent.maximum!.inclusive = true;
    }

    if (!compile(intent).supported) continue;

    const spoken: SpokenComparison | undefined =
      shape === 7
        ? { between: { min, max }, unit: !numeric }
        : !numeric && (shape === 3 || shape === 4)
          ? shape === 3
            ? { min: { value: min - 1, inclusive: false }, unit: true }
            : { max: { value: max + 1, inclusive: false }, unit: true }
          : undefined;

    let example = renderField(intent, rng, {
      order: rng.pick(ORDERS),
      numberStyle: rng.bool() ? "words" : "digits",
      ...(spoken ? { spoken } : {}),
    });

    let splitExample = example;

    let negative =
      negatives < negativeTarget && (emitted >= count - negativeTarget || rng.bool(0.35));

    let gold: FieldIntent | null = intent;
    let family = `${type}/${shape}`;
    let compileFailure: "nullable" | undefined;

    if (negative) {
      const mode = rng.int(0, 14);

      family = [
        "question",
        "description",
        "negation",
        "historical",
        "question",
        "quotation",
        "ambiguity",
        "conflict",
        "inverted",
        "incomplete",
        "unsupported-unit",
        "nullable",
        "nested",
        "unsupported-pattern",
        "approximate",
      ][mode]!;

      gold = null;

      if (mode < 7) {
        // Keep non-actionable wording paired with the statement's semantic frame.
        if (mode === 1) {
          // A report marker makes the statement non-actionable. Preserve its
          // facts rather than requiring the model to erase every field token.
          const marker = new Sentence();

          marker.add(
            rng.pick([
              "the documentation",
              "the reference guide",
              "our examples",
              "the specification",
              "the help text",
              "the notes",
            ]),
            "O",
          );

          marker.add(rng.pick(["mentions", "describes", "discusses", "shows"]), "UNSUPPORTED");
          example = prepend(splitExample, marker.text);
          example.spans = [...marker.spans, ...example.spans.slice(1)];
        } else {
          const marker = new Sentence();

          const cues = [
            ["should this field be", "would you suggest", "can we use"],
            [],
            ["do not apply", "do not use", "avoid", "never configure"],
            ["it used to be", "previously this was", "our old configuration used"],
            ["what does", "explain", "how is"],
            ['"', "the note reads", "they wrote"],
            ["perhaps", "maybe", "still undecided about"],
          ];

          marker.add(rng.pick(cues[mode]!), "UNSUPPORTED");
          example = prepend(splitExample, marker.text);
          example.spans = [...marker.spans, ...example.spans.slice(1)];

          if (mode === 0 || mode === 4) {
            example = decorate(example, "", "?");
            example.spans.at(-1)!.label = "UNSUPPORTED";
          }

          if (mode === 5) {
            example = decorate(example, "", '"');
            example.spans.at(-1)!.label = "UNSUPPORTED";
          }
        }
      } else if (mode === 8) {
        const invalid: FieldIntent = numeric
          ? {
              ...intent,
              minimum: { value: max, inclusive: true },
              maximum: { value: min, inclusive: true },
            }
          : { ...intent, minLength: max, maxLength: min };

        example = renderField(invalid, rng, {
          order: rng.pick(ORDERS),
          spoken: {
            min: { value: max, inclusive: true },
            max: { value: min, inclusive: true },
            unit: !numeric,
          },
        });

        splitExample = example;
      } else if (mode === 11) {
        // Parsing preserves this known fact; compilation explicitly refuses nullable.
        gold = { type, required: intent.required, nullable: true };
        example = renderField(gold, rng);
        splitExample = example;
        compileFailure = "nullable";
        family = `${type}/nullable`;
        negative = false;
      } else {
        const sentence = new Sentence();

        sentence.add(
          intent.required ? "required" : "optional",
          intent.required ? "REQUIRED" : "OPTIONAL",
        );

        if (mode === 7) {
          sentence.add(
            intent.required ? "optional" : "required",
            intent.required ? "OPTIONAL" : "REQUIRED",
          );
        }

        if (mode === 12) {
          sentence.add("object", "UNSUPPORTED");
          sentence.add("containing", "GLUE");
        }

        sentence.add(type, TYPE_LABELS[type]);

        if (mode === 9 || mode === 10) {
          sentence.add(mode === 9 ? "at least" : "at most", mode === 9 ? "GTE" : "LTE");

          if (mode === 10) {
            sentence.add(String(max), "NUMBER");
            sentence.add(rng.pick(["words", "bytes", "seconds", "lines"]), "UNSUPPORTED");
          }
        }

        if (mode === 14) {
          sentence.add(
            rng.pick(["around", "about", "roughly", "approximately", "~"]),
            "UNSUPPORTED",
          );

          sentence.add(String(min), "NUMBER");
          if (!numeric) sentence.add("characters", "UNIT_CHARACTER");
        }

        if (mode === 13) {
          sentence.add(
            rng.pick(["matching", "matches", "pattern", "regex", "one of"]),
            "UNSUPPORTED",
          );

          sentence.add(`value-${max}`, "O");
        }

        example = {
          text: sentence.text,
          spans: sentence.spans,
          style: "invalid",
          template: family,
        };

        splitExample = example;
      }
    }

    if (gold !== null || (negative && family !== "question" && rng.bool(0.2))) {
      example = decorate(example, rng.pick(PREFIXES), rng.pick(SUFFIXES));
    }

    const pattern = structuralPattern(splitExample.text, splitExample.spans).filter(
      (part) => part !== "O",
    );

    const group = sha(JSON.stringify(pattern));
    const bucket = Number.parseInt(group.slice(0, 8), 16) % 10;
    const split: Split = bucket < 8 ? "train" : bucket === 8 ? "validation" : "heldout";
    const text = normalize(example.text);

    if (excluded.has(sha(text.trim().toLowerCase()))) continue;
    const meaning = JSON.stringify(gold);
    const seen = meanings.get(text);

    if (seen !== undefined) {
      if (seen !== meaning) throw new Error(`conflicting supervision: ${text}`);
      continue;
    }

    meanings.set(text, meaning);
    if (negative) negatives++;

    yield {
      id: `compact-${emitted}`,
      split,
      seed,
      family: negative ? "negative" : "constraint",
      template: family,
      fingerprint: group,
      text,
      intent: gold,
      spans: example.spans,
      ...(compileFailure ? { compileFailure } : {}),
    };

    emitted++;
  }
}

export function generateCompact(count = 8_000, seed = 20261011): CorpusRecord[] {
  return [...sampleCompact(count, seed)];
}

export function writeCompact(out: string, count = 8_000, seed = 20261011): void {
  const excluded = reservedHashes(
    join(import.meta.dir, "../../evaluation/reserved"),
    join(import.meta.dir, "../../evaluation/consumed"),
  );

  const bySplit: Record<string, { positive: number; negative: number }> = {};
  const groups = new Map<string, string>();

  function* records(): Generator<CorpusRecord> {
    for (const row of sampleCompact(count, seed, excluded)) {
      const owner = groups.get(row.fingerprint);

      if (owner !== undefined && owner !== row.split) throw Error("structural split leak");
      groups.set(row.fingerprint, row.split);
      const counts = (bySplit[row.split] ??= { positive: 0, negative: 0 });

      counts[row.intent === null ? "negative" : "positive"]++;
      yield row;
    }
  }

  const written = writeJsonl(out, records());

  const manifest = {
    seed,
    rows: written.rows,
    bySplit,
    sha256: written.sha256,
    generatorSha256: sha(
      [
        import.meta.path,
        join(import.meta.dir, "renderer.ts"),
        join(import.meta.dir, "sentence.ts"),
        join(import.meta.dir, "../../src/parser/lexicon.ts"),
      ]
        .map((p) => readFileSync(p, "utf8"))
        .join("\n"),
    ),
    memory:
      "O(rows) deduplication/groups and bounded 64 KiB serialized output; at most 100000 rows",
    acceptanceUsed: false,
    excludedReservedTexts: excluded.size,
  };

  writeFileSync(`${out}.manifest.json`, JSON.stringify(manifest, null, 2) + "\n", {
    flag: "wx",
  });

  console.log(manifest);
}

if (import.meta.main) {
  writeCompact(process.argv[2] ?? "packages/validation/training/data/synth/compact.jsonl");
}
