# @syntaxless/filter

Turn natural-language filters into typed predicate intents, evaluate them in
JavaScript, and compile them to parameterized SQLite `WHERE` fragments. Parsing
runs a small bundled model locally. No user text leaves the device, and there
are no runtime dependencies.

<p align="center">
  <img alt="An editor and terminal. parse('active customers spending over 100, excluding trials') compiles to SQLite parameters ["active", 100, 1] and matches three customers; changed to 'spending between 50 and 200', it prints [50, 200] and four customers." src="https://raw.githubusercontent.com/berezovyy/syntaxless/master/docs/media/v3/films/filter.webp" width="100%" />
</p>

## Install

```sh
npm install @syntaxless/filter
```

The package is ESM-only and runs in Node 18+, Bun and modern browsers.

## Quick start

```ts
import { compile, evaluate, explain, parse } from "@syntaxless/filter";
import type { Context } from "@syntaxless/filter";

const context: Context = {
  fields: [
    {
      id: "status",
      type: "string",
      column: "status",
      terms: [{ text: "active customers", value: "active" }],
    },
    { id: "spending", type: "number", column: "spending" },
    {
      id: "trial",
      type: "boolean",
      column: "trial",
      terms: [{ text: "trials", value: true }],
    },
  ],
};

const result = await parse("active customers spending over 100, excluding trials", context);
if (result.ok) {
  const where = compile(result.intent, context);
  // where.sql uses ?1, ?2, ?3; where.parameters is ["active", 100, 1]
  evaluate(result.intent, { status: "active", spending: 101, trial: false }, context); // true
  explain(result.intent);
  // ("status" equals "active" AND "spending" is greater than 100 AND NOT ("trial" equals true))
} else {
  console.log(result.code, result.reason);
}
```

## API

### Context

```ts
interface Field {
  id: string; // stable identifier, 1–128 chars
  type: "string" | "number" | "boolean";
  aliases?: readonly string[]; // up to 16 names the user may say
  terms?: readonly { text: string; value: Scalar }[]; // up to 32 shorthand equalities
  operators?: readonly Operator[]; // restrict allowed operators
  column?: string | readonly string[]; // trusted SQL identifier segments (1–3)
}
interface Context {
  fields: readonly Field[]; // 1–64 fields
}
```

Names and terms match case-insensitively on word boundaries. A phrase may not
mean two different things.

### `parse(text, options): Promise<ParseResult>`

`options` is a `Context` plus optional `weights: Uint8Array` (a self-hosted
UTF-8 JSON artifact). The result is one of:

- `{ ok: true, intent: FilterIntent, confidence: number, source: "model" }`
- `{ ok: false, code, reason: string, source: "model" }`

Failure codes:

| Code                   | Meaning                                                                 |
| ---------------------- | ----------------------------------------------------------------------- |
| `invalid-input`        | Empty or over-long text, bad quoting, unsupported or incomplete wording |
| `ambiguous`            | Mixed AND/OR without parentheses                                        |
| `invalid-catalog`      | Malformed context                                                       |
| `invalid-intent`       | The decoded predicate is invalid (for example an empty range)           |
| `unknown-field`        | A field mention is not in the catalog                                   |
| `unsupported-operator` | The operator is not allowed for the field                               |
| `low-confidence`       | Model confidence is below its calibrated floor                          |
| `incompatible-model`   | Custom weights cannot be decoded or have the wrong format version       |
| `model-error`          | Unexpected internal failure                                             |

### `FilterIntent`

```ts
type FilterIntent =
  | {
      kind: "compare";
      field: string;
      op: "eq" | "ne" | "lt" | "le" | "gt" | "ge";
      value: Scalar;
    }
  | { kind: "in"; field: string; values: readonly Scalar[] }
  | { kind: "range"; field: string; lower: Bound; upper: Bound }
  | { kind: "null"; field: string; isNull: boolean }
  | { kind: "and" | "or"; children: readonly FilterIntent[] }
  | { kind: "not"; child: FilterIntent };
```

### `compile(intent, context & { dialect?: "sqlite" }): Result<SqlPredicate>`

Returns `{ supported: true, dialect: "sqlite", sql, parameters }` or
`{ supported: false, code, reason }` (`missing-sql-mapping`,
`unsupported-dialect` or a validation code). Values always bind as numbered
parameters; booleans bind as `0`/`1`. Each predicate carries a type guard, so
wrongly stored values yield `NULL` instead of being coerced.

### `evaluate(intent, record, context): boolean` and `evaluateTruth(...): boolean | null`

Three-valued SQL logic: missing, `null` or wrongly typed values make ordinary
predicates unknown (`null`); `NOT` preserves unknown; `false` dominates `AND`
and `true` dominates `OR`. `evaluate` returns `true` only for a true result.
Both throw `InputError` for an invalid intent or context.

### `validate(intent, context): Map<string, Field>`

Checks an edited intent against the catalog (at most 64 nodes and 8 levels,
typed values, allowed operators, non-empty ranges, membership of 1–64 values).
Throws `InputError` with a `code`.

### `explain(intent): string`

A readable description that names every predicate.

### `preload(config?)` and `dispose()`

`preload` initializes the same engine `parse` uses. `dispose` drops cached
engines. Custom weight buffers are copied and cached per buffer; treat them as
immutable after first use.

## How it works

`parse` replaces caller field names, terms and literals with typed markers,
runs a 32-unit token tagger that predicts operator and connective roles, and
resolves those roles with a deterministic grammar. The result is validated
against the catalog. There is no rules fallback: low confidence and unknown
constructions refuse. `compile` and `evaluate` are deterministic TypeScript.

The bundled build is 9,197 B of runtime and 23,525 B of lazily loaded model
(Brotli, all JavaScript chunks).

## Semantics and limitations

- Types: string, finite safe-integer-range number, boolean. Boolean ordering and
  ranges are refused.
- Strings are case-sensitive and use Unicode scalar order, matching SQLite
  `COLLATE BINARY` on valid UTF-8. Strings must be valid Unicode without NUL.
- String literals in requests use JSON double quotes.
- `between`, `from … to …` and `in the range` are inclusive; edited intents may
  use exclusive bounds.
- Compare JavaScript records with values as stored after SQLite column affinity;
  use `ANY` columns in `STRICT` tables to preserve types.
- Out of scope: relative dates, joins, aggregations, arbitrary expressions,
  fuzzy or locale-aware matching. The package does not execute SQL.

## Development

```sh
bun run --cwd packages/filter typecheck
bun run --cwd packages/filter test
bun run --cwd packages/filter build          # also runs the size gate
bun run --cwd packages/filter check:package  # packed Bun and Node consumers
bun run --cwd packages/filter test:browser   # needs Playwright Chromium
```

## Training and evaluation

- [Training](training/README.md): corpus generation, model training and promotion.
- [Evaluation](evaluation/README.md): held-out acceptance sets and gate policy.

## License

MIT
