# @syntaxless/sort

Turn natural-language ordering requests into typed sort keys, a JavaScript
comparator and SQLite `ORDER BY` fragments. Parsing runs a small bundled model
locally. No user text leaves the device, and there are no runtime dependencies.

## Install

```sh
npm install @syntaxless/sort
```

The package is ESM-only and runs in Node 18+, Bun and modern browsers.

## Quick start

```ts
import { compile, explain, parse, sortRecords } from "@syntaxless/sort";
import type { Context } from "@syntaxless/sort";

const context: Context = {
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
      column: "price",
      aliases: ["prices"],
      orderingAliases: [{ text: "cheapest", direction: "asc" }],
    },
    { id: "id", type: "number", column: "id" },
  ],
};
const records = [
  { id: 1, created: 2, price: 30 },
  { id: 2, created: 2, price: 10 },
  { id: 3, created: null, price: 5 },
];

const result = await parse("newest first, then cheapest, missing prices last", context);
if (result.ok) {
  sortRecords(records, result.intent, context).map((r) => r.id); // [2, 1, 3]
  compile(result.intent, context).supported; // true
  explain(result.intent);
  // 1. "created" descending, null values last (caller default). 2. "price" ascending, ...
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
  orderingAliases?: readonly { text: string; direction: "asc" | "desc" }[]; // up to 16, e.g. "newest"
  column?: string | readonly string[]; // trusted SQL identifier segments (1–3)
}
interface Context {
  fields: readonly Field[]; // 1–64 fields
  defaultNulls?: "first" | "last"; // used when a request does not place nulls
  tieBreaker?: string; // unique, non-null field for reproducible ties
}
```

### `parse(text, options): Promise<ParseResult>`

`options` is a `Context` plus optional `weights: Uint8Array` (a self-hosted
UTF-8 JSON artifact). The result is one of:

- `{ ok: true, intent: SortIntent, confidence: number, source: "model" }`
- `{ ok: false, code, reason: string, source: "model" }`

Failure codes:

| Code                     | Meaning                                                                |
| ------------------------ | ---------------------------------------------------------------------- |
| `invalid-input`          | Empty or over-long text (4096 chars, 128 lexemes), unsupported wording |
| `ambiguous`              | Several fields in one clause, or a null clause without a clear key     |
| `conflict`               | Contradictory directions or null placements, or a repeated field       |
| `missing-direction`      | A key has no direction and no ordering alias                           |
| `missing-null-placement` | No explicit placement and no `defaultNulls`                            |
| `invalid-catalog`        | Malformed context                                                      |
| `invalid-intent`         | The decoded intent fails validation                                    |
| `unknown-field`          | A field mention or tie breaker is not in the catalog                   |
| `low-confidence`         | Model confidence is below its calibrated floor                         |
| `incompatible-model`     | Custom weights cannot be decoded or have the wrong format version      |
| `model-error`            | Unexpected internal failure                                            |

### `SortIntent`

```ts
interface SortKey {
  field: string;
  direction: "asc" | "desc";
  nulls: "first" | "last";
  nullsSource: "explicit" | "caller-default";
}
interface SortIntent {
  keys: readonly SortKey[]; // 1–8 keys in priority order
  tieBreaker?: string; // copied from the context
}
```

### `compile(intent, context & { dialect?: "sqlite" }): Result<{ sql, dialect }>`

Returns an `ORDER BY` fragment (without the keywords) or
`{ supported: false, code, reason }`. SQL compilation requires a tie breaker
(`missing-tie-breaker`); it is appended ascending unless already a key. Each key
is type-guarded, so wrongly stored values sort as nulls, and strings use
`COLLATE BINARY`.

### `sortRecords(records, intent, context): T[]` and `comparator(intent, context)`

`sortRecords` returns a stably sorted copy. With a tie breaker, it first checks
that every record has a unique, non-null tie-breaker value. `comparator` returns
the same ordering function without that check. Both throw `InputError` for an
invalid intent or context.

### `validate(intent, context): Map<string, Field>`

Checks an edited intent against the catalog. Throws `InputError` with a `code`.

### `explain(intent): string`

Describes each key, its null placement and whether it came from the caller
default, plus the tie policy.

### `preload(config?)` and `dispose()`

`preload` initializes the same engine `parse` uses. `dispose` drops cached
engines. Custom weight buffers are copied before initialization and cached per
buffer.

## How it works

`parse` replaces caller field names and ordering aliases with typed markers,
runs a 32-unit token tagger that predicts direction, null-placement and
key-separator roles, and resolves them deterministically into keys. Directions
and null placement are never guessed from field names. There is no rules
fallback: low confidence and unknown constructions refuse.

The bundled build is 8,416 B of runtime and 23,276 B of lazily loaded model
(Brotli, all JavaScript chunks).

## Semantics and limitations

- Types: string, finite safe-integer-range number, boolean (booleans bind as
  `0`/`1` in SQL). Timestamps are numbers in the caller's units.
- Missing, null and wrongly typed values follow the key's null placement,
  regardless of direction. No value coercion occurs.
- Strings use Unicode scalar order, matching SQLite `BINARY` on valid UTF-8,
  without locale collation or case folding.
- Supported language: explicit ascending/descending, caller ordering aliases,
  keys separated by "then" or commas, and first/last null placement. A later
  null clause must name an earlier key unless there is only one.
- Out of scope: questions, random order, custom expressions, locale collation
  and implicit null policies. The package does not execute SQL or verify a
  database uniqueness constraint.

## Development

```sh
bun run --cwd packages/sort typecheck
bun run --cwd packages/sort test
bun run --cwd packages/sort build          # also runs the size gate
bun run --cwd packages/sort check:package  # packed Bun and Node consumers
bun run --cwd packages/sort test:browser   # needs Playwright Chromium
```

## Training and evaluation

- [Training](training/README.md): corpus generation, model training and promotion.
- [Evaluation](evaluation/README.md): held-out acceptance sets and gate policy.

## License

MIT
