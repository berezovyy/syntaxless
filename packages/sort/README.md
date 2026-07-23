# @syntaxless/sort

Convert English ordering requests to typed sort keys, a JavaScript comparator,
and a SQLite ORDER BY fragment. The parser uses a local model with no runtime
network requests. Callers provide field types, aliases, null defaults and SQL
identifier mappings.

Semantic ordering aliases have caller-defined directions. For example, a caller
can map "newest" to a timestamp field descending and "cheapest" to a price field
ascending. The parser never infers either meaning from a field name. Directions
must be stated or supplied by an ordering alias. Null placement must be explicit
or supplied as `defaultNulls`; explanations label caller defaults.

## Usage

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
      aliases: ["prices"],
      column: "price",
      orderingAliases: [{ text: "cheapest", direction: "asc" }],
    },
    { id: "id", type: "number", column: "id" },
  ],
};
const result = await parse("newest first, then cheapest, missing prices last", context);
if (result.ok) {
  console.log(explain(result.intent));
  const ordered = sortRecords(records, result.intent, context);
  const sql = compile(result.intent, context);
}
```

## Comparison and ties

The initial types are string, finite safe number and boolean. Numeric timestamps
use the caller's existing units. Missing properties, null and wrongly typed values
use the key's null placement, independent of direction. No value coercion occurs.
Strings use Unicode scalar order with SQLite BINARY, without locale or case folding.
JavaScript strings must contain valid Unicode without NUL; SQLite text must have
valid UTF-8. Boolean SQL values use integer 0/1. Compare JavaScript records with
values after SQLite column affinity; use ANY columns in STRICT tables to preserve
storage types.

Without a tie breaker, `sortRecords` preserves JavaScript input order on equal
keys. Reproducible SQL requires a caller-provided unique, non-null tie-breaker
field. The compiler appends it ascending unless already included in the keys.
`sortRecords` checks uniqueness and non-null values; a standalone comparator
assumes the caller has checked the tie-breaker contract. The package doesn't
execute SQL or verify a database uniqueness constraint.

SQL mappings are trusted identifier segments, never expressions. A string maps
one identifier; an array maps qualified segments. The compiler quotes each segment
and uses type guards and explicit null ordering. Unsupported dialects, missing
mappings, unknown fields, duplicate keys and conflicting directions refuse.
Edited intents are checked against the same caller context.

## Supported language

The first scope includes explicit ascending/descending direction, caller ordering
aliases, ordered keys separated by "then" or commas, and first/last null placement.
A later null clause must name an earlier key unless only one key exists. The
initial maximum is eight keys. Questions, descriptions, uncertain requests,
unknown fields, random ordering, custom expressions, locale collation and implicit
null policies are unsupported. No rules fallback supplies a model refusal.

`preload` initializes the same engine as parsing; `dispose` releases cached engines.
Custom `weights` bytes own an isolated engine and are copied before asynchronous
initialization. The bundled model passed the complete gate and is the default parser.

## Current evidence

The bundled hidden-32 model uses 8,000 generated rows. Synthetic heldout matches
512/512 positives exactly, with no wrong accepted meanings or false accepts among
295 negatives. Fresh AI-authored acceptance matches 548/548 positives, with no wrong
accepted meanings or false accepts among 637 negatives. Every declared family
and all nine critical controls pass. Exported-wire parity, actual paired SQLite
order, packed offline consumers and Chromium checks pass.

Runtime code is 8,421 B and the model is 23,288 B Brotli, counting every JavaScript
chunk. Acceptance was authored by the same implementation assistant through a
separate writer; bounded generated variations do not establish independent human
evidence, population accuracy or absence of errors. Future candidates need fresh
reserved wording. See [training](training/README.md) and
[evaluation](evaluation/README.md) for provenance and gates.
