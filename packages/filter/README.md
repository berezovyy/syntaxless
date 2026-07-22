# @syntaxless/filter

Typed predicate intents, JavaScript evaluation, and parameterized SQLite
predicate fragments. `parse()` uses a bundled local model; exact comparisons
and compilation run in TypeScript. There are no runtime dependencies and no
user text leaves the device.

Callers provide stable field identifiers, types, semantic aliases, allowed
operators, shorthand values, and trusted SQL identifier mappings. No field name
or value is guessed. SQL is a fragment plus bound parameters; this package does
not execute database queries.

The initial types are string, finite safe number, and boolean. Intents express
equality/inequality, ordered comparisons, membership, inclusive/exclusive ranges,
null checks, AND, OR, and NOT. Boolean ordering refuses. Relative dates, joins,
aggregations, arbitrary expressions and fuzzy search are outside this scope.

Missing properties and null produce unknown on ordinary predicates. Null checks
test both missing and null. NOT preserves unknown; false dominates AND, true
dominates OR. `evaluateTruth` exposes true/false/null; `evaluate` matches only
true. This follows [SQLite expression semantics](https://www.sqlite.org/lang_expr.html).
No JavaScript value coercion occurs. SQLite type guards reject wrongly stored
types; booleans use integer 0/1. Compare JavaScript records to the values actually
stored after column affinity, using ANY in STRICT tables when preserving types.

Strings are case-sensitive, use Unicode scalar ordering with SQLite BINARY
collation, and must contain valid Unicode without NUL. SQLite text must have valid
UTF-8. Explicit COLLATE BINARY overrides a column's default collation. No locale
or case-folding is inferred.

SQL identifier strings are quoted as single identifiers. Pass an array of
segments for a qualified name. Raw SQL expressions are never accepted as mappings.
Values always bind as numbered parameters. The compiler validates edited intents
and bounds expression depth, node count, and membership size.

## Usage

```ts
import { compile, evaluate, explain, parse } from "@syntaxless/filter";

const context = {
  fields: [
    {
      id: "status",
      type: "string" as const,
      column: "status",
      terms: [{ text: "active customers", value: "active" }],
    },
    { id: "spending", type: "number" as const, column: "spending" },
    {
      id: "trial",
      type: "boolean" as const,
      column: "trial",
      terms: [{ text: "trials", value: true }],
    },
  ],
};
const result = await parse("active customers spending over 100, excluding trials", context);
if (result.ok) {
  const sql = compile(result.intent, context);
  // sql.parameters: ["active", 100, 1]; identifiers come from context.
  evaluate(result.intent, { status: "active", spending: 101, trial: false }, context); // true
  explain(result.intent); // includes every comparison and exclusion
}
```

Strings use JSON double quotes. Names match caller aliases without case
differences; literal string values preserve case and punctuation. Caller-defined
terms explicitly expand to equality facts, including boolean membership phrases.
Missing field types, unknown fields and unsupported operators refuse. Adjoining
predicates mean AND. Mixed AND/OR requires explicit parentheses; `excluding`
adds an AND-NOT predicate. `between`, `from ... to ...` and `in the range` are
inclusive; edited range intents can specify exclusive bounds. No locale-based
string ordering is inferred.

`preload()` initializes lazily. `{...context, weights}` accepts a caller-owned
UTF-8 artifact buffer; treat it as immutable after first use. `dispose()` releases
caches. There is no rules fallback. Low confidence refuses an intent; ambiguous
grouping asks for parentheses. Edited intents validate before compilation.

## Evidence

The format-v2 hidden-32 model trained on 8,000 rows passes synthetic family
gates, runtime/PyTorch parity, packed offline consumers and Chromium checks.
Fresh AI-authored acceptance matched 842/842 positive intents, accepted none
of 705 negatives, and passed all seven critical controls. Prior consumed critical
regressions also pass. Each positive has independently authored three-valued
truth targets over eight witness rows, checked against actual SQLite execution
and the JavaScript evaluator. All 14 declared semantic families have at least
60 fresh positive cases.

Runtime is 9,189 B and model is 23,525 B Brotli, counting every JavaScript chunk.
The acceptance author is the implementation assistant using a separate writer;
these finite, generated variations are not independent human evidence or a
population accuracy estimate. Consumed sets remain evaluation-only; future
candidates need new reserved wording.
