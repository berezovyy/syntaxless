# @syntaxless/validation

Natural-language field constraints to a typed intent AST and JSON Schema.

`parse(text)` returns a `FieldIntent`; `compile(intent)` returns JSON Schema or an
explicit unsupported result. `schema(text)` is parse then compile. The package has
no runtime dependencies and does not send user text off-device.

Default `parse()` uses the bundled on-device model. Use
`parse(text, { source: "rules" })` for the explicit strict grammar. There is no
silent model-to-rules fallback. Models predict semantic roles; TypeScript
resolves exact numbers, checks constraints, and compiles the resulting intent.

```ts
import { compile, parse, schema } from "@syntaxless/validation";

const parsed = await parse("required email with max 100 characters");
// { ok: true, intent: { type: "email", required: true, maxLength: 100 }, source: "model" }

const compiled = compile({ type: "email", required: true, maxLength: 100 });
// { supported: true, schema: { type: "string", format: "email", minLength: 1, maxLength: 100 } }

const result = await schema("required email with max 100 characters");
// { supported: true, schema, intent }

const rejected = await schema("every weekday at 9");
// { supported: false, intent: null, code: "unrecognized" }
```

v0 recognizes presence (`required` / `optional`) plus a field type and an optional
comparison (`max 100 characters`, `at least 18`). It does not guess a missing
presence, invent nested objects, or compile `nullable`.

## Field and object semantics

`compile(intent)` describes an existing field value. A required string-like field
also rejects the empty string using a default `minLength: 1`; optional strings
permit empty values unless an explicit length says otherwise. Required strings
with `maxLength: 0` are conflicting constraints. Required numeric and boolean
fields have no standalone presence keyword.

Use `compileObject(fields)` to enforce missing properties:

```ts
import { compileObject, explain } from "@syntaxless/validation";

const object = compileObject({
  email: { type: "email", required: true, maxLength: 100 },
  enabled: { type: "boolean", required: false },
});
// { supported: true, schema: {
//   type: "object",
//   properties: {
//     email: { type: "string", format: "email", minLength: 1, maxLength: 100 },
//     enabled: { type: "boolean" },
//   },
//   required: ["email"]
// } }

explain({ type: "number", required: false, minimum: { value: 10, inclusive: false } });
// "Optional number field; greater than 10."
```

The helper assembles caller-provided names; parsing nested object descriptions
is outside the scope. Extra object properties remain allowed. Presence belongs
to the enclosing object's `required` array, as described in the
[JSON Schema object reference](https://json-schema.org/understanding-json-schema/reference/object).

Targets follow JSON Schema 2020-12. Email and UUID emit their standard format
names; URL emits `uri` and represents an absolute URI without imposing an HTTP
scheme. `format` is an annotation by default in that dialect. Enable assertions
in the consuming validator when formats must reject values; this package does
not execute a schema validator. See the
[JSON Schema string reference](https://json-schema.org/understanding-json-schema/reference/type).

Numeric bounds preserve inclusive/exclusive meaning, signed decimal literals,
and supported word numbers from zero to one hundred. Integer compilation rounds
bounds to the equivalent integer interval and refuses an empty interval.
Nonfinite and unsafe numeric values refuse. Character lengths must be
nonnegative safe integers. Unknown intent properties refuse during editing.

## Models and current evidence

`schema(text, options)` forwards the same options as `parse`. Models load lazily
from the package and work offline. A caller can provide UTF-8 JSON artifact bytes
through `{ source: "model", weights }`; treat the buffer as immutable after first
use. Each buffer owns isolated tensors and calibration. `dispose()` clears the
cache; engines already held by callers remain usable.

The bundled format-v2 model trained on 25,000 rows passed numerical parity,
synthetic family gates, packed offline consumers, and Chromium checks for lazy
loading, concurrent initialization, and offline parsing. Fresh AI-authored
acceptance exactly matched 673/702 positive intents (95.87%), with no wrong
accepted meanings and no false accepts among 709 negatives. All 11 fresh critical
controls and 34 previously consumed critical regression cases passed. The model
and validation-only confidence calibration were promoted together.

Runtime code is 11,190 B and the model is 56,889 B Brotli, counting all JavaScript
chunks. Questions, quotations, approximate bounds, unsupported units/patterns,
and conflicting constraints are included in refusal evaluation. Every future
candidate requires newly reserved acceptance wording.
AI-authored acceptance is not independent human evidence, and these finite
observations do not establish population accuracy or absence of errors.
