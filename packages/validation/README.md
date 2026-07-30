# @syntaxless/validation

Natural-language field constraints to a typed intent and JSON Schema, on device.

```text
"required email with max 100 characters"
  → { type: "email", required: true, maxLength: 100 }
  → { type: "string", format: "email", minLength: 1, maxLength: 100 }
```

The package has no runtime dependencies and never sends text off the device.

## Install

```sh
npm install @syntaxless/validation
```

ESM only. Requires Node.js 18 or later, Bun, or a modern browser.

## Quick start

```ts
import { compile, parse, schema } from "@syntaxless/validation";

const parsed = await parse("required email with max 100 characters");
// { ok: true, intent: { type: "email", required: true, maxLength: 100 },
//   confidence: 0.999…, diagnostics: [], source: "model" }

const compiled = compile({ type: "email", required: true, maxLength: 100 });
// { supported: true, schema: { type: "string", format: "email", minLength: 1, maxLength: 100 } }

const result = await schema("required email with max 100 characters");
// { supported: true, schema: { … }, intent: { … } }

const rejected = await schema("every weekday at 9");
// { supported: false, reason: "unsupported or non-actionable input", code: "unrecognized", intent: null }
```

## API

### `parse(text, options?): Promise<ParseResult>`

Parses one field description into a `FieldIntent`.

| Option    | Type                 | Default   | Meaning                                                     |
| --------- | -------------------- | --------- | ----------------------------------------------------------- |
| `source`  | `"model" \| "rules"` | `"model"` | Bundled model, or the strict rule grammar                   |
| `weights` | `Uint8Array`         | bundled   | UTF-8 JSON model artifact to use instead of the bundled one |

```ts
type ParseResult =
  | {
      ok: true;
      intent: FieldIntent;
      confidence: number;
      diagnostics: Diagnostic[];
      source: "model" | "rules";
    }
  | {
      ok: false;
      code: ParseFailureCode;
      reason: string;
      confidence: number;
      diagnostics: Diagnostic[];
      source: "model" | "rules";
    };
```

Failure codes: `empty`, `unrecognized`, `missing-type`, `missing-presence`,
`unrecognized-slot`, `conflicting-bounds`, `conflicting-presence`,
`inverted-range`, `low-confidence`, `model-error`, `incompatible-model`.
`parse` does not throw for bad input or a bad artifact; it returns a failure.
A successful model parse with a margin below the clean band carries a
`low-margin` suggestion in `diagnostics`.

There is no fallback between sources. A model refusal is final unless the
caller explicitly asks for `{ source: "rules" }`.

### `compile(intent): CompileResult`

Synchronously compiles one `FieldIntent` to a JSON Schema 2020-12 fragment, or
returns `{ supported: false, code, reason }` with code `invalid-type`,
`invalid-length`, `invalid-bound`, `conflicting-bounds`, `wrong-slot`, or
`nullable`.

### `compileObject(fields)`

Assembles named fields into an object schema and returns
`{ supported: true, schema: ObjectSchema }` or a failure that names the
offending `field`. Presence goes into the object's `required` array; extra
properties stay allowed.

```ts
import { compileObject } from "@syntaxless/validation";

compileObject({
  email: { type: "email", required: true, maxLength: 100 },
  enabled: { type: "boolean", required: false },
});
// { supported: true, schema: {
//     type: "object",
//     properties: {
//       email: { type: "string", format: "email", minLength: 1, maxLength: 100 },
//       enabled: { type: "boolean" } },
//     required: ["email"] } }
```

### `schema(text, options?): Promise<SchemaResult>`

`parse` followed by `compile`. On a compile refusal the parsed intent is
returned with the failure; on a parse refusal `intent` is `null`.

### `explain(intent): string`

Describes an intent in English, for example
`explain({ type: "number", required: false, minimum: { value: 10, inclusive: false } })`
returns `"Optional number field; greater than 10."`. Throws `TypeError` for an
invalid intent.

### `validateField(intent): Failure | null`

Checks an intent without compiling it. Returns `null` when valid, otherwise
`{ supported: false, code, reason }`.

### `preload(config?)`, `dispose()`, `getEngine(config?)`

`preload` loads the model ahead of the first `parse`; `config` is `{ weights? }`.
Each distinct `weights`
buffer gets its own engine; treat a buffer as immutable after first use.
`dispose` clears the engine cache; engines already held stay usable.
`getEngine` returns the low-level inference engine.

### Types

`FieldIntent` is `{ type, required, nullable?, minLength?, maxLength?, minimum?, maximum? }`
where `type` is `string`, `email`, `url`, `uuid`, `number`, `integer`, or
`boolean`, and `minimum`/`maximum` are `{ value: number; inclusive: boolean }`.

## How it works

Text is normalized and tokenized. The bundled model (a small int6 token
tagger) predicts a role for every token: field type, presence, comparison
operator, number, unit, or unsupported. TypeScript decodes those roles into an
intent, reads the exact numbers from the source text, validates the
constraints, and compiles deterministically. The model never produces numbers
or schema text. Low-confidence predictions and sentences containing
unsupported material are refused.

The rule parser (`{ source: "rules" }`) is a strict grammar over the same
vocabulary.

The model loads lazily from a separate chunk and works offline. Measured with
Brotli (quality 11): runtime code 11,201 B, model 56,889 B.

## Semantics

- A sentence must state presence (`required`/`optional`); it is never guessed.
- A required string-like field compiles to `minLength: 1`, so the empty string
  is rejected. Optional strings permit it. `required` with `maxLength: 0` is a
  conflict.
- Email and UUID emit their standard `format`; URL emits `uri` (absolute URI,
  any scheme). In JSON Schema 2020-12, `format` is an annotation unless the
  consuming validator enables format assertion.
- Numeric bounds keep inclusive/exclusive meaning, signed decimals, and word
  numbers from zero to one hundred. Integer bounds are rounded to the
  equivalent integer interval; an empty interval is refused.
- Non-finite or unsafe numbers are refused. Lengths must be nonnegative safe
  integers.
- Each field accepts at most one length comparison or one value comparison.

## Limitations

- One field per sentence; nested objects, patterns, enums, and units other
  than characters are refused.
- `nullable` is parsed but not compiled.
- English only.
- On the AI-authored reserved set used for promotion, the bundled model
  matched 673 of 702 positive intents exactly, accepted no wrong meanings,
  and refused all 709 negatives. That set is not independent human evidence
  or an estimate of accuracy on real-world wording. See the
  [evaluation protocol](evaluation/README.md).

## Development

| Script                  | Purpose                                             |
| ----------------------- | --------------------------------------------------- |
| `bun run typecheck`     | TypeScript check of sources, tests, and scripts     |
| `bun run test`          | Unit and regression tests                           |
| `bun run build`         | Code-split ESM bundle, declarations, and size gate  |
| `bun run size:gate`     | Brotli size budgets (runtime 20 KiB, model 64 KiB)  |
| `bun run check:package` | Build, pack, and run Bun and Node consumers offline |
| `bun run test:browser`  | Build and run the Chromium check (Playwright)       |

## Training and evaluation

- [Training](training/README.md): corpus generation, training, and promotion.
- [Evaluation](evaluation/README.md): reserved acceptance protocol and results.

## License

[MIT](LICENSE)
