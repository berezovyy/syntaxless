# @syntaxless/format

Turn English formatting instructions into typed number, currency, percentage and
date/time intents, and compile them into validated `Intl` formatters. Parsing
runs locally on a small bundled model, so no text or intents are sent to a
server.

<p align="center">
  <img alt="An editor and terminal. 'euros, two decimals, negatives in parentheses' formats a ledger as (€1,234.50), €98,000.00 and so on; changing only the locale to de-DE prints (1.234,50 €), 98.000,00 € and so on." src="https://raw.githubusercontent.com/berezovyy/syntaxless/master/docs/media/v3/films/format.webp" width="100%" />
</p>

## Install

```sh
npm install @syntaxless/format
```

The package is ESM-only and has no runtime dependencies. It needs Node.js 18 or
newer, Bun, or a modern browser with full ICU `Intl` data.

## Quick start

```ts
import { compile, explain, parse } from "@syntaxless/format";

const parsed = await parse("euros, two decimals, negatives in parentheses");
if (!parsed.ok) throw new Error(parsed.reason);
// parsed.intent → { kind: "currency", currency: "EUR", currencyDisplay: "symbol",
//                   precision: { minimum: 2, maximum: 2 }, negative: "parentheses" }

const compiled = compile(parsed.intent, { locale: "en-US" });
if (!compiled.supported) throw new Error(compiled.reason);

compiled.formatter.format(-1234.5); // { supported: true, text: "(€1,234.50)" }
compiled.formatter.formatToParts(-1234.5); // { supported: true, parts: [...] }
explain(parsed.intent); // "Format currency in EUR with symbol display. …"

const time = await parse("long date and short time, 24-hour");
if (time.ok) {
  const clock = compile(time.intent, { locale: "en-GB", timeZone: "Europe/London" });
  if (clock.supported) clock.formatter.format(Date.UTC(2024, 6, 1, 13, 5));
  // { supported: true, text: "1 July 2024 at 14:05" }
}
```

## API

### `parse(text, options?): Promise<ParseResult>`

Parses one instruction with the bundled model, or with `options.weights` (UTF-8
JSON bytes of a compatible artifact). Input must be valid Unicode of 1 to 4096
characters with at most 64 preserved lexemes.

```ts
type ParseResult =
  | { ok: true; intent: FormatIntent; confidence: number; source: "model" }
  | {
      ok: false;
      code: FailureCode | "low-confidence" | "incompatible-model" | "model-error";
      reason: string;
      source: "model";
    };
```

`parse` never throws. Failure codes:

| Code                 | Meaning                                                         |
| -------------------- | --------------------------------------------------------------- |
| `invalid-input`      | Empty, oversized or invalid Unicode input                       |
| `unsupported`        | Unsupported, uncertain, ambiguous or non-actionable instruction |
| `conflict`           | The instruction states incompatible facts                       |
| `low-confidence`     | Model confidence is below its calibrated floor                  |
| `incompatible-model` | `options.weights` could not be loaded or has the wrong version  |
| `model-error`        | Unexpected inference failure                                    |
| `invalid-context`    | A literal locale or timezone in the instruction is invalid      |

### `compile(intent, context?): Result<{ formatter: Formatter }>`

Validates an intent and a caller context (`{ locale?, timeZone? }`) and returns a
formatter or `{ supported: false, code, reason }`.

- Every formatter needs a locale. Date/time formatters also need a timezone. The
  value can come from the instruction or from `context`. Conflicting values
  return `conflict`, and a missing one returns `missing-context`.
- A timezone in context for a numeric intent returns `unsupported`.
- `formatter.format(value)` and `formatter.formatToParts(value)` return
  `{ supported: true, text }` or `{ supported: true, parts }`, or a failure.
  Numeric formatters accept finite numbers. Date/time formatters accept
  whole-millisecond timestamps. Strings and `Date` objects are not coerced.
- `formatter.kind`, `formatter.locale` and `formatter.options` expose the
  frozen `Intl` options. Later edits to the intent do not affect a compiled
  formatter.

### `explain(intent): string`

Describes every significant option, default and required caller context in
plain English. Throws on an invalid intent.

### `validate(intent): void`

Throws when an edited intent is invalid. Edit intents directly and compile them
again; don't parse an explanation.

### `preload(config?): Promise<void>` and `dispose(): void`

`preload` initializes the lazily loaded model (or `config.weights`) ahead of the
first `parse`. Concurrent calls share one initialization. `dispose` releases
cached engines. References that callers already hold remain usable.

### Types

`FormatIntent`, `NumericIntent`, `DateTimeIntent`, `Context`, `Currency`, `Style`, `FailureCode`,
`Precision`, `Formatter`, `Part`, `Result`, `ParseResult` and `ParseOptions`.

## Supported formats

**Numbers.** Number, currency and percentage formats support:

- Exact or minimum/maximum decimal places, from 0 to 20.
- Grouping on or off.
- Negatives shown with a minus sign or in parentheses. Parentheses wrap the
  positive formatted magnitude, including for negative zero.
- Currencies EUR, USD, GBP, JPY, CAD, AUD, CHF and CNY, as codes or unambiguous
  English names, displayed as symbol, code or name.
- Percent values are fractions: `0.125` formats as 12.5%.

Bare "dollars", "pounds", `$` and `¥` are ambiguous and refuse. Unspecified
precision and grouping use the `Intl` defaults for the style. Explicit native
precision refuses when combined with a stated decimal count.

**Dates and times.** Supported:

- `short`, `medium`, `long` and `full` date and time styles, either shared or
  set separately. The default is `medium`.
- 12-hour or 24-hour time.
- The Gregorian calendar and Latin digits.

**Locale and timezone.** An instruction can name a literal locale (`en-US`) or
IANA timezone (`America/New_York`). A locale never implies a timezone. Locale
extensions that request other calendars or numbering systems are unsupported.

**Not supported.** Rounding modes and increments, significant digits,
scientific or compact notation, custom patterns, relative dates and date
parsing. These refuse rather than being silently dropped.

## How it works

1. `parse` normalizes the text into lexemes, and a 32-wide int6 tagger labels
   each lexeme with one of 25 instruction roles.
2. A deterministic resolver turns the roles into a typed intent. It keeps
   numbers, currencies, styles, locales and timezones exactly as written, and
   refuses on conflicts, unsupported words or low confidence.
3. `compile` maps the intent to `Intl.NumberFormat` or `Intl.DateTimeFormat`
   options and wraps the result in a validated formatter.

The model is loaded on first use. The runtime is about 10 KB and the model about
23 KB (Brotli). The model abstains below a confidence floor calibrated on
validation data.

## Limitations

- English instructions only, within the families listed above.
- Output punctuation, spacing and names come from the runtime's locale data and
  can differ between engines and ICU versions. Accounting sign display does not
  guarantee parentheses in every locale, so the formatter adds them explicitly.
- Acceptance sets are AI-authored. Results are regression evidence, not an
  independent accuracy estimate. See [evaluation](evaluation/README.md).

## Development

From `packages/format`:

| Script                  | Purpose                                                    |
| ----------------------- | ---------------------------------------------------------- |
| `bun run typecheck`     | Type-check sources, tests and scripts                      |
| `bun run test`          | Runtime unit tests                                         |
| `bun run build`         | Code-split bundle, public declarations and size gate       |
| `bun run size:gate`     | Brotli size budget of `dist/` (20 KB runtime, 64 KB model) |
| `bun run check:package` | Pack the tarball and run it in Bun and Node consumers      |
| `bun run test:browser`  | Offline lazy-loading check in Chromium (needs Playwright)  |

## Training and evaluation

- [Training](training/README.md): corpus generation, model training and the
  release gate.
- [Evaluation](evaluation/README.md): reserved acceptance protocol and results.

## License

[MIT](LICENSE)
