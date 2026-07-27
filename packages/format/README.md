# Format

Format converts English instructions to serializable number, currency, percentage
and date/time intents, with a local model and deterministic Intl compilation.
The default parser uses the promoted model, without a rules fallback. Inference
sends no text or intents to a server.

## Public API

```ts
import { compile, explain, parse } from "@syntaxless/format";

const parsed = await parse("euros, two decimals, negatives in parentheses");
if (!parsed.ok) throw new Error(parsed.reason);

const compiled = compile(parsed.intent, { locale: "en-US" });
if (!compiled.supported) throw new Error(compiled.reason);

compiled.formatter.format(-1234.5); // { supported: true, text: "(€1,234.50)" }
compiled.formatter.formatToParts(-1234.5);
explain(parsed.intent);
```

`parse` returns a typed intent or a failure with a reason. `compile` validates
the intent and caller context, then returns a formatter or a structured failure.
`explain` describes every significant fact, missing context and default, and
`validate` rejects invalid edited intents. Edit the intent directly and compile
it again instead of parsing an explanation. A compiled formatter snapshots
options, so later edits cannot change its behavior.

`preload()` initializes the lazy model, and `dispose()` releases it.
`parse(text, { weights })` accepts an isolated compatible artifact. Parser input
is bounded to 4096 characters and 64 preserved lexemes. Numeric formatters accept
only finite numbers, and date/time formatters accept valid whole-millisecond
timestamps. Strings and Date objects are not coerced.

## Number formats

Number formats support exact decimal places or minimum/maximum decimal places,
grouping, and negative minus or parentheses presentation. Fraction digit counts
range from zero to twenty for portability. Currency formats support explicit
EUR, USD, GBP, JPY, CAD, AUD, CHF and CNY codes or unambiguous English names.
Bare dollars, pounds, dollar signs and yen signs are ambiguous and refuse.
Currency display supports symbol, code or name. Percentage values are fractions,
so a value of 0.125 represents 12.5 percent.

The compiler uses standard Intl notation and rounding. Unspecified precision
and grouping use Intl defaults for the selected style, and the default negative
presentation is a minus sign. Explicit native precision refuses when combined
with a stated decimal count. Parentheses wrap the positive formatted magnitude
for every negative value, including negative zero. Custom rounding modes,
rounding increments, significant digits, scientific/compact notation and custom
patterns are unsupported, rather than silently discarded.

## Date and time formats

Date/time formats support short, medium, long and full date/time styles, and
12-hour or 24-hour time. A shared style can apply to both date and time, while
separate styles can be stated explicitly. The default style is medium.
Custom component patterns, relative dates and date parsing are unsupported.
The formatter accepts an exact timestamp in milliseconds, not a date string.
The compiler uses the Gregorian calendar and Latin digits.

Every compiler requires an explicit supported locale. Every date/time compiler
also requires an explicit supported timezone, including date-only output.
An instruction may contain a literal locale such as en-US or a timezone such
as America/New_York. Caller context supplies any missing value, and conflicting
instruction/context values refuse. Locale does not imply timezone. Locale
extensions requesting other calendars or numbering systems are unsupported.

## Implementation and verification

The compiler returns supported Intl options and a small validated formatter
wrapper. Invalid edits, nonfinite numeric values, invalid timestamps and missing
context return specific failures. Explanations identify every option and default.
The wrapper exposes formatted parts for inspecting signs and locale separators.
Runtime locale data can change punctuation, names or spaces, so portability
checks compare semantic parts and options alongside pinned examples.

The active model uses 8000 rows and hidden width 32, with literal substitutions
and paired negatives grouped before splitting. The complete promotion gate passed
550/550 synthetic positives and 767/767 fresh acceptance positives. It observed
zero wrong accepted meanings and refused all 642 fresh negatives. The same
implementation assistant authored acceptance, so the result is not independent
human evidence or a population accuracy estimate. See [training](training/README.md)
and [acceptance evidence](evaluation/README.md) for commands, hashes and limits.

The supported Intl behavior follows the
[ECMA-402 specification](https://tc39.es/ecma402/2025/). Currency placement and
group separators depend on locale data, and accounting signs don't guarantee
parentheses in every locale. Explicit parentheses therefore use a wrapper.
