# @syntaxless/cron

Turn English schedule text into a typed intent and a 5-field cron expression,
entirely on-device.

`parse(text)` returns a `ScheduleIntent`. `compile(intent)` returns a cron
expression, or an explicit unsupported result when cron can't represent the
intent. The package has no runtime dependencies.

## Install

```sh
npm install @syntaxless/cron
```

The package is ESM-only and targets ES2022 runtimes (Node.js 18+, Bun and
modern browsers).

## Quick start

```ts
import { compile, cron, explain, parse } from "@syntaxless/cron";

await cron("every monday at 5pm");
// { supported: true, expression: "0 17 * * 1",
//   intent: { frequency: { unit: "week", interval: 1 }, weekdays: ["monday"],
//             time: { hour: 17, minute: 0 } } }

const parsed = await parse("every 15 minutes");
// { ok: true, intent: { frequency: { unit: "minute", interval: 15 } },
//   confidence: 0.999…, diagnostics: [], source: "model" }

compile({ frequency: { unit: "minute", interval: 7 } });
// { supported: false, code: "non-unit-interval",
//   reason: "cron minute steps would change the interval at hour boundaries" }

await cron("remind me tomorrow at 3");
// { supported: false, code: "unsafe-context", intent: null,
//   reason: "the text does not request a recurring schedule" }

explain({
  frequency: { unit: "week", interval: 1 },
  weekdays: ["monday"],
  time: { hour: 17, minute: 0 },
});
// "The recurrence interval is 1 week. The local time is 17:00. … The cron expression is 0 17 * * 1."
```

## API

### Parsing

| Export      | Signature                                                              |
| ----------- | ---------------------------------------------------------------------- |
| `parse`     | `(text: string, options?: ModelParserOptions) => Promise<ParseResult>` |
| `cron`      | `(text: string) => Promise<CronResult>`                                |
| `preload`   | `(config?: EngineConfig) => Promise<void>`                             |
| `getEngine` | `(config?: EngineConfig) => Promise<InferenceEngine>`                  |
| `dispose`   | `() => void`                                                           |

`ModelParserOptions`:

- `source`: `"model"` (default) uses the bundled model. `"rules"` uses a small
  deterministic grammar that fails explicitly on anything outside it.
- `weights`: self-hosted artifact bytes (UTF-8 JSON) instead of the bundled model.
- `energyThreshold`: overrides the artifact's calibrated out-of-distribution threshold.

`ParseResult` is `{ ok: true, intent, confidence, diagnostics, source }` or
`{ ok: false, reason, code, confidence, diagnostics, source }`. `reason` is a
readable sentence; `code` is one of `empty`, `unrecognized`,
`unrecognized-time`, `unrecognized-weekday`, `unrecognized-interval`,
`conflicting-selectors`, `low-confidence`, `unsafe-context` or `model-error`.
`unsafe-context` covers text that names a schedule without requesting one:
questions, quotations, descriptions, cancellations, incomplete requests and
one-off events. Diagnostics carry the machine-readable detail.

`cron(text)` parses and compiles. It returns `{ supported: true, expression,
intent }` or `{ supported: false, reason, code, intent, clarification? }`.
Parse failures have `intent: null`; compile failures keep the recognized intent.

`preload` initializes the engine that `parse` uses. Repeated calls with the same
weight buffer share initialization, so treat that buffer as immutable.
`dispose` releases cached engines; an engine a caller already holds stays usable.

### Intents

| Export                 | Signature                                                   |
| ---------------------- | ----------------------------------------------------------- |
| `validate`             | `(intent: unknown) => asserts intent is ScheduleIntent`     |
| `compile`              | `(intent: ScheduleIntent) => CompileResult`                 |
| `compileWeekdays`      | `(days: Weekday[]) => string`                               |
| `WEEKDAY_NUMBER`       | `Readonly<Record<Weekday, number>>` (Sunday is 0)           |
| `explain`              | `(intent: ScheduleIntent) => string`                        |
| `clarify`              | `(intent: ScheduleIntent) => ScheduleClarification \| null` |
| `resolveClarification` | `(intent, answer: ClarificationAnswer) => ScheduleIntent`   |
| `IntentError`          | `TypeError` subclass with a typed `code`                    |

A `ScheduleIntent` has a `frequency` (`unit`: minute, hour, day, week or month;
positive safe-integer `interval`), plus optional `time`, `weekdays`,
`dayOfMonth`, `ordinalWeekday`, `day` (last day or last business day), `bounds`
(inclusive `start`, exclusive `end`, ISO dates from 0001 to 9999) and
`exceptions`.

`validate` throws `IntentError` for malformed facts: unknown fields, invalid
dates, conflicting selectors or incomplete clocks. `compile` runs the same
checks and then returns `{ supported: true, expression }` or `{ supported:
false, reason, code }`. Unsupported codes are `non-unit-interval`,
`ordinal-weekday`, `day-spec`, `bounds`, `exceptions` and `missing-anchor`.

Cron fields are local and zero-based. Minute intervals must divide 60 and hour
intervals must divide 24, because `*/7` would leave a 4-minute gap at each hour
boundary. A complete cycle compiles to `0`: 60 minutes is `0 * * * *` and
24 hours is `0 0 * * *`. Calendar schedules default to 00:00, and hourly
schedules default to minute 0. Cron can't encode bounds, exceptions or ordinal
weekday rules. The scheduler supplies the timezone and daylight-saving behavior.

`explain` describes every fact, the defaults and the cron support without
loading the model. `clarify` returns a typed question when a weekly or monthly
intent only lacks the day that cron needs, and `null` otherwise:

```ts
import { clarify, compile, resolveClarification } from "@syntaxless/cron";

const intent = {
  frequency: { unit: "week", interval: 1 },
  time: { hour: 9, minute: 25 },
} as const;
const question = clarify(intent); // { field: "weekdays", allowed: [...], ... }
const resolved = resolveClarification(intent, { weekdays: ["tuesday", "thursday"] });
compile(resolved); // { supported: true, expression: "25 9 * * 2,4" }
```

`resolveClarification` accepts only the requested field, validates it and
returns a new intent. It never picks a day on the caller's behalf.

### Recurrence previews

```ts
import { recurrences } from "@syntaxless/cron";

recurrences(
  { frequency: { unit: "month", interval: 1 }, dayOfMonth: 31 },
  {
    timezone: "UTC",
    reference: "2028-01-31T12:00:00Z",
    count: 2,
    policies: { monthEnd: "constrain" },
  },
);
// ok: true, occurrences at 2028-02-29T00:00Z and 2028-03-31T00:00Z
```

`recurrences(intent, options)` returns a bounded preview. It never starts timers
or runs tasks. Options:

- `timezone` and `reference` are required. `reference` is an exclusive start:
  epoch milliseconds, or an ISO instant with an explicit offset.
- `count` or `before` is required. `before` is an exclusive horizon.
- `anchor` sets the interval phase and defaults to `reference`.
- `policies` sets the DST and month-end behavior described below.
- `calendar` is required for IANA timezones.
- `maxSteps` limits candidate work (default 10000, maximum 100000).

Policies:

| Policy     | Default   | Other values                 | Applies to               |
| ---------- | --------- | ---------------------------- | ------------------------ |
| `gap`      | `skip`    | `earlier`, `later`, `reject` | Nonexistent local times  |
| `fold`     | `earlier` | `later`, `both`, `reject`    | Repeated local times     |
| `monthEnd` | `skip`    | `constrain`, `reject`        | Days such as February 31 |

The result is either `{ ok: true, occurrences, stop, examinedSteps, policies }`
or `{ ok: false, code, reason }`. Each occurrence has its instant, its actual
local fields and the requested local fields. A preview returns at most 1000
occurrences. The failure codes `work-limit` and `output-limit` report an
exhausted budget instead of returning partial results. Malformed intents throw
`IntentError`.

Units follow local calendar fields rather than fixed durations, and weeks begin
on Monday. The default calendar supports UTC and fixed offsets such as
`+05:45`. For IANA timezones, pass an adapter from the `@syntaxless/cron/temporal`
entry:

```ts
import { Temporal } from "@js-temporal/polyfill";
import { createTemporalCalendar } from "@syntaxless/cron/temporal";

const calendar = createTemporalCalendar(Temporal); // native Temporal also works
```

The adapter doesn't import a polyfill or change globals, and the host supplies
the timezone database. A custom adapter must return valid Gregorian fields,
offsets below 24 hours, and ordered earlier/later resolutions.

## How it works

1. `normalize` canonicalizes punctuation, meridiem forms and dotted clocks.
2. A small model tags every token. It is a 64-wide int6-quantized
   convolutional and bidirectional-recurrent tagger. It predicts 77 schedule
   roles, span boundaries, actionable spans, a schedule kind and a speech act.
3. A deterministic decoder builds the intent from the role labels and closed
   lexicon tables. The speech-act head decides whether the text is a request at
   all. Calibrated energy and Mahalanobis thresholds reject inputs unlike the
   training data, and a confidence floor of 0.5 rejects uncertain parses.
4. `validate` and `compile` are deterministic TypeScript and never call the model.

The compiler entry (`dist/index.js`) is about 5.6 KB with Brotli. The weights
live in a separate, lazily imported `dist/weights.gen.js`, which is about 100 KB
minified and 58 KB with Brotli. Nothing loads until the first `parse`,
`preload` or `getEngine` call.

**Privacy.** Inference runs on-device. User text, tokens, intents and confidence
never leave the process, and there is no hosted fallback. Self-hosters can pass
their own weights through `parse`'s `weights` option or `preload`/`getEngine`.

**Weights format.** Each artifact declares a format version. An unsupported
version fails with `unsupported-weights-version` before any decoding. Artifacts
in the earlier 17-slot feature layout without character n-grams still load.

## Limitations

- The bundled model reads a bare hour in `every weekday at 9` as 21:00 instead
  of 09:00. Explicit `9am` and `9pm` work. `bun run smoke` keeps this case as a
  failing regression check until a retrained model fixes it.
- English only, with no locale-specific date formats.
- `last_business_day` means the last Monday-to-Friday of the month. Holidays
  aren't excluded.
- The rules parser refuses one-off wording by keyword (`today`, `tomorrow`,
  `tonight`, `next friday`, `in 10 minutes`, …). It recognizes only a small grammar.
- The evaluation corpora are written by the implementation assistant, not by
  independent people. Scores measure regression, not population accuracy. See
  [evaluation](evaluation/README.md).

## Development

| Script                  | Purpose                                                                        |
| ----------------------- | ------------------------------------------------------------------------------ |
| `bun run typecheck`     | Type-check sources, tests and scripts                                          |
| `bun run test`          | Unit and numerical-parity tests                                                |
| `bun run build`         | Build `dist/` and enforce the Brotli size gate                                 |
| `bun run size:gate`     | Re-check the size budget of an existing build                                  |
| `bun run check:package` | Pack the tarball, verify its file list, run a consumer                         |
| `bun run test:browser`  | Browser check for lazy loading and request privacy (needs Playwright Chromium) |
| `bun run smoke`         | Known-regression check for the bare-hour case above                            |

## Training and evaluation

- [Training](training/README.md): corpus preparation, model training and export.
- [Evaluation](evaluation/README.md): held-out evaluation corpora and acceptance policy.

## License

[MIT](LICENSE)
