# @syntaxless/retry

Natural-language retry policies to typed intents and deterministic, bounded delay
schedules. Parsing runs locally with a small bundled model. Nothing leaves the
device. The package never executes requests, classifies errors, sleeps or handles
cancellation.

<p align="center">
  <img alt="An editor and terminal. parse('retry three times, double the delay, cap at one minute') prints missing-initial-delay; after adding { defaults: { initialDelayMs: 250 } } it prints delays of 250, 500 and 1000 ms as bars." src="https://raw.githubusercontent.com/berezovyy/syntaxless/master/docs/media/v3/films/retry.webp" width="100%" />
</p>

## Install

```sh
npm install @syntaxless/retry
```

The package is ESM-only and runs in Node.js 18+, Bun and modern browsers.

## Quick start

```ts
import { compile, explain, parse } from "@syntaxless/retry";

const result = await parse("retry three times, double the delay, cap at one minute", {
  defaults: { initialDelayMs: 250 },
});
if (result.ok) {
  console.log(explain(result.intent));
  // "3 retries after the initial attempt. Exponential delays, starting at 250 ms (caller default). ..."
  console.log(compile(result.intent));
  // { supported: true, delaysMs: [250, 500, 1000] }
}
```

Without the caller default, `parse` returns `{ ok: false, code: "missing-initial-delay" }`.
The package never invents an initial delay or a strategy.

## API

### `parse(text, options?): Promise<ParseResult>`

- `options.defaults?: Partial<RetryPolicy>` supplies caller defaults for fields the
  text omits. The intent lists defaulted fields in `defaults`.
- `options.weights?: Uint8Array` takes custom UTF-8 JSON model bytes. The bytes are
  copied, and each buffer gets its own engine.
- Returns `{ ok: true, intent, confidence, source: "model" }` or
  `{ ok: false, code, reason, source: "model" }`. Refusal codes are the
  `FailureCode` values plus `ambiguous`, `low-confidence`, `incompatible-model` and
  `model-error`. `parse` never throws.

### `compile(intent, options?): Result<{ delaysMs: readonly number[] }>`

- `options.seed?: number` is an unsigned 32-bit seed. It is required when jitter is
  enabled.
- Returns `{ supported: true, delaysMs }` or `{ supported: false, code, reason }`,
  for example `missing-seed` or `overflow`. No partial schedule is returned.

### `validate(intent): void`

Throws an error with a `code` (`invalid-intent`, `missing-multiplier`, `conflict`)
for an invalid or edited intent. `compile` and `explain` validate too.

### `explain(intent): string`

Returns a plain-English description that labels caller defaults.

### `preload(config?): Promise<void>` and `dispose(): void`

`preload` initializes the same engine `parse` uses. It is optional, and it accepts
`{ weights }`. `dispose` clears the cached engines.

### Types

`RetryIntent`, `RetryPolicy`, `Jitter`, `PolicyField`, `Context`, `FailureCode`,
`Result`, `ParseOptions`, `ParseResult` and `CompileOptions`.

```ts
interface RetryPolicy {
  retries: number; // retries after the initial attempt, 0..1000
  strategy: "constant" | "exponential";
  initialDelayMs: number;
  multiplier?: number; // required for exponential, 1..1e6
  maxDelayMs?: number;
  jitter?:
    | { kind: "none" }
    | { kind: "full" }
    | { kind: "equal" }
    | { kind: "proportional"; fraction: number };
}
```

## How it works

1. A tokenizer and a 17-role sequence model label the request: counts, delays,
   caps, factors, units, strategy and jitter markers.
2. TypeScript resolves exact values. It handles word numbers up to one hundred,
   "once" and "twice", decimal durations in milliseconds through days, and
   "double" and "triple". Then it validates the intent.
3. Requests below the model's calibrated confidence floor refuse
   (`low-confidence`). There is no rules fallback.
4. `compile` computes the schedule with exact decimal fractions:
   - **Retry count.** Retries count attempts after the first, so three retries
     give three delays and four attempts in total.
   - **Exponential delays.** The delay at index `i` is `initialDelayMs × multiplier^i`,
     rounded up to whole milliseconds.
   - **Cap.** The cap applies from the first retry, even when it is below the
     initial delay. Uncapped values above the safe integer range refuse.
   - **Jitter.**
     - Full jitter samples `[0, base]`.
     - Equal jitter samples `[ceil(base/2), base]`.
     - Proportional jitter samples `[ceil(base·(1−f)), floor(base·(1+f))]`. Its
       upper bound is capped again.
     - "No jitter" produces `{ kind: "none" }`. It overrides a jitter default and
       needs no seed.
   - **Sampling.** Samples use the LCG `state = (1664525·state + 1013904223) mod 2^32`.
     Two outputs are combined as `(first << 21) + (second >> 11)`. The generator is
     reproducible, not secure.

The bundled model has hidden width 32 and was trained on 8,000 synthetic examples.
It compresses to 23,309 Brotli bytes. The runtime chunks add 9,101 Brotli bytes
(`bun run size:gate`). The model loads lazily on first use.

## Limitations

- English only. Compound durations ("1 minute 30 seconds"), calendar periods,
  fractional milliseconds and total-attempt wording ("three attempts including
  the first") refuse.
- Exponential backoff requires an explicit multiplier, "double" or "triple", or a
  caller default.
- Descriptions of past configurations are not always recognized. For example,
  "we previously retried three times with constant delay of one second" is
  accepted as a request. Review parsed intents before you act on them.
- Accuracy figures come from AI-authored evaluation sets. They are not independent
  human evidence. See [evaluation](evaluation/README.md).

## Development

```sh
bun run typecheck       # TypeScript
bun run test            # unit and bundled-model tests
bun run build           # dist/ with declarations
bun run size:gate       # Brotli budgets for runtime and model
bun run check:package   # packed tarball consumers in Bun and Node
bun run test:browser    # Chromium lazy, concurrent and offline checks
```

## Training and evaluation

- [Training](training/README.md) covers how the bundled model is produced.
- [Evaluation](evaluation/README.md) covers the reserved acceptance protocol and its
  datasets. The bundled model matches every positive and refuses every negative in
  all seven consumed acceptance sets: 4541 positives and 4457 negatives.

## License

MIT
