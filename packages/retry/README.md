# @syntaxless/retry

Convert English retry requests to typed policies and deterministic delay schedules.
The local model identifies policy roles. Exact values, units, arithmetic and
schedule generation run in TypeScript. The package never executes requests,
classifies errors, sleeps or handles cancellation.

Retries mean attempts after the initial attempt. A request for three retries
produces three delays and permits four total attempts. The initial delay and
strategy must be stated or provided by caller defaults. An explicit multiplier,
"double" or "triple" supplies exponential strategy; no initial delay is invented.
Explanations label caller defaults. All edited policies use the same validator.

## Usage

```ts
import { compile, explain, parse } from "@syntaxless/retry";

const result = await parse("retry three times, double the delay, cap at one minute", {
  defaults: { initialDelayMs: 250 },
});
if (result.ok) {
  console.log(explain(result.intent));
  const schedule = compile(result.intent);
  // schedule.delaysMs is [250, 500, 1000] when supported.
}
```

Without the caller's initial delay, the reference returns `missing-initial-delay`.
The parser returns a valid intent independently of compilation; an uncapped
schedule may later return `overflow`. There is no rules fallback.

## Arithmetic contract

Counts range from zero to 1000 retries. Delays are nonnegative safe whole
milliseconds. Ordinary decimal durations resolve exactly through milliseconds,
seconds, minutes, hours and days, including common abbreviations. Word numbers
from zero to one hundred, "once" and "twice" are supported. A duration has one
amount and one unit; compound durations, calendar periods and fractional
milliseconds are unsupported. Multipliers are finite numbers from one to one
million; literal values must survive the public number representation exactly.

Constant strategy repeats the initial delay. Exponential delay at index `i`,
starting at zero, is the initial delay multiplied by the factor to power `i`.
The compiler uses exact decimal fractions, rounds each result up to whole
milliseconds, and applies the maximum delay. Intermediate delays are not rounded.
A cap below the initial delay is allowed and applies from the first retry.
Uncapped values above the safe integer range refuse; no partial schedule escapes.

## Jitter contract

Jitter is optional. Explicit "no jitter" or "without jitter" produces
`{ kind: "none" }` and overrides caller jitter defaults without requiring a seed.
Full jitter samples between zero and the capped base delay.
Equal jitter samples between half the base rounded up and the base. Proportional
jitter with fraction `f` samples between `ceil(base * (1-f))` and
`floor(base * (1+f))`. The final upper bound is capped again, so jitter never
exceeds a stated maximum. Fractions range from zero to one; unknown jitter kinds
and unspecified jitter refuse. Proportional bounds above the safe range refuse.

`compile(intent, { seed })` requires an unsigned 32-bit caller seed when jitter
is enabled. Explicit `none` does not require a seed. Seeded pseudorandom integer samples reproduce the same schedule;
the generator is not for security. It uses the LCG recurrence
`state = (1664525 * state + 1013904223) mod 2^32` and combines consecutive outputs
as `(first << 21) + (second >> 11)` to sample integer bounds through exact arithmetic.

`preload` initializes the same model as parsing. Custom UTF-8 JSON `weights` bytes
own isolated engines and calibration; initialization copies caller bytes.
`dispose` clears caches. No user text or intent leaves the device.

## Verification

The promoted default model uses 8,000 training examples and hidden width 32.
It passed synthetic heldout checks with 488/488 positive matches and no false
accepts among 266 negatives. Fresh reserved acceptance matched 680/680 positive
intents across nine families, with no false accepts among 637 negatives.
All prior and fresh critical controls passed. The independently calculated
schedule targets, 32-sequence exported-wire parity, actual packed consumer and
real Chromium lazy/concurrent/offline checks passed.

The aggregate runtime is 9,063 Brotli bytes and the model is 23,318 Brotli bytes.
The complete report is
`training/retry/runs/descriptions32-2026-10-11T21-31-19-147Z/gate-report.json`.
Reserved examples were AI-authored by the same implementation assistant through
a separate evaluation writer. Bounded constructions and numeric variations are
not independent human evidence or a population accuracy estimate.
