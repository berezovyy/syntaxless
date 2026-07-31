# @syntaxless/retention

Natural-language backup retention policies to typed intents and deterministic
keep/remove previews. Parsing runs locally with a small bundled model. No backup
metadata leaves the device. The package only recommends; it never deletes files
or cloud objects.

## Install

```sh
npm install @syntaxless/retention
```

The package is ESM-only and runs in Node.js 18+, Bun and modern browsers.

## Quick start

```ts
import { compile, explain, parse } from "@syntaxless/retention";

const parsed = await parse("keep daily backups for a week and monthly backups for a year");
if (parsed.ok) {
  console.log(explain(parsed.intent));
  const preview = compile(parsed.intent, {
    timeZone: "UTC",
    referenceInstant: Date.parse("2024-03-15T12:00:00Z"),
    backups: [
      { id: "current", timestamp: Date.parse("2024-03-15T00:00:00Z") },
      { id: "old", timestamp: Date.parse("2022-01-01T00:00:00Z") },
    ],
  });
  // { supported: true, keep: [{ id: "current", tierIndexes: [0, 1] }], remove: ["old"] }
}
```

## API

### `parse(text, options?): Promise<ParseResult>`

- `options.weights?: Uint8Array` takes custom UTF-8 JSON model bytes. The bytes are
  copied, and each buffer gets its own engine.
- Parsing needs no calendar context or backup metadata.
- Returns `{ ok: true, intent, confidence, source: "model" }` or
  `{ ok: false, code, reason, source: "model" }`. Refusal codes are the
  `FailureCode` values plus `ambiguous`, `low-confidence`, `incompatible-model` and
  `model-error`. `parse` never throws.

### `compile(intent, options): Result<Preview>`

- `options.timeZone` is an IANA time zone.
- `options.referenceInstant` is in epoch milliseconds.
- `options.backups` is a list of `{ id, timestamp }`.
- Returns `{ supported: true, keep, remove }`. Each `keep` entry lists the
  zero-based `tierIndexes` that selected it. Otherwise it returns
  `{ supported: false, code, reason }` with code `missing-context`,
  `invalid-context`, `invalid-backups`, `invalid-intent`, `conflict` or `unsupported`.
- Invalid metadata refuses the whole preview.

### `validate(intent): void` and `explain(intent): string`

`validate` throws an error with a `code` for an invalid or edited intent. `explain`
validates the intent and returns a plain-English description.

### `preload(config?): Promise<void>` and `dispose(): void`

`preload` initializes the same engine `parse` uses. It is optional, and it accepts
`{ weights }`. `dispose` clears the cached engines.

### Types

`RetentionIntent`, `Tier`, `Bucket`, `Window`, `Backup`, `Context`,
`CompileOptions`, `Preview`, `FailureCode`, `Result`, `ParseOptions` and
`ParseResult`.

```ts
interface RetentionIntent {
  tiers: readonly {
    bucket: "day" | "week" | "month" | "year";
    window:
      | { kind: "calendar"; unit: "day" | "week" | "month" | "year"; count: number }
      | { kind: "elapsed"; milliseconds: number };
  }[];
}
```

## How it works

1. A tokenizer and a role sequence model label bucket words, counts, units and
   calendar/elapsed markers.
2. TypeScript resolves exact counts and units, then validates the policy. A
   policy has one to four tiers, with at most one tier per bucket.
3. Requests below the model's calibrated confidence floor refuse
   (`low-confidence`). There is no rules fallback.
4. `compile` keeps the newest eligible backup in each local bucket for each tier.
   - **Calendar windows** include the current period. "Daily for a week" counts
     seven local dates, and "monthly for a year" counts twelve local months. Weeks
     start on Monday, and years on January 1.
   - **Mixed units.** A weekly or yearly window in a weekly or yearly tier counts
     Monday weeks or calendar years. In other tiers, a week counts seven days and a
     year counts twelve months. Calendar months follow the Gregorian calendar.
   - **Elapsed windows** compare exact instants and include the cutoff.
     Milliseconds through hours are always elapsed. Days and weeks become elapsed
     with "last", "past", "rolling" or "elapsed". Elapsed months and years refuse.
   - **Ties** use ascending Unicode scalar `id` order, and both lists keep input
     order.

The bundled model has hidden width 32 and was trained on 8,000 synthetic examples.
It compresses to 23,301 Brotli bytes. The runtime chunks add 8,993 Brotli bytes
(`bun run size:gate`). The model loads lazily on first use. Time zone handling
uses the runtime's `Intl` data.

## Limitations

- English only. Text is limited to 4096 UTF-16 code units and 64 lexemes.
- Numbers are limited to numeric literals and word numbers up to one hundred.
  Calendar counts range from 1 to 10000.
- These requests refuse:
  - multiple representatives per bucket;
  - duplicate bucket tiers;
  - questions;
  - uncertain requests.
- Previews accept at most 100000 backups. These refuse:
  - identifiers longer than 256 UTF-16 code units;
  - NUL characters or lone surrogates in identifiers;
  - future timestamps;
  - dates outside years 1 to 9999.
- Descriptions of past policies are not always recognized. For example, "we
  previously kept daily backups for a week" is accepted as a request. Review parsed
  intents before you act on them.
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
  all six consumed acceptance sets: 4338 positives and 3819 negatives.

## License

MIT
