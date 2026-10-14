<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/media/logo/logo-dark.svg" />
    <img src="docs/media/logo/logo-light.svg" height="96" alt="Syntaxless" />
  </picture>
</h1>

<p align="center">
  <strong>Plain English in. Typed, deterministic output out.</strong><br />
  Small natural-language parsers for TypeScript and JavaScript that run on the device and refuse instead of guessing.
</p>

<p align="center">
  <a href="https://github.com/berezovyy/syntaxless/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/berezovyy/syntaxless/ci.yml?branch=master&style=flat-square&label=CI" alt="CI" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-8250df?style=flat-square" alt="MIT license" /></a>
  <img src="https://img.shields.io/badge/runtime%20deps-0-238636?style=flat-square" alt="Zero runtime dependencies" />
  <img src="https://img.shields.io/badge/size-%3C100%20kB%20gzip-58a6ff?style=flat-square" alt="Under 100 kB gzip per package, model included" />
</p>

<p align="center">
  <img alt="A browser app with a customers table, DevTools Network and a size card. Typing 'active customers spending over 100, excluding trials' strikes out the rows that fail; Network shows the whole package loading as 9 requests, 36.1 kB gzip with the model. With the network set to Offline, a second query, 'spending between 50 and 200', still filters the table with 0 requests. The same app then uses the other packages, each loading in Network and adding a bar to the size card, all under 100 kB: 'biggest spenders first' sorts the table, 'euros, two decimals' formats it, 'every monday at 9am' becomes 0 9 * * 1, 'required email with max 100 characters' becomes JSON Schema, 'retry three times, start at 250ms, double the delay' becomes delays of 250, 500 and 1000 ms, and a retention policy keeps 18 of 420 backups." src="docs/media/v3/films/hero.webp" width="100%" />
</p>

Each Syntaxless package turns one kind of English instruction into a typed,
editable intent with a small bundled model, then compiles that intent with
ordinary, deterministic code. Nothing is sent to a server. When a package can't
interpret the input safely, it returns a failure with a code and a reason
instead of guessing.

- **Small.** Each package is under 100 kB gzip with its model included;
  `@syntaxless/filter` is 36.1 kB.
- **Local.** The model ships inside the package, loads lazily and runs in
  Node.js, Bun and the browser, offline. No API keys, no server, no runtime
  dependencies.
- **Typed.** Every parse returns a typed intent: plain data you can store, show,
  edit and compile again without re-parsing the text.
- **Deterministic.** The same intent always compiles to the same output.
- **Honest.** Unsupported wording, conflicts and low-confidence predictions come
  back as failures with a code and a reason your UI can act on.

## Quick start

```sh
npm install @syntaxless/cron @syntaxless/format
```

```ts
import { cron } from "@syntaxless/cron";
import { compile, parse } from "@syntaxless/format";

await cron("every monday at 5pm");
// { supported: true, expression: "0 17 * * 1", intent: { … } }

await cron("remind me tomorrow at 3");
// { supported: false, code: "unsafe-context",
//   reason: "the text does not request a recurring schedule", intent: null }

const parsed = await parse("euros, two decimals, negatives in parentheses");
if (parsed.ok) {
  const compiled = compile(parsed.intent, { locale: "en-US" });
  if (compiled.supported) compiled.formatter.format(-1234.5);
  // { supported: true, text: "(€1,234.50)" }
}
```

## How it works

<p align="center">
  <img alt="The string 'weekdays at 9:30am' is split into six lexemes, tagged with roles such as DAYGROUP_WEEKDAY and HOUR_9 with their confidences, resolved into a typed ScheduleIntent and compiled to the cron expression 30 9 * * 1-5. The string 'remind me tomorrow at 3' is refused with unsafe-context." src="docs/media/v3/films/how-it-works.webp" width="100%" />
</p>

1. **Tokenize.** A tokenizer splits the text into lexemes with fixed features.
2. **Tag.** A small int6-quantized tagger, shipped as generated TypeScript,
   labels each lexeme with a role and a confidence.
3. **Resolve.** A deterministic resolver builds a typed intent from the roles. It
   refuses conflicts, unsupported wording and low-confidence predictions.
4. **Compile.** A compiler turns the intent into the target output: a cron
   expression, JSON Schema, a SQL fragment, a delay schedule, an `Intl`
   formatter and so on.

Models are trained on generated corpora. A model is promoted only through a
release gate that runs one-shot, reserved evaluation sets. Each package's
`evaluation/README.md` documents its protocol, results and limitations.

## Where to use it

Every value on screen is the package's real output for the code shown, run
with Bun.

### [`@syntaxless/cron`](packages/cron/README.md): schedule fields

Recurring schedules to 5-field cron expressions, plus the next occurrences.
When cron can't express a schedule, it says so and keeps the intent.

<p align="center">
  <img alt="An editor and terminal. cron('weekdays at 9:30am') prints 30 9 * * 1-5 and the next four runs. Changed to 'first friday of every month at noon', it prints 'cron cannot represent an ordinal weekday', while recurrences() still lists the dates." src="docs/media/v3/films/cron.webp" width="100%" />
</p>

### [`@syntaxless/validation`](packages/validation/README.md): form builders

Field constraints to a typed intent and JSON Schema.

<p align="center">
  <img alt="An editor and terminal. schema() turns 'required email with max 100 characters' and 'optional integer between 1 and 10' into JSON Schema objects, and refuses 'every weekday at 9' with unrecognized." src="docs/media/v3/films/validation.webp" width="100%" />
</p>

### [`@syntaxless/filter`](packages/filter/README.md): search over your data

Predicates you can evaluate in JavaScript or compile to parameterized SQLite
`WHERE` fragments.

<p align="center">
  <img alt="An editor and terminal. parse('active customers spending over 100, excluding trials') compiles to SQLite parameters ["active", 100, 1] and matches three customers; changed to 'spending between 50 and 200', it prints [50, 200] and four customers." src="docs/media/v3/films/filter.webp" width="100%" />
</p>

### [`@syntaxless/sort`](packages/sort/README.md): ordering lists and tables

Sort keys, a JavaScript comparator and SQLite `ORDER BY` fragments.

<p align="center">
  <img alt="An editor with a live preview. parse('newest first, then cheapest, missing prices last') reorders a product table to ids 3, 2, 6, 5, 1, 4." src="docs/media/v3/films/sort.webp" width="100%" />
</p>

### [`@syntaxless/retry`](packages/retry/README.md): retry policies

Retry policies to deterministic, bounded delay schedules. It never invents a
missing value.

<p align="center">
  <img alt="An editor and terminal. parse('retry three times, double the delay, cap at one minute') prints missing-initial-delay; after adding { defaults: { initialDelayMs: 250 } } it prints delays of 250, 500 and 1000 ms as bars." src="docs/media/v3/films/retry.webp" width="100%" />
</p>

### [`@syntaxless/retention`](packages/retention/README.md): backup retention

Retention policies to keep/remove previews. It recommends; it never deletes.

<p align="center">
  <img alt="An editor with a live preview of 420 daily backups. 'keep daily backups for a week and monthly backups for a year' keeps 18, the last seven days and the newest of each month, and recommends removing 402." src="docs/media/v3/films/retention.webp" width="100%" />
</p>

### [`@syntaxless/format`](packages/format/README.md): number and date formats

Number, currency, percentage and date/time formats as validated `Intl`
formatters.

<p align="center">
  <img alt="An editor and terminal. 'euros, two decimals, negatives in parentheses' formats a ledger as (€1,234.50), €98,000.00 and so on; changing only the locale to de-DE prints (1.234,50 €), 98.000,00 € and so on." src="docs/media/v3/films/format.webp" width="100%" />
</p>

## Packages

| Package                                                   | Turns text into                                                                   | Size, gzip |
| --------------------------------------------------------- | --------------------------------------------------------------------------------- | ---------- |
| [`@syntaxless/cron`](packages/cron/README.md)             | Recurring schedules and 5-field cron expressions                                  | 92.6 kB    |
| [`@syntaxless/validation`](packages/validation/README.md) | Field constraints and JSON Schema                                                 | 75.5 kB    |
| [`@syntaxless/filter`](packages/filter/README.md)         | Predicates, a JavaScript matcher and parameterized SQLite `WHERE` fragments       | 36.1 kB    |
| [`@syntaxless/sort`](packages/sort/README.md)             | Sort keys, a JavaScript comparator and SQLite `ORDER BY` fragments                | 35.0 kB    |
| [`@syntaxless/retry`](packages/retry/README.md)           | Retry policies and deterministic, bounded delay schedules                         | 35.9 kB    |
| [`@syntaxless/retention`](packages/retention/README.md)   | Backup retention policies and keep/remove previews                                | 35.7 kB    |
| [`@syntaxless/format`](packages/format/README.md)         | Number, currency, percentage and date/time formats as validated `Intl` formatters | 37.5 kB    |

All packages are ESM-only, have no runtime dependencies, and run in Node.js 18+,
Bun and modern browsers. The model loads lazily on first use. Sizes are the
whole built package, model included, at `gzip -9`.

## Development

Requirements: Bun 1.4 or newer. Python training also needs Python 3.13 and
[uv](https://docs.astral.sh/uv/).

```sh
bun install --frozen-lockfile
bun run check          # formatting, type checking, TypeScript tests and builds with size gates
bun run format         # format sources and docs
```

The checks also run individually: `bun run format:check`, `bun run typecheck`,
`bun run test` and `bun run build`. Package-level checks run from each package
directory, for example:

```sh
bun run --cwd packages/format check:package   # packed tarball in Bun and Node consumers
bun run --cwd packages/format test:browser    # offline Chromium check (needs Playwright's Chromium)
```

Python tests:

```sh
uv sync --project training/cron --locked
bun run test:python
```

See [TESTING.md](TESTING.md) for what the tests cover and the review standard,
and [CONTRIBUTING.md](CONTRIBUTING.md) for the contribution workflow.

<details>
<summary><strong>Repository layout</strong></summary>

```
packages/<package>/            published runtime: src/, tests/, scripts/ (build, size, package and browser checks)
packages/<package>/training/   private TypeScript workspace: corpus, featurization, evaluation and release gate
packages/<package>/evaluation/ reserved evaluation sets, consumption records and protocol
packages/shared/               private TypeScript shared by the runtimes (model decoding, inference, tokenizer)
                               and by training (JSONL, binary shards, reserved-wording checks, statistics)
training/cron/                 Python training and export harness for cron, and the shared uv environment
training/shared/               Python training and export harness for the other six packages
training/validation/           validation's boundary-calibrating exporter
scripts/                       build helpers shared by every package (declarations, tarball check)
docs/media/                    README media: logo/ holds the logo and its generator, v3/ the films and their HyperFrames sources
```

</details>

## Training

Each package has one coordinator command, run from the repository root:

```sh
bun run train:cron
bun run train:validation   # also: train:filter, train:sort, train:retry, train:retention, train:format
```

A coordinator generates a fresh corpus, then trains, exports and evaluates a
candidate in a new run directory. It never changes the bundled model; promotion
requires the package's full release gate. See each package's
`training/README.md` for inputs, outputs and the recipe.

Corpora, runs, caches and build outputs are ignored by Git. Regenerate them
through the training pipeline; never hand-edit generated model artifacts or
reserved evaluation data.

## License

[MIT](LICENSE)
