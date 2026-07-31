# Syntaxless

Local natural-language parsers with typed intents and deterministic compilers.
Each package turns one kind of English instruction into a typed, editable intent
using a small bundled model, then compiles that intent with ordinary,
deterministic code. Nothing is sent to a server, and every package refuses
input it cannot interpret safely instead of guessing.

## Packages

| Package                                                   | Turns text into                                                                   |
| --------------------------------------------------------- | --------------------------------------------------------------------------------- |
| [`@syntaxless/cron`](packages/cron/README.md)             | Recurring schedules and 5-field cron expressions                                  |
| [`@syntaxless/validation`](packages/validation/README.md) | Field constraints and JSON Schema                                                 |
| [`@syntaxless/filter`](packages/filter/README.md)         | Predicates, a JavaScript matcher and parameterized SQLite `WHERE` fragments       |
| [`@syntaxless/sort`](packages/sort/README.md)             | Sort keys, a JavaScript comparator and SQLite `ORDER BY` fragments                |
| [`@syntaxless/retry`](packages/retry/README.md)           | Retry policies and deterministic, bounded delay schedules                         |
| [`@syntaxless/retention`](packages/retention/README.md)   | Backup retention policies and keep/remove previews                                |
| [`@syntaxless/format`](packages/format/README.md)         | Number, currency, percentage and date/time formats as validated `Intl` formatters |

All packages are ESM-only, have no runtime dependencies, and run in Node.js 18+,
Bun and modern browsers. The model loads lazily on first use.

## Example

```ts
import { cron } from "@syntaxless/cron";
import { compile, parse } from "@syntaxless/format";

await cron("every monday at 5pm");
// { supported: true, expression: "0 17 * * 1", intent: { … } }

const parsed = await parse("euros, two decimals, negatives in parentheses");
if (parsed.ok) {
  const compiled = compile(parsed.intent, { locale: "en-US" });
  if (compiled.supported) compiled.formatter.format(-1234.5);
  // { supported: true, text: "(€1,234.50)" }
}
```

Every parse returns either a typed intent or a failure with a code and a reason.
Intents are plain data: edit them and compile again instead of re-parsing text.

## How it works

1. A tokenizer turns text into lexemes with fixed features.
2. A small int6-quantized tagger, shipped as generated TypeScript, labels each
   lexeme with a role.
3. A deterministic resolver builds a typed intent from the roles and refuses
   conflicts, unsupported wording and low-confidence predictions.
4. A compiler turns the intent into the target output (cron expression, JSON
   Schema, SQL fragment, delay schedule, `Intl` formatter and so on).

Models are trained from generated corpora and promoted only through a release
gate that runs one-shot, reserved evaluation sets. Each package's
`evaluation/README.md` documents its protocol, results and limitations.

## Repository layout

```
packages/<package>/          published runtime: src/, tests/, scripts/ (build, size, package and browser checks)
packages/<package>/training/ private TypeScript workspace: corpus, featurization, evaluation and release gate
packages/<package>/evaluation/ reserved evaluation sets, consumption records and protocol
packages/shared/             private TypeScript shared by the runtimes (model decoding, inference, tokenizer)
                             and by training (JSONL, binary shards, reserved-wording checks, statistics)
training/cron/               Python training and export harness for cron, and the shared uv environment
training/shared/             Python training and export harness for the other six packages
training/validation/         validation's boundary-calibrating exporter
scripts/                     build helpers shared by every package (declarations, tarball check)
```

## Development

Requirements: Bun 1.4 or newer. Python training additionally needs Python 3.13
and [uv](https://docs.astral.sh/uv/).

```sh
bun install --frozen-lockfile
bun run check          # formatting, type checking, TypeScript tests and builds with size gates
bun run format         # format sources and docs
```

The checks can also run individually: `bun run format:check`, `bun run typecheck`,
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

## Training

Each package has one coordinator command, run from the repository root:

```sh
bun run train:cron
bun run train:validation   # also: train:filter, train:sort, train:retry, train:retention, train:format
```

A coordinator generates a fresh corpus, trains, exports and evaluates a
candidate in a new run directory. It never changes the bundled model; promotion
requires the package's full release gate. See each package's
`training/README.md` for inputs, outputs and the recipe.

Corpora, runs, caches and build outputs are ignored by Git. Regenerate them
through the training pipeline; never hand-edit generated model artifacts or
reserved evaluation data.

## License

[MIT](LICENSE)
