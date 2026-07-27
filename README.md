# Syntaxless

Local natural-language parsers with typed intents and deterministic compilers.
Cron handles schedules; validation handles field constraints, filter handles
predicates, sort handles ordering, retry handles bounded delay schedules, and
retention produces backup keep/remove previews, and format compiles numeric and
date/time presentation instructions.

## Training

From the repository root:

```sh
bun run train
```

`bun run train` is cron's single coordinator command. It prepares the corpus, trains the fixed
recipe, selects and evaluates the checkpoint, calibrates, and exports weights
into a fresh run directory. It takes no flags and needs no separate preparation,
conversion, build, or export command.

See [training](packages/cron/training/README.md) for required tools, pinned input
files, and outputs.

## Repository layout

- [`packages/cron`](packages/cron/README.md): dependency-free TypeScript runtime,
  compiler, and package checks.
- [`packages/cron/training`](packages/cron/training/README.md): private TypeScript
  internal corpus preparation, featurization, and evaluation.
- `training/cron`: Python model training and export harness.
- [`packages/cron/evaluation`](packages/cron/evaluation/README.md): isolated human
  evaluation data and acceptance policy; never use these corpora for training.

The completed language packages are [validation](packages/validation/README.md),
[filter](packages/filter/README.md), [sort](packages/sort/README.md),
[retry](packages/retry/README.md), [retention](packages/retention/README.md), and
[format](packages/format/README.md).
Their coordinators are `bun run train:validation`, `bun run train:filter`, `bun run train:sort`, `bun run train:retry`, `bun run train:retention`, and `bun run train:format`. Each creates an inactive candidate; promotion requires the
package's full gate. Each package's evaluation README documents its acceptance
policy and the limits of its recorded evidence.

## Development

Use Bun 1.4.0 or newer. Python training additionally requires Python 3.13 and uv.

```sh
bun install --frozen-lockfile
bun run check           # formatting, type checking, TypeScript tests, build/size gate
bun run format          # format source and docs, excluding generated/ignored files
```

Checks can also be run individually with `bun run format:check`, `bun run typecheck`,
`bun run test`, and `bun run build`. The runtime and training workspaces each run
their own tests once. Tests use temporary fixtures, not an untracked trained model.
See [Useful tests](TESTING.md) for coverage boundaries and the test review standard.

For the Python harness:

```sh
uv sync --project training/cron --locked
bun run test:python
```

The Python test runner discovers the harness's argument-free `test_*` functions
using the standard library; no additional test dependency is required.

Additional package checks (run after building):

```sh
bun run --cwd packages/cron check:package
bun run --cwd packages/cron smoke
bun run --cwd packages/cron test:browser
```

The browser check needs Playwright's Chromium installed:
`(cd packages/cron && bunx playwright install chromium)`.

## Cron bundled model

The committed weight artifact is a hidden-64 export. Its 77 role labels match
the typed vocabulary and include speech-act, actionable, and schedule-kind
heads, so default `parse()` and `cron()` run that model. The compiler and
`parse(text, { source: "rules" })` remain the deterministic baseline.

`model-backed-parse.test.ts` checks one explicit weekday schedule. A passing
unit suite is not a release verdict on the human evaluation set. The bare-clock
smoke check still reports a known failure, as described in [Useful tests](TESTING.md).

## Generated files

Corpora, model candidates, training runs, caches, and build outputs are ignored.
Formatting also excludes the generated weight module and export report. Keep
source fixtures and evaluation data; regenerate outputs through the documented
training pipeline rather than deleting or hand-editing model artifacts.
