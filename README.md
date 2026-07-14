# Syntaxless

Natural-language schedules to a typed intent AST and cron expressions.

## Repository layout

- [`packages/cron`](packages/cron/README.md): dependency-free TypeScript runtime,
  compiler, and package checks.
- [`packages/cron/training`](packages/cron/training/README.md): private TypeScript
  corpus generation, featurization, and evaluation tools.
- `training/cron`: Python model training and export harness.
- [`packages/cron/evaluation`](packages/cron/evaluation/README.md): isolated human
  evaluation data and acceptance policy; never use these corpora for training.

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

## Bundled model

The committed weight artifact is a hidden-64 export. Its 77 role labels match
the typed vocabulary and include speech-act, actionable, and schedule-kind
heads, so default `parse()` and `cron()` run that model. The compiler and
`parse(text, { source: "rules" })` remain the deterministic baseline.

`model-backed-parse.test.ts` checks one explicit weekday schedule. A passing
unit suite is not a release verdict on the human evaluation set. Smoke and the
packed-consumer check assert that same schedule and stay strict.

## Generated files

Corpora, model candidates, training runs, caches, and build outputs are ignored.
Formatting also excludes the generated weight module and export report. Keep
source fixtures and evaluation data; regenerate outputs through the documented
training pipeline rather than deleting or hand-editing model artifacts.
