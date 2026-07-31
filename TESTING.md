# Testing

`bun run check` runs formatting, type checking, every TypeScript test suite and
every build (including the size gates). `bun run test:python` runs the Python
suites for `training/cron` and `training/shared` in separate processes using the
cron uv environment.

## What to keep

Keep a test when an independently chosen expected result exposes a plausible
wrong behavior. Prefer public behavior and distinct boundaries over repeated
synonyms, values, random seeds, corpus inventories and private structure.

- Runtime tests check exact output, invalid or conflicting input, refusal of
  unsafe model predictions, and incompatible weight artifacts.
- Training tests check aligned supervision, evaluation arithmetic, split
  separation, reserved-wording exclusion, promotion and publication integrity.
- Python tests check reference logits, quantization and wire decoding, export,
  padding neutrality, gradients and shard loading. The small cron fixture keeps
  its original Python reference values.
- Package and browser checks verify the shipped declarations and runtime, lazy
  loading, concurrent initialization, offline use and request privacy.

Give rejection and filtering tests a positive control, and test both sides of
safety limits. Don't keep expected values computed by the function under test,
truthiness checks, or assertions that accept any outcome. Delete unused fixtures
and helpers together with their last test rather than hiding tests in loops or
skips. There is no target count of tests; keep distinct useful behavior.

## Package and browser checks

Run these after `bun run build`, for each package (`cron`, `validation`,
`filter`, `sort`, `retry`, `retention`, `format`):

```sh
bun run --cwd packages/<package> check:package   # packed tarball in isolated Bun and Node consumers
bun run --cwd packages/<package> test:browser    # offline lazy loading in Chromium
```

The browser checks live in `packages/<package>/scripts/check-browser.ts`. They
need Playwright's Chromium (`bunx playwright install chromium` from a package
directory) and permission to listen on a local port.

`bun run --cwd packages/cron smoke` keeps a known cron model regression visible
(see the cron README). Report it separately; don't change its expected value.

## Evaluation replays

Consumed evaluation sets are regression evidence, never new acceptance evidence.

- `packages/validation/tests/evaluation.test.ts` replays consumed critical rows
  and development negatives as part of `bun run test`.
- `bun run --cwd packages/retry/training replay` and
  `bun run --cwd packages/retention/training replay` replay every consumed set
  against the bundled model.
- Every promotion gate replays the archived critical rows of all earlier
  consumed sets before it consumes a new one.

## Limits of passing tests

Unit tests do not measure acceptance of fresh language. Do not weaken release
limits, retune against acceptance wording, or relabel a failure as a success.

For important numerical and safety checks, introduce a targeted fault in a small
disposable copy and confirm that an assertion fails. A missing import or a
compilation failure does not prove the behavior is tested.
