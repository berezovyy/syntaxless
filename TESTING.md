# Useful tests

Run `bun run check` for formatting, types, TypeScript tests, and builds. Run
`bun run test:python` for both cron and validation's Python tests, using the
existing cron environment in separate processes. Validation's tests must not be
omitted from Python verification.

## What to keep

Keep a test when an independently chosen expected result exposes a plausible
wrong behavior. Prefer public behavior and distinct boundaries over repeated
synonyms, values, random seeds, corpus inventories, and private structure.

- Runtime tests check exact output, invalid or conflicting input, unsafe model
  acceptance, and incompatible artifacts.
- Training tests check aligned supervision, actual evaluation calculations,
  split separation, evidence consumption, promotion, and publication integrity.
- Numerical tests check reference logits, quantization, export, padding, and
  gradients. The small cron fixture retains original Python reference values.
- Package and browser checks verify the shipped declarations and runtime,
  lazy loading, concurrent initialization, offline use, and request privacy.

Give rejection and filtering tests a positive control, and test both sides of
safety limits. Don't keep expected values calculated by the function under test,
truthiness checks, or checks accepting any outcome. Delete unused fixtures and
helpers with their last useful test rather than hiding tests in loops or skips.

There is no required percentage of tests to delete. Keep distinct useful
behavior rather than retaining repeated examples or enforcing a numerical quota.

## Package and browser commands

First, run `bun run build`. Second, run `bun run --cwd packages/<package>
check:package` for each of cron, validation, filter, sort, retry, retention, and
format. The checks use isolated tarball consumers in Bun and Node and remove
their temporary directories.

Run `bun packages/cron/tests/browser.ts` for cron. Run
`bun packages/<package>/scripts/check-browser.ts` for each other package.
Browser checks require Playwright's Chromium and permission to listen on a
local port. See the [development instructions](README.md#development) for setup.

`bun run --cwd packages/cron smoke` retains a useful known regression. The
current model interprets "every weekday at 9" as 21:00 instead of the expected
09:00. Keep the expected morning schedule unchanged and report the failure
separately from the passing explicit "9am" reference.

## Limits of passing tests

Unit tests do not measure fresh language acceptance. Passing consumed evaluation
examples is regression evidence, not new promotion evidence. Do not weaken
release limits, retune against acceptance wording, or relabel a failure as success.

For important numerical and safety checks, introduce a targeted fault in a
small disposable copy and verify an assertion fails. A missing import or a
compilation failure does not prove the behavior is tested. A few checks do not
claim exhaustive mutation coverage.
