# Useful tests

Run `bun run check` for formatting, types, TypeScript tests, and the build. Run
`bun run test:python` for the Python training/export harness. See the root
[development instructions](README.md#development) for setup and package checks.

## What a test should prove

- Exercise production behavior and assert an independently chosen result. A
  hand-built report matching its own literals does not test the evaluator.
- Name the regression it catches: wrong schedule, unsafe acceptance, corrupted
  supervision, training/evaluation leakage, or incorrect numerical output.
- Give rejection and filtering tests a positive control. A leak detector that
  always returns `[]` must fail, not make every isolation test green.
- Test decision boundaries on both sides. Prefer explicit expected intents,
  counts, and tensors over truthiness or broad “success or any failure” checks.
- Use deterministic seeds and isolated temporary fixtures. Do not depend on
  downloaded corpora, a local candidate, or reports left by an earlier run.
- Do not test source spelling, README wording, private helper names, or arbitrary
  inventory counts. Frozen binary vocabularies and evidence-consumption rules
  are compatibility contracts, so their exact values do matter.

Before adding a test, ask: **what plausible wrong implementation would fail it?**
For important gates, temporarily introduce that fault in a disposable copy and
check that an assertion fails. Compilation/import failures are not evidence that
the behavioral test works. A few targeted mutations are useful checks, not a
claim of exhaustive mutation coverage.

## Suite boundaries

- **Runtime:** compiler and rules examples, model-result safety and confidence,
  correction/actionable selection, tokenization, and numerical inference parity.
- **Training tools:** actual evaluation report aggregation, promotion decisions,
  generation/exclusion behavior, aligned supervision, and sampling diversity.
- **Evaluation data:** shared corpus checks validate schema, compiler gold,
  isolation, consumed acceptance, and wording independence without copying a
  test block for each corpus. Human wording remains evaluation-only.
- **Python:** batch order, duplicates, padding and all supervision tensors;
  quantization, export round-trips, gradients, and padding neutrality.

The bundled weights match the typed label vocabulary. `model-backed-parse.test.ts`
checks that public parse of one explicit weekday schedule succeeds. That check
does not measure the human evaluation set. Keep smoke and packed-consumer checks
strict about the schedule they assert, and keep rejection coverage for
incompatible weights.

Passing consumed-corpus regressions is not fresh acceptance evidence. Do not
weaken a release gate, retune against acceptance wording, or relabel a failure
as success merely to keep tests green.
