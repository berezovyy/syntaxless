# Evaluation fixtures

The gate records source code and the evaluation policy in the candidate's frozen
identity before reserved evaluation. Changes to either invalidate the freeze.
Historical freeze reports remain unchanged, and new candidates need a new freeze.

These files are **not training data**. Do not pass them to `backward.ts`,
`featurize.ts`, the Python trainer, checkpoint selection, or calibration.

Every record carries `trainingEligible: false`. Acceptance hashes, when they
exist, are one-shot: do not consume a hash and then patch decode so the same
wording scores 100%. That score is not promotion.

`dev-negatives.jsonl` is development only. No promotion claim.

`reserved/ai-v1` through `ai-v5` have been consumed. Preserve their wordings,
metadata and hashes as evidence; they can now supply regression checks, never
fresh acceptance. `ai-v5` passed for the 25k format-v2 candidate with 673/702
exact positives, zero wrong accepted meanings and zero/709 negative false
accepts. All 11 critical controls passed. The author is the same implementation
assistant using a separate writer, so these results are not independent human
evidence or a population accuracy estimate.

The gate rejects fresh wording overlap with training or consumed sets. Previously
consumed critical cases must pass before fresh acceptance is consumed. Package
and browser checks follow language acceptance; infrastructure failures do not
authorize reusing that acceptance as fresh evidence. The v5 package-phase resume
retains its original language report and verifies the unchanged scope/artifact.
