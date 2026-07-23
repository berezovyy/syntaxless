# Filter acceptance evidence

The gate records source code and the evaluation policy in the candidate's frozen
identity before reserved evaluation. Changes to either invalidate the freeze.
Historical freeze reports remain unchanged, and new candidates need a new freeze.

Every reserved row is evaluation-only (`trainingEligible: false`). Do not train,
select checkpoints or calibrate on any reserved wording.

`ai-v1` was consumed and failed positive coverage. It is now regression evidence.
`ai-v2` was rejected before model inference because some wording overlapped v1;
it remains unconsumed. `ai-v2-revised` passed for the frozen history32 model:
842/842 positives exactly matched, zero wrong meanings, zero/705 false accepts,
and all seven critical controls passed. Preserve consumed raw rows, metadata and
hashes. Future candidates need newly authored wording.

The authoring writer imports no training renderer, parser or model inference.
Each positive carries independent truth targets over eight fixed witness rows.
The full gate checks actual SQLite and JavaScript truth against those targets.
The same implementation assistant authored these limited generated variations;
they are not independent human evidence or a population accuracy estimate.
