# Sort acceptance evidence

The gate records source code and the evaluation policy in the candidate's frozen
identity before reserved evaluation. Changes to either invalidate the freeze.
Historical freeze reports remain unchanged, and new candidates need a new freeze.

Reserved rows are evaluation-only and must never enter training, checkpoint
selection or calibration. Consumption records bind each hash to one frozen
candidate. Earlier consumed wording is regression evidence, not fresh acceptance.

The first `ai-v1` authoring attempt was rejected on one training-wording overlap
before model inference and remains unconsumed. The revised v1 set was consumed
for compact32 and failed coverage and safety. It matched 288/548 positives, with
zero wrong accepted meanings but 180/637 negative false accepts. The model stayed
inactive. Its critical controls passed, showing why critical examples alone are
insufficient evidence.

Fresh v2 passed for context32. It matched 548/548 positives, had zero wrong
accepted meanings and zero/637 false accepts, and passed all nine critical controls.
Every major positive family has at least 78 rows. The first compact32 set is now
regression-only. Future candidates require newly authored wording.

Fresh v3 passed for groups32 after correcting base/negative split assignment.
It matched 548/548 positives with no wrong meanings, zero/637 negative accepts
and no critical failures. Each of the seven major positive families has at least
78 rows. The consumed v1-revised and v2 sets also passed as regression evidence.
The v3 wording is now consumed and cannot authorize another candidate's promotion.

Positive rows include independent expected order over eight fixed witnesses.
The gate checks actual SQLite and JavaScript order against those targets before
consumption. Hand-checked critical orders cover nulls and the reference priority.
The writer imports no production parser/compiler, model or training renderer.

The same implementation assistant authored the bounded generated variations.
Caller identifiers provide much of the variation; the evidence is neither an
independent human evaluation nor a population accuracy estimate. Preserve raw
rows, metadata, consumed hashes and run reports so these limits remain auditable.
