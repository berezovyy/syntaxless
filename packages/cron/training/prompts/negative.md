---
id: negative
version: 1
angle: negative
schemaVersion: 1
---

# negative

You generate ordinary English that looks like it might describe a schedule but
does not. These negatives teach the parser to stay quiet when there is nothing to
parse, so they must not contain any schedule meaning.

## Hard rules (non-negotiable)

1. Respond with exactly one JSON object per line, and nothing else. Each line must
   match this schema: {{schema}}
2. The text must contain no schedule meaning. Do not use words from the lexicon
   below as schedule words; when in doubt, avoid them entirely. Lexicon:
   {{lexicon}}
3. Never invent a schedule the compiler does not support. For this family the
   `intent` is always `null` and the `output` is always `"negative"`.

## Task

Produce {{count}} negative examples for the family `{{family}}`. They are plain
prose — reminders, notes, or observations that mention time-like words without
stating a recurrence. The seed is intentionally empty:

```json
{{seedIntent}}
```

Do not add explanatory text, markdown, or code fences.
