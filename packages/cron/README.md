# @syntaxless/cron

Natural-language schedules to a typed intent AST and cron expressions.

`parse(text)` returns a `ScheduleIntent`; `compile(intent)` returns a cron expression or an
explicit unsupported result. The package has no runtime dependencies.

## Privacy

Inference runs entirely on-device. Fetching static model bytes is allowed; sending user text,
tokens, intents, or confidence anywhere is not. There is no silent hosted fallback: the bundled
default weights work offline, and a self-hoster supplies bytes through `parse`'s `weights` option
or `preload`/`getEngine` config. Importing the types or the compiler downloads and initializes
nothing; the model module is imported on the first `parse()`, `preload()`, or `getEngine()` call.
