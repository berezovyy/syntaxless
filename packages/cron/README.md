# @syntaxless/cron

Natural-language schedules to a typed intent AST and cron expressions.

`parse(text)` returns a `ScheduleIntent`; `compile(intent)` returns a cron expression or an
explicit unsupported result. The package has no runtime dependencies.
