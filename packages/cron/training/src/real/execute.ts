/**
 * Track C execute: compile the agreed intent. Compile failure drops the
 * record unless the compiler itself declares `supported: false`.
 */
import { compile } from "../../../src/compiler/cron";
import type { CompileResult } from "../../../src/compiler/cron";
import type { ScheduleIntent } from "../../../src/types/intent";
import { scheduleIntentSchema } from "../llm/schema";

export interface ExecuteOutcome {
  keep: boolean;
  compile: CompileResult | { supported: true; expression: null };
}

export function executeIntent(intent: ScheduleIntent | null): ExecuteOutcome {
  if (intent === null) return { keep: true, compile: { supported: true, expression: null } };
  const result = compile(intent);

  if (result.supported) return { keep: true, compile: result };

  return { keep: true, compile: result };
}

export function executeKeep<T extends { intent: ScheduleIntent | null }>(
  records: readonly T[],
): T[] {
  return records.filter((record) => {
    if (record.intent === null) return true;
    if (!scheduleIntentSchema.safeParse(record.intent).success) return false;
    compile(record.intent);

    return true;
  });
}

/** Drop only programming-invalid intents that throw, not compiler-unsupported. */
export function executeDropCompileCrash<T extends { intent: ScheduleIntent | null }>(
  records: readonly T[],
): { kept: T[]; rejected: T[] } {
  const kept: T[] = [];
  const rejected: T[] = [];

  for (const record of records) {
    try {
      if (record.intent !== null) compile(record.intent);
      kept.push(record);
    } catch {
      rejected.push(record);
    }
  }

  return { kept, rejected };
}
