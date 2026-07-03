/**
 * Hard Brotli byte budget for the emitted bundle (Phase 21).
 *
 * The compiler entry must stay free of the weights, and the separately emitted
 * model chunks must stay bounded. Both budgets are enforced at Brotli quality
 * 11; a breach fails the build and is fixed by the import boundary, never by
 * relaxing a number.
 */
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { brotliCompressSync, constants } from "node:zlib";

/** `dist/index.js` budget in Brotli q11 bytes. */
export const INDEX_BUDGET = 50_000;
/** Per-chunk budget for every `dist/model-*.js`, in Brotli q11 bytes. */
export const MODEL_BUDGET = 64_000;

const packageRoot = join(import.meta.dirname, "..");
const distDir = join(packageRoot, "dist");

/** Pure budget predicate: `true` when `brotliBytes` fits in `limit`. */
export function checkBudget(brotliBytes: number, limit: number): boolean {
  return brotliBytes <= limit;
}

/** Brotli q11 size of an emitted file. */
export function brotliSize(file: string): number {
  return brotliCompressSync(readFileSync(file), {
    params: { [constants.BROTLI_PARAM_QUALITY]: 11 },
  }).length;
}

export interface Budget {
  file: string;
  bytes: number;
  limit: number;
  ok: boolean;
}

export interface BudgetReport {
  index: Budget;
  model: Budget[];
  ok: boolean;
}

/** Measure the compiler entry and every model chunk against the budgets. */
export function measureBudgets(): BudgetReport {
  const indexBytes = brotliSize(join(distDir, "index.js"));

  const index: Budget = {
    file: join(distDir, "index.js"),
    bytes: indexBytes,
    limit: INDEX_BUDGET,
    ok: checkBudget(indexBytes, INDEX_BUDGET),
  };

  const model = readdirSync(distDir)
    .filter((name) => /^model-.*\.js$/.test(name))
    .sort()
    .map((name) => {
      const file = join(distDir, name);
      const bytes = brotliSize(file);

      return { file, bytes, limit: MODEL_BUDGET, ok: checkBudget(bytes, MODEL_BUDGET) };
    });

  return {
    index,
    model,
    ok: index.ok && model.length > 0 && model.every((chunk) => chunk.ok),
  };
}

function format(budget: Budget): string {
  const name = budget.file.slice(budget.file.lastIndexOf("/") + 1);

  return `${name} ${budget.bytes} B / ${budget.limit} B ${budget.ok ? "ok" : "over"}`;
}

/** Measure, report, and throw when either budget is exceeded. */
export function runSizeGate(): BudgetReport {
  const report = measureBudgets();

  console.log(`size:gate index ${format(report.index)}`);
  for (const chunk of report.model) console.log(`size:gate model ${format(chunk)}`);

  if (report.model.length === 0) {
    throw new Error(
      "size:gate: no dist/model-*.js chunk was emitted; the weights are not isolated",
    );
  }

  if (!report.ok) {
    const over = [report.index, ...report.model].filter((budget) => !budget.ok);

    throw new Error(`size:gate: ${over.length} budget(s) exceeded: ${over.map(format).join(", ")}`);
  }

  return report;
}

if (import.meta.main) {
  runSizeGate();
}
