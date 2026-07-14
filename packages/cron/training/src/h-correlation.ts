/**
 * Epoch-1 vs final H rank correlation over training/cron/runs (STACK-PLAN S.3).
 */
import { readdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, isAbsolute, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

function repoRoot(): string {
  return resolve(dirname(fileURLToPath(import.meta.url)), "../../../..");
}

function resolveRepo(path: string): string {
  return isAbsolute(path) ? path : resolve(repoRoot(), path);
}

export interface RunHPoint {
  run: string;
  epoch1: number;
  final: number;
}

function hRate(entry: unknown): number | undefined {
  if (entry === null || typeof entry !== "object") return undefined;
  const h = (entry as { h?: { ok?: boolean; correct?: number; total?: number } }).h;

  if (h === undefined || h.ok !== true) return undefined;
  const total = Number(h.total ?? 0);

  if (total <= 0) return undefined;

  return Number(h.correct ?? 0) / total;
}

export function collectRunHPoints(runsDir: string): RunHPoint[] {
  const dir = resolveRepo(runsDir);
  const points: RunHPoint[] = [];
  let names: string[] = [];

  try {
    names = readdirSync(dir, { withFileTypes: true })
      .filter((entry) => entry.isDirectory())
      .map((entry) => entry.name);
  } catch {
    return points;
  }

  for (const name of names) {
    const reportPath = join(dir, name, "report.json");

    try {
      const raw = readFileSync(reportPath, "utf8")
        .replace(/\bNaN\b/g, "null")
        .replace(/\b-?Infinity\b/g, "null");

      const report = JSON.parse(raw) as {
        history?: unknown[];
      };

      const history = report.history ?? [];

      if (history.length < 2) continue;
      const epoch1 = hRate(history[0]);
      const final = hRate(history[history.length - 1]);

      if (epoch1 === undefined || final === undefined) continue;
      points.push({ run: name, epoch1, final });
    } catch {
      continue;
    }
  }

  return points.sort((left, right) => left.run.localeCompare(right.run));
}

/** Spearman rank correlation; undefined when fewer than 2 comparable runs. */
export function spearman(xs: number[], ys: number[]): number | undefined {
  if (xs.length !== ys.length || xs.length < 2) return undefined;

  const rank = (values: number[]): number[] => {
    const order = values
      .map((value, index) => ({ value, index }))
      .sort((left, right) => left.value - right.value);

    const ranks = new Array<number>(values.length);

    for (let index = 0; index < order.length; index++) ranks[order[index]!.index] = index;

    return ranks;
  };

  const rx = rank(xs);
  const ry = rank(ys);
  const n = xs.length;
  let sum = 0;

  for (let index = 0; index < n; index++) {
    const d = rx[index]! - ry[index]!;

    sum += d * d;
  }

  return 1 - (6 * sum) / (n * (n * n - 1));
}

export interface HCorrelationReport {
  runs: number;
  spearman: number | undefined;
  points: RunHPoint[];
}

export function correlateEpoch1VsFinal(runsDir: string): HCorrelationReport {
  const points = collectRunHPoints(runsDir);

  const rho = spearman(
    points.map((point) => point.epoch1),
    points.map((point) => point.final),
  );

  return { runs: points.length, spearman: rho, points };
}

if (import.meta.main) {
  const dir = process.argv[2] ?? "training/cron/runs";
  const out = process.argv[3];
  const report = correlateEpoch1VsFinal(dir);
  const body = `${JSON.stringify(report, null, 2)}\n`;

  if (out !== undefined) writeFileSync(resolveRepo(out), body);
  process.stdout.write(`h-correlation: n=${report.runs} spearman=${report.spearman ?? "n/a"}\n`);
}
