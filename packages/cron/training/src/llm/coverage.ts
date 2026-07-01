/**
 * Coverage matrix for the angle registry.
 *
 * "Cover everything" becomes measurable by expanding the angle-dimension
 * product into cells. A cell exists only for a tuple the curated registry
 * declared legal; illegal combinations are dropped here, not silently counted.
 * Each cell carries a target count. Production (Phase 10 onward) reports counts
 * per angle id, and {@link gaps} names the cells still short of target.
 *
 * The plan is serialized to `packages/cron/training/data/llm/coverage.json`
 * (untracked) with every cell at `produced: 0`; later phases fill the counts.
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, isAbsolute, join } from "node:path";
import { fileURLToPath } from "node:url";

import type { Family } from "../sampler";
import { ANGLE_DIMENSIONS, ANGLES } from "./angles";
import type { Angle, AngleDimensions } from "./angles";

/** Records requested per cell by default. */
export const DEFAULT_TARGET = 50;

export interface CoverageCell {
  /** Angle id; unique across the registry and the plan. */
  key: string;
  family: Family;
  dimensions: AngleDimensions;
  target: number;
  /** Records produced so far. Zero at planning time. */
  produced: number;
}

export interface CoveragePlan {
  cells: CoverageCell[];
}

export interface CoverageReport {
  cells: CoverageCell[];
  totalTarget: number;
  totalProduced: number;
  covered: number;
  missing: number;
}

export type CoverageCounts = ReadonlyMap<string, number> | Readonly<Record<string, number>>;

type DimensionKey = keyof typeof ANGLE_DIMENSIONS;

/** Stable, order-fixed identity for a dimension tuple. */
function tupleKey(dimensions: AngleDimensions): string {
  return [
    dimensions.output,
    dimensions.register,
    dimensions.domain,
    dimensions.numberStyle,
    dimensions.timeframe,
    dimensions.noise,
    dimensions.difficulty,
  ].join("|");
}

/** Cartesian product of {@link ANGLE_DIMENSIONS}, in declaration order. */
function expandProduct(): AngleDimensions[] {
  let combos: AngleDimensions[] = [{} as AngleDimensions];

  for (const key of Object.keys(ANGLE_DIMENSIONS) as DimensionKey[]) {
    const values = ANGLE_DIMENSIONS[key] as readonly string[];
    const next: AngleDimensions[] = [];

    for (const combo of combos) {
      for (const value of values) {
        next.push({ ...combo, [key]: value } as AngleDimensions);
      }
    }

    combos = next;
  }

  return combos;
}

function countFor(counts: CoverageCounts, key: string): number {
  if (typeof (counts as ReadonlyMap<string, number>).get === "function") {
    return (counts as ReadonlyMap<string, number>).get(key) ?? 0;
  }

  return (counts as Readonly<Record<string, number>>)[key] ?? 0;
}

/**
 * Expand the product into cells and keep the legal, registered tuples. Throws
 * if two angles share a dimension tuple, because that would hide a cell.
 */
export function planCoverage(): CoveragePlan {
  const byTuple = new Map<string, Angle>();

  for (const angle of ANGLES) {
    const key = tupleKey(angle);
    const existing = byTuple.get(key);

    if (existing !== undefined) {
      throw new Error(`coverage: ${angle.id} and ${existing.id} share tuple ${key}`);
    }

    byTuple.set(key, angle);
  }

  const cells: CoverageCell[] = [];

  for (const dimensions of expandProduct()) {
    const angle = byTuple.get(tupleKey(dimensions));

    if (angle === undefined) continue;

    cells.push({
      key: angle.id,
      family: angle.family,
      dimensions,
      target: DEFAULT_TARGET,
      produced: 0,
    });
  }

  return { cells };
}

/** Attach produced counts to a plan and summarize coverage. */
export function applyCoverage(plan: CoveragePlan, counts: CoverageCounts): CoverageReport {
  const cells = plan.cells.map((cell) => ({
    ...cell,
    produced: countFor(counts, cell.key),
  }));

  let totalTarget = 0;
  let totalProduced = 0;
  let covered = 0;
  let missing = 0;

  for (const cell of cells) {
    totalTarget += cell.target;
    totalProduced += cell.produced;
    if (cell.produced >= cell.target) covered++;
    else missing++;
  }

  return { cells, totalTarget, totalProduced, covered, missing };
}

/** Cells whose produced count has not reached the target. */
export function gaps(report: CoverageReport): CoverageCell[] {
  return report.cells.filter((cell) => cell.produced < cell.target);
}

/** Walk up to the workspace root (the package.json that lists `workspaces`). */
function repoRoot(): string {
  let dir = dirname(fileURLToPath(import.meta.url));

  for (;;) {
    const manifest = join(dir, "package.json");

    if (existsSync(manifest)) {
      try {
        const pkg = JSON.parse(readFileSync(manifest, "utf8")) as {
          workspaces?: unknown;
        };

        if (Array.isArray(pkg.workspaces)) return dir;
      } catch {
        // Not a readable manifest; keep walking up.
      }
    }

    const parent = dirname(dir);

    if (parent === dir) return process.cwd();
    dir = parent;
  }
}

const DEFAULT_OUT = "packages/cron/training/data/llm/coverage.json";

function resolveOut(value: string | undefined): string {
  const target = value ?? DEFAULT_OUT;

  return isAbsolute(target) ? target : join(repoRoot(), target);
}

if (import.meta.main) {
  const argv = process.argv.slice(2);
  let out: string | undefined;

  for (let index = 0; index < argv.length; index++) {
    const arg = argv[index]!;

    if (arg === "--out") {
      out = argv[index + 1];
      if (out === undefined) throw new RangeError("missing value for --out");
      index++;
    } else {
      throw new RangeError(`unknown flag: ${arg}`);
    }
  }

  const report = applyCoverage(planCoverage(), new Map());
  const path = resolveOut(out);

  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, `${JSON.stringify(report, null, 2)}\n`);

  process.stdout.write(
    `coverage: ${report.cells.length} cells, ${report.totalTarget} target, ` +
      `0 produced -> ${path}\n`,
  );
}
