import { expect, test } from "bun:test";

import { ANGLES } from "../src/llm/angles";
import { applyCoverage, DEFAULT_TARGET, gaps, planCoverage } from "../src/llm/coverage";

test("the plan fans the product out to exactly the registered angles", () => {
  const plan = planCoverage();

  expect(plan.cells.length).toBe(ANGLES.length);
  expect(new Set(plan.cells.map((cell) => cell.key)).size).toBe(ANGLES.length);

  for (const cell of plan.cells) {
    expect(cell.target).toBe(DEFAULT_TARGET);
    expect(cell.produced).toBe(0);
  }
});

test("plan tuples correspond to registered angles' dimensions", () => {
  const plan = planCoverage();
  const byId = new Map(ANGLES.map((angle) => [angle.id, angle]));

  for (const cell of plan.cells) {
    const angle = byId.get(cell.key);

    expect(angle).toBeDefined();
    expect(cell.family).toBe(angle!.family);
    expect(cell.dimensions.output).toBe(angle!.output);
    expect(cell.dimensions.register).toBe(angle!.register);
    expect(cell.dimensions.domain).toBe(angle!.domain);
    expect(cell.dimensions.numberStyle).toBe(angle!.numberStyle);
    expect(cell.dimensions.timeframe).toBe(angle!.timeframe);
    expect(cell.dimensions.noise).toBe(angle!.noise);
    expect(cell.dimensions.difficulty).toBe(angle!.difficulty);
  }
});

test("applyCoverage attaches produced counts and summarizes", () => {
  const plan = planCoverage();
  const first = plan.cells[0]!;
  const counts = new Map<string, number>([[first.key, DEFAULT_TARGET]]);
  const report = applyCoverage(plan, counts);

  expect(report.cells.length).toBe(plan.cells.length);

  for (const cell of report.cells) {
    expect(cell.produced).toBe(cell.key === first.key ? DEFAULT_TARGET : 0);
  }

  expect(report.totalTarget).toBe(plan.cells.length * DEFAULT_TARGET);
  expect(report.totalProduced).toBe(DEFAULT_TARGET);
  expect(report.covered).toBe(1);
  expect(report.missing).toBe(plan.cells.length - 1);
});

test("applyCoverage accepts a plain record as well as a map", () => {
  const plan = planCoverage();
  const first = plan.cells[0]!;
  const report = applyCoverage(plan, { [first.key]: 3 });

  expect(report.totalProduced).toBe(3);
  expect(report.covered).toBe(0);
});

test("gaps are exactly the cells below target", () => {
  const plan = planCoverage();
  const first = plan.cells[0]!;

  const report = applyCoverage(
    plan,
    new Map<string, number>([
      [first.key, DEFAULT_TARGET],
      [plan.cells[1]!.key, DEFAULT_TARGET + 10],
    ]),
  );

  const short = gaps(report);

  expect(short.length).toBe(plan.cells.length - 2);
  expect(short.some((cell) => cell.key === first.key)).toBe(false);
  for (const cell of short) expect(cell.produced).toBeLessThan(cell.target);
});

test("the report serializes deterministically", () => {
  const plan = planCoverage();
  const first = JSON.stringify(applyCoverage(plan, new Map()));
  const second = JSON.stringify(applyCoverage(planCoverage(), new Map()));

  expect(first).toBe(second);
});
