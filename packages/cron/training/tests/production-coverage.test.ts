import { expect, test } from "bun:test";

import { semanticCases } from "../src/corpus/production/cases";
import { loadConstructions, renderConstruction } from "../src/corpus/production/constructions";
import { ConstructionCoverage } from "../src/corpus/production/coverage";

test("negative rows and invented interaction tags cannot fill positive semantic/value coverage", () => {
  const source = loadConstructions();
  const group = source.groups.find((g) => g.id === "interval-repeat-action")!;
  const semantic = semanticCases("interval").next().value!;

  const negative = renderConstruction(
    source,
    group,
    group.forms.find((f) => f.speechAct === "descriptive")!,
    semantic,
  );

  const coverage = new ConstructionCoverage();

  coverage.add(negative, semantic.semanticFamily, [
    "direct-request",
    "interval-and-clock-where-valid",
    "descriptive-report",
  ]);

  let report = coverage.finish();

  expect(
    report.missing.some(
      (m) => m.split === "train" && m.axis === "semantics" && m.key === "minute-interval",
    ),
  ).toBe(true);

  expect(
    report.missing.some(
      (m) => m.split === "train" && m.axis === "negatives" && m.key === "descriptive-report",
    ),
  ).toBe(false);

  const positive = renderConstruction(source, group, group.forms[0]!, semantic);

  coverage.add(positive, semantic.semanticFamily, [
    "direct-request",
    "interval-and-clock-where-valid",
  ]);

  report = coverage.finish();

  expect(
    report.missing.some(
      (m) =>
        m.split === "train" &&
        m.axis === "interactions" &&
        m.key === "interval-and-clock-where-valid",
    ),
  ).toBe(true);

  expect(
    report.missing.some((m) => m.split === "train" && m.axis === "values" && m.key === "hour/0"),
  ).toBe(true);

  // Many value substitutions of one source group do not meet source diversity.
  for (const c of semanticCases("interval")) {
    coverage.add(renderConstruction(source, group, group.forms[0]!, c), c.semanticFamily, [
      "direct-request",
    ]);
  }

  expect(
    coverage
      .finish()
      .missing.some(
        (m) => m.split === "train" && m.axis === "semantics" && m.key === "minute-interval",
      ),
  ).toBe(true);
});
