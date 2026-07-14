import { writeFileSync } from "node:fs";

import type { KindExample } from "../kinds";
import { bandFailures, countPopulations, type Population, sharesInBand, sharesOf } from "./quotas";
import type { EasySourceKind } from "./tatoeba";

export interface MixReport {
  total: number;
  counts: Record<Population, number>;
  shares: Record<Population, number>;
  inBand: boolean;
  failures: string[];
  uniqueE: number;
  uniqueH: number;
  easySource: EasySourceKind;
  bootstrap: boolean;
}

export function mixReportFromExamples(
  examples: readonly KindExample[],
  easySource: EasySourceKind,
): MixReport {
  const classifiable = examples.map((example) => ({
    intent: example.intent,
    speechAct: example.example.speechAct,
    spans: example.example.spans,
  }));

  const counts = countPopulations(classifiable);
  const shares = sharesOf(counts);

  const uniqueE = new Set(
    examples
      .filter((example) => example.example.speechAct === "none" && example.intent === null)
      .map((example) => example.example.text),
  ).size;

  const uniqueH = new Set(
    examples
      .filter((example) => {
        const hasHole = example.example.spans.some((span) => span.label !== "O");

        return (
          hasHole &&
          example.example.speechAct !== "request" &&
          example.example.speechAct !== "correction"
        );
      })
      .map((example) => example.template),
  ).size;

  const failures = bandFailures(shares);

  return {
    total: examples.length,
    counts,
    shares,
    inBand: sharesInBand(shares),
    failures,
    uniqueE,
    uniqueH,
    easySource,
    bootstrap: easySource === "bootstrap",
  };
}

export function mixReportFromCounts(
  counts: Readonly<Record<Population, number>>,
  uniqueE: number,
  uniqueH: number,
  easySource: EasySourceKind,
): MixReport {
  const total = counts.R + counts.H + counts.E;
  const shares = sharesOf(counts);

  return {
    total,
    counts: { ...counts },
    shares,
    inBand: sharesInBand(shares),
    failures: bandFailures(shares),
    uniqueE,
    uniqueH,
    easySource,
    bootstrap: easySource === "bootstrap",
  };
}

export function writeMixReport(path: string, report: MixReport): void {
  writeFileSync(path, `${JSON.stringify(report, null, 2)}\n`);
}
