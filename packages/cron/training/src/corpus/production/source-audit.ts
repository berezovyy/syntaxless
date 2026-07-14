/** Exhaustive source inventory validation. This does not publish a production corpus. */
import { existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";

import type { Split } from "../../generate";
import { CorpusAudit } from "./audit";
import { intervalDomainExclusions, semanticCases } from "./cases";
import {
  CONSTRUCTIONS_PATH,
  loadConstructions,
  renderConstruction,
  verifyDiscarded,
} from "./constructions";
import { ConstructionCoverage } from "./coverage";
import { loadProtectedText } from "./protected-text";
import { corpusCodeHashes, fileHash } from "./source-hashes";

export { verifyDiscarded } from "./constructions";

const bump = (counts: Record<string, number>, key: string): void => {
  counts[key] = (counts[key] ?? 0) + 1;
};

export function auditConstructionSource(onProgress?: (n: number) => void) {
  const inputHashes = () => ({
    constructions: fileHash(CONSTRUCTIONS_PATH),
    specification: fileHash(new URL("../../../data/production/coverage-v1.json", import.meta.url)),
    code: corpusCodeHashes(),
  });

  const hashes = inputHashes();
  const source = loadConstructions();
  const coverage = new ConstructionCoverage();
  const protectedText = loadProtectedText();
  const directory = mkdtempSync(join(tmpdir(), "cron-source-audit-"));

  const audit = new CorpusAudit({
    enforceMix: false,
    protectedText,
    indexPath: join(directory, "identities.sqlite"),
  });

  const families: Record<Split, Record<string, number>> = {
    train: {},
    validation: {},
    heldout: {},
  };

  const tags: Record<Split, Record<string, number>> = {
    train: {},
    validation: {},
    heldout: {},
  };

  const exclusions: Record<string, number> = {};
  let candidates = 0;
  let discardedChecks = 0;

  try {
    for (const group of source.groups) {
      for (const semantic of semanticCases(group.caseSet)) {
        verifyDiscarded(semantic);
        if (semantic.discarded) discardedChecks++;

        for (const form of group.forms) {
          const row = renderConstruction(source, group, form, semantic);

          candidates++;
          if (candidates % 100_000 === 0) onProgress?.(candidates);

          if (protectedText.matches(row.text)) {
            bump(exclusions, `${group.id}/${form.id}/protected-wording`);
            continue;
          }

          audit.add(row);
          coverage.add(row, semantic.semanticFamily, [...form.tags, ...semantic.tags]);
          bump(families[group.split], semantic.semanticFamily);
          for (const tag of new Set([...form.tags, ...semantic.tags])) bump(tags[group.split], tag);
        }
      }
    }

    const integrity = audit.finish();

    if (
      JSON.stringify(hashes) !== JSON.stringify(inputHashes()) ||
      protectedText.sources.some((source) => fileHash(source.file) !== source.sha256)
    ) {
      throw new Error(
        "Source, implementation or protected corpus changed during audit; discard this run and retry on stable inputs",
      );
    }

    return {
      version: 1,
      scope: "complete-authored-source-inventory-not-production-corpus",
      sourceId: source.sourceId,
      authorship: source.authorship,
      review: source.review,
      hashes,
      semanticDomainExclusions: intervalDomainExclusions(),
      coverage: coverage.finish(),
      candidateRecords: candidates,
      excludedRecords: Object.values(exclusions).reduce((a, b) => a + b, 0),
      exclusions,
      independentlyDecodedDiscardedClauses: discardedChecks,
      semanticFamiliesBySplit: families,
      tagsBySplit: tags,
      integrity,
    };
  } finally {
    audit.close();
    rmSync(directory, { recursive: true, force: true });
  }
}

if (import.meta.main) {
  const output = process.argv[2];

  if (!output || process.argv.length !== 3) {
    throw new Error("Usage: source-audit.ts NEW_REPORT.json");
  }

  const path = resolve(output);

  if (existsSync(path)) throw new Error(`Refusing to overwrite: ${path}`);

  const report = auditConstructionSource((n) =>
    console.error(`audited ${n.toLocaleString()} source candidates`),
  );

  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, `${JSON.stringify(report, null, 2)}\n`, { flag: "wx" });

  console.log(
    JSON.stringify(
      {
        records: report.integrity.records,
        passed: report.integrity.passed,
        coveragePassed: report.coverage.passed,
        missingCoverageCells: report.coverage.missing.length,
        violations: report.integrity.violations,
        exclusions: report.excludedRecords,
        output: path,
      },
      null,
      2,
    ),
  );

  if (!report.integrity.passed || !report.coverage.passed) process.exitCode = 1;
}
