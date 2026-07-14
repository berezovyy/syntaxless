/** Inspect every clean catalog candidate before grouped production selection. */
import { existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";

import { KINDS } from "../../kinds";
import { CorpusAudit } from "./audit";
import { CATALOG_SOURCE_ID, catalogCandidates } from "./catalog";
import { loadProtectedText } from "./protected-text";
import { diskPreflight } from "./publication";
import { corpusCodeHashes, fileHash } from "./source-hashes";

export function auditCatalogSource(onProgress?: (n: number) => void) {
  const hashes = corpusCodeHashes();
  const protectedText = loadProtectedText();

  // Two disk identity indices plus IDs; no full JSONL payload is materialized here.
  const potential = KINDS.reduce(
    (n, kind) => n + kind.intents.filter((i) => i !== null).length * kind.templates.length,
    0,
  );

  diskPreflight(tmpdir(), potential * 700);
  const directory = mkdtempSync(join(tmpdir(), "cron-catalog-audit-"));

  const audit = new CorpusAudit({
    enforceMix: false,
    protectedText,
    indexPath: join(directory, "identities.sqlite"),
  });

  const templates: Record<string, { attempted: number; rendered: number; eligible: number }> = {};
  const intents = new Map<string, { rendered: number; eligible: number }>();

  let rendered = 0;
  let excluded = 0;

  try {
    for (const candidate of catalogCandidates(KINDS, (attempt) => {
      const template = (templates[attempt.constructionId] ??= {
        attempted: 0,
        rendered: 0,
        eligible: 0,
      });

      template.attempted++;
      const key = `${attempt.kind}/${attempt.intentId}`;
      const intent = intents.get(key) ?? { rendered: 0, eligible: 0 };

      intents.set(key, intent);

      if (attempt.rendered) {
        template.rendered++;
        intent.rendered++;
      }
    })) {
      rendered++;
      if (rendered % 100_000 === 0) onProgress?.(rendered);

      if (protectedText.matches(candidate.record.text)) {
        excluded++;
        continue;
      }

      templates[candidate.constructionId]!.eligible++;
      intents.get(`${candidate.kind}/${candidate.intentId}`)!.eligible++;
      // Inventory-only: this dummy split must not be reused as production ownership.
      audit.add({ ...candidate.record, split: "train" });
    }

    const integrity = audit.finish();

    if (
      JSON.stringify(hashes) !== JSON.stringify(corpusCodeHashes()) ||
      protectedText.sources.some((source) => fileHash(source.file) !== source.sha256)
    ) {
      throw new Error("Catalog implementation or protected inputs changed during audit");
    }

    const uncovered = [...intents].filter(([, n]) => n.eligible === 0);

    return {
      version: 1,
      scope: "complete-systematic-candidate-inventory-not-split-or-production-ready",
      sourceId: CATALOG_SOURCE_ID,
      authorship: "programmatic",
      hashes,
      attemptedTemplateValues: potential,
      renderedCandidates: rendered,
      protectedExclusions: excluded,
      templates,
      intentCoverage: {
        catalogKindIntents: intents.size,
        eligibleKindIntents: intents.size - uncovered.length,
        uncovered: uncovered.map(([key, counts]) => ({ key, ...counts })),
      },
      isolationLimitation:
        "All candidates have a temporary train split for integrity analysis. This is NOT a split-isolation certificate: construction/fingerprint unions and cross-source ownership must be assigned before production selection.",
      integrity,
    };
  } finally {
    audit.close();
    rmSync(directory, { recursive: true, force: true });
  }
}

if (import.meta.main) {
  if (process.argv.length !== 3) throw new Error("Usage: catalog-audit.ts NEW_REPORT.json");
  const output = resolve(process.argv[2]!);

  if (existsSync(output)) throw new Error(`Refusing to overwrite: ${output}`);

  const report = auditCatalogSource((n) =>
    console.error(`catalog: ${n.toLocaleString()} rendered candidates`),
  );

  mkdirSync(dirname(output), { recursive: true });
  writeFileSync(output, `${JSON.stringify(report, null, 2)}\n`, { flag: "wx" });

  console.log(
    JSON.stringify(
      {
        records: report.integrity.records,
        passed: report.integrity.passed,
        violations: report.integrity.violations,
        intentCoverage: report.intentCoverage,
        output,
      },
      null,
      2,
    ),
  );

  if (!report.integrity.passed || report.intentCoverage.uncovered.length) process.exitCode = 1;
}
