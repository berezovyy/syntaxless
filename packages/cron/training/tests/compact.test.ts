import { expect, test } from "bun:test";
import { createHash } from "node:crypto";
import { mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync } from "node:fs";
import { join } from "node:path";

import { compile } from "../../src/compiler/cron.js";
import { loadCompactProtection } from "../src/compact-protection.js";
import { HOURLY_MINUTE } from "../src/compact-scope.js";
import { COMPACT_RECIPE, prepareCompactCorpus, writeCompactCorpus } from "../src/compact.js";
import type { CorpusRecord } from "../src/corpus/merge.js";
import { KINDS } from "../src/kinds.js";
import { oracleAgrees, oracleDecode } from "../src/oracle.js";
import { Sentence } from "../src/sentence.js";

function digest(bytes: string | Buffer): string {
  return createHash("sha256").update(bytes).digest("hex");
}

test(
  "numeric clock supervision preserves literal hours and explicit meridiem at midnight, " +
    "noon, and clock endpoints",
  () => {
    const kind = KINDS.find((kind) => kind.id === "time-of-day")!;

    for (const hour of [0, 9, 12, 23]) {
      for (const style of ["digits", "24h", "hour-meridiem", "hour-unqualified"]) {
        const intent = {
          frequency: { unit: "day" as const, interval: 1 },
          time: {
            hour,
            minute: style.startsWith("hour-") ? 0 : 23,
          },
        };

        const sentence = new Sentence();

        sentence.clause();
        expect(kind.render(sentence, intent, style)).toBe(true);
        expect(sentence.spans.some((span) => span.label === "HOUR_LIT")).toBe(true);

        expect(
          oracleAgrees(
            oracleDecode({
              id: style,
              ...sentence,
              text: sentence.text,
              spans: sentence.spans,
              intent,
              speechAct: "request",
              source: "kind-seed",
            }),
            intent,
          ),
        ).toBe(true);

        if (style === "hour-unqualified") expect(sentence.text).toBe(`at ${hour}`);
      }
    }
  },
);

test("hourly minute annotations reconstruct minute endpoints without inventing an hour", () => {
  for (const minute of [0, 59]) {
    for (const { id } of HOURLY_MINUTE.templates) {
      const intent = {
        frequency: { unit: "hour" as const, interval: 1 },
        time: { minute },
      };

      const sentence = new Sentence();

      sentence.clause();
      expect(HOURLY_MINUTE.render(sentence, intent, id)).toBe(true);

      expect(
        oracleDecode({
          id,
          text: sentence.text,
          spans: sentence.spans,
          intent,
          speechAct: "request",
          source: "kind-seed",
        }),
      ).toEqual({ ok: true, intent });

      expect(compile(intent)).toEqual({
        supported: true,
        expression: `${minute} * * * *`,
      });
    }
  }
});

test(
  "compact preparation separates protected " + "wording and splits and refuses overwrites",
  async () => {
    const directory = mkdtempSync("/tmp/syntaxless-cron-compact-test-");

    try {
      const protection = loadCompactProtection();

      const archive = JSON.parse(
        readFileSync(
          new URL("../../evaluation/historical-acceptance.json", import.meta.url),
          "utf8",
        ),
      );

      expect(protection.matches(archive.partitions[0].records[0].text)).toBe(true);
      const result = await prepareCompactCorpus(directory);
      const text = readFileSync(result.corpus, "utf8");

      const rows: CorpusRecord[] = text
        .trim()
        .split("\n")
        .map((line) => JSON.parse(line));

      expect(result.proof.sha256).toBe(digest(text));
      expect(result.proof.trainingEligible).toBe(true);
      expect(result.proof.consumption).toEqual(protection.consumption);
      expect(rows.length).toBeGreaterThan(0);
      expect(rows.length).toBeLessThanOrEqual(COMPACT_RECIPE.maximumRows);
      expect(new Set(rows.map((row) => row.text)).size).toBe(rows.length);
      expect(rows.filter((row) => protection.matches(row.text))).toEqual([]);
      const incomplete = join(directory, "incomplete");

      mkdirSync(incomplete);

      await expect(
        writeCompactCorpus(
          incomplete,
          rows.filter(
            (row) => row.intent?.time?.hour !== undefined || row.intent?.time === undefined,
          ),
          protection.sources,
        ),
      ).rejects.toThrow("intent scope time/hour-minute");

      expect(readdirSync(incomplete)).toEqual([]);
      const missingNoise = join(directory, "missing-noise");

      mkdirSync(missingNoise);

      await expect(
        writeCompactCorpus(
          missingNoise,
          rows.filter((row) => !row.template?.includes("/spelling-")),
          protection.sources,
        ),
      ).rejects.toThrow("paired noise spelling-deletion");

      expect(readdirSync(missingNoise)).toEqual([]);
      const groups = new Map<string, string>();

      for (const row of rows) {
        const split = groups.get(row.fingerprint);

        if (split !== undefined && split !== row.split) {
          throw new Error(`Construction group crossed splits: ${row.fingerprint}`);
        }

        groups.set(row.fingerprint, row.split);
      }

      for (const split of ["train", "validation", "heldout"] as const) {
        const manifest = JSON.parse(readFileSync(`${result.prefix}-${split}.json`, "utf8"));

        expect(manifest.sequenceMetadata.rows).toBe(
          rows.filter((row) => row.split === split).length,
        );
      }

      rmSync(incomplete, { recursive: true });
      rmSync(missingNoise, { recursive: true });

      const before = readdirSync(directory)
        .sort()
        .map((name) => [name, digest(readFileSync(join(directory, name)))]);

      await expect(prepareCompactCorpus(directory)).rejects.toThrow("must be empty");

      expect(
        readdirSync(directory)
          .sort()
          .map((name) => [name, digest(readFileSync(join(directory, name)))]),
      ).toEqual(before);
    } finally {
      rmSync(directory, { recursive: true });
    }
  },
  30000,
);
