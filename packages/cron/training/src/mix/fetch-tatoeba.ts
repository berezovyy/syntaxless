import { type ChildProcess, spawn } from "node:child_process";
/**
 * Download Tatoeba English, filter as easy-no, pin sha256 + sentences.txt.
 *
 * Usage: bun run src/mix/fetch-tatoeba.ts
 */
import { createHash } from "node:crypto";
import { createReadStream, existsSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createInterface } from "node:readline";

import { humanEvalTexts } from "../slots/leak";
import { EASY_FILTER_VERSION, type EasyDropReason, easyNoDropReason } from "./filter";
import { TATOEBA_DIR, TATOEBA_MANIFEST, TATOEBA_SENTENCES } from "./tatoeba";

export const TATOEBA_URL =
  "https://downloads.tatoeba.org/exports/per_language/eng/eng_sentences.tsv.bz2";

export const TATOEBA_LICENSE = "CC-BY 2.0 FR";
export const TATOEBA_LIMIT = 50_000;
export const TATOEBA_MIN_KEEP = 10_000;

async function sha256File(path: string): Promise<string> {
  const hash = createHash("sha256");

  for await (const chunk of createReadStream(path)) hash.update(chunk);

  return hash.digest("hex");
}

function processCompleted(child: ChildProcess, name: string): Promise<void> {
  return new Promise((resolve, reject) => {
    child.once("error", reject);

    child.once("close", (code, signal) => {
      if (code === 0) resolve();
      else reject(new Error(`${name} failed with ${signal ?? code}`));
    });
  });
}

/** Filter a TSV stream and count the actual rejection reason for each row. */
export async function filterTatoebaLines(
  lines: AsyncIterable<string> | Iterable<string>,
  blocked: ReadonlySet<string>,
): Promise<{ kept: string[]; dropped: Record<EasyDropReason | "language", number> }> {
  const seen = new Set<string>();
  const kept: string[] = [];

  const dropped: Record<EasyDropReason | "language", number> = {
    language: 0,
    shape: 0,
    length: 0,
    normalization: 0,
    timeShaped: 0,
    timeWord: 0,
    reserved: 0,
    blocked: 0,
    duplicate: 0,
  };

  for await (const line of lines) {
    const parts = line.split("\t");
    const language = parts[1];
    const text = parts[2];

    if (language !== "eng" || text === undefined || text.length === 0) {
      dropped.language += 1;
      continue;
    }

    const reason = easyNoDropReason(text, { blocked, seen });

    if (reason !== null) {
      dropped[reason] += 1;
      continue;
    }

    kept.push(text);
  }

  return { kept, dropped };
}

export async function fetchAndPinTatoeba(archivePath?: string): Promise<{
  kept: number;
  dropped: Record<string, number>;
  sha256: string;
}> {
  const archive = archivePath ?? join(tmpdir(), "tatoeba-eng-sentences.tsv.bz2");

  if (!existsSync(archive)) {
    if (archivePath !== undefined) throw new Error(`Tatoeba archive not found: ${archivePath}`);

    const curl = spawn("curl", ["-f", "-sS", "-L", TATOEBA_URL, "-o", archive], {
      stdio: ["ignore", "inherit", "inherit"],
    });

    try {
      await processCompleted(curl, "curl Tatoeba");
    } catch (error) {
      rmSync(archive, { force: true });
      throw error;
    }
  }

  const sha256 = await sha256File(archive);

  const bunzip = spawn("bunzip2", ["-dc", archive], {
    stdio: ["ignore", "pipe", "inherit"],
  });

  const completed = processCompleted(bunzip, "bunzip2 Tatoeba");
  const lines = createInterface({ input: bunzip.stdout, crlfDelay: Infinity });
  let filtered: Awaited<ReturnType<typeof filterTatoebaLines>>;

  try {
    [filtered] = await Promise.all([
      filterTatoebaLines(lines, new Set(humanEvalTexts())),
      completed,
    ]);
  } finally {
    lines.close();
    if (bunzip.exitCode === null) bunzip.kill();
  }

  const { kept, dropped } = filtered;
  const stride = Math.max(1, Math.floor(kept.length / TATOEBA_LIMIT));
  const selected = kept.filter((_, index) => index % stride === 0).slice(0, TATOEBA_LIMIT);

  if (selected.length < TATOEBA_MIN_KEEP) {
    throw new Error(`Tatoeba keep ${selected.length} < ${TATOEBA_MIN_KEEP} after filter`);
  }

  mkdirSync(TATOEBA_DIR, { recursive: true });
  writeFileSync(TATOEBA_SENTENCES, `${selected.join("\n")}\n`);

  writeFileSync(
    TATOEBA_MANIFEST,
    `${JSON.stringify(
      {
        url: TATOEBA_URL,
        license: TATOEBA_LICENSE,
        attribution: "Sentences from Tatoeba (https://tatoeba.org), CC-BY 2.0 FR.",
        retrievedAt: new Date().toISOString().slice(0, 10),
        sha256,
        filterVersion: EASY_FILTER_VERSION,
        kept: selected.length,
        dropped,
        limit: TATOEBA_LIMIT,
      },
      null,
      2,
    )}\n`,
  );

  return { kept: selected.length, dropped, sha256 };
}

if (import.meta.main) {
  fetchAndPinTatoeba()
    .then((result) => {
      process.stdout.write(`tatoeba kept ${result.kept} sha256 ${result.sha256}\n`);
    })
    .catch((error: unknown) => {
      process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
      process.exit(1);
    });
}
