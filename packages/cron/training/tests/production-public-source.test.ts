import { Database } from "bun:sqlite";
import { expect, test } from "bun:test";
import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import type { CorpusRecord } from "../src/corpus/merge";
import { ProtectedText } from "../src/corpus/production/protected-text";
import {
  buildPublicSource,
  PUBLIC_CONFIG_PATH,
  type PublicSourceConfig,
  visitBzipLines,
} from "../src/corpus/production/public-source";
import {
  diskPreflight,
  hashArtifact,
  publishDirectory,
} from "../src/corpus/production/publication";
import { fileHash } from "../src/corpus/production/source-hashes";
import { tempDir } from "./helpers/temp-dir";

test("decompression consumes fast empty streams and preserves CRLF, split Unicode, final lines and callback failures", async () => {
  const directory = tempDir("cron-source-lines-");
  const path = join(directory, "lines.bz2");

  for (const expected of [[], [""], ["α\t☺️", "末尾"], ["x".repeat(65_535) + "𐍈é", "end"]]) {
    const contents = expected.length ? `${expected.join("\r\n")}\r\n` : "";

    writeFileSync(path, execFileSync("bzip2", ["-c"], { input: contents }));

    for (let repetition = 0; repetition < 3; repetition++) {
      const actual: string[] = [];

      await visitBzipLines(path, (line, number) => {
        expect(number).toBe(actual.length + 1);
        actual.push(line);
      });

      expect(actual).toEqual(expected);
    }
  }

  writeFileSync(path, execFileSync("bzip2", ["-c"], { input: "first\nlast" }));
  const lines: string[] = [];

  await visitBzipLines(path, (line) => lines.push(line));
  expect(lines).toEqual(["first", "last"]);

  await expect(
    visitBzipLines(path, () => {
      throw new Error("callback sentinel");
    }),
  ).rejects.toThrow("callback sentinel");
});

function sentence(id: number, text: string, owner = "fixture-owner") {
  return `${id}\teng\t${text}\t${owner}\t2026-01-01 00:00:00\t2026-02-01 00:00:00`;
}

function fixture(lines: string[], links: string[] = []) {
  const directory = tempDir("cron-public-source-");
  const config = JSON.parse(readFileSync(PUBLIC_CONFIG_PATH, "utf8")) as PublicSourceConfig;

  config.sourceId = "public-fixture";

  for (const [key, text] of [
    ["sentences", lines.join("\n") + "\n"],
    ["links", links.length ? links.join("\n") + "\n" : ""],
  ] as const) {
    const path = join(directory, `${key}.bz2`);

    writeFileSync(path, execFileSync("bzip2", ["-c"], { input: text }));
    config[key].file = path;
    config[key].sha256 = fileHash(path);
  }

  const configPath = join(directory, "config.json");

  writeFileSync(configPath, JSON.stringify(config));

  return {
    directory,
    config,
    configPath,
    output: join(directory, "published"),
    protectedText: new ProtectedText([]),
  };
}

function records(output: string): CorpusRecord[] {
  return readFileSync(join(output, "corpus.jsonl"), "utf8")
    .trim()
    .split("\n")
    .map((line) => JSON.parse(line) as CorpusRecord);
}

test("the full public importer retains attribution and groups links, value variants, and duplicate aliases before splitting", async () => {
  const f = fixture(
    [
      sentence(201, "Tom owns 2 red bicycles.", "duplicate-owner"),
      sentence(104, "Please run this check every day."),
      sentence(102, "Mary owns 8 red bicycles."),
      sentence(105, "Please open the large box."),
      sentence(101, "Tom owns 2 red bicycles.", "canonical-owner"),
      sentence(103, "Please lift the wooden lid."),
      sentence(106, "Several bronze gears filled the crate.", "\\N"),
    ],
    ["103\t104", "104\t105", "201\t102"],
  );

  const result = await buildPublicSource(f);

  expect(result.records).toBe(5);
  expect(result.dropped).toEqual({ timeWord: 1, "normalized-duplicate": 1 });
  const rows = records(f.output);
  const a = rows.find((r) => r.provenance!.sourceRecordId === "101")!;
  const b = rows.find((r) => r.provenance!.sourceRecordId === "102")!;

  expect(a.provenance!.groupId).toBe(b.provenance!.groupId);
  expect(a.split).toBe(b.split);
  expect(a.provenance!.attribution).toContain("canonical-owner");
  expect(a.provenance!.attribution).toContain("https://tatoeba.org/en/sentences/show/101");
  expect(rows.find((r) => r.id.endsWith("/201"))).toBeUndefined();
  const linked = rows.filter((r) => ["103", "105"].includes(r.provenance!.sourceRecordId!));

  expect(linked.map((r) => r.provenance!.groupId)).toEqual(["component-103", "component-103"]);
  expect(result.concentration.unowned).toBe(1);

  for (const row of rows) {
    expect(row).toMatchObject({
      speechAct: "none",
      intent: null,
      provenance: { authorship: "public-source", license: "CC-BY-2.0-FR" },
    });
  }

  const db = new Database(join(f.output, "source.sqlite"), { readonly: true });

  try {
    expect(
      db.query<{ original: string }, []>("SELECT original FROM sentences WHERE id=101").get()!
        .original,
    ).toBe("Tom owns 2 red bicycles.");
  } finally {
    db.close();
  }

  expect(result.artifacts["corpus.jsonl"]!.sha256).toBe(
    await hashArtifact(join(f.output, "corpus.jsonl")),
  );

  // Reordered source input must retain the same representative and exact output.
  const reversed = fixture(
    [
      sentence(106, "Several bronze gears filled the crate.", "\\N"),
      sentence(103, "Please lift the wooden lid."),
      sentence(101, "Tom owns 2 red bicycles.", "canonical-owner"),
      sentence(105, "Please open the large box."),
      sentence(102, "Mary owns 8 red bicycles."),
      sentence(104, "Please run this check every day."),
      sentence(201, "Tom owns 2 red bicycles.", "duplicate-owner"),
    ],
    ["201\t102", "104\t105", "103\t104"],
  );

  const repeat = await buildPublicSource(reversed);

  expect(repeat.artifacts["corpus.jsonl"]!.sha256).toBe(result.artifacts["corpus.jsonl"]!.sha256);
});

test("protection excludes the whole related component, including twins whose wording is not protected", async () => {
  const f = fixture(
    [
      sentence(10, "Tom carries 4 heavy boxes."),
      sentence(11, "Mary carries 9 heavy boxes."),
      sentence(12, "These cartons are very heavy."),
      sentence(20, "Several bronze gears filled the crate."),
    ],
    ["11\t12"],
  );

  f.protectedText = new ProtectedText(["Tom carries 4 heavy boxes."]);
  const result = await buildPublicSource(f);

  expect(records(f.output).map((r) => r.provenance!.sourceRecordId)).toEqual(["20"]);

  expect(result.dropped).toEqual({
    "protected-wording": 1,
    "protected-related-group": 2,
  });
});

test("corrupt or duplicate-ID input never publishes a partial source or damages a prior version", async () => {
  const f = fixture([sentence(1, "Several bronze gears filled the crate.")]);

  writeFileSync(f.config.sentences.file, "not bzip2");
  await expect(buildPublicSource(f)).rejects.toThrow("checksum mismatch");
  expect(existsSync(f.output)).toBe(false);
  f.config.sentences.sha256 = fileHash(f.config.sentences.file);
  writeFileSync(f.configPath, JSON.stringify(f.config));
  await expect(buildPublicSource(f)).rejects.toThrow("bunzip2 failed");

  expect(
    readdirSync(f.directory).some((name) => name.includes("building-") || name.endsWith(".lock")),
  ).toBe(false);

  const duplicate = fixture([
    sentence(1, "Several bronze gears filled the crate."),
    sentence(1, "Another wooden crate held the gears."),
  ]);

  await expect(buildPublicSource(duplicate)).rejects.toThrow();
  expect(existsSync(duplicate.output)).toBe(false);
  mkdirSync(f.output);
  writeFileSync(join(f.output, "keep.txt"), "prior version");
  await expect(buildPublicSource(f)).rejects.toThrow("Refusing to overwrite");
  expect(readFileSync(join(f.output, "keep.txt"), "utf8")).toBe("prior version");
});

test("staged publication is invisible until success and an active version lock prevents another writer", async () => {
  const directory = tempDir("cron-publication-");
  const output = join(directory, "v1");

  const result = await publishDirectory(output, 1, async (staging) => {
    expect(existsSync(output)).toBe(false);

    await expect(
      publishDirectory(output, 1, async () => {
        throw new Error("must not run");
      }),
    ).rejects.toThrow();

    writeFileSync(join(staging, "manifest.json"), "{}\n");

    return "published";
  });

  expect(result).toBe("published");
  expect(readFileSync(join(output, "manifest.json"), "utf8")).toBe("{}\n");
  const failed = join(directory, "v2");

  await expect(
    publishDirectory(failed, 1, async (staging) => {
      writeFileSync(join(staging, "partial"), "unfinished");
      throw new Error("injected failure");
    }),
  ).rejects.toThrow("injected failure");

  expect(existsSync(failed)).toBe(false);
  expect(readdirSync(directory)).toEqual(["v1"]);

  expect(() => diskPreflight(directory, Number.MAX_SAFE_INTEGER)).toThrow(
    "Insufficient disk space",
  );
});

test("a changed source configuration invalidates publication rather than attaching new metadata to old rows", async () => {
  const f = fixture([sentence(1, "Several bronze gears filled the crate.")]);
  const pending = buildPublicSource(f);

  f.config.sourceId = "changed-after-start";
  writeFileSync(f.configPath, JSON.stringify(f.config));
  await expect(pending).rejects.toThrow("Inputs or implementation changed");
  expect(existsSync(f.output)).toBe(false);
});

test("source import excludes normalization-exposed schedule evidence and unstable punctuation before assigning easy-negative gold", async () => {
  const f = fixture([
    sentence(1, "The valve closes at six p.m."),
    sentence(2, "Please put these boxes aside . . ."),
    sentence(3, "Several bronze gears filled the crate."),
  ]);

  const result = await buildPublicSource(f);

  expect(result.records).toBe(1);
  expect(result.dropped).toEqual({ timeWord: 1, normalization: 1 });
  expect(records(f.output)[0]!.provenance!.sourceRecordId).toBe("3");
});
