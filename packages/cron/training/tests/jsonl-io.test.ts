import { expect, test } from "bun:test";
import { spawnSync } from "node:child_process";
import { randomBytes } from "node:crypto";
import { readdirSync, readFileSync, readlinkSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { gzipSync } from "node:zlib";

import { jsonlByteChunks, jsonlReader, writeGzip } from "../src/corpus/jsonl-io";
import { tempDir } from "./helpers/temp-dir";

function inputDescriptors(path: string): string[] {
  return readdirSync("/proc/self/fd").filter((name) => {
    try {
      return readlinkSync(`/proc/self/fd/${name}`) === path;
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") return false;
      throw error;
    }
  });
}

test.skipIf(process.platform !== "linux")(
  "a plain or gzip consumer failure preserves the error and closes its source",
  async () => {
    const directory = tempDir("syntaxless-gzip-failure-");
    const payload = randomBytes(256 * 1024).toString("hex");
    const failure = new Error("consumer failed");

    for (const gzip of [false, true]) {
      const path = join(directory, gzip ? "input.jsonl.gz" : "input.jsonl");

      writeFileSync(path, gzip ? gzipSync(payload) : payload);

      try {
        for await (const _chunk of jsonlByteChunks(path)) throw failure;
        throw new Error("no chunk was read");
      } catch (error) {
        expect(error).toBe(failure);
      }

      expect(inputDescriptors(path)).toEqual([]);
    }
  },
);

test(
  "JSONL lines preserve split Unicode, CRLF, " + "blank lines and a final line without a newline",
  async () => {
    const directory = tempDir("syntaxless-jsonl-lines-");
    const middle = "x".repeat(65_534) + "🍀";
    const text = `\n${middle}\r\nlast`;

    for (const gzip of [false, true]) {
      const path = join(directory, gzip ? "input.jsonl.gz" : "input.jsonl");

      writeFileSync(path, gzip ? gzipSync(text) : text);
      const lines: string[] = [];

      {
        await using reader = jsonlReader(jsonlByteChunks(path));

        for await (const line of reader.lines) lines.push(line);
      }

      expect(lines).toEqual(["", middle, "last"]);
      if (process.platform === "linux") expect(inputDescriptors(path)).toEqual([]);
    }
  },
);

test(
  "gzip writing reports an earlier output failure " +
    "without an unhandled rejection or stalled drain",
  () => {
    const path = join(tempDir("syntaxless-gzip-output-error-"), "missing", "output.gz");
    const module = fileURLToPath(new URL("../src/corpus/jsonl-io.ts", import.meta.url));

    // A child isolates unhandled background errors and makes a stalled write bounded.
    const result = spawnSync(
      process.execPath,
      [
        "--eval",
        `
        import { writeGzip } from ${JSON.stringify(module)};
        import { setTimeout } from "node:timers/promises";
        process.on("unhandledRejection", (error) => {
          console.error(error);
          process.exitCode = 1;
        });
        try {
          await writeGzip(${JSON.stringify(path)}, (async function* () {
            await setTimeout(50);
            yield Buffer.alloc(4 * 1024 * 1024);
          })());
          throw new Error("Missing output directory was accepted");
        } catch (error) {
          if (error.code !== "ENOENT") throw error;
        }
        console.log("output failure observed");
      `,
      ],
      { encoding: "utf8", timeout: 2000 },
    );

    expect(result.error).toBeUndefined();
    expect(result.status).toBe(0);
    expect(result.stdout).toBe("output failure observed\n");
    expect(result.stderr).toBe("");
  },
);

test("failed gzip output preserves a prior artifact and finishes cleanup", async () => {
  const path = join(tempDir("syntaxless-gzip-existing-"), "output.gz");
  const prior = Buffer.from("prior artifact");

  writeFileSync(path, prior);
  await expect(writeGzip(path, [Buffer.alloc(256 * 1024)])).rejects.toThrow("EEXIST");
  expect(readFileSync(path)).toEqual(prior);
  if (process.platform === "linux") expect(inputDescriptors(path)).toEqual([]);
});
