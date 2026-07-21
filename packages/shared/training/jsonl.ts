/** Bounded UTF-8 JSONL input and output, without a whole serialized corpus. */
import { createHash } from "node:crypto";
import { closeSync, mkdirSync, openSync, readSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { StringDecoder } from "node:string_decoder";

export function* readJsonl(path: string): Generator<unknown> {
  const file = openSync(path, "r");
  const buffer = Buffer.alloc(65536);
  const decoder = new StringDecoder("utf8");

  let pending = "";

  try {
    for (;;) {
      const count = readSync(file, buffer);

      pending += count ? decoder.write(buffer.subarray(0, count)) : decoder.end();
      let end: number;

      while ((end = pending.indexOf("\n")) !== -1) {
        const line = pending.slice(0, end).trim();

        pending = pending.slice(end + 1);
        if (Buffer.byteLength(line, "utf8") > 1048576) throw Error("JSONL record exceeds one MiB");
        if (line) yield JSON.parse(line);
      }

      if (Buffer.byteLength(pending, "utf8") > 1048576) throw Error("JSONL record exceeds one MiB");
      if (!count) break;
    }

    if (pending.trim()) yield JSON.parse(pending);
  } finally {
    closeSync(file);
  }
}

export function writeJsonl(
  path: string,
  records: Iterable<unknown>,
): { sha256: string; rows: number } {
  mkdirSync(dirname(path), { recursive: true });

  const file = openSync(path, "wx");
  const digest = createHash("sha256");

  let buffer = "";
  let rows = 0;

  function flush() {
    if (buffer) {
      writeFileSync(file, buffer);
      digest.update(buffer);
      buffer = "";
    }
  }

  try {
    for (const record of records) {
      const line = JSON.stringify(record) + "\n";

      if (Buffer.byteLength(line, "utf8") > 1048576) throw Error("JSONL record exceeds one MiB");
      buffer += line;
      rows++;
      if (buffer.length >= 65536) flush();
    }

    flush();
  } finally {
    closeSync(file);
  }

  return { sha256: digest.digest("hex"), rows };
}
