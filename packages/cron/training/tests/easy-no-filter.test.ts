import { expect, test } from "bun:test";
import { writeFileSync } from "node:fs";
import { join } from "node:path";

import { normalizeText, RESERVED_PHRASES } from "../src/background";
import { easyKindExample } from "../src/mix/easy";
import { acceptEasyNo, easyNoDropReason, hasTimeWord, timeShapedNumber } from "../src/mix/filter";
import { loadEasySource } from "../src/mix/tatoeba";
import { oracleDecode, oracleMatches, type OracleRecord } from "../src/oracle";
import { humanEvalTexts, humanWordingHits } from "../src/slots/leak";
import { tempDir } from "./helpers/temp-dir";

test("filter drops Monday / every / 8pm / 12:30 and keeps the required fixtures", () => {
  expect(acceptEasyNo("Monday is a fine day for a picnic with friends.")).toBe(false);
  expect(hasTimeWord("Monday is a fine day for a picnic with friends.")).toBe(true);
  expect(acceptEasyNo("Please do this every time you save the document.")).toBe(false);
  expect(hasTimeWord("Please do this every time you save the document.")).toBe(true);
  expect(timeShapedNumber("Meet me at 8pm near the office lobby.")).toBe(true);
  expect(acceptEasyNo("Meet me at 8pm near the office lobby.")).toBe(false);
  expect(timeShapedNumber("The train is listed at 12:30 in the booklet.")).toBe(true);
  expect(acceptEasyNo("The train is listed at 12:30 in the booklet.")).toBe(false);

  expect(acceptEasyNo("Please close the window.")).toBe(true);
  expect(easyNoDropReason("Please close the window.")).toBeNull();
  expect(acceptEasyNo("Room 10 contains 8 chairs.")).toBe(true);
  expect(easyNoDropReason("Room 10 contains 8 chairs.")).toBeNull();
});

test("a produced E record is all-O, speechAct none, intent null", () => {
  const example = easyKindExample("Please close the window.", 0);

  expect(example.intent).toBeNull();
  expect(example.example.speechAct).toBe("none");
  expect(example.example.spans.every((span) => span.label === "O")).toBe(true);
  expect(example.example.spans.every((span) => span.actionable === false)).toBe(true);

  const record: OracleRecord = {
    id: "easy-no",
    source: "kind-seed",
    text: example.example.text,
    spans: example.example.spans,
    intent: null,
    speechAct: "none",
  };

  expect(oracleDecode(record).ok).toBe(false);
  expect(oracleMatches(record)).toBe(true);
});

test("pinned sentences are the easy-no source once a sufficient corpus exists", () => {
  const path = join(tempDir("syntaxless-easy-source-"), "sentences.txt");

  const sentences = Array.from({ length: 10_000 }, (_, index) => {
    const tag = index
      .toString()
      .padStart(4, "0")
      .replace(/\d/g, (digit) => String.fromCharCode(97 + Number(digit)));

    return `Please inspect the wooden panel marked tag${tag}.`;
  });

  writeFileSync(path, `${sentences.join("\n")}\n`);
  const source = loadEasySource({}, path);

  expect(source.kind).toBe("tatoeba");
  expect(source.sentences).toEqual(sentences);
});

test("missing or insufficient pinned corpora use bootstrap and honor blocked texts", () => {
  const path = join(tempDir("syntaxless-easy-fallback-"), "sentences.txt");
  const blocked = "Please close the window.";
  const options = { blocked: new Set([normalizeText(blocked)]) };
  const missing = loadEasySource(options, path);

  expect(missing.kind).toBe("bootstrap");
  expect(missing.sentences.length).toBeGreaterThanOrEqual(2000);
  expect(missing.sentences).not.toContain(blocked);
  writeFileSync(path, "Please open the wooden door.\n");
  expect(loadEasySource(options, path)).toEqual(missing);
});

test("easy source does not leak human-v or reserved phrases", () => {
  const human = humanEvalTexts();

  expect(human.length).toBeGreaterThan(100);
  const source = loadEasySource();

  for (const text of source.sentences) {
    const normalized = normalizeText(text);

    expect(RESERVED_PHRASES.has(normalized)).toBe(false);
    expect(humanWordingHits(text, human)).toEqual([]);
  }
});

test("easy negatives are checked after runtime normalization, including dotted meridiems and Unicode ordinals", () => {
  expect(easyNoDropReason("The valve closes at six p.m.")).toBe("timeWord");
  expect(easyNoDropReason("The valve closes at 9 p. m.")).toBe("timeShaped");
  expect(easyNoDropReason("This is the 89ᵗʰ entry in the catalogue.")).toBe("timeShaped");
  expect(easyNoDropReason("Please put these boxes aside . . .")).toBe("normalization");
  expect(easyNoDropReason("Please put these boxes aside...")).toBeNull();
});
