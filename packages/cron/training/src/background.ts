/**
 * Carrier prose for the corpus.
 *
 * Carriers give positive expressions an ordinary-language context and supply
 * the all-`O` negatives. Every candidate is normalized and rejected while it
 * collides with {@link RESERVED_PHRASES}; the reserved list is registered here
 * exactly once and reused by Phase 08's held-out split.
 */
import { ORDINAL_WORDS } from "../../src/parser/lexicon.js";
import type { Rng } from "./rng";

/** Lowercase and collapse whitespace, matching how reserved text is compared. */
export function normalizeText(text: string): string {
  return text.toLowerCase().split(/\s+/).filter(Boolean).join(" ");
}

/**
 * Wording that only the held-out renderer may emit. Train and validation
 * carriers must never produce one of these, so the held-out split tests
 * reordering rather than memorized phrases.
 */
export const RESERVED_PHRASES: ReadonlySet<string> = new Set([
  "remind me to",
  "please add",
  "the train leaves at",
  "could you arrange a reminder for",
  "our rehearsal begins at",
  "please put this in my diary for",
  "note that the office opens at",
  "the meeting starts at",
]);

const ACTORS = ["I", "we", "you", "they"] as const;
const MODALS = ["'ll", " will", " may", " might", " should"] as const;
const STATES = ["be out", "be back", "be available", "go back"] as const;
const PLACES = ["the office", "our clinic", "the store", "the library"] as const;
const OPENINGS = ["is open", "is closed", "opens", "closes"] as const;
const DETERMINERS = ["the", "our", "my"] as const;
const EVENTS = ["meeting", "appointment", "interview", "call", "lesson"] as const;
const PREDICATES = ["is", "starts", "is scheduled"] as const;
const LEADS = ["", "please", "could you", "can you", "I'd like to"] as const;
const ASIDES = ["", "", "", "please,", "note:", "could you check this:"] as const;
const NAMES = ["May", "Alex", "Jordan", "Riley", "Sam", "Taylor", "Casey"] as const;
const RECIPIENTS = ["me", "us", "the team", ...NAMES] as const;

const SUFFIXES = [
  "works for {who}",
  "for {who}",
  "if that works for {who}",
  "is the deadline",
  "please",
  "and back before dark",
] as const;

const NOUNS = [
  "file",
  "document",
  "report",
  "page",
  "section",
  "list",
  "table",
  "message",
  "draft",
  "option",
] as const;

const VERBS = [
  "open",
  "close",
  "read",
  "review",
  "print",
  "select",
  "send",
  "check",
  "copy",
  "approve",
] as const;

const ORDERS = [...ORDINAL_WORDS.filter((word) => word !== "fourth"), "next", "previous"] as const;

const MAX_ATTEMPTS = 100;

function terminate(text: string): string {
  return /[.!?]$/.test(text) ? text : `${text}.`;
}

function request(rng: Rng): string {
  const event = rng.pick(EVENTS);

  const asks = [
    `schedule a ${event}`,
    `book a ${event}`,
    `reserve the ${event}`,
    `book room ${rng.int(1, 50)}`,
    "set an alarm",
    "add a note",
    `send a reminder about the ${event}`,
  ];

  const lead = rng.pick(LEADS);
  const ask = rng.pick(asks);

  return lead ? `${lead} ${ask}` : ask;
}

function availability(rng: Rng): string {
  return `${rng.pick(ACTORS)}${rng.pick(MODALS)} ${rng.pick(STATES)}`;
}

function hours(rng: Rng): string {
  return `${rng.pick(PLACES)} ${rng.pick(OPENINGS)}`;
}

function event(rng: Rng): string {
  return `${rng.pick(DETERMINERS)} ${rng.pick(EVENTS)} ${rng.pick(PREDICATES)}`;
}

/** Ordinary prose that may precede a rendered expression. */
export function prefix(rng: Rng): string {
  for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
    const body = rng.pick([request(rng), availability(rng), hours(rng), event(rng)]);
    const aside = rng.pick(ASIDES);
    const text = aside ? `${aside} ${body}` : body;

    if (!RESERVED_PHRASES.has(normalizeText(text))) return text;
  }

  throw new Error("background.prefix: no reserved-free candidate");
}

/** Ordinary prose that may follow a rendered expression. */
export function suffix(rng: Rng): string {
  for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
    const text = rng.pick(SUFFIXES).replace("{who}", rng.pick(RECIPIENTS));

    if (!RESERVED_PHRASES.has(normalizeText(text))) return text;
  }

  throw new Error("background.suffix: no reserved-free candidate");
}

function sentenceBody(rng: Rng): string {
  const order = rng.pick(ORDERS);
  const noun = rng.pick(NOUNS);
  const verb = rng.pick(VERBS);
  const name = rng.pick(NAMES);
  const count = rng.int(1, 99);

  const templates = [
    `Please ${verb} the ${order} ${noun}`,
    `Could you ${verb} the ${noun} for ${name}`,
    `The ${order} ${noun} contains ${count} examples`,
    `The ${noun} has ${count} rows and ${rng.int(1, 31)} columns`,
    `The beginning of the ${noun} explains the format`,
    `At the end of the ${noun}, the author signs it`,
    `${name} wrote the ${order} ${noun}`,
    `Send the ${order} ${noun} to ${name}`,
    `Choose option ${count} from section ${rng.int(1, 12)}`,
    `The ${order} attempt succeeded`,
    `Each ${noun} needs a title`,
    `The field named ${rng.pick(["year", "timestamp", "date", "duration"])} contains a string`,
    `The word ${rng.pick(["midnight", "tomorrow", "morning", "weekend"])} appears in the glossary`,
    `${name} reviewed the ${order} ${noun} before the ` +
      rng.pick(["review", "meeting", "deadline"]),
  ];

  return rng.pick(templates);
}

/** A full all-`O` sentence for the negative family. */
export function sentence(rng: Rng): string {
  for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
    const text = terminate(sentenceBody(rng));

    if (!RESERVED_PHRASES.has(normalizeText(text))) return text;
  }

  throw new Error("background.sentence: no reserved-free candidate");
}
