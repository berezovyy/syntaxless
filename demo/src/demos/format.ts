import { compile, explain, parse, preload } from "../../../packages/format/src/index.ts";
import type { Context, FormatIntent } from "../../../packages/format/src/index.ts";
import { browserTimeZone, card, esc, jsonCard, options } from "../ui.ts";
import type { Demo } from "../ui.ts";

const LOCALES = ["en-US", "en-GB", "de-DE", "fr-FR", "uk-UA", "ja-JP", "hi-IN", "pt-BR"];
const ZONES = [browserTimeZone, "UTC", "America/New_York", "Europe/London", "Asia/Tokyo"];
const NUMBERS = [1234.5, -1234.5, 0, 0.125, 98000, 1 / 3, -0.5, 1e9];
const now = Date.now() - (Date.now() % 1000);

const DATES: [string, number][] = [
  ["now", now],
  ["2024-07-01T13:05Z", Date.UTC(2024, 6, 1, 13, 5)],
  ["2000-01-01T00:00Z", Date.UTC(2000, 0, 1)],
  ["2026-12-31T23:59:59Z", Date.UTC(2026, 11, 31, 23, 59, 59)],
];

function contextFor(
  intent: FormatIntent,
  opts: Record<string, string>,
): { context: Context; notes: string[] } {
  const context: Context = {};
  const notes: string[] = [];

  if (intent.locale) {
    notes.push(`locale <code>${esc(intent.locale)}</code> comes from the instruction`);
  } else {
    context.locale = opts.locale!;
  }

  if (intent.kind === "date-time") {
    if (intent.timeZone) {
      notes.push(`timezone <code>${esc(intent.timeZone)}</code> comes from the instruction`);
    } else {
      context.timeZone = opts.tz!;
    }
  }

  return { context, notes };
}

function customValue(raw: string, kind: string): number | null {
  const text = raw.trim();

  if (text === "") return null;

  if (kind === "date-time") {
    const n = /^-?\d+$/.test(text) ? Number(text) : Date.parse(text);

    return Number.isFinite(n) ? n : NaN;
  }

  return Number(text);
}

const VALUE_INPUT =
  `<input class="small wide" data-opt="value" ` +
  `placeholder="-42.5 or 2024-02-29T12:00Z" spellcheck="false">`;

export const formatDemo: Demo = {
  id: "format",
  pkg: "@syntaxless/format",
  blurb:
    "Number, currency, percentage and date/time instructions to validated <code>Intl</code> " +
    "formatters. Locale and timezone come from the instruction or from the caller context below, " +
    "and a conflict between them is refused.",
  placeholder: "euros, two decimals, negatives in parentheses",
  examples: [
    ["euros, two decimals, negatives in parentheses"],
    ["percent with one decimal"],
    ["US dollars shown as code, no grouping"],
    ["JPY shown as a name"],
    ["plain number with three decimals and no grouping"],
    ["long date and short time, 24-hour"],
    ["full date in de-DE"],
    ["short date and time in America/New_York"],
    ["dollars", true],
    ["scientific notation", true],
    ["round to the nearest 5 cents", true],
  ],
  controls: () =>
    `<label>Context locale <select data-opt="locale">${options(LOCALES)}</select></label>
     <label>Context timezone <select data-opt="tz">${options(ZONES)}</select></label>
     <label>Your value ${VALUE_INPUT}</label>`,
  preload: () => preload(),
  parse: (text) => parse(text),
  render(parsed, ctx) {
    if (!parsed.ok) return;
    const intent: FormatIntent = parsed.intent;
    const { context, notes } = contextFor(intent, ctx.opts);
    const compiled = compile(intent, context);

    if (!compiled.supported) {
      ctx.setStage({ ok: false, code: compiled.code, reason: compiled.reason });

      ctx.out.innerHTML =
        jsonCard("Intent", intent) + card("explain(intent)", `<p>${esc(explain(intent))}</p>`);

      return;
    }

    const f = compiled.formatter;

    ctx.setStage({ ok: true, text: `${f.kind} formatter, ${f.locale}` });

    const samples: [string, number][] =
      f.kind === "date-time" ? DATES : NUMBERS.map((n) => [String(n), n]);

    const custom = customValue(ctx.opts.value!, f.kind);

    if (custom !== null) samples.unshift([`your value: ${ctx.opts.value!.trim()}`, custom]);

    const rows = samples
      .map(([label, value]) => {
        const r = f.format(value);

        const status = r.supported ? "" : "no";
        const text = r.supported ? esc(r.text) : `${esc(r.code)}: ${esc(r.reason)}`;
        const output = `<td class="mono ${status}" style="font-size:15px">${text}</td>`;

        return `<tr><td class="mono muted">${esc(label)}</td>${output}</tr>`;
      })
      .join("");

    const partsValue = custom ?? samples[0]![1];
    const parts = f.formatToParts(partsValue);

    const partSpans = parts.supported
      ? parts.parts.map((p) => `<span>${esc(p.value)}<i>${esc(p.type)}</i></span>`).join("")
      : "";

    const partsHtml = parts.supported
      ? `<div class="parts">${partSpans}</div>`
      : `<p class="no">${esc(parts.reason)}</p>`;

    const zoneSuffix = f.kind === "date-time" ? ` · ${f.options.timeZone}` : "";
    const table = `<div class="scroll"><table><tbody>${rows}</tbody></table></div>`;
    const noteHtml = notes.length ? `<p class="note">${notes.join("; ")}.</p>` : "";

    ctx.out.innerHTML =
      card(`formatter.format() · ${f.locale}${zoneSuffix}`, table + noteHtml) +
      jsonCard("Intl options (frozen)", f.options) +
      card("explain(intent)", `<p>${esc(explain(intent))}</p>`) +
      jsonCard("Intent", intent) +
      card(`formatToParts(${esc(String(partsValue))})`, partsHtml, true);
  },
};
