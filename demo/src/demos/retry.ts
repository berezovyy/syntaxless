import { compile, explain, parse, preload } from "../../../packages/retry/src/index.ts";
import type { RetryPolicy } from "../../../packages/retry/src/index.ts";
import { card, esc, formatMs, jsonCard, options } from "../ui.ts";
import type { Demo } from "../ui.ts";

const MAX_BARS = 40;

const STRATEGIES = ["", "constant", "exponential"];

const JITTERS = ["", "full", "equal"];

const NONE = { "": "(none)" };

const INITIAL_INPUT = `<input class="small" data-opt="initial" value="250" placeholder="none">`;

function defaults(opts: Record<string, string>): Partial<RetryPolicy> {
  const result: Partial<RetryPolicy> = {};
  const number = (key: string) => (opts[key]!.trim() === "" ? undefined : Number(opts[key]));
  const initial = number("initial");
  const multiplier = number("multiplier");
  const cap = number("cap");

  if (initial !== undefined) result.initialDelayMs = initial;
  if (multiplier !== undefined) result.multiplier = multiplier;
  if (cap !== undefined) result.maxDelayMs = cap;

  if (opts.strategy === "constant" || opts.strategy === "exponential") {
    result.strategy = opts.strategy;
  }

  if (opts.jitter === "full" || opts.jitter === "equal") result.jitter = { kind: opts.jitter };

  return result;
}

export const retryDemo: Demo = {
  id: "retry",
  pkg: "@syntaxless/retry",
  blurb:
    "Retry policies to deterministic, bounded delay schedules. It never invents a missing value: " +
    "what the text leaves out must come from the caller defaults below, " +
    "and the intent records which fields did.",
  placeholder: "retry three times, double the delay, cap at one minute",
  examples: [
    ["retry three times, double the delay, cap at one minute"],
    ["retry 5 times with a constant delay of 2 seconds"],
    ["retry up to 10 times starting at 100ms, triple the delay, full jitter"],
    ["retry twice with a fixed delay of 500 milliseconds"],
    ["retry 6 times, start at 1 second, multiply by 1.5, cap at 5 seconds, 20% jitter"],
    ["retry three times", true],
    ["retry 1 minute 30 seconds later", true],
    ["three attempts including the first", true],
  ],
  controls: () =>
    `<span>Caller defaults:</span>
     <label>initialDelayMs ${INITIAL_INPUT}</label>
     <label>strategy <select data-opt="strategy">${options(STRATEGIES, NONE)}</select></label>
     <label>multiplier <input class="small" data-opt="multiplier" placeholder="none"></label>
     <label>maxDelayMs <input class="small" data-opt="cap" placeholder="none"></label>
     <label>jitter <select data-opt="jitter">${options(JITTERS, NONE)}</select></label>
     <label>seed <input class="small" data-opt="seed" value="42"></label>
     <button class="act reroll" type="button">new seed</button>`,
  bind(panel, rerun) {
    panel.querySelector(".reroll")!.addEventListener("click", () => {
      panel.querySelector<HTMLInputElement>('[data-opt="seed"]')!.value = String(
        Math.floor(Math.random() * 2 ** 32),
      );

      rerun();
    });
  },
  preload: () => preload(),
  parse: (text, opts) => parse(text, { defaults: defaults(opts) }),
  render(parsed, ctx) {
    if (!parsed.ok) return;
    const seed = ctx.opts.seed!.trim() === "" ? undefined : Number(ctx.opts.seed);
    const schedule = compile(parsed.intent, seed === undefined ? {} : { seed });

    if (!schedule.supported) {
      ctx.setStage({ ok: false, code: schedule.code, reason: schedule.reason });

      ctx.out.innerHTML =
        card("explain(intent)", `<p>${esc(explain(parsed.intent))}</p>`) +
        jsonCard("Intent", parsed.intent);

      return;
    }

    const delays = schedule.delaysMs;
    const total = delays.reduce((a, b) => a + b, 0);

    ctx.setStage({ ok: true, text: `${delays.length} delays` });
    const max = Math.max(1, ...delays);
    let elapsed = 0;

    const bars = delays
      .slice(0, MAX_BARS)
      .map((d, i) => {
        elapsed += d;

        const bar = `<div class="bar" style="width:${(d / max) * 100}%"></div>`;
        const timing = `${formatMs(d)} <span class="muted">· at ${formatMs(elapsed)}</span>`;

        return `<span>retry ${i + 1}</span>${bar}<span>${timing}</span>`;
      })
      .join("");

    const more =
      delays.length > MAX_BARS ? `<p class="note">…and ${delays.length - MAX_BARS} more.</p>` : "";

    const defaultedFields = parsed.intent.defaults
      .map((f: string) => `<code>${esc(f)}</code>`)
      .join(", ");

    const defaulted = parsed.intent.defaults.length
      ? `<p class="note">From caller defaults: ${defaultedFields}</p>`
      : `<p class="note">Every value came from the text.</p>`;

    ctx.out.innerHTML =
      card(
        "Delay schedule",
        `<div class="stats">
           <div><b>${delays.length}</b><span>retries</span></div>
           <div><b>${delays.length + 1}</b><span>attempts in total</span></div>
           <div><b>${formatMs(total)}</b><span>total wait</span></div>
         </div>
         <div class="bars">${bars}</div>${more}`,
        true,
      ) +
      card("explain(intent)", `<p>${esc(explain(parsed.intent))}</p>${defaulted}`) +
      jsonCard("Intent", parsed.intent);
  },
};
