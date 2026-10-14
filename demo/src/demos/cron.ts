import {
  clarify,
  compile,
  explain,
  parse,
  preload,
  recurrences,
  resolveClarification,
} from "../../../packages/cron/src/index.ts";
import type { Occurrence, ScheduleIntent } from "../../../packages/cron/src/index.ts";
import { createTemporalCalendar } from "../../../packages/cron/src/temporal.ts";
import { browserTimeZone, card, errorText, esc, jsonCard, options } from "../ui.ts";
import type { Ctx, Demo } from "../ui.ts";

const temporal = (globalThis as { Temporal?: any }).Temporal;

const calendar = temporal?.Instant?.fromEpochMilliseconds
  ? createTemporalCalendar(temporal)
  : undefined;

function fixedOffset(): string {
  const minutes = -new Date().getTimezoneOffset();
  const abs = Math.abs(minutes);
  const sign = minutes < 0 ? "-" : "+";
  const hours = String(Math.floor(abs / 60)).padStart(2, "0");
  const mins = String(abs % 60).padStart(2, "0");

  return `${sign}${hours}:${mins}`;
}

// Without native Temporal only UTC and fixed offsets are available.
const zones = calendar
  ? [browserTimeZone, "UTC", "America/New_York", "Europe/London", "Asia/Tokyo"]
  : ["UTC", fixedOffset()];

const WEEKDAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
const MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split(" ");
const FIELDS = ["minute", "hour", "day of month", "month", "weekday"];
const pad = (n: number) => String(n).padStart(2, "0");

function relative(ms: number): string {
  const minutes = Math.round(ms / 60000);

  if (minutes < 60) return `in ${minutes} min`;
  const hours = Math.round(minutes / 60);

  if (hours < 48) return `in ${hours} h`;

  return `in ${Math.round(hours / 24)} days`;
}

function clarifyBox(question: string, answer: string): string {
  const prompt = `<p>${esc(question)}</p>`;
  const apply = `<button class="act" data-apply>Apply</button>`;

  return `<div class="clarify">${prompt}<div class="row">${answer}${apply}</div></div>`;
}

function runItem(run: Occurrence): string {
  const l = run.local;
  const dow = WEEKDAYS[new Date(Date.UTC(l.year, l.month - 1, l.day)).getUTCDay()];
  const when = `${dow} ${l.day} ${MONTHS[l.month - 1]} ${l.year} · ${pad(l.hour)}:${pad(l.minute)}`;
  const until = relative(Date.parse(run.instant) - Date.now());

  return `<li><span>${when}</span><span>${until}</span></li>`;
}

const noteParagraph = (html: string) => `<p class="note">${html}</p>`;

const KEPT_NOTE = noteParagraph("The intent is kept, so the upcoming runs below still work.");

const NO_TEMPORAL = "No native Temporal in this browser, so only UTC and a fixed offset.";

function draw(intent: ScheduleIntent, ctx: Ctx, note = ""): void {
  const compiled = compile(intent);

  ctx.setStage(
    compiled.supported
      ? { ok: true, text: compiled.expression }
      : { ok: false, code: compiled.code, reason: compiled.reason },
  );

  let head: string;

  if (compiled.supported) {
    const fields = compiled.expression
      .split(" ")
      .map((v, i) => `<div><b>${esc(v)}</b><span>${FIELDS[i]}</span></div>`)
      .join("");

    head = card("Cron expression", `<div class="cronfields">${fields}</div>${note}`);
  } else {
    const question = clarify(intent);
    let ask = "";

    if (question?.field === "weekdays") {
      const dayButtons = question.allowed
        .map((d) => `<button class="act" data-day="${d}">${d.slice(0, 3)}</button>`)
        .join("");

      ask = clarifyBox(question.question, dayButtons);
    } else if (question?.field === "dayOfMonth") {
      const range = `min="${question.minimum}" max="${question.maximum}"`;
      const dayInput = `<input class="small" type="number" ${range} value="1">`;

      ask = clarifyBox(question.question, dayInput);
    }

    const reason = `<p class="no">Cron can't express this schedule: ${esc(compiled.reason)}</p>`;

    head = card("Cron expression", reason + ask + KEPT_NOTE);
  }

  let runs: string;

  try {
    const zone = ctx.opts.tz!;
    const isNamedZone = zone !== "UTC" && !/^[+-]/.test(zone);

    const preview = recurrences(intent, {
      timezone: zone,
      reference: Date.now(),
      count: 8,
      ...(calendar && isNamedZone ? { calendar } : {}),
    });

    runs = preview.ok
      ? `<ul class="runs">${preview.occurrences.map(runItem).join("")}</ul>`
      : `<p class="no">${esc(preview.code)}: ${esc(preview.reason)}</p>`;
  } catch (error) {
    runs = `<p class="no">${esc(errorText(error))}</p>`;
  }

  ctx.out.innerHTML =
    head +
    card(`Next runs · ${ctx.opts.tz}`, runs) +
    card("explain(intent)", `<p>${esc(explain(intent))}</p>`, true) +
    jsonCard("Intent", intent, true);

  const box = ctx.out.querySelector(".clarify");

  if (!box) return;
  const days = new Set<string>();

  box.querySelectorAll<HTMLButtonElement>("[data-day]").forEach((b) =>
    b.addEventListener("click", () => {
      const day = b.dataset.day!;

      if (!days.delete(day)) days.add(day);
      b.classList.toggle("on", days.has(day));
    }),
  );

  box.querySelector("[data-apply]")!.addEventListener("click", () => {
    const input = box.querySelector<HTMLInputElement>("input");

    try {
      const resolved = resolveClarification(
        intent,
        input ? { dayOfMonth: Number(input.value) } : { weekdays: [...days] as any },
      );

      draw(
        resolved,
        ctx,
        noteParagraph("Resolved with your answer through <code>resolveClarification()</code>."),
      );
    } catch (error) {
      alert(errorText(error));
    }
  });
}

export const cronDemo: Demo = {
  id: "cron",
  pkg: "@syntaxless/cron",
  blurb:
    "Recurring schedules to a typed intent and a 5-field cron expression. " +
    "When cron can't express a schedule, it says so, keeps the intent, " +
    "and asks for a missing day when that is all cron needs. " +
    "Known model issue: a bare hour, as in <code>every weekday at 9</code>, reads as 21:00.",
  placeholder: "every monday at 5pm",
  examples: [
    ["weekdays at 9:30am"],
    ["every 15 minutes"],
    ["every monday at 5pm"],
    ["every other hour"],
    ["on the 1st of every month at 8am"],
    ["every week at 9:25am"],
    ["first friday of every month at noon"],
    ["last day of every month at 6pm"],
    ["every weekday at 9"],
    ["remind me tomorrow at 3", true],
    ["what time is the standup?", true],
  ],
  controls: () =>
    `<label>Parser <select data-opt="source">${options(["model", "rules"])}</select></label>
     <label>Timezone <select data-opt="tz">${options(zones)}</select></label>
     ${calendar ? "" : `<span>${NO_TEMPORAL}</span>`}`,
  preload: () => preload(),
  parse: (text, opts) => parse(text, { source: opts.source as "model" | "rules" }),
  render(parsed, ctx) {
    if (parsed.ok) draw(parsed.intent, ctx);
  },
};
