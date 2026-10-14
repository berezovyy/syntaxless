import { compile, explain, parse, preload } from "../../../packages/retention/src/index.ts";
import type { Backup } from "../../../packages/retention/src/index.ts";
import { browserTimeZone, card, esc, jsonCard, options } from "../ui.ts";
import type { Demo } from "../ui.ts";

const HOUR = 3_600_000;

const SETS: Record<string, { label: string; step: number; count: number; offset: number }> = {
  daily: {
    label: "daily at 02:00 UTC, 420 backups",
    step: 24 * HOUR,
    count: 420,
    offset: 2 * HOUR,
  },
  hourly: { label: "hourly, 21 days", step: HOUR, count: 504, offset: 0 },
  six: { label: "every 6 hours, 3 years", step: 6 * HOUR, count: 4380, offset: 0 },
};

const SET_NAMES = Object.keys(SETS);

const SET_LABELS = Object.fromEntries(Object.entries(SETS).map(([k, v]) => [k, v.label]));

const ZONES = [browserTimeZone, "UTC", "America/New_York", "Asia/Tokyo"];

const KEPT_HEAD = "<thead><tr><th>id</th><th>local time</th><th>tiers</th></tr></thead>";

const TIER_COLORS = ["var(--entity)", "var(--string)", "var(--constant)", "var(--accent)"];

function backups(set: string, now: number): Backup[] {
  const { step, count, offset } = SETS[set]!;
  const newest = Math.floor((now - offset) / step) * step + offset;

  return Array.from({ length: count }, (_, i) => {
    const timestamp = newest - (count - 1 - i) * step;

    return {
      id: `backup-${new Date(timestamp).toISOString().slice(0, 16)}`,
      timestamp,
    };
  });
}

export const retentionDemo: Demo = {
  id: "retention",
  pkg: "@syntaxless/retention",
  blurb:
    "Backup retention policies to a typed intent and a deterministic keep/remove preview " +
    "over your backup list. It only recommends; it never deletes anything.",
  placeholder: "keep daily backups for a week and monthly backups for a year",
  examples: [
    ["keep daily backups for a week and monthly backups for a year"],
    ["keep daily backups for 30 days"],
    ["keep weekly backups for 3 months and yearly backups for 5 years"],
    ["keep daily backups for the last 14 days"],
    [
      "keep daily backups for 7 days, weekly backups for 4 weeks, " +
        "monthly backups for 12 months and yearly backups for 3 years",
    ],
    ["keep the two newest backups of each day", true],
    ["delete everything older than a month", true],
    ["what is our retention policy?", true],
  ],
  controls: () =>
    `<label>Backups <select data-opt="set">${options(SET_NAMES, SET_LABELS)}</select></label>
     <label>Timezone <select data-opt="tz">${options(ZONES)}</select></label>`,
  preload: () => preload(),
  parse: (text) => parse(text),
  render(parsed, ctx) {
    if (!parsed.ok) return;
    const now = Date.now();
    const list = backups(ctx.opts.set!, now);

    const preview = compile(parsed.intent, {
      timeZone: ctx.opts.tz!,
      referenceInstant: now,
      backups: list,
    });

    if (!preview.supported) {
      ctx.setStage({ ok: false, code: preview.code, reason: preview.reason });

      ctx.out.innerHTML =
        card("explain(intent)", `<p>${esc(explain(parsed.intent))}</p>`) +
        jsonCard("Intent", parsed.intent);

      return;
    }

    ctx.setStage({
      ok: true,
      text: `keep ${preview.keep.length}, remove ${preview.remove.length}`,
    });

    const kept = new Map(preview.keep.map((k) => [k.id, k.tierIndexes]));

    const time = new Intl.DateTimeFormat("en-GB", {
      timeZone: ctx.opts.tz,
      dateStyle: "medium",
      timeStyle: "short",
    });

    const cells = list
      .map((b) => {
        const tiers = kept.get(b.id);
        const color = tiers ? TIER_COLORS[tiers[0]!] : "";

        const verdict = tiers ? ` · kept by tier ${tiers.join(", ")}` : " · remove";
        const style = color ? ` style="background:${color}"` : "";

        return `<i title="${esc(time.format(b.timestamp))}${verdict}"${style}></i>`;
      })
      .join("");

    const legend = parsed.intent.tiers
      .map((t: any, i: number) => {
        const window =
          t.window.kind === "calendar"
            ? `${t.window.count} ${t.window.unit}(s), calendar`
            : `${t.window.milliseconds / HOUR} h elapsed`;

        const swatch = `<i style="background:${TIER_COLORS[i]}"></i>`;

        return `<span>${swatch}tier ${i}: ${t.bucket} for ${window}</span>`;
      })
      .join("");

    const byTimestamp = new Map(list.map((b) => [b.id, b.timestamp]));

    const rows = preview.keep
      .slice()
      .sort((a, b) => byTimestamp.get(b.id)! - byTimestamp.get(a.id)!)
      .map((k) => {
        const localTime = esc(time.format(byTimestamp.get(k.id)!));

        const tiers = k.tierIndexes
          .map((i) => `<span style="color:${TIER_COLORS[i]}">${i}</span>`)
          .join(", ");

        const cells = [
          `<td class="mono">${esc(k.id)}</td>`,
          `<td>${localTime}</td>`,
          `<td class="mono">${tiers}</td>`,
        ];

        return `<tr>${cells.join("")}</tr>`;
      })
      .join("");

    const removeKey = `<span><i style="background:var(--line)"></i>remove</span>`;
    const keptTable = `<table>${KEPT_HEAD}<tbody>${rows}</tbody></table>`;

    ctx.out.innerHTML =
      card(
        "Keep / remove preview",
        `<div class="stats">
           <div><b>${preview.keep.length}</b><span>keep</span></div>
           <div><b>${preview.remove.length}</b><span>recommend removing</span></div>
           <div><b>${list.length}</b><span>backups, oldest → newest</span></div>
         </div>
         <div class="cells">${cells}</div>
         <div class="legend">${legend}${removeKey}</div>`,
        true,
      ) +
      card("explain(intent)", `<p>${esc(explain(parsed.intent))}</p>`) +
      jsonCard("Intent", parsed.intent) +
      card(
        `Kept backups · ${ctx.opts.tz}`,
        `<div class="scroll" style="max-height:360px;overflow:auto">${keptTable}</div>`,
        true,
      );
  },
};
