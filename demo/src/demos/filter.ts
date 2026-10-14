import {
  compile,
  evaluateTruth,
  explain,
  parse,
  preload,
} from "../../../packages/filter/src/index.ts";
import type { Context } from "../../../packages/filter/src/index.ts";
import { card, cellValue, esc, jsonCard } from "../ui.ts";
import type { Demo } from "../ui.ts";

export const context: Context = {
  fields: [
    { id: "name", type: "string", column: "name", aliases: ["name", "customer name"] },
    {
      id: "status",
      type: "string",
      column: "status",
      aliases: ["status"],
      terms: [
        { text: "active customers", value: "active" },
        { text: "paused customers", value: "paused" },
        { text: "churned customers", value: "churned" },
      ],
    },
    {
      id: "spending",
      type: "number",
      column: "spending",
      aliases: ["spending", "spend"],
    },
    {
      id: "trial",
      type: "boolean",
      column: "trial",
      aliases: ["trial"],
      terms: [
        { text: "trials", value: true },
        { text: "paying customers", value: false },
      ],
    },
    { id: "seats", type: "number", column: "seats", aliases: ["seats"] },
    { id: "country", type: "string", column: "country", aliases: ["country"] },
  ],
};

const customers = [
  {
    id: 1,
    name: "Acme Corp",
    status: "active",
    spending: 240,
    trial: false,
    seats: 25,
    country: "US",
  },
  {
    id: 2,
    name: "Globex",
    status: "active",
    spending: 95,
    trial: false,
    seats: 8,
    country: "DE",
  },
  {
    id: 3,
    name: "Initech",
    status: "active",
    spending: 130,
    trial: true,
    seats: 4,
    country: "US",
  },
  {
    id: 4,
    name: "Umbrella",
    status: "paused",
    spending: 510,
    trial: false,
    seats: 60,
    country: "GB",
  },
  {
    id: 5,
    name: "Hooli",
    status: "active",
    spending: 180,
    trial: false,
    seats: 14,
    country: "US",
  },
  {
    id: 6,
    name: "Stark Industries",
    status: "churned",
    spending: 75,
    trial: false,
    seats: 3,
    country: "US",
  },
  {
    id: 7,
    name: "Wayne Enterprises",
    status: "active",
    spending: null,
    trial: false,
    seats: 40,
    country: "US",
  },
  {
    id: 8,
    name: "Soylent",
    status: "active",
    spending: 55,
    trial: true,
    seats: 2,
    country: "FR",
  },
  {
    id: 9,
    name: "Tyrell",
    status: "paused",
    spending: 210,
    trial: false,
    seats: 18,
    country: "JP",
  },
  {
    id: 10,
    name: "Cyberdyne",
    status: "active",
    spending: 320,
    trial: false,
    seats: 33,
    country: "DE",
  },
  {
    id: 11,
    name: "Wonka",
    status: "active",
    spending: 49,
    trial: false,
    seats: 6,
    country: "GB",
  },
  {
    id: 12,
    name: "Vandelay",
    status: "churned",
    spending: 150,
    trial: true,
    seats: 1,
    country: "US",
  },
];

export function sqlHtml(sql: string): string {
  return esc(sql).replace(
    /\b(AND|OR|NOT|IN|IS|NULL|BETWEEN|CASE|WHEN|THEN|ELSE|END|COLLATE|BINARY|ASC|DESC|NULLS|FIRST|LAST)\b/g,
    `<span class="kw">$1</span>`,
  );
}

function vocabulary(): string {
  const items = context.fields
    .map((f) => {
      const names = [f.id, ...(f.aliases ?? [])].filter((v, i, a) => a.indexOf(v) === i);

      const terms = (f.terms ?? []).map(
        (t) => `<code>${esc(t.text)}</code> = ${esc(JSON.stringify(t.value))}`,
      );

      const nameList = names.map((n) => `<code>${esc(n)}</code>`).join(", ");
      const termList = terms.length ? `; terms ${terms.join(", ")}` : "";

      return `<li><b>${esc(f.id)}</b> (${f.type}): ${nameList}${termList}</li>`;
    })
    .join("");

  const summary = `<summary>What this demo's context knows</summary>`;

  const example = `<code>country is "US"</code>`;
  const quoting = `<p>String literals use JSON double quotes, for example ${example}.</p>`;

  return `<details class="vocab">${summary}<ul>${items}</ul>${quoting}</details>`;
}

const COLUMNS = ["", "name", "status", "spending", "trial", "seats", "country"];

const TABLE_HEAD = `<thead><tr>${COLUMNS.map((c) => `<th>${c}</th>`).join("")}</tr></thead>`;

const TRUTH_LEGEND =
  `<p class="note">✓ true · ✗ false · ? unknown ` +
  `(a null value makes the predicate unknown, as in SQL)</p>`;

export const filterDemo: Demo = {
  id: "filter",
  pkg: "@syntaxless/filter",
  blurb:
    "Predicates over your own fields, evaluated in JavaScript with SQL's three-valued logic " +
    "and compiled to a parameterized SQLite <code>WHERE</code> fragment.",
  placeholder: "active customers spending over 100, excluding trials",
  examples: [
    ["active customers spending over 100, excluding trials"],
    ["spending between 50 and 200"],
    ["paying customers and seats at least 10"],
    ['country is "US" or country is "DE"'],
    ['country in "GB", "FR", "JP"'],
    ["spending is missing"],
    ["churned customers or paused customers"],
    ["not trials and seats under 5"],
    ["spending over 100 and seats under 10 or trials", true],
    ["revenue over 100", true],
    ["customers who signed up last week", true],
  ],
  aside: vocabulary,
  preload: () => preload(),
  parse: (text) => parse(text, context),
  render(parsed, ctx) {
    if (!parsed.ok) return;
    const where = compile(parsed.intent, context);
    const truths = customers.map((c) => evaluateTruth(parsed.intent, c, context));
    const matched = truths.filter((t) => t === true).length;

    ctx.setStage({
      ok: true,
      text: `${matched} of ${customers.length} customers match`,
    });

    let sql: string;

    if (where.supported) {
      const clause = `<div class="sql"><span class="kw">WHERE</span> ${sqlHtml(where.sql)}</div>`;

      const params = where.parameters
        .map((p, i) => `<span><i>?${i + 1}</i> ${esc(JSON.stringify(p))}</span>`)
        .join("");

      sql = `${clause}<div class="params">${params}</div>`;
    } else {
      sql = `<p class="no">${esc(where.code)}: ${esc(where.reason)}</p>`;
    }

    const rows = customers
      .map((c, i) => {
        const t = truths[i];
        const cls = t === true ? "" : t === null ? "unknown" : "miss";

        const mark =
          t === true
            ? `<span class="yes">✓</span>`
            : t === null
              ? `<span class="muted">?</span>`
              : `<span class="no">✗</span>`;

        const cells = [
          `<td class="mark">${mark}</td>`,
          `<td>${esc(c.name)}</td>`,
          `<td>${esc(c.status)}</td>`,
          `<td class="num">${cellValue(c.spending)}</td>`,
          `<td>${c.trial}</td>`,
          `<td class="num">${c.seats}</td>`,
          `<td>${esc(c.country)}</td>`,
        ];

        return `<tr class="${cls}">${cells.join("")}</tr>`;
      })
      .join("");

    const table = `<div class="scroll"><table>${TABLE_HEAD}<tbody>${rows}</tbody></table></div>`;

    ctx.out.innerHTML =
      card("SQLite WHERE", sql) +
      card("explain(intent)", `<p>${esc(explain(parsed.intent))}</p>`) +
      card(`Customers · ${matched} match`, table + TRUTH_LEGEND, true) +
      jsonCard("Intent", parsed.intent, true);
  },
};
