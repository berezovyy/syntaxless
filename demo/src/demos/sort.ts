import { compile, explain, parse, preload, sortRecords } from "../../../packages/sort/src/index.ts";
import type { Context } from "../../../packages/sort/src/index.ts";
import { card, cellValue, errorText, esc, jsonCard, options } from "../ui.ts";
import type { Demo } from "../ui.ts";
import { sqlHtml } from "./filter.ts";

const fields: Context["fields"] = [
  {
    id: "created",
    type: "number",
    column: "created",
    aliases: ["created", "date", "creation date"],
    orderingAliases: [
      { text: "newest", direction: "desc" },
      { text: "oldest", direction: "asc" },
    ],
  },
  {
    id: "price",
    type: "number",
    column: "price",
    aliases: ["price", "prices"],
    orderingAliases: [
      { text: "cheapest", direction: "asc" },
      { text: "most expensive", direction: "desc" },
    ],
  },
  {
    id: "rating",
    type: "number",
    column: "rating",
    aliases: ["rating", "ratings"],
    orderingAliases: [{ text: "best rated", direction: "desc" }],
  },
  { id: "name", type: "string", column: "name", aliases: ["name", "names"] },
  { id: "stock", type: "number", column: "stock", aliases: ["stock"] },
  { id: "id", type: "number", column: "id", aliases: ["id"] },
];

function contextFor(nulls: string): Context {
  return {
    fields,
    tieBreaker: "id",
    ...(nulls === "first" || nulls === "last" ? { defaultNulls: nulls } : {}),
  };
}

const day = (iso: string) => Date.parse(`${iso}T00:00:00Z`);

const products = [
  {
    id: 1,
    name: "Desk lamp",
    price: 39,
    created: day("2026-03-02"),
    rating: 4.1,
    stock: 12,
  },
  {
    id: 2,
    name: "Mechanical keyboard",
    price: 129,
    created: day("2026-09-18"),
    rating: 4.7,
    stock: 4,
  },
  {
    id: 3,
    name: "USB-C hub",
    price: 49,
    created: day("2026-09-18"),
    rating: null,
    stock: 30,
  },
  {
    id: 4,
    name: "Monitor arm",
    price: null,
    created: day("2025-11-20"),
    rating: 4.4,
    stock: 0,
  },
  { id: 5, name: "Webcam", price: 89, created: null, rating: 3.9, stock: 7 },
  {
    id: 6,
    name: "Cable ties",
    price: 5,
    created: day("2026-07-01"),
    rating: 4.4,
    stock: 250,
  },
  {
    id: 7,
    name: "Standing desk",
    price: 499,
    created: day("2026-01-15"),
    rating: 4.8,
    stock: 2,
  },
  {
    id: 8,
    name: "Mouse pad",
    price: 15,
    created: day("2026-05-09"),
    rating: null,
    stock: 80,
  },
];

function vocabulary(): string {
  const items = fields
    .map((f) => {
      const ordering = (f.orderingAliases ?? []).map(
        (o) => `<code>${esc(o.text)}</code> = ${o.direction}`,
      );

      const aliasList = (f.aliases ?? []).map((a) => `<code>${esc(a)}</code>`).join(", ");
      const orderingList = ordering.length ? `; ordering ${ordering.join(", ")}` : "";

      return `<li><b>${esc(f.id)}</b>: ${aliasList}${orderingList}</li>`;
    })
    .join("");

  const summary = `<summary>What this demo's context knows</summary>`;
  const tieBreaker = `<p>Tie breaker: <code>id</code>.</p>`;

  return `<details class="vocab">${summary}<ul>${items}</ul>${tieBreaker}</details>`;
}

const NULL_PLACEMENTS = ["last", "first", "none"];

const NULL_LABELS = { none: "(not set)" };

const COLUMNS = ["#", "moved", "id", "name", "price", "created", "rating", "stock"];

const TABLE_HEAD = `<thead><tr>${COLUMNS.map((c) => `<th>${c}</th>`).join("")}</tr></thead>`;

export const sortDemo: Demo = {
  id: "sort",
  pkg: "@syntaxless/sort",
  blurb:
    "Ordering requests to typed sort keys, a stable JavaScript sort " +
    "and an SQLite <code>ORDER BY</code> fragment. " +
    "Directions and null placement are never guessed from field names.",
  placeholder: "newest first, then cheapest, missing prices last",
  examples: [
    ["newest first, then cheapest, missing prices last"],
    ["most expensive first"],
    ["best rated first, missing ratings last"],
    ["name ascending"],
    ["stock descending, then price ascending"],
    ["oldest first, missing dates first"],
    ["price", true],
    ["newest first, oldest first", true],
    ["random order", true],
  ],
  controls: () => {
    const select = `<select data-opt="nulls">${options(NULL_PLACEMENTS, NULL_LABELS)}</select>`;

    return `<label>Context <code>defaultNulls</code> ${select}</label>`;
  },
  aside: vocabulary,
  preload: () => preload(),
  parse: (text, opts) => parse(text, contextFor(opts.nulls!)),
  render(parsed, ctx) {
    if (!parsed.ok) return;
    const context = contextFor(ctx.opts.nulls!);
    const orderBy = compile(parsed.intent, context);
    let sorted: typeof products;

    try {
      sorted = sortRecords(products, parsed.intent, context);
      ctx.setStage({ ok: true, text: `ids ${sorted.map((p) => p.id).join(", ")}` });
    } catch (error) {
      ctx.setStage({ ok: false, code: "sortRecords", reason: errorText(error) });

      return;
    }

    const rows = sorted
      .map((p, i) => {
        const was = products.indexOf(p);

        const moved =
          was === i
            ? ""
            : was > i
              ? `<span class="yes">↑${was - i}</span>`
              : `<span class="no">↓${i - was}</span>`;

        const created = p.created === null ? null : new Date(p.created).toISOString().slice(0, 10);

        const cells = [
          `<td class="num">${i + 1}</td>`,
          `<td class="mono">${moved}</td>`,
          `<td class="num">${p.id}</td>`,
          `<td>${esc(p.name)}</td>`,
          `<td class="num">${cellValue(p.price)}</td>`,
          `<td class="mono">${cellValue(created)}</td>`,
          `<td class="num">${cellValue(p.rating)}</td>`,
          `<td class="num">${p.stock}</td>`,
        ];

        return `<tr>${cells.join("")}</tr>`;
      })
      .join("");

    ctx.out.innerHTML =
      card(
        "SQLite ORDER BY",
        orderBy.supported
          ? `<div class="sql"><span class="kw">ORDER BY</span> ${sqlHtml(orderBy.sql)}</div>`
          : `<p class="no">${esc(orderBy.code)}: ${esc(orderBy.reason)}</p>`,
      ) +
      card("explain(intent)", `<p>${esc(explain(parsed.intent))}</p>`) +
      card(
        "Products · sortRecords()",
        `<div class="scroll"><table>${TABLE_HEAD}<tbody>${rows}</tbody></table></div>`,
        true,
      ) +
      jsonCard("Intent", parsed.intent, true);
  },
};
