import {
  compile,
  compileObject,
  explain,
  parse,
  preload,
} from "../../../packages/validation/src/index.ts";
import type { FieldIntent, JsonSchema } from "../../../packages/validation/src/index.ts";
import { card, esc, jsonCard, jsonHtml, options } from "../ui.ts";
import type { Ctx, Demo } from "../ui.ts";

const FORMATS: Record<string, (v: string) => boolean> = {
  email: (v) => /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(v),
  uri: (v) => {
    try {
      return /^[a-z][a-z0-9+.-]*:/i.test(v) && Boolean(new URL(v));
    } catch {
      return false;
    }
  },
  uuid: (v) => /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(v),
};

/** A small checker for the JSON Schema subset the package emits. Not part of the package. */
function check(schema: JsonSchema, required: boolean, raw: string, missing: boolean) {
  const s = schema as Record<string, any>;

  if (missing) {
    return [
      {
        pass: !required,
        label: required ? "required, but no value was sent" : "optional, may be omitted",
      },
    ];
  }

  const results: { pass: boolean; label: string }[] = [];
  let value: unknown = raw;

  if (s.type === "number" || s.type === "integer") {
    const n = raw.trim() === "" ? NaN : Number(raw);
    const isType = Number.isFinite(n) && (s.type === "number" || Number.isInteger(n));

    results.push({ pass: isType, label: `type ${s.type}` });
    if (!isType) return results;
    value = n;
  } else if (s.type === "boolean") {
    const isType = raw === "true" || raw === "false";

    results.push({ pass: isType, label: `type boolean (true or false)` });

    return results;
  } else {
    results.push({ pass: true, label: "type string" });
  }

  if (typeof value === "string") {
    const length = [...value].length;

    if (s.minLength !== undefined) {
      results.push({
        pass: length >= s.minLength,
        label: `minLength ${s.minLength} (is ${length})`,
      });
    }

    if (s.maxLength !== undefined) {
      results.push({
        pass: length <= s.maxLength,
        label: `maxLength ${s.maxLength} (is ${length})`,
      });
    }

    if (s.format && FORMATS[s.format]) {
      results.push({ pass: FORMATS[s.format]!(value), label: `format ${s.format}` });
    }
  } else if (typeof value === "number") {
    if (s.minimum !== undefined) {
      results.push({ pass: value >= s.minimum, label: `≥ ${s.minimum}` });
    }

    if (s.exclusiveMinimum !== undefined) {
      results.push({
        pass: value > s.exclusiveMinimum,
        label: `> ${s.exclusiveMinimum}`,
      });
    }

    if (s.maximum !== undefined) {
      results.push({ pass: value <= s.maximum, label: `≤ ${s.maximum}` });
    }

    if (s.exclusiveMaximum !== undefined) {
      results.push({
        pass: value < s.exclusiveMaximum,
        label: `< ${s.exclusiveMaximum}`,
      });
    }
  }

  return results;
}

/** Fields added to the object-schema builder, kept across re-parses. */
const objectFields = new Map<string, FieldIntent>();
let current: FieldIntent | null = null;

function drawObject(root: HTMLElement): void {
  const target = root.querySelector<HTMLElement>(".object-out")!;

  if (objectFields.size === 0) {
    target.innerHTML = `<p class="hint">No fields yet. Name the current field and add it.</p>`;

    return;
  }

  const result = compileObject(Object.fromEntries(objectFields));

  const list = [...objectFields.keys()]
    .map(
      (name) =>
        `<button class="act" data-remove="${esc(name)}" title="Remove">${esc(name)} ×</button>`,
    )
    .join(" ");

  const schema = `<pre style="margin-top:10px">${jsonHtml(result)}</pre>`;

  target.innerHTML = `<div class="row">${list}</div>${schema}`;

  target.querySelectorAll<HTMLButtonElement>("[data-remove]").forEach((b) =>
    b.addEventListener("click", () => {
      objectFields.delete(b.dataset.remove!);
      drawObject(root);
    }),
  );
}

function render(parsed: any, ctx: Ctx): void {
  if (!parsed.ok) {
    current = null;

    return;
  }

  const intent: FieldIntent = parsed.intent;

  current = intent;
  const compiled = compile(intent);

  ctx.setStage(
    compiled.supported
      ? { ok: true, text: "JSON Schema 2020-12" }
      : { ok: false, code: compiled.code, reason: compiled.reason },
  );

  const tryBox = compiled.supported
    ? `<div class="controls" style="margin-top:0">
         <input class="small wide try-value" placeholder="a value to check" spellcheck="false">
         <label><input type="checkbox" class="try-missing"> send no value</label>
       </div>
       <ul class="checks"></ul>
       <p class="note">Checked by this page's tiny JSON Schema checker, not by the package.</p>`
    : `<p class="hint">Nothing to check: the intent did not compile.</p>`;

  ctx.out.innerHTML =
    (compiled.supported
      ? jsonCard("JSON Schema", compiled.schema)
      : card("JSON Schema", `<p class="no">${esc(compiled.code)}: ${esc(compiled.reason)}</p>`)) +
    jsonCard("Intent", intent) +
    card("explain(intent)", `<p>${esc(explain(intent))}</p>`, true) +
    card("Try a value against the schema", tryBox) +
    card(
      "compileObject() builder",
      `<div class="controls" style="margin-top:0">
         <input class="small field-name" placeholder="field name" spellcheck="false">
         <button class="act add-field">Add current field</button>
       </div>
       <div class="object-out" style="margin-top:10px"></div>`,
    );

  const root = ctx.out;

  drawObject(root);

  root.querySelector(".add-field")!.addEventListener("click", () => {
    const name = root.querySelector<HTMLInputElement>(".field-name")!.value.trim();

    if (!name || !current) return;
    objectFields.set(name, current);
    drawObject(root);
  });

  if (!compiled.supported) return;
  const value = root.querySelector<HTMLInputElement>(".try-value")!;
  const missing = root.querySelector<HTMLInputElement>(".try-missing")!;
  const list = root.querySelector<HTMLElement>(".checks")!;

  const update = () => {
    const results = check(compiled.schema, intent.required, value.value, missing.checked);
    const pass = results.every((r) => r.pass);

    list.innerHTML =
      `<li><b class="${pass ? "yes" : "no"}">${pass ? "VALID" : "INVALID"}</b></li>` +
      results
        .map(
          (r) =>
            `<li><b class="${r.pass ? "yes" : "no"}">${r.pass ? "✓" : "✗"}</b>${esc(r.label)}</li>`,
        )
        .join("");
  };

  value.addEventListener("input", update);
  missing.addEventListener("change", update);
  update();
}

export const validationDemo: Demo = {
  id: "validation",
  pkg: "@syntaxless/validation",
  blurb:
    "Field constraints to a typed intent and JSON Schema. " +
    "Presence (required or optional) is never guessed, " +
    "and anything outside one field's constraints is refused.",
  placeholder: "required email with max 100 characters",
  examples: [
    ["required email with max 100 characters"],
    ["optional integer between 1 and 10"],
    ["required url"],
    ["optional string up to 20 characters"],
    ["required number greater than 0"],
    ["required uuid"],
    ["optional boolean"],
    ["required string with at least eight characters"],
    ["email with max 100 characters", true],
    ["every weekday at 9", true],
    ["required email matching a regex", true],
  ],
  controls: () =>
    `<label>Parser <select data-opt="source">${options(["model", "rules"])}</select></label>`,
  preload: () => preload(),
  parse: (text, opts) => parse(text, { source: opts.source as "model" | "rules" }),
  render,
};
