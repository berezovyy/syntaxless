/** Syntaxless playground: one tab per package, each running the real bundled model. */
import { DEMOS } from "./demos/index.ts";
import { errorText, esc, jsonHtml } from "./ui.ts";
import type { Ctx, Demo, Stage } from "./ui.ts";

interface PanelState {
  seq: number;
  loaded: Promise<void> | null;
}

const tabs = document.querySelector<HTMLElement>("nav .tabs")!;
const main = document.querySelector<HTMLElement>("main")!;
const states = new Map<string, PanelState>();

function readOpts(panel: HTMLElement): Record<string, string> {
  const opts: Record<string, string> = {};

  for (const el of panel.querySelectorAll<HTMLInputElement | HTMLSelectElement>("[data-opt]")) {
    opts[el.dataset.opt!] =
      el instanceof HTMLInputElement && el.type === "checkbox" ? (el.checked ? "1" : "") : el.value;
  }

  return opts;
}

function setTabStatus(id: string, text: string, ready = false): void {
  const small = tabs.querySelector<HTMLElement>(`button[data-id="${id}"] small`);

  if (!small) return;
  small.textContent = text;
  small.classList.toggle("ready", ready);
}

function ensureLoaded(demo: Demo): Promise<void> {
  const state = states.get(demo.id)!;

  if (!state.loaded) {
    setTabStatus(demo.id, "loading model…");
    const started = performance.now();

    state.loaded = demo.preload().then(
      () => setTabStatus(demo.id, `model ${Math.round(performance.now() - started)} ms`, true),
      (error) => setTabStatus(demo.id, `load failed: ${errorText(error)}`),
    );
  }

  return state.loaded;
}

function stepHtml(name: string, stage: Stage): string {
  if (stage.ok) {
    const meta = stage.text ? `<span class="meta">${esc(stage.text)}</span>` : "";

    return `<span class="step ok">${name} ✓</span>${meta}`;
  }

  const step = `<span class="step bad">${name} ✗ ${esc(stage.code)}</span>`;

  return `${step}<span class="reason">${esc(stage.reason)}</span>`;
}

async function run(demo: Demo, panel: HTMLElement): Promise<void> {
  const state = states.get(demo.id)!;
  const seq = ++state.seq;
  const input = panel.querySelector<HTMLInputElement>("input.q")!;
  const verdict = panel.querySelector<HTMLElement>(".verdict")!;
  const out = panel.querySelector<HTMLElement>(".out")!;
  const raw = panel.querySelector<HTMLElement>("details.raw pre")!;
  const text = input.value;

  if (text.trim() === "") {
    verdict.innerHTML = `<span class="hint">Type an instruction or pick an example.</span>`;
    out.innerHTML = "";
    raw.textContent = "";

    return;
  }

  const opts = readOpts(panel);

  await ensureLoaded(demo);
  const started = performance.now();
  let parsed: any;

  try {
    parsed = await demo.parse(text, opts);
  } catch (error) {
    parsed = { ok: false, code: "threw", reason: errorText(error) };
  }

  const ms = performance.now() - started;

  if (seq !== state.seq) return;

  const refusedConfidence =
    typeof parsed.confidence === "number" && parsed.confidence > 0
      ? ` (confidence ${parsed.confidence.toFixed(3)})`
      : "";

  const parseStage: Stage = parsed.ok
    ? {
        ok: true,
        text: `confidence ${Number(parsed.confidence).toFixed(3)} · ${ms.toFixed(1)} ms`,
      }
    : {
        ok: false,
        code: parsed.code,
        reason: `${parsed.reason}${refusedConfidence}`,
      };

  const drawVerdict = (next: Stage | null) => {
    verdict.innerHTML =
      stepHtml("parse", parseStage) +
      (next ? `<span class="meta">→</span>${stepHtml("compile", next)}` : "");
  };

  drawVerdict(null);
  raw.innerHTML = jsonHtml(parsed);
  out.innerHTML = "";

  const ctx: Ctx = {
    out,
    opts,
    setStage: drawVerdict,
    rerun: () => void run(demo, panel),
  };

  try {
    demo.render(parsed, ctx);
  } catch (error) {
    drawVerdict({ ok: false, code: "threw", reason: errorText(error) });
  }
}

function mount(demo: Demo): HTMLElement {
  states.set(demo.id, { seq: 0, loaded: null });
  const tab = document.createElement("button");

  tab.role = "tab";
  tab.dataset.id = demo.id;
  tab.innerHTML = `${esc(demo.id)}<small>not loaded</small>`;
  tab.addEventListener("click", () => activate(demo.id));
  tabs.append(tab);

  const panel = document.createElement("section");

  panel.className = "panel";
  panel.id = `panel-${demo.id}`;
  panel.hidden = true;

  const chips = demo.examples
    .map(([text, refuse]) => {
      const title = refuse ? "Should be refused" : "Should parse";

      return `<button class="${refuse ? "neg" : ""}" title="${title}">${esc(text)}</button>`;
    })
    .join("");

  const firstExample = esc(demo.examples[0]![0]);
  const queryAttrs = `class="q" type="text" spellcheck="false" autocomplete="off"`;
  const queryValues = `placeholder="${esc(demo.placeholder)}" value="${firstExample}"`;

  panel.innerHTML = `
    <div class="intro"><h2>${esc(demo.pkg)}</h2><p>${demo.blurb}</p></div>
    <div class="ask">
      <input ${queryAttrs} ${queryValues}>
      <div class="chips"><span class="label">Examples</span>${chips}</div>
      ${demo.controls ? `<div class="controls">${demo.controls()}</div>` : ""}
      ${demo.aside ? demo.aside() : ""}
    </div>
    <div class="verdict"></div>
    <div class="out grid"></div>
    <details class="raw"><summary>Raw parse() result</summary><pre></pre></details>`;

  main.append(panel);

  const input = panel.querySelector<HTMLInputElement>("input.q")!;
  const rerun = () => void run(demo, panel);
  let timer: ReturnType<typeof setTimeout> | undefined;

  input.addEventListener("input", () => {
    clearTimeout(timer);
    timer = setTimeout(rerun, 120);
  });

  panel.querySelectorAll<HTMLButtonElement>(".chips button").forEach((button, index) =>
    button.addEventListener("click", () => {
      input.value = demo.examples[index]![0];
      rerun();
    }),
  );

  panel
    .querySelectorAll("[data-opt]")
    .forEach((el) => el.addEventListener(el.tagName === "SELECT" ? "change" : "input", rerun));

  demo.bind?.(panel, rerun);

  return panel;
}

const panels = new Map(DEMOS.map((demo) => [demo.id, mount(demo)]));
const ran = new Set<string>();

function activate(id: string): void {
  if (!panels.has(id)) id = DEMOS[0]!.id;
  for (const [key, panel] of panels) panel.hidden = key !== id;

  for (const tab of tabs.querySelectorAll<HTMLElement>("button")) {
    tab.setAttribute("aria-selected", String(tab.dataset.id === id));
  }

  if (location.hash !== `#${id}`) history.replaceState(null, "", `#${id}`);

  if (!ran.has(id)) {
    ran.add(id);

    void run(
      DEMOS.find((demo) => demo.id === id)!,
      panels.get(id)!,
    );
  }
}

addEventListener("hashchange", () => activate(location.hash.slice(1)));
activate(location.hash.slice(1));
