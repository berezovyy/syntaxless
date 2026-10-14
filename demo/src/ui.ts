/** Shared types and HTML helpers for the playground panels. */
export type Stage = { ok: true; text: string } | { ok: false; code: string; reason: string };

export interface Ctx {
  out: HTMLElement;
  opts: Record<string, string>;
  /** Report the compile/evaluate stage that follows parsing. */
  setStage(stage: Stage | null): void;
  rerun(): void;
}

export interface Demo {
  id: string;
  pkg: string;
  blurb: string;
  placeholder: string;
  /** [text, shouldRefuse] */
  examples: [string, boolean?][];
  controls?: () => string;
  aside?: () => string;
  bind?: (panel: HTMLElement, rerun: () => void) => void;
  preload: () => Promise<void>;
  parse: (text: string, opts: Record<string, string>) => Promise<any>;
  render: (parsed: any, ctx: Ctx) => void;
}

export function esc(value: unknown): string {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

export function jsonHtml(value: unknown): string {
  const text = JSON.stringify(value, null, 2) ?? "undefined";

  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replace(
      /("(?:\\.|[^"\\])*")(\s*:)?|\b(true|false|null)\b|(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)/g,
      (_m, str, colon, lit, num) =>
        str
          ? colon
            ? `<span class="k">${str}</span>${colon}`
            : `<span class="s">${str}</span>`
          : lit
            ? `<span class="l">${lit}</span>`
            : `<span class="n">${num}</span>`,
    );
}

export function card(title: string, body: string, full = false, extra = ""): string {
  const className = full ? "card full" : "card";
  const heading = `<h3><span>${esc(title)}</span>${extra}</h3>`;

  return `<div class="${className}">${heading}${body}</div>`;
}

export const jsonCard = (title: string, value: unknown, full = false) =>
  card(title, `<pre>${jsonHtml(value)}</pre>`, full);

export function errorText(error: unknown): string {
  if (error && typeof error === "object" && "message" in error) {
    const code = "code" in error ? `${String(error.code)}: ` : "";

    return code + String(error.message);
  }

  return String(error);
}

export const browserTimeZone = Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC";

/** `<option>` list; labels default to the values. Duplicates are dropped. */
export function options(values: readonly string[], labels: Record<string, string> = {}): string {
  return [...new Set(values)]
    .map((v) => `<option value="${esc(v)}">${esc(labels[v] ?? v)}</option>`)
    .join("");
}

export function formatMs(ms: number): string {
  if (ms < 1000) return `${ms} ms`;
  if (ms < 60_000) return `${+(ms / 1000).toFixed(2)} s`;
  if (ms < 3_600_000) return `${+(ms / 60_000).toFixed(2)} min`;

  return `${+(ms / 3_600_000).toFixed(2)} h`;
}

export function cellValue(value: unknown): string {
  if (value === null || value === undefined) return `<span class="null">null</span>`;

  return esc(value);
}
