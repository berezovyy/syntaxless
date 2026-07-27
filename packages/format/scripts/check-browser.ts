import { resolve, sep } from "node:path";

/** Verify the built package in Chromium, with lazy loading and offline parsing. */
import { chromium } from "playwright";

const directory = resolve(process.argv[2] ?? resolve(import.meta.dir, "../dist"));
const requests: string[] = [];

const server = Bun.serve({
  hostname: "127.0.0.1",
  port: 0,
  async fetch(request) {
    const url = new URL(request.url);

    if (url.pathname === "/") {
      return new Response("<!doctype html><title>Format browser check</title>", {
        headers: { "Content-Type": "text/html" },
      });
    }

    const path = resolve(directory, `.${url.pathname}`);

    if (!path.startsWith(directory + sep)) return new Response(null, { status: 403 });
    const file = Bun.file(path);

    return (await file.exists()) ? new Response(file) : new Response(null, { status: 404 });
  },
});

const origin = `http://127.0.0.1:${server.port}`;
let browser;

try {
  browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();

  await page.route("**/*", async (route) => {
    const r = route.request();

    if (new URL(r.url()).origin !== origin || r.method() !== "GET" || r.postData() !== null) {
      throw new Error("unexpected browser request");
    }

    requests.push(new URL(r.url()).pathname);
    await route.continue();
  });

  await page.goto(origin);

  await page.evaluate(async (entry) => {
    await import(entry);
  }, "/index.js");

  if (requests.some((path) => path.endsWith("weights.gen.js"))) {
    throw new Error("entry import eagerly loaded weights");
  }

  const initial = await page.evaluate(async (entry) => {
    const api = await import(entry);

    const [parsed] = await Promise.all([
      api.parse("euros, two decimals, negatives in parentheses"),
      api.preload(),
    ]);

    if (!parsed.ok) throw new Error("browser reference failed: " + JSON.stringify(parsed));

    return {
      parsed,
      compiled: (() => {
        const c = api.compile(parsed.intent, { locale: "en-US" });

        return c.supported ? c.formatter.format(-1234.5) : c;
      })(),
    };
  }, "/index.js");

  if (!initial.compiled.supported || initial.compiled.text !== "(€1,234.50)") {
    throw new Error("browser meaning mismatch");
  }

  if (requests.filter((path) => path.endsWith("weights.gen.js")).length !== 1) {
    throw new Error("concurrent operations did not share the model load");
  }

  await page.context().setOffline(true);

  const offline = await page.evaluate(async (entry) => {
    const api = await import(entry);

    const accepted = await api.parse(
      "short date and long time with 24-hour clock in en-US timezone America/New_York",
    );

    const rejected = await api.parse("format number with negative two decimals");
    const compiled = accepted.ok ? api.compile(accepted.intent) : null;

    const shown = compiled?.supported
      ? compiled.formatter.formatToParts(Date.parse("2024-03-10T07:30:00Z"))
      : null;

    return {
      accepted,
      rejected,
      compiled: compiled?.supported
        ? { supported: true, options: compiled.formatter.options, shown }
        : compiled,
    };
  }, "/index.js");

  if (
    !offline.accepted.ok ||
    offline.rejected.ok ||
    !offline.compiled?.supported ||
    offline.compiled.options.timeZone !== "America/New_York" ||
    !offline.compiled.shown?.supported ||
    offline.compiled.shown.parts.find(
      (part: { type: string; value: string }) => part.type === "hour",
    )?.value !== "03"
  ) {
    throw new Error("offline browser parsing/formatting failed: " + JSON.stringify(offline));
  }

  console.log({
    chromium: browser.version(),
    requests: requests.length,
    locale: "en-US",
    timeZone: "America/New_York",
    currency: initial.compiled.text,
    offlineHour: offline.compiled?.shown?.parts?.find(
      (part: { type: string; value: string }) => part.type === "hour",
    )?.value,
  });

  const coverage = "lazy weights, concurrent initialization, offline parsing";

  console.log(`browser: passed (${requests.length} local GET requests; ${coverage})`);
} finally {
  await browser?.close();
  server.stop(true);
}
