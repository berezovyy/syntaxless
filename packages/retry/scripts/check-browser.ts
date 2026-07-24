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
      return new Response("<!doctype html><title>Retry browser check</title>", {
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
      api.parse("retry three times, double the delay, cap at one minute", {
        defaults: { initialDelayMs: 250 },
      }),
      api.preload(),
    ]);

    if (!parsed.ok) throw new Error("browser reference failed: " + JSON.stringify(parsed));

    return { parsed, compiled: api.compile(parsed.intent) };
  }, "/index.js");

  if (
    !initial.compiled.supported ||
    JSON.stringify(initial.compiled.delaysMs) !== "[250,500,1000]" ||
    !initial.parsed.intent.defaults.includes("initialDelayMs")
  ) {
    throw new Error("browser meaning mismatch");
  }

  if (requests.filter((path) => path.endsWith("weights.gen.js")).length !== 1) {
    throw new Error("concurrent operations did not share the model load");
  }

  await page.context().setOffline(true);

  const offline = await page.evaluate(async (entry) => {
    const api = await import(entry);

    return {
      accepted: await api.parse("retry three times with constant delay of one second"),
      rejected: await api.parse("retry three times with constant delay of one second?"),
    };
  }, "/index.js");

  if (!offline.accepted.ok || offline.rejected.ok) {
    throw new Error("offline browser parsing failed: " + JSON.stringify(offline));
  }

  const coverage = "lazy weights, concurrent initialization, offline parsing";

  console.log(`browser: passed (${requests.length} local GET requests; ${coverage})`);
} finally {
  await browser?.close();
  server.stop(true);
}
