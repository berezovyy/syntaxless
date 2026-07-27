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
      return new Response("<!doctype html><title>Retention browser check</title>", {
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
      api.parse("keep daily backups for a week and monthly backups for a year"),
      api.preload(),
    ]);

    if (!parsed.ok) throw new Error("browser reference failed: " + JSON.stringify(parsed));

    return {
      parsed,
      compiled: api.compile(parsed.intent, {
        timeZone: "UTC",
        referenceInstant: Date.parse("2024-03-15T12:00:00Z"),
        backups: [{ id: "current", timestamp: Date.parse("2024-03-15T00:00:00Z") }],
      }),
    };
  }, "/index.js");

  if (
    !initial.compiled.supported ||
    JSON.stringify(initial.compiled.keep) !== '[{"id":"current","tierIndexes":[0,1]}]'
  ) {
    throw new Error("browser meaning mismatch");
  }

  if (requests.filter((path) => path.endsWith("weights.gen.js")).length !== 1) {
    throw new Error("concurrent operations did not share the model load");
  }

  await page.context().setOffline(true);

  const offline = await page.evaluate(async (entry) => {
    const api = await import(entry);
    const accepted = await api.parse("keep daily backups for the last 48 hours");
    const rejected = await api.parse("keep daily backups for a week?");

    const compiled = accepted.ok
      ? api.compile(accepted.intent, {
          timeZone: "America/New_York",
          referenceInstant: Date.parse("2024-03-11T04:30:00Z"),
          backups: [{ id: "old-local-date", timestamp: Date.parse("2024-03-09T05:00:00Z") }],
        })
      : null;

    return { accepted, rejected, compiled };
  }, "/index.js");

  if (
    !offline.accepted.ok ||
    offline.rejected.ok ||
    !offline.compiled?.supported ||
    offline.compiled.keep[0]?.id !== "old-local-date"
  ) {
    throw new Error("offline browser parsing/calendar failed: " + JSON.stringify(offline));
  }

  const coverage = "lazy weights, concurrent initialization, offline parsing";

  console.log(`browser: passed (${requests.length} local GET requests; ${coverage})`);
} finally {
  await browser?.close();
  server.stop(true);
}
