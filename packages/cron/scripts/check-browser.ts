import { resolve } from "node:path";

/** Browser checks for lazy artifact loading, engine reuse, and on-device privacy. */
import { chromium } from "playwright";
import { createServer } from "vite";

const workspaceRoot = resolve(import.meta.dirname, "../../..");
const ENTRY = "/packages/cron/dist/index.js";
const SENTENCE = "every weekday at 9am";
const WEIGHTS = "/packages/cron/dist/weights.gen.js";

interface RecordedRequest {
  url: string;
  body: string | null;
}

interface Recording {
  requests: RecordedRequest[];
  weightRequests: number;
}

function check(condition: boolean, message: string): void {
  if (!condition) throw new Error(`browser: ${message}`);
}

const server = await createServer({
  configFile: false,
  root: workspaceRoot,
  server: { host: "127.0.0.1", port: 0 },
  logLevel: "error",
  optimizeDeps: { noDiscovery: true, include: [] },
});

try {
  await server.listen();
  const base = server.resolvedUrls!.local[0]!;
  const browser = await chromium.launch({ headless: true });

  try {
    const recordings: Recording[] = [];

    async function recordedPage(): Promise<{
      page: Awaited<ReturnType<typeof browser.newPage>>;
      recording: Recording;
    }> {
      const page = await browser.newPage();
      const recording: Recording = { requests: [], weightRequests: 0 };

      recordings.push(recording);

      await page.route("**/*", (route) => {
        const request = route.request();
        const url = request.url();

        check(
          new URL(url).origin === new URL(base).origin,
          "browser attempted an external request",
        );

        check(
          request.method() === "GET" && request.postData() === null,
          "browser attempted to transmit request data",
        );

        if (url.endsWith("/blank.html")) {
          return route.fulfill({
            contentType: "text/html",
            body: "<!doctype html><title>cron browser</title>",
          });
        }

        if (new URL(url).pathname === WEIGHTS) recording.weightRequests += 1;
        recording.requests.push({ url, body: request.postData() });

        return route.continue();
      });

      return { page, recording };
    }

    const origin = base + "blank.html";

    // Zero model requests from importing the entry point alone.
    {
      const { page, recording } = await recordedPage();

      await page.goto(origin);

      await page.evaluate(async (entry) => {
        const module = await import(entry);

        const intent = {
          frequency: { unit: "hour", interval: 2 },
          time: { minute: 15 },
        };

        module.validate(intent);

        if (!module.explain(intent).includes("15 */2 * * *")) {
          throw new Error("browser hourly explanation mismatch");
        }
      }, ENTRY);

      check(
        recording.weightRequests === 0,
        `importing the entry requested the model ${recording.weightRequests} times, expected 0`,
      );

      await page.addScriptTag({
        path: resolve(
          import.meta.dirname,
          "../node_modules/@js-temporal/polyfill/dist/index.umd.js",
        ),
      });

      await page.evaluate(async (entry) => {
        const module = await import(entry);
        const adapterEntry = "/packages/cron/dist/temporal.js";
        const adapter = await import(adapterEntry);

        const provider = (
          globalThis as unknown as {
            temporal: {
              Temporal: Parameters<typeof adapter.createTemporalCalendar>[0];
            };
          }
        ).temporal.Temporal;

        const result = module.recurrences(
          { frequency: { unit: "day", interval: 1 }, time: { hour: 2, minute: 30 } },
          {
            timezone: "America/New_York",
            reference: "2026-03-07T12:00:00Z",
            count: 1,
            calendar: adapter.createTemporalCalendar(provider),
            policies: { gap: "later" },
          },
        );

        if (!result.ok || result.occurrences[0]?.instant !== "2026-03-08T07:30:00.000Z") {
          throw new Error("browser zoned recurrence mismatch");
        }
      }, ENTRY);

      check(recording.weightRequests === 0, "recurrence evaluation loaded the model");
      await page.close();
    }

    // Exactly one model request on the first parse.
    {
      const { page, recording } = await recordedPage();

      await page.goto(origin);

      await page.evaluate(
        async ({ entry, sentence }) => {
          const module = await import(entry);
          const result = await module.parse(sentence);

          if (!result.ok || result.intent.time?.hour !== 9) {
            throw new Error("browser parse changed the explicit morning clock");
          }
        },
        { entry: ENTRY, sentence: SENTENCE },
      );

      check(
        recording.weightRequests === 1,
        `first parse requested weights ${recording.weightRequests} times, expected 1`,
      );

      await page.close();
    }

    // Concurrent parses coalesce into one model request.
    {
      const { page, recording } = await recordedPage();

      await page.goto(origin);

      await page.evaluate(async (entry) => {
        const module = await import(entry);

        await Promise.all([
          module.parse("every day"),
          module.parse("every week"),
          module.parse("every hour"),
        ]);
      }, ENTRY);

      check(
        recording.weightRequests === 1,
        `concurrent parses requested the model ${recording.weightRequests} times, expected 1`,
      );

      await page.close();
    }

    // preload() initializes once; a later parse reuses it.
    {
      const { page, recording } = await recordedPage();

      await page.goto(origin);

      await page.evaluate(async (entry) => {
        const module = await import(entry);

        await module.preload();
      }, ENTRY);

      check(
        recording.weightRequests === 1,
        `preload requested the model ${recording.weightRequests} times, expected 1`,
      );

      await page.evaluate(
        async ({ entry, sentence }) => {
          const module = await import(entry);

          await module.parse(sentence);
        },
        { entry: ENTRY, sentence: SENTENCE },
      );

      check(
        recording.weightRequests === 1,
        `parse after preload re-initialized the model (${recording.weightRequests} requests), ` +
          "expected no re-init",
      );

      await page.context().setOffline(true);

      await page.evaluate(
        async ({ entry, sentence }) => {
          const module = await import(entry);
          const result = await module.parse(sentence);

          if (!result.ok || result.intent.time?.hour !== 9) {
            throw new Error("preloaded parsing failed offline");
          }
        },
        { entry: ENTRY, sentence: SENTENCE },
      );

      check(recording.weightRequests === 1, "offline parsing requested weights again");
      await page.close();
    }

    // dispose() releases the engine; a fresh document initializes again.
    {
      const { page, recording } = await recordedPage();

      await page.goto(origin);

      await page.evaluate(async (entry) => {
        const module = await import(entry);

        await module.preload();
      }, ENTRY);

      check(
        recording.weightRequests === 1,
        `preload requested the model ${recording.weightRequests} times, expected 1`,
      );

      await page.evaluate(async (entry) => {
        const module = await import(entry);

        module.dispose();
      }, ENTRY);

      await page.reload();

      await page.evaluate(
        async ({ entry, sentence }) => {
          const module = await import(entry);

          await module.parse(sentence);
        },
        { entry: ENTRY, sentence: SENTENCE },
      );

      check(
        recording.weightRequests === 2,
        "dispose did not allow re-initialization " +
          `(${recording.weightRequests} requests, expected 2)`,
      );

      await page.close();
    }

    // Privacy: no request URL or body carries the sentence or its tokens.
    const needles = [SENTENCE, "every", "weekday"];

    for (const recording of recordings) {
      for (const request of recording.requests) {
        for (const needle of needles) {
          check(
            !request.url.toLowerCase().includes(needle),
            `request URL leaked user text ${JSON.stringify(needle)}: ${request.url}`,
          );

          check(
            request.body === null || !request.body.toLowerCase().includes(needle),
            `request body leaked user text ${JSON.stringify(needle)}: ${request.body}`,
          );
        }
      }
    }

    const modelRequests = recordings.reduce(
      (total, recording) => total + recording.weightRequests,
      0,
    );

    console.log(
      `browser: ok (${modelRequests} weight requests across ${recordings.length} recordings)`,
    );
  } finally {
    await browser.close();
  }
} finally {
  await server.close();
}
