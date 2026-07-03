import { resolve } from "node:path";

/**
 * Browser inference test (Phase 20).
 *
 * A real Chromium page loads the built package over a Vite dev server. Every
 * request is routed so the test can (a) count requests for the lazily imported
 * model module and (b) prove no request carries the user's sentence, tokens, or
 * confidence. The model itself is irrelevant here: a weak smoke model is fine,
 * because this test asserts lifecycle counting and privacy, not accuracy.
 *
 * Browsers cache ES modules per document, so `dispose`-then-`parse` cannot
 * produce a second network request inside one document. The dispose case is
 * therefore exercised in a fresh document (a reload) while the route counter
 * persists, which is the only browser-accurate way to observe re-initialization.
 */
import { chromium } from "playwright";
import { createServer } from "vite";

const workspaceRoot = resolve(import.meta.dirname, "../../..");
const ENTRY = "/packages/cron/dist/index.js";
const SENTENCE = "every weekday at 9";

interface RecordedRequest {
  url: string;
  body: string | null;
}

interface Recording {
  requests: RecordedRequest[];
  modelChunkCount: number;
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

await server.listen();
const base = server.resolvedUrls!.local[0]!;

const browser = await chromium.launch({ headless: true });
const recordings: Recording[] = [];

async function recordedPage(): Promise<{
  page: Awaited<ReturnType<typeof browser.newPage>>;
  recording: Recording;
}> {
  const page = await browser.newPage();
  const recording: Recording = { requests: [], modelChunkCount: 0 };

  recordings.push(recording);

  await page.route("**/*", (route) => {
    const request = route.request();
    const url = request.url();

    if (url.endsWith("/blank.html")) {
      return route.fulfill({
        contentType: "text/html",
        body: "<!doctype html><title>cron browser</title>",
      });
    }

    if (url.includes("/model/inference.")) recording.modelChunkCount += 1;
    recording.requests.push({ url, body: request.postData() });

    return route.continue();
  });

  return { page, recording };
}

try {
  const origin = base + "blank.html";

  // Zero model requests from importing the entry point alone.
  {
    const { page, recording } = await recordedPage();

    await page.goto(origin);

    await page.evaluate(async (entry) => {
      await import(entry);
    }, ENTRY);

    check(
      recording.modelChunkCount === 0,
      `importing the entry requested the model ${recording.modelChunkCount} times, expected 0`,
    );

    await page.close();
  }

  // Exactly one model request on the first parse.
  {
    const { page, recording } = await recordedPage();

    await page.goto(origin);

    await page.evaluate(
      async ({ entry, sentence }) => {
        const module = await import(entry);

        await module.parse(sentence);
      },
      { entry: ENTRY, sentence: SENTENCE },
    );

    check(
      recording.modelChunkCount === 1,
      `first parse requested the model ${recording.modelChunkCount} times, expected 1`,
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
      recording.modelChunkCount === 1,
      `concurrent parses requested the model ${recording.modelChunkCount} times, expected 1`,
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
      recording.modelChunkCount === 1,
      `preload requested the model ${recording.modelChunkCount} times, expected 1`,
    );

    await page.evaluate(
      async ({ entry, sentence }) => {
        const module = await import(entry);

        await module.parse(sentence);
      },
      { entry: ENTRY, sentence: SENTENCE },
    );

    check(
      recording.modelChunkCount === 1,
      `parse after preload re-initialized the model (${recording.modelChunkCount} requests), expected no re-init`,
    );

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
      recording.modelChunkCount === 1,
      `preload requested the model ${recording.modelChunkCount} times, expected 1`,
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
      recording.modelChunkCount === 2,
      `dispose did not allow re-initialization (${recording.modelChunkCount} requests, expected 2)`,
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
    (total, recording) => total + recording.modelChunkCount,
    0,
  );

  console.log(
    `browser: ok (${modelRequests} model-chunk requests across ${recordings.length} recordings)`,
  );
} finally {
  await browser.close();
  await server.close();
}
