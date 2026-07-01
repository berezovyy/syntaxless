/**
 * Thin functions over the opencode SDK.
 *
 * This is the only module that imports `@opencode-ai/sdk`; keeping the surface
 * small means an SDK change moves one file, and the runtime `@syntaxless/cron`
 * package never sees the dependency. The process attaches to a single opencode
 * server lazily, on first use, so importing the runner has no side effects.
 *
 * The model and provider come from the environment, never from the repo. No key
 * or token is read, written, or logged here.
 */
import { createOpencode } from "@opencode-ai/sdk";
import type {
  OpencodeClient,
  Session,
  SessionPromptResponse,
  TextPartInput,
} from "@opencode-ai/sdk";

export interface ModelRef {
  providerID: string;
  modelID: string;
}

interface Instance {
  client: OpencodeClient;
  stop(): void;
}

let instance: Instance | undefined;

async function client(): Promise<OpencodeClient> {
  if (instance === undefined) {
    const started = await createOpencode();

    instance = { client: started.client, stop: () => started.server.close() };
  }

  return instance.client;
}

/**
 * Resolve the model/provider to ask from the environment. Accepts either
 * `OPENCODE_MODEL=provider/model` or separate `OPENCODE_PROVIDER` /
 * `OPENCODE_MODEL_ID` variables. Never falls back to a hard-coded provider.
 */
export function model(): ModelRef {
  const providerID = (process.env.OPENCODE_PROVIDER ?? process.env.OPENCODE_PROVIDER_ID)?.trim();
  const modelID = (process.env.OPENCODE_MODEL_ID ?? process.env.OPENCODE_MODEL)?.trim();

  if (providerID !== undefined && providerID.length > 0) {
    if (modelID === undefined || modelID.length === 0) {
      throw new Error("opencode model not configured; set OPENCODE_MODEL_ID (provider is set)");
    }

    const slash = modelID.indexOf("/");

    return { providerID, modelID: slash === -1 ? modelID : modelID.slice(slash + 1) };
  }

  const combined = process.env.OPENCODE_MODEL?.trim();

  if (combined !== undefined && combined.includes("/")) {
    const slash = combined.indexOf("/");
    const provider = combined.slice(0, slash);
    const id = combined.slice(slash + 1);

    if (provider.length > 0 && id.length > 0) return { providerID: provider, modelID: id };
  }

  throw new Error(
    "opencode model not configured; set OPENCODE_MODEL=provider/model " +
      "(or OPENCODE_PROVIDER and OPENCODE_MODEL_ID)",
  );
}

function requestError(error: unknown): Error {
  if (error instanceof Error) return error;

  if (typeof error === "object" && error !== null) {
    const record = error as Record<string, unknown>;
    const message = record.message ?? record.name ?? "request failed";
    const wrapped = new Error(`opencode: ${String(message)}`);
    const status = record.status ?? record.statusCode;

    if (typeof status === "number") {
      (wrapped as Error & { status?: number }).status = status;
    }

    return wrapped;
  }

  return new Error(`opencode: ${String(error)}`);
}

function unwrap<T>(result: { data: T | undefined; error: unknown }): T {
  if (result.data !== undefined) return result.data;
  throw requestError(result.error);
}

export async function createSession(title: string): Promise<string> {
  const result = await (await client()).session.create({ body: { title } });
  const session = unwrap<Session>(result);

  return session.id;
}

export async function prompt(sessionId: string, system: string, text: string): Promise<string> {
  const result = await (
    await client()
  ).session.prompt({
    path: { id: sessionId },
    body: {
      model: model(),
      system,
      tools: {},
      parts: [{ type: "text", text } satisfies TextPartInput],
    },
  });

  return assistantText(unwrap<SessionPromptResponse>(result));
}

function assistantText(response: SessionPromptResponse): string {
  let text = "";

  for (const part of response.parts) {
    if (part.type === "text") text += part.text;
  }

  return text;
}

/** Stop the shared opencode server. Safe to call when no session was opened. */
export async function close(): Promise<void> {
  const current = instance;

  instance = undefined;
  current?.stop();
}
