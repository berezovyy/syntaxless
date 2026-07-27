/** Repository-relative paths for records that must stay valid on any checkout. */
import { isAbsolute, relative, resolve } from "node:path";

export const repositoryRoot = resolve(import.meta.dir, "../../..");

/** Resolve a recorded path; relative paths are anchored at the repository root. */
export const fromRepository = (path: string) => resolve(repositoryRoot, path);

/** Record a path relative to the repository root, using forward slashes. */
export function toRepository(path: string): string {
  const result = relative(repositoryRoot, resolve(path)).split("\\").join("/");

  if (result.startsWith("..") || isAbsolute(result)) {
    throw new Error(`path is outside the repository: ${path}`);
  }

  return result;
}
