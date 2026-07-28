/**
 * Reads the workspace's current uncommitted changes via `git diff`.
 *
 * Scope, deliberately: plain `git diff` (working tree vs. the index) --
 * matching plan Section 7.2's literal wording ("reads the working diff
 * (`git diff`) of the currently open workspace"). Staged-only changes
 * (`git diff --cached`) and changes against a different base (e.g. the
 * merge-base with `main`) are a reasonable future extension, not this
 * command's job.
 *
 * `ExecFn` is injected (defaulting to the real `child_process.exec`) so
 * `reviewCurrentChanges`'s logic is testable without a real git repository
 * -- see `src/test/git.test.ts`.
 */

import { exec } from "node:child_process";
import { promisify } from "node:util";

export interface ExecResult {
  stdout: string;
  stderr: string;
}

export type ExecFn = (command: string, options: { cwd: string }) => Promise<ExecResult>;

const defaultExec = promisify(exec) as ExecFn;

/** Returns the raw `git diff` output for `cwd`, or throws a readable Error
 * (e.g. "not a git repository", git not installed) with the command's
 * stderr surfaced -- `child_process.exec`'s promisified form attaches
 * `stdout`/`stderr` to the rejected Error itself. */
export async function getWorkingDiff(cwd: string, execFn: ExecFn = defaultExec): Promise<string> {
  try {
    const { stdout } = await execFn("git diff", { cwd });
    return stdout;
  } catch (err) {
    const stderr = (err as { stderr?: string }).stderr?.trim();
    const message = stderr || (err instanceof Error ? err.message : String(err));
    throw new Error(`git diff failed: ${message}`);
  }
}
