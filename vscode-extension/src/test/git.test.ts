import assert from "node:assert/strict";
import { test } from "node:test";
import { getWorkingDiff } from "../git";

test("returns stdout from `git diff` in the given cwd", async () => {
  const calls: Array<{ command: string; cwd: string }> = [];
  const fakeExec = async (command: string, options: { cwd: string }) => {
    calls.push({ command, cwd: options.cwd });
    return { stdout: "diff --git a/x b/x\n@@ -1 +1 @@\n-a\n+b\n", stderr: "" };
  };

  const diff = await getWorkingDiff("/repo", fakeExec);

  assert.equal(diff, "diff --git a/x b/x\n@@ -1 +1 @@\n-a\n+b\n");
  assert.deepEqual(calls, [{ command: "git diff", cwd: "/repo" }]);
});

test("returns an empty string for a clean working tree", async () => {
  const fakeExec = async () => ({ stdout: "", stderr: "" });
  assert.equal(await getWorkingDiff("/repo", fakeExec), "");
});

test("wraps a failing invocation using the attached stderr", async () => {
  const fakeExec = async () => {
    const err = new Error("Command failed: git diff") as Error & { stderr?: string };
    err.stderr = "fatal: not a git repository (or any of the parent directories): .git\n";
    throw err;
  };

  await assert.rejects(
    () => getWorkingDiff("/not-a-repo", fakeExec),
    /not a git repository/,
  );
});

test("falls back to the error message when stderr is absent", async () => {
  const fakeExec = async () => {
    throw new Error("spawn git ENOENT");
  };

  await assert.rejects(
    () => getWorkingDiff("/repo", fakeExec),
    /spawn git ENOENT/,
  );
});
