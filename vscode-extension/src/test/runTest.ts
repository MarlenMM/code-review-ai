/**
 * Entry point for the INTEGRATION test pass (`npm run test:integration`),
 * as opposed to `npm test`'s plain-Node unit tests for `git.ts`/`apiClient.ts`.
 *
 * `@vscode/test-electron` downloads a real VS Code build and launches it
 * with this extension loaded via `--extensionDevelopmentPath`, then runs
 * `suite/index.ts`'s Mocha suite *inside* that real extension host --
 * genuine end-to-end verification (activation, command registration, a
 * real webview panel) without driving a GUI by mouse/keyboard, which is
 * both the standard way to test a VS Code extension and the only
 * practical one in an environment where GUI automation of an IDE is
 * click-only (see `reports/vscode_extension_design.md`).
 *
 * `CODE_REVIEW_AI_TEST_WORKSPACE` lets the caller point the test VS Code
 * instance at a real git repo with uncommitted changes (the suite needs a
 * non-empty `git diff` to exercise the full command path); it defaults to
 * this extension's own folder, which has no uncommitted changes by design,
 * so the command-execution test would have nothing to review unless a
 * real workspace is supplied.
 */

import * as path from "node:path";
import { runTests } from "@vscode/test-electron";

async function main(): Promise<void> {
  const extensionDevelopmentPath = path.resolve(__dirname, "../../");
  const extensionTestsPath = path.resolve(__dirname, "./suite/index");
  const workspacePath = process.env.CODE_REVIEW_AI_TEST_WORKSPACE
    ?? path.resolve(__dirname, "../../");

  await runTests({
    extensionDevelopmentPath,
    extensionTestsPath,
    // `--user-data-dir`/`--extensions-dir` pinned to short, fixed /tmp paths:
    // the default (nested under this project's own, fairly deep path) makes
    // VS Code derive a Unix domain socket path over macOS's 103-char limit
    // ("listen EINVAL: invalid argument ... /1.13-main.sock"), which fails
    // before any test code runs.
    launchArgs: [
      workspacePath,
      "--disable-extensions",
      "--user-data-dir=/tmp/code-review-ai-vscode-test-user-data",
      "--extensions-dir=/tmp/code-review-ai-vscode-test-extensions",
    ],
  });
}

main().catch((err) => {
  console.error("Integration tests failed:", err);
  process.exit(1);
});
