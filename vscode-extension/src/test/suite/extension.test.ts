/**
 * Runs INSIDE a real VS Code extension host (via `runTest.ts` /
 * `@vscode/test-electron`) -- genuine end-to-end coverage for the parts
 * that can't be unit-tested without the real `vscode` API: activation,
 * command registration, the webview panel, and the click-to-jump
 * navigation the panel's comments trigger.
 *
 * Requires a running backend at `codeReviewAi.backendUrl`'s default
 * (`http://127.0.0.1:8000`) and a workspace with real uncommitted changes
 * (`CODE_REVIEW_AI_TEST_WORKSPACE`, see `runTest.ts`) -- without either,
 * the backend-dependent tests skip rather than fail, so `npm run
 * test:integration` still means something when run without that setup.
 */

import * as assert from "node:assert/strict";
import * as vscode from "vscode";
import { anchorComments } from "../../anchoring";
import { reviewDiff } from "../../apiClient";
import { parseDiffMap } from "../../diffMap";
import { getWorkingDiff } from "../../git";

const BACKEND_URL = "http://127.0.0.1:8000";
const PANEL_TITLE = "Code Review AI: Review Results";

async function backendIsUp(): Promise<boolean> {
  try {
    const res = await fetch(`${BACKEND_URL}/health`);
    return res.ok;
  } catch {
    return false;
  }
}

async function workingDiff(): Promise<string> {
  const folder = vscode.workspace.workspaceFolders?.[0];
  return folder ? await getWorkingDiff(folder.uri.fsPath) : "";
}

async function waitForTab(label: string, timeoutMs: number): Promise<boolean> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const tabs = vscode.window.tabGroups.all.flatMap((g) => g.tabs);
    if (tabs.some((t) => t.label === label)) {
      return true;
    }
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  return false;
}

suite("Code Review AI extension (integration)", () => {
  test("extension activates and registers its commands", async () => {
    const ext = vscode.extensions.getExtension("MarlenMM.code-review-ai");
    assert.ok(ext, "extension not found -- is it loaded via --extensionDevelopmentPath?");
    await ext!.activate();
    assert.ok(ext!.isActive);

    const commands = await vscode.commands.getCommands(true);
    for (const id of [
      "codeReviewAi.reviewCurrentChanges",
      "codeReviewAi.jumpToComment",
      "codeReviewAi.clearHighlights",
    ]) {
      assert.ok(commands.includes(id), `${id} was not registered`);
    }
  });

  test("real git diff + real backend produce a valid ReviewResponse", async function () {
    if (!(await backendIsUp())) {
      this.skip();
      return;
    }
    const diff = await workingDiff();
    if (!diff.trim()) {
      this.skip();
      return;
    }

    const result = await reviewDiff(BACKEND_URL, { diff, mode: "fast" });
    assert.ok(result.merge_probability >= 0 && result.merge_probability <= 1);
    assert.ok(["MERGE", "CLOSE"].includes(result.merge_prediction));
    assert.equal(result.ml_model, "rf_v1_balanced");
    assert.equal(result.mode, "fast");
  });

  test("executing the command opens the results panel", async function () {
    if (!(await backendIsUp())) {
      this.skip();
      return;
    }
    if (!(await workingDiff()).trim()) {
      this.skip();
      return;
    }

    await vscode.commands.executeCommand("codeReviewAi.reviewCurrentChanges");
    assert.ok(
      await waitForTab(PANEL_TITLE, 15_000),
      `expected a "${PANEL_TITLE}" webview tab to open`,
    );
  });

  test("the real working diff anchors a comment to a real file and line", async function () {
    const diff = await workingDiff();
    if (!diff.trim()) {
      this.skip();
      return;
    }

    const files = parseDiffMap(diff);
    assert.ok(files.length > 0, "expected the working diff to touch at least one file");
    const file = files[0];

    // Quote a symbol the diff genuinely added. It must be >= 4 word chars:
    // `anchoring.ts` deliberately refuses 1-2 char tokens as too weak to
    // claim a line-precise anchor, so picking e.g. "if" would (correctly)
    // resolve to nothing and would be testing the guard, not the anchor.
    let symbol: string | undefined;
    for (const added of file.addedLines) {
      const match = added.text.match(/[A-Za-z_][A-Za-z0-9_]{3,}/);
      if (match) {
        symbol = match[0];
        break;
      }
    }
    assert.ok(symbol, "expected an added line containing a usable identifier");

    // Name the file too: the same symbol may well appear in the other
    // changed file, and an ambiguous symbol is rejected by design.
    const [anchored] = anchorComments(
      [`In \`${file.path}\`, the \`${symbol}\` change needs a test.`],
      files,
    );
    assert.ok(anchored.location, "expected the comment to resolve to a location");
    assert.equal(anchored.location!.path, file.path);
    // Documented behaviour: the FIRST added line containing the symbol.
    const expected = file.addedLines.find((l) => l.text.includes(symbol!))!.line;
    assert.equal(anchored.location!.line, expected);
  });

  test("jumpToComment opens the file and puts the cursor on the right line", async function () {
    const diff = await workingDiff();
    if (!diff.trim()) {
      this.skip();
      return;
    }
    const files = parseDiffMap(diff);
    const target = files[0].addedLines[0];
    assert.ok(target, "expected the diff to add at least one line");

    await vscode.commands.executeCommand("codeReviewAi.jumpToComment", {
      path: files[0].path,
      line: target.line,
      precision: "line",
    });

    const editor = vscode.window.activeTextEditor;
    assert.ok(editor, "expected an active editor after jumping");
    assert.ok(
      editor!.document.uri.fsPath.endsWith(files[0].path),
      `expected ${files[0].path} to be open, got ${editor!.document.uri.fsPath}`,
    );
    // `line` is 1-based; the editor's Position is 0-based.
    assert.equal(editor!.selection.active.line, target.line - 1);
  });

  test("jumpToComment on a missing file warns instead of throwing", async () => {
    await vscode.commands.executeCommand("codeReviewAi.jumpToComment", {
      path: "definitely/not/here.py",
      line: 3,
      precision: "line",
    });
    // Reaching here without an unhandled rejection is the assertion.
    assert.ok(true);
  });

  test("clearHighlights runs cleanly", async () => {
    await vscode.commands.executeCommand("codeReviewAi.clearHighlights");
    assert.ok(true);
  });
});
