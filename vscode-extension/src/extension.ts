/**
 * Extension entry point: registers the three commands and keeps the
 * inline highlights in sync with whichever editors are visible.
 *
 * Still deliberately thin -- the reviewing logic lives in `git.ts`,
 * `apiClient.ts`, `diffMap.ts` and `anchoring.ts` (all unit-tested with no
 * `vscode` dependency); this module is the VS Code-API wiring between
 * those, `panel.ts` and `decorations.ts`.
 *
 * `codeReviewAi.jumpToComment` is a real registered command rather than a
 * closure private to the webview's message handler, for two reasons: the
 * panel's click handler and any future code path (a CodeLens, a tree view)
 * share one implementation, and the navigation behaviour becomes directly
 * testable from the integration suite without simulating a webview click.
 */

import * as vscode from "vscode";
import { anchorComments, type CommentLocation } from "./anchoring";
import { reviewDiff } from "./apiClient";
import {
  applyToVisibleEditors,
  clearReviewComments,
  disposeDecorations,
  setReviewComments,
} from "./decorations";
import { parseDiffMap } from "./diffMap";
import { getWorkingDiff } from "./git";
import { disposeReviewPanel, showReviewPanel } from "./panel";

export function activate(context: vscode.ExtensionContext): void {
  context.subscriptions.push(
    vscode.commands.registerCommand("codeReviewAi.reviewCurrentChanges", () =>
      reviewCurrentChanges(context),
    ),
    vscode.commands.registerCommand("codeReviewAi.jumpToComment", (location: CommentLocation) =>
      jumpToComment(location),
    ),
    vscode.commands.registerCommand("codeReviewAi.clearHighlights", () => {
      clearReviewComments();
    }),
    // Decorations live on editors, which come and go; re-apply whenever the
    // visible set changes so opening a reviewed file later still shows its
    // highlights rather than only the editors open at review time.
    vscode.window.onDidChangeVisibleTextEditors(() => applyToVisibleEditors()),
  );
}

export function deactivate(): void {
  disposeDecorations();
  disposeReviewPanel();
}

async function jumpToComment(location: CommentLocation | undefined): Promise<void> {
  const folder = vscode.workspace.workspaceFolders?.[0];
  if (!folder || !location?.path) {
    return;
  }

  const uri = vscode.Uri.joinPath(folder.uri, location.path);
  try {
    const document = await vscode.workspace.openTextDocument(uri);
    const editor = await vscode.window.showTextDocument(document, {
      viewColumn: vscode.ViewColumn.One,
      preview: false,
    });
    // Clamp: the file may have been edited since the reviewed diff was taken.
    const line = Math.max(0, Math.min((location.line ?? 1) - 1, document.lineCount - 1));
    const position = new vscode.Position(line, 0);
    editor.selection = new vscode.Selection(position, position);
    editor.revealRange(new vscode.Range(position, position), vscode.TextEditorRevealType.InCenter);
  } catch {
    vscode.window.showWarningMessage(
      `Code Review AI: could not open ${location.path} — the comment may refer to a file that no longer exists.`,
    );
  }
}

async function reviewCurrentChanges(context: vscode.ExtensionContext): Promise<void> {
  const folder = vscode.workspace.workspaceFolders?.[0];
  if (!folder) {
    vscode.window.showErrorMessage("Code Review AI: open a folder with a git repository first.");
    return;
  }

  const config = vscode.workspace.getConfiguration("codeReviewAi");
  const backendUrl = config.get<string>("backendUrl", "http://127.0.0.1:8000");
  const mode = config.get<"fast" | "deep">("mode", "fast");

  await vscode.window.withProgress(
    {
      location: vscode.ProgressLocation.Notification,
      title: "Code Review AI: reviewing current changes...",
    },
    async () => {
      let diff: string;
      try {
        diff = await getWorkingDiff(folder.uri.fsPath);
      } catch (err) {
        vscode.window.showErrorMessage(`Code Review AI: ${(err as Error).message}`);
        return;
      }

      if (!diff.trim()) {
        vscode.window.showInformationMessage("Code Review AI: no uncommitted changes to review.");
        return;
      }

      try {
        const result = await reviewDiff(backendUrl, { diff, mode });
        const comments = anchorComments(result.review_comments ?? [], parseDiffMap(diff));
        showReviewPanel(context, result, comments);
        setReviewComments(comments, folder.uri);
      } catch (err) {
        vscode.window.showErrorMessage(`Code Review AI: ${(err as Error).message}`);
      }
    },
  );
}
