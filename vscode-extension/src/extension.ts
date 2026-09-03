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
 *
 * Every outcome of a review renders into the panel -- result, nothing to
 * review, or failure -- rather than into a notification toast. Progress is
 * reported in the status bar (`ProgressLocation.Window`) instead of as a
 * notification, because the panel is already showing the loading state and
 * two simultaneous progress reports for one action is one too many.
 */

import * as vscode from "vscode";
import { anchorComments, type CommentLocation } from "./anchoring";
import { ReviewError, reviewDiff } from "./apiClient";
import {
  applyToVisibleEditors,
  clearReviewComments,
  disposeDecorations,
  initDecorations,
  setReviewComments,
} from "./decorations";
import { parseDiffMap } from "./diffMap";
import { getWorkingDiff } from "./git";
import {
  disposeReviewPanel,
  setRerunHandler,
  showEmptyPanel,
  showErrorPanel,
  showLoadingPanel,
  showReviewPanel,
} from "./panel";

/** The panel's "Try again" button and the Command Palette reach the same
 * function; this stops an impatient second click starting a second review
 * while the first is still in flight. */
let reviewInFlight = false;

export function activate(context: vscode.ExtensionContext): void {
  initDecorations(context.extensionUri);
  setRerunHandler(() => void reviewCurrentChanges(context));

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
      `Code Review AI: can't open ${location.path}. Comments are anchored to the diff you `
        + `reviewed, so a file renamed or deleted since then won't be there.`,
    );
  }
}

async function reviewCurrentChanges(context: vscode.ExtensionContext): Promise<void> {
  if (reviewInFlight) {
    return;
  }

  const config = vscode.workspace.getConfiguration("codeReviewAi");
  const backendUrl = config.get<string>("backendUrl", "http://127.0.0.1:8000");
  const mode = config.get<"fast" | "deep">("mode", "fast");

  const folder = vscode.workspace.workspaceFolders?.[0];
  if (!folder) {
    showErrorPanel(context, {
      kind: "git",
      message: "No folder is open, so there is no working tree to diff.",
      backendUrl,
    });
    return;
  }

  reviewInFlight = true;
  try {
    let diff: string;
    try {
      diff = await getWorkingDiff(folder.uri.fsPath);
    } catch (err) {
      showErrorPanel(context, { kind: "git", message: messageOf(err), backendUrl });
      return;
    }

    if (!diff.trim()) {
      showEmptyPanel(context);
      return;
    }

    showLoadingPanel(context, {
      mode,
      filesChanged: parseDiffMap(diff).length,
      backendUrl,
    });

    await vscode.window.withProgress(
      { location: vscode.ProgressLocation.Window, title: "Code Review AI: reviewing your changes" },
      async () => {
        try {
          const result = await reviewDiff(backendUrl, { diff, mode });
          const comments = anchorComments(result.review_comments ?? [], parseDiffMap(diff));
          showReviewPanel(context, result, comments);
          setReviewComments(comments, folder.uri);
        } catch (err) {
          showErrorPanel(context, {
            // Anything that isn't a `ReviewError` got past the fetch, so it
            // is a problem with the answer rather than with reaching it.
            kind: err instanceof ReviewError ? err.kind : "http",
            message: messageOf(err),
            backendUrl,
          });
        }
      },
    );
  } finally {
    reviewInFlight = false;
  }
}

function messageOf(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}
