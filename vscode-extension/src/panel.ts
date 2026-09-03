/**
 * Webview lifecycle for the review panel. The HTML itself is built by
 * `panelHtml.ts` (no `vscode` import, so it is unit-testable) -- this
 * module owns only what needs the real editor API.
 *
 * Three deliberate choices:
 *
 * * **One reused panel, not one per run.** `showReviewPanel` reveals and
 *   re-renders the existing panel if it is still open, so repeatedly
 *   running the command doesn't bury the editor under stale result tabs.
 * * **The panel opens at the *start* of a review, not the end.** It shows
 *   the loading state first and is then re-rendered in place with the
 *   result, the empty state or the error state. Every outcome therefore
 *   lands somewhere the user can read at their own pace -- a review that
 *   fails, or finds nothing, used to produce a toast that vanished and no
 *   panel at all, which is the worst version of the first run.
 * * **Scripts are enabled, so the CSP is explicit.** Click-to-jump and the
 *   retry button need `postMessage`, so the document ships a nonce-based
 *   Content-Security-Policy forbidding everything (`default-src 'none'`)
 *   except its own inline style block and its own nonced script.
 *   `localResourceRoots: []` because every asset is inlined -- the webview
 *   never needs to read from disk.
 */

import * as vscode from "vscode";
import type { AnchoredComment } from "./anchoring";
import type { ReviewResponse } from "./apiClient";
import {
  renderEmptyHtml,
  renderErrorHtml,
  renderLoadingHtml,
  renderPanelHtml,
} from "./panelHtml";

const VIEW_TYPE = "codeReviewAi.reviewPanel";
const PANEL_TITLE = "Code Review AI: Review Results";

let currentPanel: vscode.WebviewPanel | undefined;

/** Set by `extension.ts` so the panel's "Try again" button can re-run the
 * review. A callback rather than a direct `executeCommand` so the panel
 * stays ignorant of which command drives it. */
let onRerun: (() => void) | undefined;

export function setRerunHandler(handler: () => void): void {
  onRerun = handler;
}

export function showReviewPanel(
  context: vscode.ExtensionContext,
  result: ReviewResponse,
  comments: AnchoredComment[],
): void {
  render(context, (nonce) => renderPanelHtml(result, comments, nonce));
}

/** Opened immediately when a review starts; replaced in place when it
 * finishes. `filesChanged` is counted from the diff the extension already
 * read, so the wait names real work rather than an anonymous spinner. */
export function showLoadingPanel(
  context: vscode.ExtensionContext,
  info: { mode: string; filesChanged: number; backendUrl: string },
): void {
  render(context, (nonce) => renderLoadingHtml(nonce, info));
}

export function showEmptyPanel(context: vscode.ExtensionContext): void {
  render(context, (nonce) => renderEmptyHtml(nonce));
}

export function showErrorPanel(
  context: vscode.ExtensionContext,
  error: { kind: "git" | "unreachable" | "http"; message: string; backendUrl: string },
): void {
  render(context, (nonce) => renderErrorHtml(nonce, error));
}

/** Reveal-or-create, then set the HTML. Every state goes through here so
 * they all share the panel-reuse and message-wiring behaviour. */
function render(
  context: vscode.ExtensionContext,
  html: (nonce: string) => string,
): void {
  if (currentPanel) {
    currentPanel.webview.html = html(makeNonce());
    currentPanel.reveal(currentPanel.viewColumn ?? vscode.ViewColumn.Beside, true);
    return;
  }

  const panel = vscode.window.createWebviewPanel(
    VIEW_TYPE,
    PANEL_TITLE,
    { viewColumn: vscode.ViewColumn.Beside, preserveFocus: true },
    { enableScripts: true, localResourceRoots: [] },
  );
  currentPanel = panel;

  panel.webview.onDidReceiveMessage(
    (message: { type?: string; path?: string; line?: number }) => {
      if (message?.type === "jump" && message.path && typeof message.line === "number") {
        void vscode.commands.executeCommand("codeReviewAi.jumpToComment", {
          path: message.path,
          line: message.line,
        });
        return;
      }
      if (message?.type === "rerun") {
        onRerun?.();
      }
    },
    undefined,
    context.subscriptions,
  );

  panel.onDidDispose(
    () => {
      currentPanel = undefined;
    },
    undefined,
    context.subscriptions,
  );

  panel.webview.html = html(makeNonce());
  context.subscriptions.push(panel);
}

/** Exposed for `deactivate()`; safe to call when nothing is open. */
export function disposeReviewPanel(): void {
  currentPanel?.dispose();
  currentPanel = undefined;
}

function makeNonce(): string {
  const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  let nonce = "";
  for (let i = 0; i < 32; i += 1) {
    nonce += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return nonce;
}
