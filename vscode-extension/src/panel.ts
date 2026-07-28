/**
 * Webview lifecycle for the review results panel. The HTML itself is built
 * by `panelHtml.ts` (no `vscode` import, so it is unit-testable) -- this
 * module owns only what needs the real editor API.
 *
 * Two deliberate choices:
 *
 * * **One reused panel, not one per run.** `showReviewPanel` reveals and
 *   re-renders the existing panel if it is still open, so repeatedly
 *   running the command doesn't bury the editor under stale result tabs.
 * * **Scripts are enabled now, so the CSP is explicit.** Step 24 could set
 *   `enableScripts: false`; click-to-jump needs `postMessage`, so the
 *   document ships a nonce-based Content-Security-Policy forbidding
 *   everything (`default-src 'none'`) except its own inline style block
 *   and its own nonced script. `localResourceRoots: []` because every
 *   asset is inlined -- the webview never needs to read from disk.
 */

import * as vscode from "vscode";
import type { AnchoredComment } from "./anchoring";
import type { ReviewResponse } from "./apiClient";
import { renderPanelHtml } from "./panelHtml";

const VIEW_TYPE = "codeReviewAi.reviewPanel";
const PANEL_TITLE = "Code Review AI: Review Results";

let currentPanel: vscode.WebviewPanel | undefined;

export function showReviewPanel(
  context: vscode.ExtensionContext,
  result: ReviewResponse,
  comments: AnchoredComment[],
): void {
  if (currentPanel) {
    currentPanel.webview.html = renderPanelHtml(result, comments, makeNonce());
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

  panel.webview.html = renderPanelHtml(result, comments, makeNonce());
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
