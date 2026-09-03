/**
 * Inline highlights for anchored review comments: a whole-line background,
 * an overview-ruler mark, a gutter icon, and a truncated inline hint on
 * each line a comment resolved to, with the full comment text on hover.
 *
 * The marker is a real SVG (`media/review-comment-{dark,light}.svg`) in
 * the gutter, not a `💬` in the inline text. An emoji is the wrong tool
 * twice over: it renders in whatever colour and metrics the platform's
 * emoji font decides, ignoring the user's theme entirely, and it reads as
 * a placeholder next to VS Code's own icon language. The SVG is drawn to
 * the same 16px/1.5-stroke spec as the panel's icons and painted in the
 * panel's accent colour, so an anchored comment looks the same in the
 * gutter as it does in the panel.
 *
 * Only `precision: 'line'` anchors are decorated. File-level anchors
 * (`'file'` / `'inferred-file'`, see `anchoring.ts`) deliberately get no
 * inline highlight: the comment is about the file, not about whichever
 * line the anchor happened to fall back to, and painting it on that line
 * would assert a precision the heuristic never claimed. Those comments are
 * still click-to-jump in the panel -- navigation is a hint, a decoration
 * on a specific line is a claim.
 *
 * State is module-level (one review's results at a time, replacing the
 * previous) and keyed by absolute `fsPath`, so `applyToVisibleEditors` is
 * a cheap lookup that can also be re-run when the user opens a file later.
 * All colors come from `ThemeColor`, so the highlights follow the user's
 * theme instead of hard-coding light- or dark-mode values.
 */

import * as vscode from "vscode";
import type { AnchoredComment } from "./anchoring";

interface PendingDecoration {
  line: number; // 1-based
  text: string;
}

let decorationType: vscode.TextEditorDecorationType | undefined;
let byFsPath = new Map<string, PendingDecoration[]>();
/** Set once from `activate`; needed to resolve the gutter icon on disk. */
let extensionUri: vscode.Uri | undefined;

const MAX_INLINE_CHARS = 60;

/** Called from `activate` before any review runs, so the decoration type
 * can point at the packaged icon. */
export function initDecorations(uri: vscode.Uri): void {
  extensionUri = uri;
}

function iconPath(theme: "dark" | "light"): vscode.Uri | undefined {
  return extensionUri
    ? vscode.Uri.joinPath(extensionUri, "media", `review-comment-${theme}.svg`)
    : undefined;
}

function getDecorationType(): vscode.TextEditorDecorationType {
  if (!decorationType) {
    decorationType = vscode.window.createTextEditorDecorationType({
      isWholeLine: true,
      backgroundColor: new vscode.ThemeColor("editor.wordHighlightBackground"),
      overviewRulerLane: vscode.OverviewRulerLane.Right,
      overviewRulerColor: new vscode.ThemeColor("editorOverviewRuler.warningForeground"),
      gutterIconSize: "contain",
      dark: { gutterIconPath: iconPath("dark") },
      light: { gutterIconPath: iconPath("light") },
    });
  }
  return decorationType;
}

function truncate(text: string): string {
  const flat = text.replace(/\s+/g, " ").trim();
  return flat.length > MAX_INLINE_CHARS ? `${flat.slice(0, MAX_INLINE_CHARS - 1)}…` : flat;
}

/** Replace the current review's decorations and paint them immediately. */
export function setReviewComments(
  comments: AnchoredComment[],
  workspaceRoot: vscode.Uri,
): void {
  const next = new Map<string, PendingDecoration[]>();
  for (const comment of comments) {
    if (!comment.location || comment.location.precision !== "line") {
      continue;
    }
    const fsPath = vscode.Uri.joinPath(workspaceRoot, comment.location.path).fsPath;
    const list = next.get(fsPath) ?? [];
    list.push({ line: comment.location.line, text: comment.text });
    next.set(fsPath, list);
  }
  byFsPath = next;
  applyToVisibleEditors();
}

export function clearReviewComments(): void {
  byFsPath = new Map();
  applyToVisibleEditors();
}

export function applyToVisibleEditors(): void {
  const type = getDecorationType();
  for (const editor of vscode.window.visibleTextEditors) {
    editor.setDecorations(type, buildOptions(editor));
  }
}

function buildOptions(editor: vscode.TextEditor): vscode.DecorationOptions[] {
  const pending = byFsPath.get(editor.document.uri.fsPath);
  if (!pending || pending.length === 0) {
    return [];
  }

  // Group by line so two comments on the same line produce one decoration
  // with both texts, rather than two overlapping inline hints.
  const grouped = new Map<number, string[]>();
  for (const item of pending) {
    // Clamp: the file may have been edited since the diff was taken.
    const zeroBased = Math.max(0, Math.min(item.line - 1, editor.document.lineCount - 1));
    const list = grouped.get(zeroBased) ?? [];
    list.push(item.text);
    grouped.set(zeroBased, list);
  }

  return [...grouped.entries()].map(([line, texts]) => {
    const hover = new vscode.MarkdownString(
      texts.map((t) => `**Code Review AI** — ${t}`).join("\n\n---\n\n"),
    );
    hover.isTrusted = false;
    // The gutter icon already says "there is a comment here", so the inline
    // hint carries only what the icon can't: which comment, or how many.
    const label = texts.length > 1
      ? `${texts.length} review comments`
      : truncate(texts[0]);

    return {
      range: new vscode.Range(line, 0, line, 0),
      hoverMessage: hover,
      renderOptions: {
        after: {
          contentText: `   ${label}`,
          color: new vscode.ThemeColor("editorCodeLens.foreground"),
          fontStyle: "italic",
        },
      },
    } satisfies vscode.DecorationOptions;
  });
}

/** Called from `deactivate()`; disposing the type removes every decoration
 * it painted, so nothing is left behind on reload. */
export function disposeDecorations(): void {
  decorationType?.dispose();
  decorationType = undefined;
  byFsPath = new Map();
}
