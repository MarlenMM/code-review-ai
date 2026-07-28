/**
 * Pure HTML rendering for the review panel -- no `vscode` import, so it is
 * unit-testable (and previewable in a plain browser) without an extension
 * host, following the same split `git.ts`/`apiClient.ts` already use.
 *
 * Escaping matters here beyond tidiness: `review_comments` is **LLM-
 * generated text** that this module interpolates straight into a webview
 * document. Anything it emits must be escaped at every interpolation site,
 * including into HTML attributes (`data-path`, `title`) -- hence the
 * single `escapeHtml` covering `& < > " '` and its dedicated tests.
 */

import type { AnchoredComment, AnchorPrecision } from "./anchoring";
import type { ReviewResponse } from "./apiClient";

const GAUGE_RADIUS = 52;
const GAUGE_CIRCUMFERENCE = 2 * Math.PI * GAUGE_RADIUS;

const PRECISION_TOOLTIP: Record<AnchorPrecision, string> = {
  line: "Resolved to this line: the comment names a symbol or line number found in the diff.",
  file: "Resolved to this file only: the comment names the file but nothing narrower. Jumps to its first change.",
  "inferred-file":
    "Inferred: the comment names no file, but this change touches only one file. Jumps to its first change.",
};

export function renderPanelHtml(
  result: ReviewResponse,
  comments: AnchoredComment[],
  nonce: string,
): string {
  const percent = Math.round(result.merge_probability * 100);
  const accent = result.merge_prediction === "MERGE" ? "var(--crai-green)" : "var(--crai-red)";
  const arc = (result.merge_probability * GAUGE_CIRCUMFERENCE).toFixed(2);
  const rest = GAUGE_CIRCUMFERENCE.toFixed(2);

  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta http-equiv="Content-Security-Policy"
      content="default-src 'none'; style-src 'unsafe-inline'; script-src 'nonce-${nonce}';" />
<style>
  :root { --crai-green: #3fb950; --crai-red: #f85149; }
  body {
    font-family: var(--vscode-font-family, -apple-system, system-ui, sans-serif);
    font-size: var(--vscode-font-size, 13px);
    color: var(--vscode-foreground, #ccc);
    background: var(--vscode-editor-background, transparent);
    padding: 1.2em 1.5em 2em;
  }
  .summary { display: flex; align-items: center; gap: 1.4em; flex-wrap: wrap; }
  .gauge { flex: 0 0 auto; }
  .gauge-track { stroke: var(--vscode-editorWidget-border, rgba(128,128,128,0.3)); }
  .gauge-arc { stroke: ${accent}; }
  .gauge-value { fill: var(--vscode-foreground, #ccc); font-size: 25px; font-weight: 600; }
  .gauge-label { fill: ${accent}; font-size: 12px; font-weight: 600; letter-spacing: 0.06em; }
  .headline { margin: 0 0 0.35em; font-size: 1.05em; font-weight: 600; }
  .meta { color: var(--vscode-descriptionForeground, #888); margin: 0; line-height: 1.7; }
  .muted { color: var(--vscode-descriptionForeground, #888); font-style: italic; }
  .warning {
    background: var(--vscode-inputValidation-warningBackground, #5a3d00);
    border: 1px solid var(--vscode-inputValidation-warningBorder, #cca700);
    padding: 0.55em 0.85em; border-radius: 4px; margin: 1.2em 0 0;
  }
  h2 { font-size: 0.85em; margin: 1.8em 0 0.6em; text-transform: uppercase;
       letter-spacing: 0.06em; color: var(--vscode-descriptionForeground, #888); }
  ol.comments { list-style: none; padding: 0; margin: 0; }
  ol.comments li {
    border-left: 2px solid var(--vscode-editorWidget-border, rgba(128,128,128,0.3));
    padding: 0.45em 0 0.45em 0.85em; margin-bottom: 0.5em; line-height: 1.5;
  }
  ol.comments li.clickable { cursor: pointer; }
  ol.comments li.clickable:hover {
    background: var(--vscode-list-hoverBackground, rgba(128,128,128,0.12));
    border-left-color: ${accent};
  }
  .chip {
    display: block; margin-top: 0.35em;
    font-family: var(--vscode-editor-font-family, ui-monospace, monospace);
    font-size: 0.85em; color: var(--vscode-textLink-foreground, #4daafc);
  }
  .chip.weak { color: var(--vscode-descriptionForeground, #888); }
  table { border-collapse: collapse; margin-top: 0.6em; }
  td { padding: 0.15em 1.2em 0.15em 0; font-size: 0.9em; }
  td.num { font-family: var(--vscode-editor-font-family, ui-monospace, monospace);
           color: var(--vscode-descriptionForeground, #888); }
  details { margin-top: 1.8em; }
  summary { cursor: pointer; color: var(--vscode-descriptionForeground, #888); }
  .footnote { margin-top: 1.4em; font-size: 0.85em; line-height: 1.5;
              color: var(--vscode-descriptionForeground, #888); }
</style>
</head>
<body>
  <div class="summary">
    <svg class="gauge" viewBox="0 0 120 120" width="118" height="118" role="img"
         aria-label="Merge probability ${percent} percent, predicted ${escapeHtml(result.merge_prediction)}">
      <circle class="gauge-track" cx="60" cy="60" r="${GAUGE_RADIUS}" fill="none" stroke-width="11" />
      <circle class="gauge-arc" cx="60" cy="60" r="${GAUGE_RADIUS}" fill="none" stroke-width="11"
              stroke-linecap="round" stroke-dasharray="${arc} ${rest}"
              transform="rotate(-90 60 60)" />
      <text class="gauge-value" x="60" y="59" text-anchor="middle" dominant-baseline="middle">${percent}%</text>
      <text class="gauge-label" x="60" y="80" text-anchor="middle">${escapeHtml(result.merge_prediction)}</text>
    </svg>
    <div>
      <p class="headline">Merge probability</p>
      <p class="meta">
        Mode: <strong>${escapeHtml(result.mode)}</strong><br />
        Model: ${escapeHtml(result.ml_model)}<br />
        Files changed: ${result.n_files_changed}
        ${result.llm_config ? `<br />LLM: ${escapeHtml(result.llm_config)}` : ""}
      </p>
    </div>
  </div>

  ${result.llm_warning ? `<p class="warning">⚠ ${escapeHtml(result.llm_warning)}</p>` : ""}

  <h2>Review comments</h2>
  ${renderComments(result, comments)}

  <details>
    <summary>Features used by the merge prediction (${Object.keys(result.features).length})</summary>
    ${renderFeatures(result.features)}
  </details>

  <script nonce="${nonce}">
    const vscode = acquireVsCodeApi();
    document.querySelectorAll("li.clickable").forEach((li) => {
      li.addEventListener("click", () => {
        vscode.postMessage({
          type: "jump",
          path: li.dataset.path,
          line: Number(li.dataset.line),
        });
      });
    });
  </script>
</body>
</html>`;
}

function renderComments(result: ReviewResponse, comments: AnchoredComment[]): string {
  if (comments.length === 0) {
    const reason = result.mode === "deep"
      ? "The LLM did not return any review comments."
      : "Run in “deep” mode (the codeReviewAi.mode setting) to generate review comments.";
    return `<p class="muted">${reason}</p>`;
  }

  const items = comments.map((comment) => {
    const body = escapeHtml(comment.text);
    if (!comment.location) {
      return `<li>${body}</li>`;
    }
    const { path, line, precision } = comment.location;
    const weak = precision === "line" ? "" : " weak";
    return `<li class="clickable" data-path="${escapeHtml(path)}" data-line="${line}"`
      + ` title="${escapeHtml(PRECISION_TOOLTIP[precision])}">${body}`
      + `<span class="chip${weak}">${escapeHtml(path)}:${line}</span></li>`;
  });

  const anchored = comments.filter((c) => c.location !== null).length;
  const footnote =
    `<p class="footnote">${anchored} of ${comments.length} comment(s) resolved to a location in your diff — `
    + `click one to jump there. The model returns prose, not line numbers, so locations are inferred from the `
    + `comment text; hover a location chip to see how confident that match is.</p>`;

  return `<ol class="comments">${items.join("")}</ol>${footnote}`;
}

function renderFeatures(features: Record<string, number>): string {
  const rows = Object.entries(features)
    .map(([name, value]) =>
      `<tr><td>${escapeHtml(name)}</td><td class="num">${formatNumber(value)}</td></tr>`)
    .join("");
  return `<table>${rows}</table>`;
}

function formatNumber(value: number): string {
  return Number.isInteger(value) ? String(value) : value.toFixed(3);
}

export function escapeHtml(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}
