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
 *
 * FOUR STATES, NOT ONE
 * --------------------
 * The panel renders every state the command can end in -- `renderPanelHtml`
 * (a result), `renderLoadingHtml` (request in flight), `renderEmptyHtml`
 * (clean working tree) and `renderErrorHtml` (git or backend failed).
 * Earlier revisions only had the first: the other three were VS Code
 * toasts, so the two most common first-run outcomes -- a clean tree and a
 * backend that isn't running -- were a disappearing notification and no
 * panel at all. All four share `documentShell`, so they share one
 * stylesheet and one CSP rather than drifting apart.
 *
 * DESIGN CONSTRAINTS (deliberate; see `docs/design-constraints.md`)
 * ----------------------------------------------------------------
 * * Type is a five-step scale (`--t-*`) at two weights. Nothing sits at a
 *   size that isn't on the scale, so hierarchy is a decision, not an
 *   accident.
 * * Space is a 4px rhythm (`--s-*`). No ad-hoc `em` padding.
 * * Colour is semantic only: green/red mean MERGE/CLOSE, and the single
 *   accent (`--accent`) means "you can click this". Nothing is coloured
 *   for decoration, and every value comes from a `--vscode-*` token so the
 *   panel follows the user's theme.
 * * A left border on a comment encodes *anchor confidence*, not emphasis
 *   -- solid accent for a line-precise anchor, hairline for file-level,
 *   dashed for inferred, none for unanchored.
 * * Icons are a three-glyph inline-SVG set at one size and one stroke
 *   weight. No emoji: `⚠`/`💬` read as placeholder, and neither picks up
 *   the theme's foreground colour the way `currentColor` does.
 */

import type { AnchoredComment, AnchorPrecision } from "./anchoring";
import type { ReviewResponse } from "./apiClient";

const GAUGE_RADIUS = 52;
const GAUGE_CIRCUMFERENCE = 2 * Math.PI * GAUGE_RADIUS;
const GAUGE_CENTER = 60;

/**
 * The merge rate of the mined corpus this model was trained from:
 * `results/tables/exp1_summary.json` -> `merge_status.pooled`
 * (1075 merged / 419 not merged out of 1494 = 0.7195).
 *
 * It is on the gauge because a bare "61%" is unreadable on its own -- 61%
 * sounds healthy until you know that 72% of the PRs the model learned from
 * were merged, which makes 61% a *below-average* change. The tick is the
 * difference between a number and a judgement, and it is a real number
 * from a committed artifact rather than a decorative flourish.
 */
const CORPUS_MERGE_RATE = 0.7195;
const CORPUS_SIZE = 1494;

/** `src/api/main.py`: `"MERGE" if probability >= 0.5 else "CLOSE"`. */
const DECISION_THRESHOLD = 0.5;

/** Within this much of the base rate, "above"/"below" would over-read the
 * model's precision, so the reading says "level with" instead. */
const LEVEL_WITH_TOLERANCE = 0.02;

const PRECISION_TOOLTIP: Record<AnchorPrecision, string> = {
  line: "Resolved to this line: the comment names a symbol or line number found in the diff.",
  file: "Resolved to this file only: the comment names the file but nothing narrower. Jumps to its first change.",
  "inferred-file":
    "Inferred: the comment names no file, but this change touches only one file. Jumps to its first change.",
};

/** One icon family: 16x16 box, 1.5 stroke, round caps, `currentColor`, no
 * fill. Used sparingly -- one per state, never as a bullet or a label
 * decoration. */
const ICONS = {
  alert:
    `<path d="M8 2.8 1.8 13.4h12.4L8 2.8Z" /><path d="M8 6.6v3.1" /><path d="M8 11.7h.01" />`,
  diff:
    `<path d="M3.2 2.2h6.2l3.4 3.4v8.2H3.2V2.2Z" /><path d="M9.4 2.2v3.4h3.4" /><path d="M5.8 10.6h4.4" />`,
  offline:
    `<circle cx="8" cy="8" r="5.8" /><path d="M3.9 3.9l8.2 8.2" />`,
} as const;

function icon(name: keyof typeof ICONS): string {
  return `<svg class="icon" viewBox="0 0 16 16" width="16" height="16" fill="none"`
    + ` stroke="currentColor" stroke-width="1.5" stroke-linecap="round"`
    + ` stroke-linejoin="round" aria-hidden="true">${ICONS[name]}</svg>`;
}

/* ------------------------------------------------------------------ */
/* The four states                                                     */
/* ------------------------------------------------------------------ */

export function renderPanelHtml(
  result: ReviewResponse,
  comments: AnchoredComment[],
  nonce: string,
): string {
  const percent = Math.round(result.merge_probability * 100);
  const merges = result.merge_prediction === "MERGE";
  const accent = merges ? "var(--merge)" : "var(--close)";
  const arc = (result.merge_probability * GAUGE_CIRCUMFERENCE).toFixed(2);
  const rest = GAUGE_CIRCUMFERENCE.toFixed(2);

  const body = `
  <section class="verdict" style="--verdict: ${accent};">
    <svg class="gauge" viewBox="0 0 120 120" width="132" height="132" role="img"
         aria-label="Merge probability ${percent} percent, predicted ${escapeHtml(result.merge_prediction)}, against a ${formatPercent(CORPUS_MERGE_RATE)} percent corpus base rate">
      <circle class="gauge-track" cx="60" cy="60" r="${GAUGE_RADIUS}" fill="none" stroke-width="11" />
      <circle class="gauge-arc" cx="60" cy="60" r="${GAUGE_RADIUS}" fill="none" stroke-width="11"
              stroke-linecap="round" stroke-dasharray="${arc} ${rest}"
              transform="rotate(-90 60 60)" />
      ${gaugeTick(DECISION_THRESHOLD, "gauge-threshold",
        `${formatPercent(DECISION_THRESHOLD)}% — below this the call flips to CLOSE`)}
      ${gaugeTick(CORPUS_MERGE_RATE, "gauge-baserate",
        `${formatPercent(CORPUS_MERGE_RATE)}% — the merge rate of the ${withThousands(CORPUS_SIZE)} PRs this model was trained from`)}
      <text class="gauge-value" x="60" y="57" text-anchor="middle" dominant-baseline="middle">${percent}%</text>
      <text class="gauge-label" x="60" y="80" text-anchor="middle">${escapeHtml(result.merge_prediction)}</text>
    </svg>
    <div class="verdict-text">
      <h1>${merges ? "Likely to merge" : "Worth a second look"}</h1>
      <p class="reading">${renderReading(result.merge_probability, percent)}</p>
      <p class="provenance">${renderProvenance(result)}</p>
    </div>
  </section>

  ${result.llm_warning ? renderWarning(result.llm_warning) : ""}

  <h2>Review comments <span class="count">${comments.length}</span></h2>
  ${renderComments(result, comments)}

  <details>
    <summary>The ${Object.keys(result.features).length} features behind that probability</summary>
    ${renderFeatures(result.features)}
  </details>`;

  return documentShell(nonce, body, JUMP_SCRIPT);
}

/**
 * Shown the moment the command starts, before the backend answers. It is
 * not a spinner over a blank panel: it names the diff that was actually
 * read and the real pipeline the request is sitting in, so the wait is
 * legible. One indeterminate bar, because one request of genuinely unknown
 * duration is running -- there is no per-step progress to report, and
 * faking a step-by-step animation would claim knowledge this has none of.
 */
export function renderLoadingHtml(
  nonce: string,
  info: { mode: string; filesChanged: number; backendUrl: string },
): string {
  const deep = info.mode === "deep";
  const steps = deep
    ? "parse the diff, compute 28 V1 features, run rf_v1_balanced, then one live Groq call for the review comments"
    : "parse the diff, compute 28 V1 features, run rf_v1_balanced";

  return documentShell(nonce, `
  <section class="state">
    <h1>Reviewing ${countNoun(info.filesChanged, "changed file")}</h1>
    <p class="reading">Sent to <code>${escapeHtml(info.backendUrl)}</code> in
       <strong>${escapeHtml(info.mode)}</strong> mode. It will ${steps}.</p>
    <div class="progress" role="progressbar" aria-label="Waiting for the backend"><span></span></div>
    <p class="provenance">${deep
      ? "Deep mode waits on a live LLM call, so this takes a few seconds."
      : "Fast mode is local and sub-second — this should be gone almost immediately."}</p>
  </section>`);
}

/**
 * The clean-tree case, which used to be a toast that vanished. It gets a
 * panel because the useful thing to say is not "nothing found" but *why*
 * nothing was found: `git.ts` runs plain `git diff`, so staged and
 * committed work is invisible to it by design, and that trips people up
 * far more often than an actually-clean tree does.
 */
export function renderEmptyHtml(nonce: string): string {
  return documentShell(nonce, `
  <section class="state">
    <p class="state-icon">${icon("diff")}</p>
    <h1>No unstaged changes to review</h1>
    <p class="reading"><code>git diff</code> came back empty. This command reviews the
       working tree against the index, so anything you have already staged or
       committed is invisible to it.</p>
    <ul class="next">
      <li>Edit a tracked file, then run the review again.</li>
      <li>Staged it already? <code>git restore --staged .</code> puts it back in the working tree.</li>
    </ul>
    <button class="retry" data-action="rerun">Review again</button>
  </section>`, RERUN_SCRIPT);
}

/**
 * Anything that stopped the review: git failed, or the backend was
 * unreachable or unhappy. `kind` decides what to suggest, because "start
 * the backend" is the right answer to a refused connection and the wrong
 * answer to a 422.
 */
export function renderErrorHtml(
  nonce: string,
  error: { kind: "git" | "unreachable" | "http"; message: string; backendUrl: string },
): string {
  const headline = {
    git: "Couldn’t read your changes",
    unreachable: "The backend isn’t answering",
    http: "The backend rejected the request",
  }[error.kind];

  const remedy = {
    git: `<ul class="next">
       <li>Open a folder that is a git repository — this reads <code>git diff</code> in the first workspace folder.</li>
       <li>Check that <code>git</code> is on your <code>PATH</code>.</li>
     </ul>`,
    unreachable: `<ul class="next">
       <li>Start it from the repository root:
           <code>source .venv/bin/activate &amp;&amp; uvicorn src.api.main:app</code></li>
       <li>Listening somewhere else? Point <code>codeReviewAi.backendUrl</code> at it.</li>
     </ul>`,
    http: `<ul class="next">
       <li>A 422 usually means the diff wasn’t unified-diff format — check <code>git diff</code> output directly.</li>
       <li>A 500 leaves a traceback in the <code>uvicorn</code> console.</li>
     </ul>`,
  }[error.kind];

  return documentShell(nonce, `
  <section class="state">
    <p class="state-icon danger">${icon("offline")}</p>
    <h1>${headline}</h1>
    <p class="detail">${escapeHtml(error.message)}</p>
    ${remedy}
    <p><button class="retry" data-action="rerun">Try again</button></p>
    <p class="provenance">Backend: <code>${escapeHtml(error.backendUrl)}</code></p>
  </section>`, RERUN_SCRIPT);
}

/* ------------------------------------------------------------------ */
/* Pieces                                                              */
/* ------------------------------------------------------------------ */

/** A radial mark across the gauge stroke at `fraction` of the way round,
 * carrying its own `<title>` so hovering it explains what it marks. */
function gaugeTick(fraction: number, className: string, label: string): string {
  const [x1, y1] = onGauge(fraction, GAUGE_RADIUS - 9);
  const [x2, y2] = onGauge(fraction, GAUGE_RADIUS + 9);
  const width = className === "gauge-threshold" ? 2.5 : 2;
  return `<line class="${className}" x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}"`
    + ` stroke-width="${width}" stroke-linecap="butt"><title>${escapeHtml(label)}</title></line>`;
}

/** Cartesian point at `fraction` of a full turn, starting at 12 o'clock and
 * going clockwise -- matching the arc's own `rotate(-90)`. */
function onGauge(fraction: number, radius: number): [string, string] {
  const angle = (fraction * 360 - 90) * (Math.PI / 180);
  return [
    (GAUGE_CENTER + radius * Math.cos(angle)).toFixed(2),
    (GAUGE_CENTER + radius * Math.sin(angle)).toFixed(2),
  ];
}

/** The one sentence that turns the number into a judgement. */
function renderReading(probability: number, percent: number): string {
  const delta = probability - CORPUS_MERGE_RATE;
  const standing = Math.abs(delta) <= LEVEL_WITH_TOLERANCE
    ? "level with"
    : delta > 0 ? "above" : "below";

  return `${percent}% is <strong>${standing}</strong> the ${formatPercent(CORPUS_MERGE_RATE)}% `
    + `base rate of the ${withThousands(CORPUS_SIZE)} pull requests this model was trained from. `
    + `The call flips to CLOSE below ${formatPercent(DECISION_THRESHOLD)}%.`;
}

/** Quiet line: which model, which mode, how much it looked at. */
function renderProvenance(result: ReviewResponse): string {
  const parts = [
    escapeHtml(result.ml_model),
    `${escapeHtml(result.mode)} mode`,
    countNoun(result.n_files_changed, "file"),
  ];
  const llm = result.llm_config
    ? `<br />${escapeHtml(result.llm_config)}`
    : "";
  return parts.join(" · ") + llm;
}

function renderWarning(warning: string): string {
  return `<p class="warning">${icon("alert")}<span>${escapeHtml(warning)}</span></p>`;
}

function renderComments(result: ReviewResponse, comments: AnchoredComment[]): string {
  if (comments.length === 0) {
    const reason = result.mode === "deep"
      ? "The LLM did not return any review comments for this diff — deep mode ran, it just had nothing to say."
      : "Fast mode predicts the merge probability only. Switch the codeReviewAi.mode setting to “deep” to also generate review comments.";
    return `<p class="muted">${reason}</p>`;
  }

  const items = comments.map((comment) => {
    const body = escapeHtml(comment.text);
    if (!comment.location) {
      return `<li>${body}</li>`;
    }
    const { path, line, precision } = comment.location;
    const weak = precision === "line" ? "" : " weak";
    return `<li class="clickable" data-precision="${precision}"`
      + ` data-path="${escapeHtml(path)}" data-line="${line}"`
      + ` title="${escapeHtml(PRECISION_TOOLTIP[precision])}">${body}`
      + `<span class="chip${weak}">${escapeHtml(path)}:${line}</span></li>`;
  });

  const anchored = comments.filter((c) => c.location !== null).length;
  const footnote =
    `<p class="footnote">${anchored} of ${comments.length} comment(s) resolved to a place in your diff — `
    + `click one to jump there. The model writes prose, not line numbers, so each location is matched `
    + `from the comment's own wording; the bar on the left and the tooltip both say how firm that match is.</p>`;

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

/** Locale-independent, unlike `toLocaleString` -- the extension host's
 * locale is the user's, and this string is asserted on in tests. */
function withThousands(value: number): string {
  return String(value).replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

function formatPercent(fraction: number): string {
  return String(Math.round(fraction * 100));
}

function countNoun(count: number, noun: string): string {
  return `${count} ${noun}${count === 1 ? "" : "s"}`;
}

/* ------------------------------------------------------------------ */
/* Shell                                                               */
/* ------------------------------------------------------------------ */

const JUMP_SCRIPT = `
    const vscode = acquireVsCodeApi();
    document.querySelectorAll("li.clickable").forEach((li) => {
      li.addEventListener("click", () => {
        vscode.postMessage({
          type: "jump",
          path: li.dataset.path,
          line: Number(li.dataset.line),
        });
      });
    });`;

const RERUN_SCRIPT = `
    const vscode = acquireVsCodeApi();
    document.querySelectorAll("button[data-action='rerun']").forEach((button) => {
      button.addEventListener("click", () => {
        button.disabled = true;
        vscode.postMessage({ type: "rerun" });
      });
    });`;

/**
 * Every state's document: one stylesheet, one CSP, one nonce. `script` is
 * omitted for states with nothing to click, but the CSP still names the
 * nonce so the policy string is identical across all four.
 */
function documentShell(nonce: string, body: string, script?: string): string {
  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta http-equiv="Content-Security-Policy"
      content="default-src 'none'; style-src 'unsafe-inline'; script-src 'nonce-${nonce}';" />
<style>
  :root {
    /* Semantic only. Green/red mean MERGE/CLOSE; --accent means clickable. */
    --merge: var(--vscode-charts-green, #3fb950);
    --close: var(--vscode-charts-red, #f85149);
    --accent: var(--vscode-textLink-foreground, #4daafc);
    --fg: var(--vscode-foreground, #ccc);
    --fg-quiet: var(--vscode-descriptionForeground, #8b949e);
    --edge: var(--vscode-editorWidget-border, rgba(128, 128, 128, 0.3));
    --hover: var(--vscode-list-hoverBackground, rgba(128, 128, 128, 0.12));
    --mono: var(--vscode-editor-font-family, ui-monospace, SFMono-Regular, monospace);

    /* Five type steps, two weights. Nothing off this scale. */
    --t-display: 2.15em;
    --t-lead: 1.45em;
    --t-body: 1em;
    --t-small: 0.92em;
    --t-micro: 0.85em;

    /* One 4px rhythm. No ad-hoc padding. */
    --s-1: 4px;  --s-2: 8px;  --s-3: 12px; --s-4: 16px;
    --s-5: 24px; --s-6: 32px; --s-7: 48px;
  }

  body {
    font-family: var(--vscode-font-family, -apple-system, system-ui, sans-serif);
    font-size: var(--vscode-font-size, 13px);
    line-height: 1.6;
    color: var(--fg);
    background: var(--vscode-editor-background, transparent);
    margin: 0;
    padding: var(--s-6) var(--s-6) var(--s-7);
    max-width: 62ch;
  }

  h1 { font-size: var(--t-lead); font-weight: 600; line-height: 1.25; margin: 0 0 var(--s-2); }
  h2 {
    font-size: var(--t-body); font-weight: 600; color: var(--fg);
    margin: var(--s-7) 0 var(--s-3); display: flex; align-items: baseline; gap: var(--s-2);
  }
  h2 .count { font-size: var(--t-micro); font-weight: 400; color: var(--fg-quiet); }
  p { margin: 0 0 var(--s-3); }
  code {
    font-family: var(--mono); font-size: var(--t-micro);
    background: var(--hover); padding: 1px var(--s-1); border-radius: 3px;
  }

  /* --- verdict: the one dominant thing on the page ----------------- */
  .verdict { display: flex; align-items: center; gap: var(--s-6); flex-wrap: wrap; }
  .verdict-text { flex: 1 1 22ch; min-width: 22ch; }
  .gauge { flex: 0 0 auto; }
  .gauge-track { stroke: var(--edge); }
  .gauge-arc { stroke: var(--verdict); }
  /* The two marks have different jobs, so they are drawn differently: the
     decision threshold is a hard rule, cut through the ring as a notch in
     the panel's own background; the base rate is a reference point, drawn
     as a line laid over it. */
  .gauge-threshold { stroke: var(--vscode-editor-background, #1f1f1f); }
  .gauge-baserate { stroke: var(--fg); opacity: 0.85; }
  .gauge-value { fill: var(--fg); font-size: 27px; font-weight: 600; }
  .gauge-label { fill: var(--verdict); font-size: 11px; font-weight: 600; letter-spacing: 0.08em; }
  .reading { font-size: var(--t-small); color: var(--fg); }
  .reading strong { font-weight: 600; }
  .provenance { font-size: var(--t-micro); color: var(--fg-quiet); margin: 0; }

  /* --- states ------------------------------------------------------ */
  .state { max-width: 52ch; }
  .state-icon { color: var(--fg-quiet); line-height: 0; margin: 0 0 var(--s-3); }
  .state-icon.danger { color: var(--close); }
  .icon { vertical-align: -2px; }
  .detail {
    font-family: var(--mono); font-size: var(--t-micro); color: var(--fg-quiet);
    border-left: 2px solid var(--edge); padding-left: var(--s-3); margin-bottom: var(--s-4);
    overflow-wrap: anywhere;
  }
  ul.next { margin: 0 0 var(--s-4); padding-left: var(--s-4); font-size: var(--t-small); }
  ul.next li { margin-bottom: var(--s-2); }
  button.retry {
    font: inherit; font-size: var(--t-small);
    color: var(--vscode-button-foreground, #fff);
    background: var(--vscode-button-background, #0e639c);
    border: none; border-radius: 3px; padding: var(--s-2) var(--s-4); cursor: pointer;
  }
  button.retry:hover { background: var(--vscode-button-hoverBackground, #1177bb); }
  button.retry:disabled { opacity: 0.55; cursor: default; }

  /* One animation in the whole panel, and only while a request is in
     flight: it reports state, it is not there to look alive. */
  .progress {
    height: 2px; background: var(--edge); border-radius: 2px;
    overflow: hidden; margin: 0 0 var(--s-3);
  }
  .progress span {
    display: block; height: 100%; width: 35%;
    background: var(--accent); animation: slide 1.4s ease-in-out infinite;
  }
  @keyframes slide {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(320%); }
  }
  @media (prefers-reduced-motion: reduce) {
    .progress span { animation: none; width: 100%; opacity: 0.5; }
  }

  /* --- warning ----------------------------------------------------- */
  .warning {
    display: flex; gap: var(--s-2); align-items: flex-start;
    font-size: var(--t-small);
    color: var(--vscode-inputValidation-warningForeground, inherit);
    background: var(--vscode-inputValidation-warningBackground, #5a3d00);
    border: 1px solid var(--vscode-inputValidation-warningBorder, #cca700);
    border-radius: 3px; padding: var(--s-2) var(--s-3); margin: var(--s-5) 0 0;
  }
  .warning .icon { flex: 0 0 auto; margin-top: 2px; }

  /* --- comments: the left bar encodes anchor confidence ------------ */
  .muted { color: var(--fg-quiet); font-size: var(--t-small); }
  ol.comments { list-style: none; padding: 0; margin: 0; }
  ol.comments li {
    padding: var(--s-2) 0 var(--s-2) var(--s-3);
    margin-bottom: var(--s-2);
    border-left: 2px solid transparent;
  }
  ol.comments li[data-precision="line"] { border-left-color: var(--accent); }
  ol.comments li[data-precision="file"] { border-left-color: var(--edge); }
  ol.comments li[data-precision="inferred-file"] {
    border-left-style: dashed; border-left-color: var(--edge);
  }
  ol.comments li.clickable { cursor: pointer; }
  ol.comments li.clickable:hover { background: var(--hover); }
  .chip {
    display: block; margin-top: var(--s-1);
    font-family: var(--mono); font-size: var(--t-micro); color: var(--accent);
  }
  .chip.weak { color: var(--fg-quiet); }
  .footnote {
    margin-top: var(--s-4); font-size: var(--t-micro); color: var(--fg-quiet);
  }

  /* --- features ---------------------------------------------------- */
  details { margin-top: var(--s-7); }
  summary { cursor: pointer; font-size: var(--t-small); color: var(--fg-quiet); }
  summary:hover { color: var(--fg); }
  table { border-collapse: collapse; margin-top: var(--s-3); }
  td { padding: var(--s-1) var(--s-5) var(--s-1) 0; font-size: var(--t-micro); }
  td.num { font-family: var(--mono); color: var(--fg-quiet); }
</style>
</head>
<body>
${body}
${script ? `  <script nonce="${nonce}">${script}\n  </script>` : ""}
</body>
</html>`;
}

export function escapeHtml(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}
