import assert from "node:assert/strict";
import { test } from "node:test";
import type { AnchoredComment } from "../anchoring";
import type { ReviewResponse } from "../apiClient";
import {
  escapeHtml,
  renderEmptyHtml,
  renderErrorHtml,
  renderLoadingHtml,
  renderPanelHtml,
} from "../panelHtml";

function response(overrides: Partial<ReviewResponse> = {}): ReviewResponse {
  return {
    mode: "deep",
    n_files_changed: 2,
    merge_probability: 0.75,
    merge_prediction: "MERGE",
    ml_model: "rf_v1_balanced",
    features: { m_additions: 4, m_frac_added: 0.6667 },
    review_comments: [],
    llm_config: "groq/llama-3.1-8b-instant multi_turn @ diff_repo_context",
    llm_warning: null,
    ...overrides,
  };
}

const NONCE = "testnonce";

test("renders the percentage and prediction in the gauge", () => {
  const html = renderPanelHtml(response({ merge_probability: 0.75 }), [], NONCE);
  assert.ok(html.includes(">75%<"));
  assert.ok(html.includes(">MERGE<"));
});

test("gauge arc length is proportional to the probability", () => {
  const circumference = 2 * Math.PI * 52;
  const html = renderPanelHtml(response({ merge_probability: 0.25 }), [], NONCE);
  const expected = (0.25 * circumference).toFixed(2);
  assert.ok(
    html.includes(`stroke-dasharray="${expected} ${circumference.toFixed(2)}"`),
    `expected an arc of ${expected}`,
  );
});

test("a zero probability renders a zero-length arc, not a full circle", () => {
  const html = renderPanelHtml(response({ merge_probability: 0, merge_prediction: "CLOSE" }), [], NONCE);
  assert.ok(html.includes('stroke-dasharray="0.00 '));
  assert.ok(html.includes(">0%<"));
});

test("an anchored comment renders as clickable with its location data", () => {
  const comments: AnchoredComment[] = [
    { text: "Needs a test.", location: { path: "src/a.py", line: 12, precision: "line" } },
  ];
  const html = renderPanelHtml(response(), comments, NONCE);
  assert.ok(html.includes('class="clickable"'));
  assert.ok(html.includes('data-path="src/a.py"'));
  assert.ok(html.includes('data-line="12"'));
  assert.ok(html.includes("src/a.py:12"));
});

test("an unanchored comment is not clickable and has no location chip", () => {
  const comments: AnchoredComment[] = [{ text: "General remark.", location: null }];
  const html = renderPanelHtml(response(), comments, NONCE);
  assert.ok(html.includes("<li>General remark.</li>"));
  // Check the list markup specifically: the string "clickable" also appears
  // in the stylesheet, which is emitted unconditionally.
  assert.ok(!html.includes('<li class="clickable"'));
  assert.ok(!html.includes("data-path"));
});

test("weaker anchors are visually marked as such", () => {
  const comments: AnchoredComment[] = [
    { text: "A.", location: { path: "a.py", line: 1, precision: "line" } },
    { text: "B.", location: { path: "b.py", line: 2, precision: "inferred-file" } },
  ];
  const html = renderPanelHtml(response(), comments, NONCE);
  assert.ok(html.includes('class="chip"'), "a line-precise anchor uses the strong chip");
  assert.ok(html.includes('class="chip weak"'), "a weaker anchor is marked");
});

test("the footnote reports how many comments resolved", () => {
  const comments: AnchoredComment[] = [
    { text: "A.", location: { path: "a.py", line: 1, precision: "line" } },
    { text: "B.", location: null },
    { text: "C.", location: null },
  ];
  const html = renderPanelHtml(response(), comments, NONCE);
  assert.ok(html.includes("1 of 3 comment(s) resolved"));
});

test("fast mode with no comments explains how to get them", () => {
  const html = renderPanelHtml(response({ mode: "fast", llm_config: null }), [], NONCE);
  assert.ok(html.includes("codeReviewAi.mode"));
});

test("deep mode with no comments says so rather than nudging to deep mode", () => {
  const html = renderPanelHtml(response({ mode: "deep" }), [], NONCE);
  assert.ok(html.includes("did not return any review comments"));
});

test("an llm_warning is surfaced", () => {
  const html = renderPanelHtml(response({ llm_warning: "Groq quota exhausted" }), [], NONCE);
  assert.ok(html.includes("Groq quota exhausted"));
  assert.ok(html.includes("warning"));
});

test("LLM comment text is HTML-escaped", () => {
  // The comment body is model-generated and goes straight into the document.
  const comments: AnchoredComment[] = [
    { text: '<img src=x onerror="alert(1)"> & <script>bad()</script>', location: null },
  ];
  const html = renderPanelHtml(response(), comments, NONCE);
  assert.ok(!html.includes("<img src=x"), "raw tag must not survive");
  assert.ok(!html.includes("<script>bad()"), "raw script must not survive");
  assert.ok(html.includes("&lt;img src=x"));
  assert.ok(html.includes("&amp;"));
});

test("a malicious path cannot break out of an HTML attribute", () => {
  const comments: AnchoredComment[] = [
    { text: "x", location: { path: 'a.py" onmouseover="alert(1)', line: 1, precision: "line" } },
  ];
  const html = renderPanelHtml(response(), comments, NONCE);
  assert.ok(!html.includes('onmouseover="alert(1)"'));
  assert.ok(html.includes("&quot;"));
});

test("the CSP carries the supplied nonce and blocks everything else", () => {
  const html = renderPanelHtml(response(), [], "abc123");
  assert.ok(html.includes("default-src 'none'"));
  assert.ok(html.includes("script-src 'nonce-abc123'"));
  assert.ok(html.includes('<script nonce="abc123">'));
});

test("features are listed with integers and decimals formatted distinctly", () => {
  const html = renderPanelHtml(response(), [], NONCE);
  assert.ok(html.includes("m_additions"));
  assert.ok(html.includes(">4<"), "an integer feature renders without decimals");
  assert.ok(html.includes("0.667"), "a fractional feature is rounded to 3 dp");
});

test("escapeHtml covers all five entities", () => {
  assert.equal(escapeHtml(`&<>"'`), "&amp;&lt;&gt;&quot;&#39;");
});

/* ------------------------------------------------------------------ */
/* The gauge's context marks                                           */
/* ------------------------------------------------------------------ */

test("the gauge marks the 50% decision threshold and the corpus base rate", () => {
  const html = renderPanelHtml(response(), [], NONCE);
  assert.ok(html.includes('class="gauge-threshold"'), "the MERGE/CLOSE boundary is marked");
  assert.ok(html.includes('class="gauge-baserate"'), "the training-corpus base rate is marked");
  // Both marks explain themselves on hover rather than needing a legend.
  assert.ok(html.includes("below this the call flips to CLOSE"));
  assert.ok(html.includes("1,494"), "the base-rate tick names the corpus it comes from");
});

test("the reading places the probability against the base rate, not in a vacuum", () => {
  const below = renderPanelHtml(response({ merge_probability: 0.61 }), [], NONCE);
  assert.ok(below.includes("<strong>below</strong> the 72%"), "0.61 is below the 0.7195 base rate");

  const above = renderPanelHtml(response({ merge_probability: 0.9 }), [], NONCE);
  assert.ok(above.includes("<strong>above</strong> the 72%"));

  // Within the tolerance, claiming "above" would over-read the model.
  const level = renderPanelHtml(response({ merge_probability: 0.72 }), [], NONCE);
  assert.ok(level.includes("<strong>level with</strong> the 72%"));
});

test("the thousands separator does not depend on the host locale", () => {
  // `toLocaleString` would render 1.494 under a de-DE extension host.
  const html = renderPanelHtml(response(), [], NONCE);
  assert.ok(html.includes("1,494"));
  assert.ok(!html.includes("1.494"));
});

test("the verdict headline reads as a judgement, not a label", () => {
  assert.ok(renderPanelHtml(response({ merge_prediction: "MERGE" }), [], NONCE)
    .includes("Likely to merge"));
  assert.ok(renderPanelHtml(response({ merge_prediction: "CLOSE" }), [], NONCE)
    .includes("Worth a second look"));
});

/* ------------------------------------------------------------------ */
/* Anchor confidence is carried by the left bar, not just the chip      */
/* ------------------------------------------------------------------ */

test("each anchored comment exposes its precision so the left bar can encode it", () => {
  const comments: AnchoredComment[] = [
    { text: "A.", location: { path: "a.py", line: 1, precision: "line" } },
    { text: "B.", location: { path: "b.py", line: 2, precision: "file" } },
    { text: "C.", location: { path: "c.py", line: 3, precision: "inferred-file" } },
  ];
  const html = renderPanelHtml(response(), comments, NONCE);
  assert.ok(html.includes('data-precision="line"'));
  assert.ok(html.includes('data-precision="file"'));
  assert.ok(html.includes('data-precision="inferred-file"'));
  // ...and the stylesheet actually distinguishes them, rather than the
  // attribute being inert decoration.
  assert.ok(html.includes('li[data-precision="line"] { border-left-color: var(--accent); }'));
  assert.ok(html.includes("border-left-style: dashed"));
});

/* ------------------------------------------------------------------ */
/* Loading state                                                       */
/* ------------------------------------------------------------------ */

test("the loading state names the real work, not just 'loading'", () => {
  const html = renderLoadingHtml(NONCE, {
    mode: "fast",
    filesChanged: 3,
    backendUrl: "http://127.0.0.1:8000",
  });
  assert.ok(html.includes("Reviewing 3 changed files"));
  assert.ok(html.includes("http://127.0.0.1:8000"));
  assert.ok(html.includes("rf_v1_balanced"), "it names the model that will actually run");
});

test("the loading state singularises a one-file diff", () => {
  const html = renderLoadingHtml(NONCE, { mode: "fast", filesChanged: 1, backendUrl: "u" });
  assert.ok(html.includes("Reviewing 1 changed file<"));
});

test("only deep mode's loading state promises an LLM call", () => {
  const deep = renderLoadingHtml(NONCE, { mode: "deep", filesChanged: 1, backendUrl: "u" });
  assert.ok(deep.includes("live Groq call"));
  assert.ok(deep.includes("takes a few seconds"));

  const fast = renderLoadingHtml(NONCE, { mode: "fast", filesChanged: 1, backendUrl: "u" });
  assert.ok(!fast.includes("Groq"), "fast mode makes no network call beyond localhost");
});

test("the loading state's one animation is disabled under prefers-reduced-motion", () => {
  const html = renderLoadingHtml(NONCE, { mode: "fast", filesChanged: 1, backendUrl: "u" });
  assert.ok(html.includes("@media (prefers-reduced-motion: reduce)"));
});

/* ------------------------------------------------------------------ */
/* Empty state                                                         */
/* ------------------------------------------------------------------ */

test("the empty state explains why a staged change looks like no change", () => {
  const html = renderEmptyHtml(NONCE);
  assert.ok(html.includes("No unstaged changes to review"));
  // The actual gotcha: `git.ts` runs plain `git diff`, so staged work is
  // invisible. Saying only "nothing found" would leave the user stuck.
  assert.ok(html.includes("git restore --staged ."));
});

test("the empty state offers a way back in rather than being a dead end", () => {
  const html = renderEmptyHtml(NONCE);
  assert.ok(html.includes('data-action="rerun"'));
  assert.ok(html.includes(`<script nonce="${NONCE}">`));
});

/* ------------------------------------------------------------------ */
/* Error state                                                         */
/* ------------------------------------------------------------------ */

test("an unreachable backend is told how to start the backend", () => {
  const html = renderErrorHtml(NONCE, {
    kind: "unreachable",
    message: "fetch failed",
    backendUrl: "http://127.0.0.1:8000",
  });
  assert.ok(html.includes("The backend isn’t answering"));
  assert.ok(html.includes("uvicorn src.api.main:app"));
  assert.ok(html.includes("codeReviewAi.backendUrl"));
});

test("an HTTP error is not told to start the backend it just talked to", () => {
  const html = renderErrorHtml(NONCE, {
    kind: "http",
    message: "Backend returned 422 Unprocessable Entity",
    backendUrl: "u",
  });
  assert.ok(html.includes("The backend rejected the request"));
  assert.ok(!html.includes("uvicorn src.api.main:app"));
  assert.ok(html.includes("422"));
});

test("a git failure points at git, not at the backend", () => {
  const html = renderErrorHtml(NONCE, {
    kind: "git",
    message: "git diff failed: not a git repository",
    backendUrl: "u",
  });
  assert.ok(html.includes("Couldn’t read your changes"));
  assert.ok(html.includes("not a git repository"));
  assert.ok(!html.includes("uvicorn"));
});

test("the error detail is escaped — it can be a remote response body", () => {
  // `apiClient.ts` puts the backend's own response text into this message,
  // so it is as untrusted as the LLM comments are.
  const html = renderErrorHtml(NONCE, {
    kind: "http",
    message: 'Backend returned 500: <img src=x onerror="alert(1)">',
    backendUrl: '"><script>bad()</script>',
  });
  assert.ok(!html.includes("<img src=x"));
  assert.ok(!html.includes("<script>bad()"));
  assert.ok(html.includes("&lt;img src=x"));
});

/* ------------------------------------------------------------------ */
/* Consistency across all four states                                  */
/* ------------------------------------------------------------------ */

test("every state ships the same CSP and the same stylesheet", () => {
  const states = [
    renderPanelHtml(response(), [], NONCE),
    renderLoadingHtml(NONCE, { mode: "fast", filesChanged: 1, backendUrl: "u" }),
    renderEmptyHtml(NONCE),
    renderErrorHtml(NONCE, { kind: "git", message: "x", backendUrl: "u" }),
  ];
  for (const html of states) {
    assert.ok(html.includes("default-src 'none'"), "same CSP everywhere");
    assert.ok(html.includes(`script-src 'nonce-${NONCE}'`));
    // The design tokens are the stylesheet's fingerprint: if one state
    // drifted onto its own CSS, these would stop matching.
    assert.ok(html.includes("--t-display: 2.15em;"), "same type scale");
    assert.ok(html.includes("--s-4: 16px;"), "same 4px spacing rhythm");
  }
});

test("no state falls back to an emoji where an icon belongs", () => {
  const states = [
    renderPanelHtml(response({ llm_warning: "quota exhausted" }), [], NONCE),
    renderLoadingHtml(NONCE, { mode: "deep", filesChanged: 1, backendUrl: "u" }),
    renderEmptyHtml(NONCE),
    renderErrorHtml(NONCE, { kind: "unreachable", message: "x", backendUrl: "u" }),
  ];
  // Anything in the emoji/pictograph/dingbat blocks, plus the two that
  // used to be here specifically.
  const emoji = /[\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}\u{FE0F}]/u;
  for (const html of states) {
    assert.ok(!emoji.test(html), `found an emoji in: ${html.match(emoji)?.[0]}`);
  }
  assert.ok(states[0].includes('class="icon"'), "the warning uses the real SVG icon set");
});
