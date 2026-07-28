import assert from "node:assert/strict";
import { test } from "node:test";
import type { AnchoredComment } from "../anchoring";
import type { ReviewResponse } from "../apiClient";
import { escapeHtml, renderPanelHtml } from "../panelHtml";

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
