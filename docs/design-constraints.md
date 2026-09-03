# Design constraints for the review panel

These are the rules `vscode-extension/src/panelHtml.ts` is written against.
They exist because the panel is the only part of this project a user
actually *looks* at, and the default output of any AI-assisted UI pass —
gradient hero, nested rounded cards, an eyebrow badge over every heading,
one flat font size, an emoji standing in for an icon, no empty state —
converges on the same generic page regardless of what the product does.

Nothing here is a matter of taste for its own sake. Each rule is followed
by the failure it prevents and where it is enforced.

## 1. Every state is designed, not just the happy one

The panel renders **four** states, all from the same module and the same
stylesheet: a result, a review in flight, a clean working tree, and a
failure.

| State | Function | What it must answer |
|---|---|---|
| Result | `renderPanelHtml` | What is the verdict, and how confident is each comment's location? |
| Loading | `renderLoadingHtml` | What was sent, where, and what is it doing? |
| Empty | `renderEmptyHtml` | Why did it find nothing, and what would change that? |
| Error | `renderErrorHtml` | What broke, and what is the specific fix? |

*Prevents:* the demo-only product. Before this, the three non-result
outcomes were VS Code toasts and no panel — so the two most likely first
runs (a clean tree, and a backend nobody started) produced a notification
that vanished and an empty screen. The empty state now explains the actual
trap, which is that `git.ts` runs plain `git diff` and therefore cannot
see staged work.

*Enforced by:* the state tests in `src/test/panelHtml.test.ts`, including
one asserting all four ship the same CSP and the same design tokens.

## 2. Type is a five-step scale at two weights

`--t-display` / `--t-lead` / `--t-body` / `--t-small` / `--t-micro`, at 400
and 600 only. Nothing is sized off the scale.

*Prevents:* flat hierarchy — the look where a heading, a caption, a table
cell and a footnote all land within a few tenths of an em of each other
and the eye has nowhere to rest. The panel used to have six different
near-identical sizes (`1.05em`, `0.85em`, `0.9em`, `0.85em`, …) chosen one
at a time.

## 3. Space is one 4px rhythm

`--s-1` … `--s-7`. No ad-hoc `em` padding anywhere.

*Prevents:* uniform-but-arbitrary spacing, where everything is equidistant
from everything else and no grouping is visible.

## 4. One dominant element per state

In the result state that is the gauge and the verdict sentence. Model
name, mode and file count drop to `--t-micro` in the quiet colour; the
28-feature table stays behind a `<details>`.

*Prevents:* the equal-weight feature grid, where six things are presented
as equally important and the reader is left to work out which one matters.

## 5. Colour is semantic; there is exactly one accent

* Green / red mean MERGE / CLOSE. Nothing else uses them.
* `--accent` means "this is clickable / this is anchored". Nothing else
  uses it.
* Every value resolves through a `--vscode-*` token, so the panel follows
  the user's theme instead of asserting its own.

*Prevents:* decoration-coloured UI — the purple-gradient reflex, and the
one-sided coloured border applied to every card until it means nothing.

## 6. A coloured left border encodes anchor confidence

`data-precision="line"` gets a solid accent bar, `"file"` a hairline,
`"inferred-file"` a dashed hairline, and an unanchored comment gets none.

This is the rule in §5 applied to the one place a strip *does* carry
information: how firmly the anchoring heuristic in `anchoring.ts` matched
that comment to a location. The bar, the chip colour and the tooltip all
say the same thing.

*Prevents:* the coloured strip as filler.

## 7. Icons are one family; no emoji

Three inline SVGs at 16×16, 1.5 stroke, round caps, `currentColor`, no
fill — plus the matching gutter asset in `media/`. The panel previously
used `⚠` and the editor decorations `💬`.

*Prevents:* emoji-as-icon, which is the single most legible "nobody
designed this" signal. It is also functionally worse: an emoji renders in
the platform emoji font's own colour and metrics and ignores the user's
theme entirely, where `currentColor` does not.

## 8. Motion reports state, or it isn't there

There is exactly one animation in the whole panel — the indeterminate bar
in the loading state — and it exists only while a request is genuinely in
flight. It is disabled under `prefers-reduced-motion`.

*Prevents:* pulsing dots, glowing borders and drifting gradients used to
prove the page is alive.

*Not faked:* the loading state does **not** animate step-by-step progress
through the pipeline. The backend reports no per-step progress, so
claiming it would be an invented detail. It names the real steps and shows
one honest indeterminate bar instead.

## 9. Copy is specific, and every number is traceable

"Likely to merge", not "Analysis complete". "No unstaged changes to
review", not "No data found". The reading under the gauge names the
corpus, the base rate and the decision threshold, and each of those is a
real committed number (§10) rather than a confident-sounding phrase.

*Prevents:* the polished-but-empty register — "Get actionable insights
into your code quality" — that would describe a linter, a coverage tool,
a security scanner or this.

## 10. The handmade detail: the gauge is calibrated

The gauge carries two marks the arc alone cannot express:

* a **notch at 50%**, the threshold where `src/api/main.py` actually flips
  MERGE to CLOSE, cut through the ring in the panel's own background
  colour, because it is a hard rule;
* a **line at 72%**, the merge rate of the 1,494 mined PRs the model was
  trained from (`results/tables/exp1_summary.json` →
  `merge_status.pooled.merge_rate` = 0.7195), drawn over the ring, because
  it is a reference point.

Both explain themselves on hover, and the sentence beside the gauge reads
the number against them: *"61% is below the 72% base rate of the 1,494
pull requests this model was trained from. The call flips to CLOSE below
50%."*

This is the one element no component library or scaffold would ever
generate, because it requires knowing what this particular model was
trained on. It is also the most useful thing on the panel: a bare "61%"
reads as healthy, and it is in fact a below-average change.

---

## Re-checking the panel visually

`panelHtml.ts` imports nothing from `vscode`, so every state can be
rendered to a standalone HTML file and opened in a browser. Inject VS
Code's own `--vscode-*` token values (Dark Modern and Light Modern both,
since the panel must hold up in each) and stub `acquireVsCodeApi` so the
click handlers attach. That is how `docs/images/vscode_panel_demo.png` and
`docs/images/panel_states_demo.png` were produced, and how the type scale
and the gauge marks were checked at real size in both themes.
