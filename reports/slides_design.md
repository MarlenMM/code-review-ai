# Slide Deck — Design & Verification (Step 28)

*Deliverable: `reports/slides.pptx` — 10 slides, 13.33"×7.5" widescreen,
built with `pptxgenjs`. Adapts `reports/final_report.md` into presentation
form: architecture diagram, key numbers per experiment, a plugin demo, and
the ML-vs-LLM-vs-improved-LLM comparison table, per the Step 28 ask.*

---

## 1. Structure

| # | Slide | Content | Source |
|---|---|---|---|
| 1 | Title | Project name, five-stage subtitle, author/course line | — |
| 2 | Architecture | 6-stage pipeline as native shapes (icon, number, one-line result per stage) | `final_report.md` §2 diagram, redrawn as shapes rather than embedded ASCII |
| 3 | Experiment 1 | 4 stat cards (PR count, merge rate, AI-authored share, merge-rate split) + `merge_rate_ai_vs_human.png` | `results/tables/exp1_summary.json` |
| 4 | Experiment 2 | Headline stat + 2 non-obvious findings (leakage measured, V2>V1) + `exp2_model_comparison.png` | `results/tables/exp2_metrics.json`, `exp2_feature_importance.json` |
| 5 | Experiment 3 | Peak-parity stat, pooled-average stat, `exp3_merge_accuracy_by_config.png` | `results/tables/exp3_metrics.json` |
| 6 | Experiment 4 | 0.0-recall stat, self-reflection/multi-turn stat, `exp4_not_merged_recall_by_strategy.png` | `results/tables/exp4_metrics.json` |
| 7 | Synthesis | Native table = `final_report.md` §7's cross-experiment table, verbatim | same |
| 8 | System (backend) | fast/deep mode cards + `docs/images/api_docs_demo.png` | `reports/api_design.md` |
| 9 | System (plugin) | Feature callouts + `docs/images/vscode_panel_demo.png` | `reports/vscode_extension_design.md` |
| 10 | Conclusion | Report's closing throughline paragraph, verbatim | `final_report.md` §10 |

Four of the fifteen `results/figures/*.png` plots were used — the ones that
each carry one slide's specific point (leakage ROC-AUC jump, LLM
context×strategy grid, the 0.0-recall bar chart) rather than the full set,
per the instruction not to dump every plot in.

## 2. Number verification

Every stat card and table cell was checked directly against
`results/tables/{exp1_summary,exp2_metrics,exp3_metrics,exp4_metrics}.json`
before being typed into the deck — not copied from `final_report.md` on
trust. All matched exactly (see the Step 28 session's verification pass);
no discrepancies were found between the report and the underlying JSON at
the time of this build.

## 3. Assets built for this step

Nothing under `results/` or `docs/` was regenerated — all chart PNGs and
both screenshots are the real, previously-verified outputs from Steps 6,
12/13, 17, 21, and 26. New assets are deck-internal only:

* 18 icons rendered from `react-icons` (Font Awesome 6) via
  `ReactDOMServer` → raw SVG → `sharp` rasterization, in the deck's own
  navy/blue/green/red palette. Not committed as separate files — embedded
  directly in `slides.pptx`.
* The architecture diagram (slide 2) is drawn as native PowerPoint shapes,
  not an image — editable, and avoids re-rendering the report's ASCII
  diagram as a screenshot.

## 4. Verification performed

* `scripts/office/validate.py slides.pptx` — schema/relationship/content-type
  checks: **all passed**.
* Rendered to PDF via LibreOffice (`soffice`, installed via `brew` for this
  step — not previously on the machine) and to JPEG via `pdftoppm`; all 10
  slides visually inspected. Two real defects were caught and fixed this
  way, not by trusting the generator code:
  - An early icon-rendering bug (SVG reconstruction dropped the
    `currentColor` styling react-icons relies on) rendered every icon as a
    solid black blob instead of a white glyph — fixed by rasterizing
    react-icons' own SVG output directly instead of rebuilding the `<svg>`
    wrapper.
  - Slide 9's title text overflowed into the screenshot's card because its
    text box was wider than the space left of the image — fixed by
    shortening the title and narrowing its box.
* `markitdown slides.pptx`, grepped for placeholder markers
  (`TODO`, `lorem ipsum`, `[insert`, `xxx`, …) — none found.
* Slide count and canvas dimensions confirmed via `python-pptx`
  (10 slides, 13.33"×7.5").

## 5. Known limitation

The deck was visually QA'd via LibreOffice's renderer, not real
PowerPoint/Keynote — per the `pptx` skill's own guidance, fonts used here
(Calibri) are on the "renders true-to-width" safe list specifically so that
LibreOffice's text-fit preview is trustworthy, but a final look in the
user's actual presentation software before presenting is still worthwhile.
