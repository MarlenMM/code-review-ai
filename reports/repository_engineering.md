# Repository engineering — README, LICENSE, CI

*Plan Section 9, row 26: "write a real README (architecture diagram, setup
steps, demo GIF/screenshot) plus LICENSE and a basic GitHub Actions CI
workflow (lint + tests on push)." The extension-specific half of that row
(packaging and checking the packaged artifact) is in
[`vscode_extension_design.md`](vscode_extension_design.md) §9–10; this note
covers everything outside `vscode-extension/`.*

---

## 1. README — where every number comes from

Every figure in the README's Results section is read from a committed
artifact rather than restated from a report's prose:

| Claim | Source | Read via |
|---|---|---|
| 1,494 PRs, 380 AI-authored (25.4%), merge rate 72.0% / 65.3% / 74.2% | `results/tables/exp1_summary.json` | `dataset`, `ai_vs_human`, `merge_rate_by_authorship` |
| `rf_v1_balanced` 76.8% accuracy, 42.5% not-merged recall | `results/tables/exp2_metrics.json['rf_v1_balanced']` | direct read |
| Exp 3 best config 93.75% accuracy / 0.816 macro-F1; pooled macro-F1 0.525 | `results/tables/exp3_metrics.json` | `merge_prediction.per_cell` sorted by `macro_f1`; `pooled_over_all_cells` |
| Exp 4 shared strategies 0.0 not-merged recall; `self_reflection` 0.60, `multi_turn` 0.50 | `results/tables/exp4_metrics.json['central_analysis']['prompt_effect_on_merge']` | direct read |
| Lab report page counts (15 / 12 / 16 / 14 pp) | the compiled PDFs | `pdfinfo reports/lab{1,2,3,4}.pdf` |
| Test counts | `pytest`, `npm test` and `npm run test:integration` | the suites themselves, and the CI log for every push |

Re-measuring matters: lab 1's page count, for example, had been quoted as
"7 pages" in an early draft, long after the report had grown.

## 2. The architecture diagram

A Mermaid flowchart, which GitHub renders natively, so there is no image
asset to keep in sync. It follows the plan's pipeline diagram but adds the
report each stage feeds, so every numbered box leads to something a reader
can open.

## 3. Demo assets (`docs/images/`)

All screenshots are of the real system; none is drawn or mocked:

* **`vscode_panel_demo.png`** — the extension's results panel
  (`panelHtml.ts`, exactly as shipped), rendered with a real `mode: "deep"`
  backend response and captured with headless Chrome. VS Code supplies its
  `--vscode-*` CSS custom properties (font, colours, borders) at runtime, and
  a bare browser has none, so default Dark+ theme values were injected for
  the capture only; the panel's shipped CSS is untouched. The README says
  so beside the image and links to `vscode_extension_design.md` §3, which
  explains why a capture from inside VS Code was not used.
* **`panel_states_demo.png`** — the panel's in-flight, empty and
  backend-down states, rendered the same way.
* **`api_docs_demo.png`** — FastAPI's generated Swagger UI at `/docs`,
  captured unmodified from the running backend.

The temporary demo repository and backend process used for the captures
were removed afterwards; only the images and this note remain.

## 4. LICENSE

MIT. It is the standard low-friction choice for research code with no
reason to restrict reuse. A copy also lives in `vscode-extension/LICENSE`,
so it ships inside the packaged `.vsix` (`vscode_extension_design.md` §9):
someone installing the extension does not necessarily have the repository
checked out.

## 5. CI (`.github/workflows/ci.yml`)

Two independent jobs, one per runtime in the repository. The workflow file
itself explains why no secrets are needed and why the integration suite is
excluded. One decision needs more room than an inline comment:

### Why `ruff` is scoped to `F` + `E9`

Running `ruff check` with its full default rule set over `src/` and
`tests/` reported **139 findings** (ruff 0.16; the count moves with ruff's
defaults, and a later run reported 127), almost all line length and other
style points in experiment code that had already been tested and reported
on. Reformatting all of it for a style guide adopted after the fact would
have touched result-producing code for no behavioural gain, so the CI gate
is scoped to what indicates a bug: `F` (Pyflakes — unused imports,
undefined names, unused variables) and `E9` (syntax errors).

That narrower check found **12 real, if minor, issues** — nine unused
imports, one redundant f-string and two unused local variables in test
doubles — and all twelve were fixed when the gate was added, with the
full test suite passing before and after.

### Why the extension's static check is `tsc`, not ESLint

The extension has no ESLint configuration, and choosing and tuning a style
guide against roughly 2,500 lines of tested TypeScript was more than a
basic CI step justifies. `npm run compile` (`tsc -p ./` with
`"strict": true`) already catches type errors, which is most of what a
lint step buys for TypeScript. The workflow labels the step "Typecheck"
rather than "Lint" so its name does not overclaim.

## 6. Verification

```
pytest tests/ -q                                       # Python suite, fully offline
ruff check .                                           # F + E9
cd vscode-extension && npm run compile && npm test     # extension unit tests
CODE_REVIEW_AI_TEST_WORKSPACE=<throwaway repo> \
  npm run test:integration                             # 7 tests in a real VS Code
npx vsce package                                       # clean .vsix
code --install-extension ... && diff -rq ...           # installed out/ == tested out/
```

CI runs the first three on every push; the current counts are in the
README's Testing table.

## 7. Not included

* **No `notebooks/` directory.** The plan's repository sketch (Section 8.4)
  lists one, but none of the 26 checklist rows asked for exploratory
  notebooks, so none were added to match the sketch. The README's structure
  section shows the real tree.
