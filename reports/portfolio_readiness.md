# Portfolio Readiness — README, LICENSE, CI (Step 26)

*Deliverable: plan Section 9 row 26's repo-wide half — "write a real README
(architecture diagram, setup steps, demo GIF/screenshot) plus LICENSE and a
basic GitHub Actions CI workflow (lint + tests on push)." The extension-
specific half (packaging, packaged-artifact verification) is
`reports/vscode_extension_design.md` §9–10 — this doc covers what's new
outside `vscode-extension/`.*

---

## 1. README — where every number came from

Every figure in `README.md`'s Results section was re-read from the actual
committed artifact during this step, not recalled from memory or an
earlier report's prose:

| Claim | Source | Verified via |
|---|---|---|
| 1,494 PRs, 380 AI-authored (25.4%), merge rate 72.0%/65.3%/74.2% | `results/tables/exp1_summary.json` | `python3 -c "import json; ..."` read of `dataset`/`ai_vs_human`/`merge_rate_by_authorship` |
| `rf_v1_balanced` 76.8% acc / 42.5% not-merged recall | `results/tables/exp2_metrics.json['rf_v1_balanced']` | direct read |
| Exp3 best config 93.75% acc / 0.816 macro-F1, pooled macro-F1 0.525 | `results/tables/exp3_metrics.json` | `merge_prediction.per_cell` sorted by `macro_f1`; `pooled_over_all_cells` |
| Exp4 shared-strategy 0.0 not-merged recall; self_reflection 0.60 / multi_turn 0.50 recall | `results/tables/exp4_metrics.json['central_analysis']['prompt_effect_on_merge']` | direct read |
| Lab report page counts (15/12/16/14pp) | the actual compiled PDFs | `pdfinfo reports/lab{1,2,3,4}.pdf` |
| Test counts (414 / 49 / 7) | live `pytest`/`npm test`/`npm run test:integration` runs, this session | terminal output, reproduced below in §3 |

This matters because at least one number here (lab1's page count) had been
cited as "7 pages" in an earlier session's summary — stale by the time this
step ran (the report grew across later revisions). Re-measuring rather than
trusting a remembered figure is the same discipline every prior step in
this project applied to the mining/ML/LLM numbers; the README shouldn't be
the one place that discipline lapses.

## 2. The architecture diagram

A Mermaid flowchart (GitHub renders Mermaid natively in READMEs, no image
asset to keep in sync). It mirrors plan Section 2's own pipeline diagram
structurally, but adds the report each stage feeds — tying the six
numbered boxes back to something concrete a reader can click into, which
the plan's own ASCII version didn't do.

## 3. Demo assets (`docs/images/`)

Two real screenshots, both produced this step, neither hand-drawn or
mocked:

* **`vscode_panel_demo.png`** — the extension's actual results panel
  (`panelHtml.ts`, unchanged from what ships), rendered with a real
  `mode: "deep"` backend response (cached from the identical call already
  made during Step 25's verification, so this cost no additional Groq
  quota) and captured via headless Chrome
  (`google-chrome --headless --screenshot=...`). VS Code's own `--vscode-*`
  CSS custom properties (font, colors, borders) are supplied by the real
  editor at runtime and are absent in a bare browser, so a set of Dark+
  theme values matching VS Code's own defaults were injected for this
  capture only — documented inline in the capture script, and the panel's
  shipped CSS is untouched (see `panelHtml.ts`; its fallback colors are
  deliberately theme-agnostic, not tuned for this screenshot). This is
  presented in the README explicitly as *"the extension's real results
  panel, rendered with real backend + live-LLM output"* together with a
  link to `vscode_extension_design.md`'s explanation of why a literal
  inside-VS-Code capture wasn't possible (§3 there: IDE apps are
  click-only for the computer-use tooling available in this environment —
  no keyboard access, so the Command Palette can't be reached that way).
* **`api_docs_demo.png`** — FastAPI's auto-generated Swagger UI at
  `/docs`, captured the same way, unmodified (Swagger UI's own default
  styling, no injected theme). A genuinely unedited screenshot of the real
  running backend.

Both PNGs are committed (129 KB and 54 KB — small enough not to bloat the
repo). The demo git repo and backend process used to produce them were
deleted/stopped afterward; only the two image files and this write-up
persist.

## 4. LICENSE

MIT, `Marlen Melis, 2026` — the standard, low-friction default for a
coursework/portfolio repo with no third-party contributors yet and no
reason to restrict reuse. Copied into `vscode-extension/LICENSE` too so it
ships inside the packaged `.vsix` (`reports/vscode_extension_design.md`
§9) — a user installing the extension doesn't necessarily have the parent
monorepo checked out to find the root one.

## 5. CI (`.github/workflows/ci.yml`)

Two independent jobs, matching the two runtimes already in this repo —
full rationale is written inline in the workflow file itself (why no
secrets are needed, why the integration suite is deliberately excluded)
rather than duplicated here. One decision worth explaining in more depth
than the inline comment has room for:

### Why `ruff` scoped to `F` + `E9`, not the default rule set

Running `ruff check` with its full default rules against `src/`/`tests/`
surfaced **139 findings** (ruff 0.16; the exact count moves with ruff's own
default rule set — a re-run at audit time reported 127) — almost entirely
line-length and other style
nits accumulated across ~25 steps of work that were never linted as they
went. Two options: reformat the whole codebase to satisfy a style guide
invented after the fact (real risk of touching already-tested,
already-reported-on experiment code for zero behavioral benefit, this late
in the project), or scope the CI lint gate to what actually indicates a
bug. Chose the latter: `ruff.toml` selects only `F` (Pyflakes — unused
imports, undefined names, unused variables) and `E9` (syntax errors).

That narrower check found **12 real, if minor, issues** — 9 unused
imports, one redundant f-string, two unused local variables (in test-mock
classes, where the variable's *side effect*, not its value, mattered) —
and all 12 were fixed as part of adding this CI gate, verified with
`pytest tests/ -q` before and after (414 passed both times — behavior
unchanged). The alternative of adding the check without fixing what it
already finds was rejected: an immediately-red CI badge on a portfolio
repo defeats the purpose of adding CI at all.

### Why the extension's "lint" step is `tsc` (compile), not ESLint

The extension has no ESLint config, and adding one now — picking a style
guide, tuning it against ~2,500 lines of already-written, already-tested
TypeScript — is more new surface area than a "basic CI" ask justifies.
`npm run compile` (`tsc -p ./`, already the extension's own build step)
is a genuine, meaningful static check on its own: with `"strict": true`
in `tsconfig.json`, it catches type errors, which is most of what a
JS/TS lint step buys you anyway. Labelled "Typecheck" in the workflow
rather than "Lint" so the step name doesn't overclaim what it does.

## 6. Final verification (this step, end to end)

```
pytest tests/ -q                                    # 414 passed
ruff check .                                          # All checks passed!
cd vscode-extension && npm run compile && npm test    # 49 passed
CODE_REVIEW_AI_TEST_WORKSPACE=<throwaway repo> \
  npm run test:integration                            # 7 passed, real VS Code
npx vsce package                                       # clean, 21.1 KB .vsix
code --install-extension ... && diff -rq ...           # installed out/ == tested out/
```

All green. No stray background processes, temp repos, or `/tmp` VS Code
profiles were left behind (checked explicitly after each verification
pass — `pgrep -fl uvicorn`, `pgrep -fl http.server` both empty at the end
of this step).

## 7. What's still genuinely open

* **`repository` URL is a best-effort placeholder**, not a confirmed one —
  see `vscode_extension_design.md` §9. Fix once the real GitHub repo
  exists, if the name differs from `MarlenMM/code-review-ai`.
* **CI has never run on real GitHub Actions** — there is no git remote
  configured in this session, so nothing has been pushed. The workflow was
  written to the same standard as everything else here (idiomatic,
  documented, locally-equivalent commands verified to pass in §6), but
  "verify against a real run" — this project's own stated standard — could
  not be applied to the *hosted* CI run itself, only to the commands it
  invokes. Worth a first-push sanity check once a remote exists.
* **No `notebooks/` directory** — plan Section 8.4's repo-structure sketch
  lists one; Section 9's actual 26-row checklist never asked for exploratory
  notebooks as a deliverable, so none were fabricated just to match the
  sketch. `README.md`'s repository-structure section reflects the real
  tree, not the plan's aspirational one.
