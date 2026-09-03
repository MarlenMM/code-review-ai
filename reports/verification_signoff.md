# Verification Pass — Sign-Off Notes (Step 29, re-audited post-Step-30)

> **Two passes are recorded here.** Sections 1–6 are the Step 29 pass. Section 7 is an
> independent re-audit run after the project was complete, which re-derived Sections 1–6's
> own claims from source rather than trusting them, and found four further issues. Read
> Section 7 for the current state.

*Scope: re-check every number quoted in the 4 lab reports, the final consolidated
report, the README, and the slide deck against `results/tables/*.json`; spot-check
figures against the underlying parquet; flag any discrepancy before submission.*

**Verdict of this first pass: signed off.** 4 discrepancies were found and all 4 are
fixed. Details below, including what was checked and how. (The second pass in Section 7
re-derived these findings independently, confirmed all 4 fixes held, and found 4 more —
so treat "nothing outstanding" as Section 7's verdict, not this one's.)

---

## 1. Method

Numbers were **not** read across from one report to another — each claim was checked
against the artifact it derives from, and the JSON tables were themselves re-derived
from `data/processed/*.parquet` where that was possible. Specifically:

| Layer | How it was checked |
|---|---|
| `exp1_summary.json` | Every pooled statistic **recomputed independently from parquet** (20/20 matched exactly, incl. mean/median/max/zero-counts for review comments, reviewers, PR length, and label coverage) |
| `exp2_metrics.json` | All 12 model rows compared cell-by-cell against both lab2 tables (12/12 exact) |
| Exp2 train/test split | **Re-executed** `split_by_repo_time` — reproduced 890/224, the 72.2%/82.1% merge rates, and the exact per-repo test sizes (55/54/39/38/38) |
| Exp2 rejected split | **Re-executed** the global time cutoff — reproduced the `semantic-kernel` collapse to 1 PR of 274 and the 87.9%/70.8% gap quoted in Problems #1 |
| Exp2 worked example | All 14 feature values for `microsoft/vscode#326961` read back from `features_v1.parquet` (14/14 exact) |
| `exp3_metrics.json` / `exp4_metrics.json` | Per-strategy and per-context aggregates **recomputed from `per_cell`**; both lab reports' tables reproduce exactly |
| Raw grids | `exp3_raw.jsonl` = 512 rows / 0 errors / 16 PRs; `exp4_raw.jsonl` = 148 rows / 0 errors / 4 PRs (3 × 48 complete + 4 partial) |
| Figures | `merge_status.png` and `merge_rate_ai_vs_human.png` read bar-by-bar against the data; Exp2/3/4 figures cross-checked against their JSON |

**Two structural integrity checks also passed** (claims that would silently invalidate
the cross-experiment comparison if false):

* Experiment 3's 16-PR sample is a **genuine subset of Experiment 2's 224-PR held-out
  test split** (16/16 inside it), 14:2 merged, repo-stratified 4/3/3/3/3, and **all 16
  are human-authored** — so lab3's "same test set as Experiment 2" claim holds.
* Experiment 4's 3 fully-complete PRs are **all AI-authored**, 2 merged : 1 not-merged,
  one per repo — so lab4's population claim holds.

**Grid denominator reconciled.** Experiment 4's "148/384 cells" looked wrong at first
(8 PRs × 5 tiers × 5 strategies × 2 tasks = 400, not 384). It is correct: `multi_turn`
is undefined on the bare-`diff` tier (its mechanism *is* progressive context disclosure,
so there is no context to disclose), giving 24 valid tier×strategy combos per PR, not
25 — 24 × 2 × 8 = 384. Confirmed against the raw grid: `multi_turn` appears on exactly
4 tiers, every other strategy on 5.

---

## 2. Discrepancies found and fixed

### 2.1 lab1 — AI-reviewer count understated (factual error) — **FIXED**

lab1 stated `copilot-pull-request-reviewer` was present on **1,060** PRs "with
`coderabbitai` also present on 4," and in Reflection Q4 that it "accounts for 1,060 of
our 1,064 AI-reviewed PRs."

The raw string `value_counts()` does show `1060`, but that is the count of PRs where
Copilot is the **only** AI reviewer. The 4 `coderabbitai` PRs are a **subset**, not a
disjoint group — `coderabbitai` never appears alone. The true figure is that
`copilot-pull-request-reviewer` is present on **all 1,064** AI-reviewed PRs.

Both sentences were corrected to state the containment explicitly (all 1,064; 1,060
Copilot-only, 4 with both). This slightly *strengthens* the report's own Reflection Q4
argument about uniform, high-volume automated review style.

### 2.2 final_report §7 — pooled not-merged recall wrong (rounding error) — **FIXED**

The cross-experiment table gave Experiment 3's pooled not-merged recall as **9.0%**.
The value in `exp3_metrics.json` is `0.0938` (= 3 TN / 32 actual-not-merged), i.e.
**9.4%**. Corrected in `final_report.md` and in the slide deck's copy of the same table.
(The report rounds correctly everywhere else — 0.8477→84.8%, 0.5249→0.525 — so this was
an isolated slip, not a convention.)

### 2.3 final_report §7 — inference-cost row compared two different bases — **FIXED**

The row read `~60–1,100 ms/call` (Exp3) against `~290–740 ms/call` (Exp4). Both numbers
are individually traceable, but to **different aggregations**:

* Exp3's came from **task × strategy** means (min 59.6 = zero-shot merge; max 1088.4 =
  role-based review) — matching lab3's own basis.
* Exp4's came from **strategy** means pooled across tasks (min 290.4 = few-shot; max
  741.8 = self-reflection) — matching lab4's own basis.

Placed side by side this made Experiment 4 look *tighter and slower-at-minimum* than it
is. On Experiment 3's basis, Exp4 is **~120–1,240 ms**. Both cells are now stated on the
task × strategy basis (`~60–1,090` vs `~120–1,240`), and the note under the table now
says which basis is used and why Exp4's top end is higher (2 API calls per review).
Each lab report's own figure was left unchanged — both were correct in their own context.

### 2.4 Slide deck — "1,499 of 1,500" typo — **FIXED**

Slide 3's stat-card label read "1,499 of 1,500 targeted" (the card's own headline value
correctly read 1,494). Introduced during Step 28. Corrected.

---

## 3. Clarity gap closed (not an error)

**The 1,500 − 1,494 = 6 vs. "5 excluded" arithmetic.** Both lab1 and the final report
said 1,494 of 1,500 were retrieved with "5 permanently excluded" on 502s — leaving a
6th PR unaccounted for on the face of it. Investigation confirmed **the "5" is correct**:

| Repo | Shortfall | Cause |
|---|---|---|
| `dotnet/aspnetcore` | 1 | server-side 502, retries exhausted |
| `dotnet/runtime` | 1 | server-side 502, retries exhausted |
| `microsoft/semantic-kernel` | 3 | server-side 502, retries exhausted |
| `microsoft/vscode` | 1 | **pagination drift** — the same PR landed on two pages, so the 300-entry sample list held only 299 *distinct* PRs |

The 6th was never a distinct target. This was already documented in lab1's Problems #2,
but ~150 lines away from the dataset-size claim and never connected to it. Both
documents now state the two causes together at the point the shortfall is first quoted.
Verified: `microsoft/vscode#326842` is present in the cache and the tables exactly once.

---

## 4. Re-verified after the edits

* `reports/lab1.pdf` recompiled (`xelatex`, `TEXINPUTS=.:./template//:`), 0 errors —
  **14pp → 15pp**. All three corrected passages confirmed present in the **PDF text
  layer** via `pdftotext`, not just the `.tex` source.
* `reports/final_report.pdf` recompiled (`pandoc --pdf-engine=xelatex`) — **10pp → 11pp**.
  The edited §7 table was re-read with `pdftotext -layout`: no truncated cells, no
  column bleed, the new footnote wraps cleanly.
* `reports/slides.pptx` rebuilt; `validate.py` passed; slides 3 and 7 re-rendered and
  visually re-inspected.
* Page-count references updated in `README.md` (×3), `final_report.md`, and
  `portfolio_readiness.md` to match the recompiled PDFs.
* `pytest tests/ -q` → **414 passed**. `ruff check .` → **All checks passed**.

---

## 5. Checked and correct — no change needed

Recorded so a grader (or a later pass) can see what was covered rather than assumed:

* **lab1** — all 7 table row counts; merge status per repo and pooled; review-comment,
  reviewer-count, PR-length and label distributions; AI-authored share per repo; the
  361/19 detection-method split; the 59-PR / 68-assessment / 0-false-positive precision
  check and its rule-of-three ≥92.5% bound; 4.59 commits/PR.
* **lab2** — both 6-row model tables (all cells); the human-written dataset table;
  827/287 and the 2.88:1 ratio; 71.8% AST/CFG coverage and the ~27% `staticfg` CFG
  failure rate; the full feature-importance-by-category table for V0/V1/V2;
  `p_has_approved` 41.6%; `author_prior_merge_rate` 10.0%; V2-beats-V1 on all 4 matched
  configs at +0.06–0.16 ROC-AUC.
* **lab3** — per-strategy accuracy/macro-F1/minority-recall; per-context accuracy and
  macro-F1; the BLEU/ROUGE table; `role_based` ranking last on ROUGE-L; the 60/400/458/566
  ms merge-inference table; 512/512 cells, 0 failures; 5 PRs with ground-truth comments;
  11-of-16 configs never predicting CLOSE; both peak cells at 93.75% / 0.816.
* **lab4** — the 5-row context-tier table and 5-row strategy table (all cells); the
  0.0-recall-on-every-tier claim for the three shared strategies; 0.60/0.50 recall for
  the two new ones; verbosity figures (8.5 / 3.1 / 5.7); the 742/677 vs 290–602 ms
  comparison; Experiment 3's 0.125 shared-strategy mean.
* **README** — every headline number on the page.
* **Test/tooling counts** — 414 Python tests, 49 extension unit tests, 7 VS Code
  integration tests.

---

## 6. Residual limitations (pre-existing, correctly disclosed)

Not defects — each is already stated in the relevant report, and re-confirmed as still
accurate here:

* Experiment 3/4 sample sizes (16 and 3 fully-complete PRs) are small, bounded by Groq's
  shared free-tier daily quota. Every LLM-side number is a paired, directional
  observation on a stated sample, and both lab reports say so.
* `--fetch-issues` was not enabled for the Experiment 4 grid, so `diff_issue` /
  `complete_se_context` measure those tiers' *other* wiring, not real issue text. lab4
  footnotes this at the table itself.
* AST/CFG coverage is 71.8%, with a root-caused `staticfg` bug degrading ~27% of
  otherwise-parseable Python hunks — so the S-category's true ceiling is unknown.
* CI has been verified by running its exact commands locally but never on a hosted
  GitHub Actions run (no git remote is configured).

---

## 7. Second pass — full re-audit after Step 30

*A second, independent verification pass was run over the finished project after all
30 checklist rows were complete. It deliberately did **not** trust Sections 1–6 above:
every claim in this document that was in scope was re-checked from source.*

**All of Sections 1–6 held.** Specifically re-confirmed by re-derivation rather than
re-reading: all of `exp1_summary.json` recomputed from `pull_requests`/`commits`/
`reviews`/`review_comments` parquet (including the reviewer-count and PR-length
definitions, which are distinct-reviewers-per-PR and additions+deletions respectively);
all 12 Exp2 models re-scored from their committed `.joblib` files (168/168 metric cells
exact); `split_by_repo_time` re-executed (890/224, 72.2%/82.1%, per-repo 55/54/39/38/38,
40 not-merged in test); the rejected global cutoff reproduced (semantic-kernel → 1 of
274, 87.9% vs 70.8%); the 14/14 worked-example feature values; every Exp3/Exp4
per-cell, per-strategy and per-tier aggregate recomputed from `per_cell` and from the
raw JSONL; the 0.0-recall-on-every-tier claim checked cell by cell (15/15, zero
exceptions); `multi_turn` on exactly 4 tiers and the 24 × 2 × 8 = 384 denominator; the
16-PR Exp3 sample as a subset of the 224 test split, 14:2, all human-authored; Exp4's 3
complete PRs all AI-authored, 2:1, one per repo. The §7 inference-cost cells were
re-derived on the (task × strategy) basis and reproduce ~60–1,090 and ~120–1,240 ms
exactly, confirming §2.3's fix. Figures were re-read image-by-image against the JSON —
including the two apparently-missing bars in `exp4_merge_accuracy_by_config.png`, which
are genuine zeros (`diff|self_reflection` and `diff_issue|few_shot` both score 0.0
accuracy), not dropped data.

**Also verified live, not just via unit tests:** the FastAPI backend was started for
real and `/health`, `/review` (fast **and** deep, with a real Groq call returning
`groq/llama-3.1-8b-instant multi_turn @ diff_repo_context`), and every documented error
path were exercised against it; the LLM-failure degradation path was forced with an
invalid key and returns HTTP 200 with `llm_warning` set, never a 500; the 7 integration
tests ran inside a real VS Code 1.130.0 instance against that backend; and the `.vsix`
was rebuilt and diffed against the compiled source (all 8 `.js` files, `package.json`
and `LICENSE` byte-identical).

**Four further issues were found and fixed** — all of them descriptions of a correct
number rather than a wrong number, which is the failure mode this project's numbers
were already hardened against:

1. **README called Exp3's pooled macro-F1 an "average."** 0.525 is the macro-F1 of the
   pooled confusion matrix over all 256 predictions; the mean of the 16 per-config
   macro-F1 values is **0.509**. (Accuracy and not-merged recall are identical on both
   bases — every config sees the same 16 PRs — but F1 is not linear in the confusion
   counts.) Reworded; the basis is now stated in `exp3_model_evaluation.md`, in
   `final_report.md` §7's note, and as a defence talking point in `qa_prep.md`. §7's
   column labels changed from "(pooled avg.)" to "(pooled)", and slide 7 with them.
2. **The ≥92.5% detection-precision bound was quoted immediately after "68
   assessments."** The rule of three here is 3/40 on the *random* sample; the 19-PR
   census is a full census of the weakest tier and cannot widen the bound (3/68 would
   give ≥95.6% and would be an overclaim). The basis is now stated wherever the bound
   appears outside its own derivation in `exp1_precision_check.md`.
3. **`exp4_grid_run_status.md`'s run table mixed two bases.** "Cells written" is
   148 (whole run) but the single-shot/conversational split was 90/54 (the 3 complete
   PRs only), summing to 144 and silently dropping the partial PR's 4 cells — the same
   shape as the 1,500−1,494 gap closed in Section 3 above. Now 93/55, with the 90/54
   restriction named.
4. **`exp2_feature_spec.md` described V2 as adding one feature.** It adds two
   (`author_prior_merge_rate` *and* `author_prior_pr_count`), which is what makes V2
   the same 28-feature width as V1 — load-bearing, since otherwise the table implies 27
   and invites the question of whether V2-beats-V1 is a feature-count artifact. A Count
   column was added, and the file-status row now names the four implemented columns
   rather than the guide's three-way shorthand.

Smaller items fixed in the same pass: `numpy` and `pydantic` are imported directly by
`src/` but were only present transitively, so they are now declared in
`requirements.txt`; `vscode-extension/README.md`'s `../`-relative links were rewritten
by `vsce` into `…/blob/HEAD/../src/api/main.py` inside the packaged `.vsix` (a broken
URL in the shipped artifact) and are now plain code spans; README no longer attributes
the DeepSeek rejection to `.env.example`, which only documents Gemini.

**Checked and deliberately left alone.** `results/figures/exp4_inference_time_by_strategy.png`
is generated by `score_exp4.py` but is the one figure of fifteen not displayed in any
report (lab3 shows its Exp3 counterpart; lab4 states the same latencies in prose
instead). Adding it would mean adding content to a finished, compiled report, so it was
left as-is and is recorded here instead. `portfolio_readiness.md`'s "139 default-ruff
findings" re-runs at 127 today; the count moves with ruff's own default rule set, so the
original measurement was annotated rather than restated.

---

# Pass 2 — after the panel design pass and the Groq → Qwen switch

*Same discipline as the pass above: every number this pass changed was
re-checked against the artifact that produces it, not against the previous
document.*

## 7. Numbers that moved, and what re-derived them

| Claim | Was | Now | Re-derived from |
|---|---|---|---|
| Python tests | 414 | **451** | `pytest tests/ -q` |
| Extension unit tests | 49 | **66** | `cd vscode-extension && npm test` |
| `final_report.pdf` length | 11pp | **13pp** | `pdfinfo` on the recompiled PDF |
| Deep-mode provider | Groq `llama-3.1-8b-instant` | **Qwen `qwen-plus`** | `src/api/llm_review.py`; label asserted in `tests/test_api_llm_review.py` |

Updated in: `README.md` (test counts ×3, page count, provider section),
`reports/qa_prep.md` (core-numbers table + a new provider talking point),
`reports/final_report.md` (§8 verification paragraph, §9 limitations),
`reports/vscode_extension_design.md` (header + §12–14),
`reports/api_design.md` (§8), `vscode-extension/README.md`.

**Deliberately not changed**, because they are dated records of the state at
the time rather than claims about the repository now: §4 above (10pp → 11pp),
`reports/api_design.md` §6.1 ("414 total"), `reports/portfolio_readiness.md`'s
counts, and `reports/vscode_extension_design.md` §7.1 ("49 passing", Step 25).
The Experiment 3/4 provider statements in `final_report.md` §5 and the lab
reports are likewise correct as written — those experiments *were* run on Groq,
and still reproduce from cache under `--provider groq`.

## 8. The PDF recompile was verified, not assumed

The exact original invocation was not recorded anywhere, so it was recovered
before editing rather than guessed: `pandoc final_report.md
--pdf-engine=xelatex -o final_report.pdf`, run from `reports/`, was used to
rebuild the **unmodified** source and the result diffed against the committed
PDF via `pdftotext -layout` — **515 lines, zero differences**. Only then was
the source edited and rebuilt, so the 11pp → 13pp change is attributable to
the edits and not to a different toolchain.

Every new passage was then confirmed present in the recompiled PDF's own
**text layer** (`pdftotext -layout`), not merely in the Markdown source: the
451/66 test counts, the Qwen/DashScope paragraphs, the calibrated-gauge
passage, and the `401 Incorrect API key provided` verification note.

## 9. What is verified, and the one thing that is not

* `pytest tests/ -q` → **451 passed**. `ruff check .` → **All checks passed**.
* `npm test` → **66 passed**. `npm run test:integration` → **7 passed** in a
  real VS Code 1.130.0 against a real backend and a real dirty repo.
* `npx vsce ls` confirms the new gutter icons (`media/review-comment-*.svg`)
  are actually inside the package, not just on disk.
* DashScope's endpoint and request shape were confirmed by a **real** HTTP
  round-trip returning `401 Incorrect API key provided` — a wrong URL returns
  404 and a malformed body 400, so 401 is the response that validates both.
  Both regional endpoints answer 401 rather than 404.
* A real `POST /review` with `mode: "deep"` and no key configured returns
  **HTTP 200** with the fast-mode prediction and an actionable `llm_warning`,
  confirming the degrade-don't-fail contract still holds after the provider
  change.
* **Not verified: a live `qwen-plus` completion.** No DashScope key exists in
  this environment. Everything up to the API-key boundary is covered; the
  quality of the comments Qwen actually returns is not, and
  `final_report.md` §9 says so.
