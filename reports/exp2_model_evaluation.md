# Experiment 2 — Model Evaluation & Discussion (Step 12)

*Held-out test-set results for the 12 SVM/RF models trained in Step 11 (3 feature
variants × plain/`balanced` class-weight), Random Forest feature importance by
category, and the V0/V1/V2 comparison the Step 8 spec calls for. All numbers below
are real, produced by `python -m src.ml.evaluate` — see
`results/tables/exp2_metrics.{json,csv}`, `results/tables/exp2_feature_importance.json`,
and `results/figures/exp2_{feature_importance_by_category,model_comparison}.png`.*

---

## 1. Test set

Per-repo time-based split (`src/ml/common.py`; see that module's docstring for why a
single global cutoff was rejected): **890 train / 224 test**, train merge rate 72.3%,
test merge rate 82.1%. Test composition is proportional across all 5 repos
(semantic-kernel 55, home-assistant 54, vscode 39, aspnetcore 38, runtime 38) — no
repo is degenerate, unlike the rejected global-cutoff split.

---

## 2. Headline: V1 (per-spec, compliant)

| Model | class_weight | Accuracy | ROC-AUC | Recall (merged) | **Recall (not-merged)** | F1 (not-merged) |
|---|---|---|---|---|---|---|
| SVM | plain | 0.817 | 0.561 | 0.973 | **0.100** | 0.163 |
| SVM | balanced | 0.817 | 0.607 | 0.989 | **0.025** | 0.047 |
| RF | plain | 0.821 | 0.678 | 0.946 | **0.250** | 0.333 |
| RF | balanced | 0.768 | 0.677 | 0.842 | **0.425** | 0.395 |

The compliant, spec-exact feature set gets real but modest lift over chance
(ROC-AUC 0.56–0.68) — a legitimate, if unglamorous, headline result. Random Forest
clearly outperforms SVM here on every metric that matters for the minority class.

---

## 3. V0 vs V1 — quantifying *why* §2.4.3 bans review-process features

| Model | class_weight | Accuracy | ROC-AUC | Recall (not-merged) |
|---|---|---|---|---|
| SVM (V0) | balanced | 0.987 | **0.984** | 0.950 |
| RF (V0) | balanced | 0.982 | **0.989** | 0.950 |
| RF (V1) | balanced | 0.768 | **0.677** | 0.425 |

V0 (V1 plus the forbidden `p_*`/`num_labels` process features) reaches
**ROC-AUC 0.98–0.99** — a dramatic, model-independent jump from V1's 0.56–0.68. Random
Forest's feature importances make the mechanism concrete rather than asserted:

| Rank | V0 top feature | Importance |
|---|---|---|
| 1 | `p_has_approved` | **41.6%** |
| 2 | `p_num_reviewers` | 10.4% |
| 3 | `p_num_reviews` | 7.7% |
| 4 | `p_num_issue_comments` | 3.3% |

**`p_has_approved` alone accounts for 41.6% of total importance** — exactly the
failure mode the Step 8 spec predicted ("an `APPROVED` review essentially *is* the
merge decision"). Aggregated by category, the forbidden **P** features account for
**68.9%** of V0's total RF importance (figure:
`exp2_feature_importance_by_category.png`), crowding out the legitimate M/S/T
signal that V1 is built entirely from. V0's headline numbers are real, reproducible,
and worthless for pre-merge prediction — which is the whole point of measuring it.

---

## 4. V1 vs V2 — a genuinely surprising result

| Model | class_weight | V1 ROC-AUC | V2 ROC-AUC | Δ |
|---|---|---|---|---|
| SVM | plain | 0.561 | 0.718 | **+0.157** |
| SVM | balanced | 0.607 | 0.740 | **+0.133** |
| RF | plain | 0.678 | 0.739 | **+0.061** |
| RF | balanced | 0.677 | 0.744 | **+0.067** |

**V2 (pre-review-strict) outperforms V1 on every one of the 4 matched
model/class-weight configurations** — not a single fluke. This runs against the
naive assumption that a *stricter*, more leakage-resistant feature set must cost
predictive power. It doesn't, here, and the RF feature importances explain why:
V2 drops two features that grow during review (`m_num_commits`,
`t_commit_msg_len_total`) and adds `author_prior_merge_rate` /
`author_prior_pr_count` — and the added feature turns out to be the single most
important feature in the whole V2 model:

| Rank | V2 top feature | Importance |
|---|---|---|
| 1 | `author_prior_merge_rate` | **10.0%** |
| 2 | `m_frac_added` | 6.1% |
| 3 | `t_commit_msg_len_first` | 6.0% |
| … | | |
| 8 | `author_prior_pr_count` | 5.2% |

Together the two author-history features contribute **15.3%** of V2's total
importance (vs. 0% in V1, where they don't exist) — more than any single category
in V1. The two dropped features apparently weren't pulling much weight in V1 to
begin with (neither appears in V1's top-3), so their removal cost little while
`author_prior_merge_rate` — a genuinely known-before-review, non-leaky signal about
who is submitting the code — added real, measurable signal. **The lesson: the
"pre-review-only" framing isn't just a stricter, more honest subset of V1 — it's an
opportunity to add better features that the leakage-averse framing naturally
surfaces**, worth stating plainly rather than assuming stricter always means worse.

---

## 5. SVM vs RF — a suitability finding, not just a leaderboard

Random Forest's `class_weight='balanced'` behaves exactly as the textbook predicts —
a clean accuracy-for-minority-recall trade (§6):

| Variant | RF plain: acc / recall(not-merged) | RF balanced: acc / recall(not-merged) |
|---|---|---|
| V1 | 0.821 / 0.250 | 0.768 / **0.425** |
| V2 | 0.875 / 0.375 | 0.826 / **0.425** |

SVM's `class_weight='balanced'` does not behave the same way here — it barely moves
ROC-AUC and, on V1, makes hard-threshold minority recall *worse*, not better:

| Variant | SVM plain: acc / recall(not-merged) | SVM balanced: acc / recall(not-merged) |
|---|---|---|
| V1 | 0.817 / **0.100** | 0.817 / **0.025** |
| V2 | 0.844 / 0.200 | 0.835 / 0.225 |

The likely mechanism: SVM's probability output here comes from
`CalibratedClassifierCV` (see `src/ml/train_svm.py` — `SVC(probability=True)` is
deprecated as of scikit-learn 1.9), which re-derives calibrated probabilities from
the `class_weight`-shifted decision function via a *separate* cross-validated
calibration pass. `class_weight` shifts the SVM's margin; calibration then
re-maps decision-function values back onto [0, 1] using its own cross-validated
fit — the two stages aren't jointly optimized, so `class_weight`'s intended effect
on the 0.5-threshold recall isn't guaranteed to survive calibration intact. Random
Forest's `class_weight` instead directly reweights the impurity criterion each
tree splits on, with no separate calibration stage to potentially undo it — a
plausible, concrete answer to reflection Q3 ("for which application scenarios are
SVM and Random Forest respectively suitable"): **for an imbalanced target where
class-weighting is the intended lever, RF's effect is more direct and predictable
than SVM's here**, independent of which one has the higher raw ROC-AUC.

---

## 6. Feature importance by category — answering "which type of feature matters most"

| Category | V0 | V1 | V2 |
|---|---|---|---|
| P — process/forbidden | 68.9% | — | — |
| M — code modification | 14.6% | **44.2%** | 37.9% |
| T — textual | 11.8% | **40.0%** | 32.2% |
| S — code structure | 4.7% | 15.8% | 14.6% |
| author history | — | — | 15.3% |

In both compliant variants, **M (code modification) and T (textual) dominate,
roughly evenly, together accounting for ~84% (V1) / ~70% (V2) of importance — no
single feature or category quietly does all the work**, unlike V0. **S (code
structure) consistently contributes the least** (15.8% V1, 14.6% V2) despite being
the most implementation-heavy category (Step 9's AST/CFG pipeline). This connects
directly to Step 9's own fidelity findings rather than being a new mystery:
`s_has_parseable_code` is true for only ~72% of PRs to begin with, and even among
parseable Python hunks, `staticfg`'s CFG stage fails on ~27% of them (the
`ast.NameConstant` bug documented in `reports/exp2_ast_cfg_fidelity.md`) — so a
meaningful fraction of S values are honestly-reported zeros rather than measured
structure, which plausibly dilutes their predictive signal relative to M/T features
that are available for every PR. This is a concrete, evidence-linked answer to
reflection Q4 ("which type of feature contributes the most, why"): M and T, roughly
tied, because they're always measurable; S trails because its own upstream
pipeline (honestly) can't always measure it.

---

## 7. Artifacts

- `results/tables/exp2_metrics.json` / `.csv` — full per-model metrics (12 configs).
- `results/tables/exp2_feature_importance.json` — RF per-feature + per-category
  importances, all three variants.
- `results/figures/exp2_feature_importance_by_category.png` — the §3/§4/§6 story
  in one chart.
- `results/figures/exp2_model_comparison.png` — SVM vs RF × V0/V1/V2, balanced mode.
- `src/ml/evaluate.py` — reproduces all of the above from the Step 11 saved models.
