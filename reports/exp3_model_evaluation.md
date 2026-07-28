# Experiment 3 — LLM Evaluation & Analysis (Step 17)

*Scoring of the Experiment 3 grid (`results/tables/exp3_raw.jsonl` →
`results/tables/exp3_metrics.json`, via `src/llm/score_exp3.py`), with the
generated-vs-real side-by-side in `results/tables/exp3_qualitative.md`. All
numbers below are computed from the real run, not estimated. Read
`reports/exp3_grid_run_status.md` first for how the dataset was produced and its
free-tier limits.*

Model: **Groq `llama-3.1-8b-instant`** (free tier). Tasks: merge prediction and
review-comment generation, over **4 code contexts × 4 prompt strategies**, on
**16 PRs** — the complete 512-cell grid, 0 failures, 0 unparseable responses.

---

## 0. Read this first: the sample, honestly

The run is a **repo-stratified 16-PR subset** of Experiment 2's 224-PR test set
(the full 224-PR grid is infeasible on the free tier — §4 of the run-status
doc). Every one of the 16 PRs is complete across all 32 cells, so every cell
below is scored on exactly the same PRs. Two consequences still bound how
strong any *absolute* number can be:

- **Merge labels are 14 merged : 2 not-merged.** A model that blindly predicts
  MERGE scores 14/16 = 0.875 accuracy. So **accuracy alone is close to
  uninformative** — the signal is in *which* configs also catch the not-merged
  PRs (minority-class recall / macro-F1), and in the *relative* ordering of
  contexts and strategies.
- **Only 5 of 16 PRs carry human review comments**, so BLEU/ROUGE is averaged
  over a handful of PRs.

With 2 negatives rather than 1, minority-class recall is now a real measurement
(0.0 / 0.5 / 1.0) instead of a single-PR coin flip — enough to separate configs,
though still small. **0 of 512 responses were unparseable** — the prompt output
contracts (Step 14) held perfectly in practice.

---

## 1. Merge prediction

### 1.1 The majority-class trap, made visible

**11 of the 16 configs score exactly 0.875 accuracy with 0.00 not-merged
recall** — they always say MERGE and never catch a negative. This is precisely
the class-imbalance failure Experiment 1's reflection Q5 is about, now
reproduced for the LLM. Three configs break the pattern and actually caught a
not-merged PR:

| Config | Accuracy | Not-merged recall | Macro-F1 |
|---|---|---|---|
| `diff_commit_message` + `few_shot` | **0.938** | **0.50** | **0.816** |
| `diff_metadata` + `role_based` | **0.938** | **0.50** | **0.816** |
| `diff_only` + `few_shot` | 0.750 | **0.50** | 0.590 |
| *(the 11 always-MERGE cells)* | 0.875 | 0.00 | 0.467 |
| `diff_description` + `few_shot` | 0.750 | 0.00 | 0.429 |
| `diff_metadata` + `few_shot` | 0.563 | 0.00 | 0.360 |

(3 + 11 + 2 = all 16 configs.)

The two best configs achieve an identical confusion matrix (14 TP, 1 FP, 1 TN,
0 FN): they keep every merged PR right *and* catch one of the two negatives.
That is a **+0.35 macro-F1 improvement over the always-MERGE baseline** at no
accuracy cost — the clearest single result in this experiment.

### 1.2 Relative ordering (the part worth reading)

Averaging across the four contexts per strategy, and the four strategies per
context:

| Prompt strategy | Avg acc | Avg not-merged recall | Avg macro-F1 |
|---|---|---|---|
| `role_based` | **0.891** | 0.125 | **0.554** |
| `few_shot` | 0.750 | **0.250** | 0.549 |
| `cot` | 0.875 | 0.000 | 0.467 |
| `zero_shot` | 0.875 | 0.000 | 0.467 |

| Code context | Avg acc | Avg not-merged recall | Avg macro-F1 |
|---|---|---|---|
| `diff + commit message` | **0.891** | 0.125 | **0.554** |
| `diff only` | 0.844 | 0.125 | 0.498 |
| `diff + metadata` | 0.813 | 0.125 | 0.527 |
| `diff + description` | 0.844 | 0.000 | 0.457 |

Two interpretable findings:

- **`few_shot` is the recall specialist.** It has the *lowest* accuracy (0.750)
  but by far the *highest* minority recall (0.250) — its balanced MERGE/CLOSE
  demonstrations push the model to predict CLOSE more often, which costs
  accuracy under an 87.5%-merged reality but is the only strategy that
  consistently produces minority-class predictions. This is the
  accuracy-vs-recall trade-off from Experiment 2 re-appearing as a
  *prompt-design* dial rather than a `class_weight` dial — a genuinely useful
  cross-experiment observation for reflection Q3.9(2).
- **`role_based` is the best all-rounder** (top accuracy *and* top macro-F1),
  and **`diff + commit message` is the strongest context**. Commit messages
  carry the author's own statement of intent and completeness, which is exactly
  the signal a merge decision turns on — a satisfying answer to reflection
  Q3.9(3) ("which context contributes most to merge prediction?").
- `zero_shot` and `cot` are *identical* on every merge metric here: both simply
  always predict MERGE. Chain-of-thought reasoning did **not** change the
  decision on this task for this 8B model, only the tokens spent getting there
  (§3).

### 1.3 LLM vs. the Experiment 2 ML models (same 16 PRs)

`compare_exp2` re-scored the saved Exp2 models on *exactly* these PRs:

| Model | Accuracy | Not-merged recall | Macro-F1 |
|---|---|---|---|
| **RF V1 (plain)** | **0.938** | 0.50 | **0.816** |
| **SVM V1 (plain)** | **0.938** | 0.50 | **0.816** |
| RF V1 (balanced) | 0.750 | 0.50 | 0.590 |
| SVM V1 (balanced) | 0.875 | 0.00 | 0.467 |
| **Best LLM config** (`diff_commit_message`+`few_shot`) | **0.938** | 0.50 | **0.816** |
| LLM pooled over all 16 configs | 0.848 | 0.09 | 0.525 |

*The pooled row is a single confusion matrix over all 16 × 16 = 256 predictions,
not the mean of the 16 per-config scores. For accuracy and not-merged recall the
two are identical (every config sees the same 16 PRs, 14:2), so 0.848 / 0.09 hold
on either basis; macro-F1 is not linear in the confusion counts, so the mean of
the 16 per-config macro-F1 values is **0.509**, slightly below the pooled 0.525.
Both describe the same "average configuration is far below the peak" finding —
the pooled figure is the one quoted throughout the reports.*

The honest reading, which is more nuanced than "one wins":

- **The best LLM configuration exactly matches the best ML models** — identical
  accuracy, macro-F1, and confusion matrix (14/1/1/0). A zero-training,
  prompt-only approach reaches the same operating point as a trained SVM/RF on
  this subset.
- **But the *typical* LLM configuration is well below them** (pooled macro-F1
  0.525 vs 0.816; 0.509 as a mean over the 16 configs — see the note above).
  Most prompt/context combinations never predict CLOSE at all. So the
  LLM's competitiveness is entirely contingent on *choosing the right prompt and
  context* — which is precisely the thing Experiment 3 exists to measure, and
  the strongest argument for why prompt engineering is a first-class engineering
  concern rather than cosmetic.
- The trained models get there **for free at inference time** (milliseconds, no
  API), whereas the LLM needs a network round-trip per PR (§3). For reflection
  Q3.9(5), that is the real trade-off: comparable peak quality, very different
  cost, and the LLM additionally generates review *text*, which SVM/RF cannot.

At n=16 with 2 negatives, all of this remains indicative rather than
conclusive — one PR flipping moves macro-F1 noticeably. It is stated as a
measured observation on a stated sample, not a general claim.

---

## 2. Review-comment generation

BLEU/ROUGE of generated vs. real human comments (averaged over the 5 PRs with
ground truth), per strategy:

| Strategy | BLEU | ROUGE-1 | ROUGE-L |
|---|---|---|---|
| `few_shot` | **1.15** | **0.165** | **0.107** |
| `cot` | 0.79 | 0.149 | 0.095 |
| `zero_shot` | 0.40 | 0.101 | 0.068 |
| `role_based` | 0.79 | 0.118 | 0.067 |

Absolute values are **very low** (best cell: ROUGE-L 0.126, BLEU ~1.0) —
expected, and exactly why the plan (§5.4) pairs them with a qualitative read: a
generated review and a human review can both be excellent yet share almost no
n-grams. The ordering is nonetheless consistent with intent: **`few_shot` tops
every text metric**, which is what it is designed to do (it conditions on real
human comments, so it mimics their style and length), with `cot` second.
`role_based` — the best *merge* strategy — is near the bottom here, a concrete
demonstration that **the best prompt is task-dependent**, not universal.

The qualitative side-by-side (`results/tables/exp3_qualitative.md`,
`role_based`+`diff_metadata`) is the honest evidence: the 8B model produces
plausible, on-topic review remarks (naming, missing tests, edge cases), but they
are more verbose and generic than the terse, specific human comments — exactly
what BLEU/ROUGE punishes. **The weakness of the metric is itself a finding**
(reflection Q3.9(4)): prompt design visibly changes comment *character* (a
maintainer persona is terser and more merge-focused than zero-shot) far more
than it moves the n-gram score.

---

## 3. Inference time (server-side, per call)

Measured from Groq's own `usage.total_time` — **not** wall-clock latency, which
on the free tier is dominated by client-side rate-limit waiting (tens of
seconds) and is not a valid inference time (this distinction is enforced in
`score_exp3._inference_ms`). Mean ms per strategy:

| Strategy | Merge prediction | Review generation |
|---|---|---|
| `zero_shot` | **60 ms** | 894 ms |
| `role_based` | 400 ms | 1088 ms |
| `few_shot` | 458 ms | 619 ms |
| `cot` | 566 ms | 978 ms |

The ordering is exactly what the reasoning strategies predict: **zero-shot merge
prediction is ~10× faster** than the others because it emits only a two-line
verdict, while CoT and role-based spend tokens (hence time) on reasoning or a
persona-shaped review. Note the sharpest cost-quality point: `cot` is the
**slowest** merge strategy (566 ms, ~9× zero-shot) while being **metrically
identical** to zero-shot (§1.2) — pure cost for no gain on this task. This is
the cost axis the lab guide's §3.7.5 asks for, and a concrete input to the
FastAPI "fast vs deep mode" design (plan §7.1): merge prediction can be
near-instant, review generation is inherently ~1 s+.

---

## 4. What Step 18 (the Lab 3 report) should say

- Lead with the **methodology and the comparisons**, not the absolute accuracy —
  and state the 14:2 sample skew up front (it is the honest framing the rubric
  rewards, and ties straight back to Exp1 Q5).
- The headline results: (a) two configs (`diff_commit_message`+`few_shot`,
  `diff_metadata`+`role_based`) reach macro-F1 0.816, **+0.35 over the
  always-MERGE baseline and exactly matching the best trained SVM/RF**;
  (b) the accuracy-vs-minority-recall trade-off is a *prompt-design* dial
  (few-shot ↔ recall); (c) the best prompt is **task-dependent** — `role_based`
  wins merge prediction, `few_shot` wins comment generation; (d) BLEU/ROUGE
  badly under-measures review quality, so the qualitative side-by-side carries
  that argument; (e) CoT costs ~9× zero-shot on merge prediction for identical
  predictions.
- State the limits plainly: 16 PRs with 2 negatives, and 2,500-character
  contexts (a free-tier cap, materially smaller than what Exp 2's ML pipeline
  saw). Both are named, not hidden, and neither invalidates the *relative*
  comparisons the reflection questions turn on.
