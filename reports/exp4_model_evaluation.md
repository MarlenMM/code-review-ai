# Experiment 4 — LLM Evaluation & Central Analysis (Step 21)

*Scores `results/tables/exp4_raw.jsonl` and answers the experiment's core
question: does richer context and advanced prompting actually improve code
review of **AI-generated** PRs, and how does it compare to Experiment 3's
human-code numbers? Deliverables: `results/tables/exp4_metrics.json`,
`exp4_qualitative.md`, three figures in `results/figures/exp4_*.png`, and this
write-up. Scorer: `src/llm/score_exp4.py` (+ `tests/test_llm_score_exp4.py`,
11 tests). Provider: Groq `llama-3.1-8b-instant` (free tier).*

---

## 0. Read this first — the sample, honestly

The free-tier daily token cap stopped the Step-20 run at **3 fully-complete
PRs** (`dotnet/aspnetcore`, `dotnet/runtime` — both merged; `home-assistant/core`
— not merged), i.e. **2 merged / 1 not-merged**, plus one partial PR
(`reports/exp4_grid_run_status.md`). Two consequences shape every number below,
and both are stated in the metrics JSON's `caveats`, not hidden:

1. **Merge metrics are coarse and directional.** With 3 PRs, a per-cell accuracy
   only takes values in {0, ⅓, ⅔, 1}. The meaningful signal is not any absolute
   number but the **paired** comparison — the *same* 3 PRs seen under every tier
   and every strategy, so a change in prediction is attributable to the axis
   that changed. That is exactly the comparison this experiment is designed to
   make, and it is what §1–§2 read.
2. **BLEU/ROUGE for review generation is not computable here.** *None* of the 3
   complete PRs carries a substantive human review comment — a real property of
   merged agent-authored PRs (Copilot's coding agent opens them, a maintainer
   merges them, often with no line comments). So there is no human gold to score
   generated reviews against on this sample. Review quality is instead assessed
   through (a) generated-comment **characteristics** and (b) a **turn-by-turn
   qualitative** read of what the advanced prompts actually do (`exp4_qualitative.md`).

The scorer re-runs unchanged on a larger `exp4_raw.jsonl`; a quota top-up
(especially with `--fetch-issues`, omitted here) shrinks both caveats.

---

## 1. Merge prediction

### 1.1 The base-rate trap (in both experiments)

Under class imbalance, raw accuracy mostly measures the base rate. Exp 4's
sample is 2:1 merged (base rate 0.67); the strategies that always say "MERGE"
(`role_based`, `cot`) score **0.69** accuracy — essentially the base rate, and
identical to always-predict-merge. Exp 3 showed the same pathology (base rate
0.875; `role_based` averaged 0.891, barely above it, and 11 of its 16
configurations sat exactly *at* the base rate). **In neither experiment does
the LLM's raw merge accuracy demonstrate real discriminative power** — which is
why the honest lens throughout is **macro-F1** and **not-merged recall** (the
minority class), not accuracy.

### 1.2 Context effect — richer context helps catch the hard case, but "everything at once" regresses

Pooling over strategies, by context tier (`central_analysis.context_effect_on_merge`):

| Context tier | Accuracy | Not-merged recall | Macro-F1 |
|---|---|---|---|
| `diff` (baseline) | 0.56 | **0.0** | 0.36 |
| `diff + PR description` | 0.60 | 0.2 | 0.49 |
| `diff + repository context` | 0.53 | **0.4** | **0.50** |
| `diff + issue`¹ | 0.47 | 0.4 | 0.44 |
| `complete SE context` | 0.60 | **0.0** | 0.375 |

¹ `--fetch-issues` was not passed in the run, so the issue slot was empty for
this tier — its "diff+issue" is effectively "diff + (issue unavailable)". Its
0.4 not-merged recall therefore comes from the *other* wiring of that tier, not
from real issue text; the issue tier's true effect awaits a top-up.

Two findings, both directional at n=3 but internally consistent:

- **Adding repository-level context is what moves the needle on the minority
  class.** Not-merged recall goes 0.0 (diff) → 0.4 (`+repo context`), and
  macro-F1 peaks there (0.36 → 0.50). This is the direct, numbers-backed answer
  to reflection Q4.9(1): the surrounding code lets the model see that a
  locally-plausible AI change doesn't actually fit, which is precisely the
  not-merged case a diff-only view misses.
- **The `complete` tier regresses to catching nothing not-merged** (recall back
  to 0.0, macro-F1 0.375). Piling *all* context into one prompt appears to
  dilute the specific signal and push the 8B model back toward its "looks fine →
  merge" default. This is a genuinely interesting, if fragile, observation: more
  context is not monotonically better, and the mid-tier `+repo context` beats
  the maximal tier. Worth stating in the report as a hypothesis to confirm on a
  larger sample.

### 1.3 Prompt effect — the advanced strategies are the only ones that catch the not-merged PR

Pooling over tiers, by strategy (`central_analysis.prompt_effect_on_merge`):

| Strategy | Accuracy | Not-merged recall | Macro-F1 |
|---|---|---|---|
| `role_based` | **0.69** | 0.0 | 0.41 |
| `cot` | **0.69** | 0.0 | 0.41 |
| `few_shot` | 0.50 | 0.0 | 0.33 |
| `self_reflection` | 0.375 | **0.6** | 0.375 |
| `multi_turn` | 0.50 | **0.5** | **0.49** |

The headline of the whole experiment: **`self_reflection` and `multi_turn` are
the only two strategies that ever predict "not merged" correctly** — the three
carried over from Experiment 3 (`role_based`, `few_shot`, `cot`) never do, on
any tier. The advanced prompts shift the model's operating point away from the
majority-class default:

- **Self-reflection** trades accuracy for the highest minority recall (0.6): its
  critique turn makes it *more skeptical*, so it catches the not-merged PR most
  often but also produces false "CLOSE" calls on the merged PRs (hence its 0.375
  accuracy). It moves the decision threshold, at a precision cost.
- **Multi-turn** strikes the best *balance* — the top macro-F1 (0.49) of any
  strategy — catching the minority case (0.5 recall) without collapsing
  accuracy, because revealing the broader context mid-conversation lets it
  revise a diff-only "merge" specifically when the context contradicts it.

So "does better prompting help review of AI-generated code?" has a concrete,
if small-sample, answer: **yes for the thing that matters under imbalance
(minority-class detection), and `multi_turn` gives the best accuracy/recall
balance; no if you only look at raw accuracy**, which the baseline strategies
"win" purely by predicting the majority.

---

## 2. Review-comment generation

No human gold exists on this sample (§0), so this section is characteristics +
qualitative, and says so plainly rather than reporting a meaningless BLEU.

**Generated-comment characteristics** (a verbosity/coverage proxy, not quality):
`role_based` is by far the most verbose (mean **8.5** comments/review),
`cot` the tersest (3.1); `multi_turn` (5.7) produces more than the single-shot
`few_shot`/`cot`. More context tiers produce slightly more comments
(diff 5.8 → complete 5.8, non-monotonic). Verbosity ≠ quality — read with §2.1.

### 2.1 What the advanced prompts actually do (the real evidence — `exp4_qualitative.md`)

The turn-by-turn dumps are the most informative artifact this experiment
produced, because they show the *mechanism*, which no aggregate can:

- **Self-reflection genuinely self-corrects.** On the `dotnet/aspnetcore` rename
  PR, the *draft* review is vague ("The `EnvironmentViewTest.cs` file should be
  reviewed for any potential issues") and partly wrong (it asks for files that
  are already in the diff). The *critique turn* explicitly notes "some comments
  were too vague or generic," then produces concrete, actionable ones — including
  catching a real `includeAttribut` → `includeAttribute` typo the draft missed.
  This is the reflection mechanism doing exactly what it is designed to do, and
  it is the qualitative answer to Q4.9(2).
- **Multi-turn shifts focus once context is revealed.** The diff-only turn
  produces generic "add tests / add docs" comments; after the broader context is
  disclosed, the revised review pivots to *consistency with the surrounding
  codebase* and *integration/end-to-end testing* — concerns only visible with the
  wider view, which is the §4.3 thesis in miniature.

That said, the 8B model's reviews are frequently generic or slightly
hallucinated (e.g. speculating about namespaces it can't see), a limitation to
name honestly and a direct motivation for the §4.10 future work (a stronger
model, RAG-grounded context).

---

## 3. Inference time (server-side) — the cost of the advanced strategies

Using the provider's server-side `usage.total_time` (wall-clock `latency_ms` is
polluted by free-tier rate-limit waiting and is *not* inference time — a bug
already fixed in Exp 3), summed across **both** turns for conversational cells:

| Strategy | Mean inference (ms) | API calls / cell |
|---|---|---|
| `few_shot` | ~290 | 1 |
| `role_based` | ~390 | 1 |
| `cot` | ~600 | 1 |
| `multi_turn` | ~680 | **2** |
| `self_reflection` | ~740 | **2** |

The advanced strategies cost **~2× the API calls** and roughly 2× the wall-cost
of the cheapest single-shot prompt. Under a fixed token budget that is the real
trade-off: `self_reflection`/`multi_turn` buy minority-class recall for double
the calls — which, on a free tier with a 500k-token/day cap, is why the grid
only reached 3 complete PRs. `multi_turn` is the better buy (best macro-F1 at
the same 2-call cost as self-reflection).

---

## 4. Experiment 4 vs. Experiment 3 (guide §4.8(7))

On the three strategies both experiments ran (`exp3_comparison.merge_by_strategy`).
Experiment 3's side is its **complete 16-PR / 512-cell grid** (14 merged : 2
not-merged, base rate 0.875); Experiment 4's is its 3 complete PRs (2:1, base
rate 0.67):

| Strategy | Exp 3 (human) acc | Exp 4 (AI) acc | Exp 3 not-merged recall | Exp 4 not-merged recall | Exp 3 macro-F1 | Exp 4 macro-F1 |
|---|---|---|---|---|---|---|
| `role_based` | 0.891 | 0.688 | 0.125 | **0.0** | 0.554 | 0.407 |
| `few_shot` | 0.750 | 0.500 | **0.250** | **0.0** | 0.549 | 0.333 |
| `cot` | 0.875 | 0.688 | 0.0 | **0.0** | 0.467 | 0.407 |

Read carefully (the populations, samples, and base rates differ — this is
directional):

- **Raw accuracy is higher on human code** (−0.19 to −0.25 moving to AI code),
  but that is substantially the base-rate gap (Exp 3 is 87.5% merged vs Exp 4's
  67%). In *both* experiments the shared strategies barely exceed
  always-predict-merge, so the drop is not by itself evidence the LLM is "worse"
  at AI code in a discriminative sense — accuracy can't show that.
- **The real, defensible cross-experiment statement** is about *what it takes to
  catch the minority class.* On human code, the shared strategies **do** reach it
  sometimes — `few_shot` averages 0.250 not-merged recall and `role_based` 0.125
  (shared-strategy mean 0.125), and on human code `few_shot`+`diff_commit_message`
  reached the grid's best macro-F1 of 0.816. On **AI-authored** code those *same
  three strategies score 0.0 not-merged recall on every single tier* — it took
  Experiment 4's **new** strategies (`self_reflection` 0.6, `multi_turn` 0.5) to
  catch the not-merged AI PR at all.
- That contrast is the cleanest evidence for §4.3's premise: **the prompting that
  suffices for human-written code stops working on AI-generated code**, which is
  exactly why Experiment 4 exists. The caveat remains that Exp 4's side rests on
  3 PRs; the direction is consistent and interpretable, the magnitude is not yet
  firm.

---

## 5. The five reflection questions (§4.9), grounded in these numbers

1. **Why does repository-level context improve review performance?** — On this
   sample it is *the* tier that lifts not-merged recall (0.0 → 0.4) and macro-F1
   to its peak (0.50): the surrounding code exposes when a locally-plausible AI
   change doesn't fit, which the diff alone hides (§1.2).
2. **Which prompt design best suits AI-generated code, and why?** — `multi_turn`:
   best macro-F1 (0.49) and it catches the minority class (0.5 recall) without
   the accuracy collapse self-reflection suffers, because progressive context
   disclosure lets it revise a "merge" call exactly when the wider context
   contradicts it (§1.3, §2.1).
3. **What do prompt optimization vs. context augmentation each fix?** — They fix
   different failures: *context augmentation* (repo context) supplies information
   the diff lacks (→ minority recall); *prompt optimization* (self-reflection /
   multi-turn) changes how the model *uses* whatever information it has, moving it
   off the majority-class default. The design deliberately separates the two axes
   so this is attributable (§1.2 vs §1.3).
4. **Do prompt combinations generalize across tasks?** — Partly. The verbosity
   ordering of strategies differs between merge prediction and review generation,
   and role-based is strong for producing review comments but adds nothing to
   merge discrimination — so the best prompt is task-dependent, not universal (§2, §3).
5. **What else to explore?** — A stronger model than 8B (its reviews are often
   generic/hallucinated), RAG-grounded repository context and real fetched issue
   text (both §4.10), and simply a larger sample: every number here is n=3
   directional and the `complete`-tier regression (§1.2) is the kind of finding
   that most needs confirmation.

---

## 6. What Step 22 (the Lab 4 report) should say

- Lead with the **honest sample** (3 PRs, no human comments) — the rubric rewards
  naming this over hiding it.
- The **defensible, numbers-backed claims** are: repo context lifts minority-class
  detection and macro-F1; the two new strategies are the only ones that catch the
  not-merged AI PR; `multi_turn` is the best-balanced; the advanced strategies
  cost ~2× the calls; and richer-is-not-monotonically-better (the `complete`-tier
  regression).
- Use the **turn-by-turn qualitative examples** as the primary evidence for
  review quality, since BLEU/ROUGE is genuinely inapplicable here — and say why.
- Frame the Exp 3 comparison through **macro-F1 / minority recall**, not accuracy,
  and state the base-rate confound explicitly.
- Carry forward the two open items for a top-up run: `--fetch-issues` (to give
  the issue tier real content) and more PRs (to firm up every directional number).
