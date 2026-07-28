# Experiment 3 — Prompt-Template Library Design

*LLM-based code review of human-written code. Design deliverable for Step 14;
feeds Steps 15–17 (context builder, inference grid, scoring).*
*Everything about population sizes and pools below is computed from the real
1,494-PR mined dataset in `data/processed/*.parquet`, not estimated. No API
calls happen in Step 14 — this is prompt design only.*

Deliverable: `src/llm/prompts/` — `schema.py`, `templates.py`, `few_shot.py`,
`parsing.py`, `__init__.py`. Tests: `tests/test_llm_prompts.py`,
`tests/test_llm_few_shot.py` (52 tests).

---

## 1. What the lab guide actually asks for (§3, verified against the PDF)

Read directly from `实验指导书_代码审查_英文.pdf` §3, not the plan's paraphrase:

- **Two tasks** (§3.4.4): **Merge Prediction** and **Review Comment Generation**.
- **Four prompt strategies** (§3.4.3): **zero-shot**, **few-shot**,
  **Chain-of-Thought**, **role-based**.
- **Four code contexts** (§3.7.2): diff-only, diff + PR description,
  diff + commit message, diff + other information (labels/file list/metadata).
  (§3.4.2 additionally lists content-of-modified-files, historical snippets, and
  historical review comments — those richer contexts are Experiment 4's territory,
  §4.4.2.)
- **Required grid** (§3.7.3): "*For each context type, design* zero-shot / few-shot /
  CoT / role-based" — i.e. **4 contexts × 4 prompts**, run for **both tasks**.
- **Required results** (§3.8): examples of different context constructions;
  examples of different prompt designs; merge-prediction results; examples of
  generated review comments; performance comparison across prompts; across
  contexts; and a comparison against the model from the previous experiment.

This module owns exactly one axis of that grid — **the four prompt strategies,
for the two tasks** — and is written to compose cleanly with the other two axes
(context, and eventually the LLM provider), so the §3.7 grid is a nested loop in
Step 16, not 32 hand-copied scripts.

### 1.1 The one non-obvious wording note

The guide's own §3.7.1 says "Load the test data from Experiment 3" and §3.8(7)
says "comparison against the deep learning model from Experiment 3" — both are
self-referential typos in the source PDF. In this project's build they mean
**Experiment 2**: the LLM is scored on the *same held-out test PRs* as the SVM/RF
models, and the comparison in reflection Q5 is LLM-vs-ML. Section 6 below pins
this down concretely.

---

## 2. Design axes and how they compose

A prompt in this library is a pure function of three orthogonal inputs:

```
build_prompt(task, strategy, context, few_shot_examples) -> RenderedPrompt
             └─ §3.4.4   └─ §3.4.3  └─ Step 15   └─ from the Exp-2 train split
```

- **`task`** — `Task.MERGE_PREDICTION` | `Task.REVIEW_COMMENT`.
- **`strategy`** — `Strategy.{ZERO_SHOT, FEW_SHOT, COT, ROLE_BASED}`.
- **`context`** — a `PromptContext` carrying an already-rendered `context_text`
  string (plus optional `repo`/`title` header fields). **This library is
  deliberately blind to how rich that string is** — whether it is diff-only or
  diff+description is Step 15's `context_builder`'s decision. That blindness is
  what lets the same four templates serve all four contexts.
- **`few_shot_examples`** — required iff `strategy is FEW_SHOT`, and drawn only
  from the Experiment-2 train split (Section 5).

The output is a `RenderedPrompt` with separate `system` / `user` message parts
(so a provider can map them onto Gemini's `system_instruction` + content, or two
chat messages), a `.messages()` helper, and a `.label` (e.g.
`merge_prediction:cot`) for cache keys and result-table columns.

---

## 3. The four strategies — the intellectual core

The grading rubric and reflection questions §3.9(2) and §3.9(4) ask *why*
few-shot beats zero-shot and *why* prompt design changes comment quality. Those
questions only have real answers if the four prompts differ **in reasoning
mechanism**, not in wording. Each strategy is built around a different mechanism:

| Strategy | Mechanism it relies on | What is varied | What is held constant |
|---|---|---|---|
| **Zero-shot** | the model's **raw prior** — no scaffold | nothing added | — (this is the baseline) |
| **Few-shot** | **in-context learning by analogy** + calibration to *this* dataset | K labelled demonstrations prepended | task framing, output contract |
| **Chain-of-Thought** | **serial decomposition** of the judgment into ordered sub-steps | an explicit reasoning procedure in the user message + a "reason first" system nudge | task framing, output contract |
| **Role-based** | a **shifted evaluative frame** — a senior maintainer's standards & threshold | a persona + value system in the *system* message | task framing, output contract |

Two invariants make the comparison scientifically clean:

1. **The output contract is identical across all four strategies of a task**
   (Section 4). Only the reasoning scaffold changes; the thing Step 17 parses and
   scores does not. So a metric difference between cells is attributable to the
   strategy, not to differing parse success.
2. **Each mechanism is localised.** Role-based puts its persona in `system` and
   changes *standards*; CoT puts its procedure in `user` and changes *process*;
   few-shot changes *evidence*; zero-shot changes *nothing*. They are additive and
   independent, so Experiment 4's advanced prompts (self-reflection, multi-turn)
   can layer on top without colliding.

### 3.1 Merge prediction — how each strategy differs

- **Zero-shot** — states the task, shows the PR, asks for the decision directly.
  Probes what the model believes about merge-worthiness with no help. Baseline.
- **Few-shot** — prepends K **balanced** (context → `DECISION: MERGE/CLOSE`)
  demonstrations from the train split, then asks the model to *infer the pattern
  that separates merged from closed PRs*. Mechanism: analogy to demonstrated
  cases, plus calibration to this dataset's boundary (the model's untutored prior
  about "what gets merged on GitHub" may not match these five repos).
- **Chain-of-Thought** — forces an explicit four-step analysis before the answer:
  (1) scope & risk, (2) completeness & correctness, (3) alignment with stated
  intent, (4) maturity signals — then a *calibrated* conclusion with an explicit
  base-rate nudge ("a mature repo merges most PRs, so treat CLOSE as the claim
  needing stronger evidence"). Mechanism: allocate serial reasoning and surface
  intermediate judgments a single-shot answer skips; the base-rate nudge guards
  against CoT's tendency to over-hunt for reasons to reject.
- **Role-based** — a system-message persona: *a senior maintainer of {repo} with
  commit access* who gates on correctness, testing, scope discipline, and
  conventions, and is wary of large/risky changes. Mechanism: shifts the decision
  threshold toward a real gatekeeper's, rather than a generically helpful
  assistant's.

### 3.2 Review comment generation — how each strategy differs

- **Zero-shot** — "write the review comments a reviewer would leave." Bare.
- **Few-shot** — prepends K real **human** review comments from train-split PRs,
  each with the change it commented on, and asks the model to *match their style*.
  Mechanism: style/altitude transfer — real project comments are terse, specific,
  and line-anchored, unlike an LLM's default verbose prose. This directly targets
  the BLEU/ROUGE metric (comment *style* is most of what those n-gram metrics
  actually measure) and is the concrete basis for reflection Q3.9(2).
- **Chain-of-Thought** — three explicit stages: **analyze** (enumerate every
  concrete observation), **prioritize** (keep only merge-relevant issues, drop
  nitpicks), **write** (one actionable comment per kept issue). Mechanism:
  separate discovery from articulation, so comments are grounded in a scan rather
  than free-associated.
- **Role-based** — a maintainer persona with an explicit **review rubric**
  (correctness > security > tests > maintainability > style), told to focus on
  merge-blocking issues and stay concise. Mechanism: a prioritization frame,
  producing gatekeeper-style comments rather than an exhaustive linter dump. That
  prompts differ *this much* in what they optimise for is the answer to Q3.9(4).

---

## 4. Output contracts and parsing (`parsing.py`)

A prompt that asks for a format is only useful if something parses that format
back. The contract and its parser are two halves of one decision, so they live
together in the library; the shared tests fail if they drift.

- **Merge prediction** ends with:
  ```
  DECISION: MERGE   (or CLOSE)
  CONFIDENCE: <0.0–1.0 probability it was merged>
  ```
  `parse_merge_prediction` prefers the **last** explicit `DECISION:` line (a CoT
  answer mentions "merge" many times while reasoning; only the final labelled line
  is the commitment), tolerates markdown (`**MERGE**`) and aliases
  (merged/accepted/yes ↔ closed/rejected/no). If there is **no** `DECISION:` line
  it scans only the *final non-empty line* for **strong** decision words
  (merge/close/accept/reject…), never the weak yes/no tokens — so ordinary prose
  containing "no" is not misread as CLOSE. A genuinely format-less answer returns
  `label=None` (**abstention**), because coercing it to a class would corrupt the
  accuracy/precision/recall numbers Step 17 reports; how to treat abstentions
  (wrong, or a separate coverage figure) is Step 17's policy, not the parser's.
- **Review comments** are wrapped in `<review>…</review>` with one `- ` bullet
  per comment. `parse_review_comments` isolates that block so BLEU/ROUGE compare
  the **comment body** to the human comments, not a CoT preamble or a role-based
  persona's throat-clearing. Missing sentinels → fall back to the whole response
  (nothing silently dropped).

Holding these contracts identical across strategies is what makes §3.8's
"performance comparison across different prompt designs" a fair comparison.

---

## 5. Few-shot example pools (`few_shot.py`) — the leakage guarantee

This is the part most likely to be quietly wrong, so it is the part most heavily
tested.

### 5.1 The rule: demonstrations come only from the Experiment-2 *train* split

Experiment 3's merge prediction is scored on the **same held-out test set as
Experiment 2** (Section 6). A few-shot prompt that demonstrated a test PR — or,
worse, revealed its merge outcome — would be handing the model the answer it is
about to be graded on. So `load_train_pool()` returns exactly
`split_by_repo_time(load_variant("v1"))[0]` — the **train** slice — and every
demonstration is selected from it. Because that split is deterministic and the
train/test slices are provably disjoint (`test_ml_common.py` already proves the
disjointness; `test_llm_few_shot.py` proves the examples stay inside train), the
guarantee holds end-to-end.

The same logic covers review-comment generation: BLEU/ROUGE compare against the
real human comments on the *test* PRs, so the human comments used as
demonstrations are pulled from *train* PRs only.

**Real pool sizes** (from the actual split):

| Pool | Size | Notes |
|---|---|---|
| Exp-2 train split (human-written, closed) | **890 PRs** | merge rate 72.2 % |
| Exp-2 test split (the scored set) | **224 PRs** | merge rate 82.1 %; **never** used as examples |
| Train PRs with a substantive human review comment | **275 PRs** | the comment-demonstration pool (after the §5.3 human/length filter) |

### 5.2 Merge examples: balanced by default (a real trade-off, hence a flag)

`select_merge_examples(..., balanced=True)` draws an equal-as-possible number of
MERGE and CLOSE demonstrations and interleaves them. Rationale: a few-shot set
exists to *demonstrate the decision boundary*; an all-MERGE set (which random
sampling of a 72 %-merged pool tends toward) teaches the model only the majority
prior it already holds. Showing both classes is what makes the demonstrations
informative. The counter-argument — that the true base rate is itself signal — is
real, so `balanced=False` (raw base rate) is a supported option rather than a
hidden choice. Selection is **seeded/deterministic** so every downstream LLM call
caches under a stable key and the grid is reproducible.

### 5.3 Comment examples: human, substantive, representative

The comment-demonstration pool excludes non-human authors by reusing
Experiment 1's own allowlists (`AI_CODING_AGENT_LOGINS ∪ AI_REVIEW_BOT_LOGINS`,
imported from `src/mining/ai_detection.py` so the two never drift) plus non-AI
automation (`github-actions`, `codecov`, …) and any `*[bot]` login, and requires
≥40 characters (so the gold style is a real maintainer's substantive comment, not
a bot's or a bare "LGTM"). For each qualifying PR the single longest qualifying
comment (capped) is used as the representative example. This matters concretely:
in this dataset `copilot-pull-request-reviewer` alone authored **2,962** of the
review comments — without this filter the "human style" demonstrations would be
dominated by an AI reviewer's voice.

### 5.4 Context rendering of examples (decoupling from Step 15)

A demonstration must be rendered in the *same context kind* as the query, or the
analogy is a format mismatch. So `select_*` takes a `render_context` callable; in
the Step 16 grid that callable is Step 15's context builder at the current context
kind. Until Step 15 lands (and in tests), a minimal built-in
`default_render_context` (title + truncated diff) keeps the module self-contained
and runnable. This is the one seam where Step 14 anticipates Step 15 without
depending on it.

---

## 6. Test-set alignment with Experiment 2 (plan §5.4 / guide §3.7.1)

The apples-to-apples ML-vs-LLM comparison requires both experiments to be scored
on identical test PRs. This is achieved structurally, not by convention:

- Experiment 2 evaluates on `split_by_repo_time(load_variant("v1"))[1]` (test).
- Experiment 3's few-shot pool is `split_by_repo_time(load_variant("v1"))[0]`
  (train) — the exact complement.
- Step 16 will run the LLM on that same test slice, giving 224 human-written
  test PRs scored by both the SVM/RF (Step 12) and the LLM (Step 17) — the basis
  for §3.8(7) and reflection Q3.9(5).

Nothing in this library re-derives or re-splits the data; it imports
`split_by_repo_time` and `load_variant` from `src.ml.common`, so the split can
never silently diverge from Experiment 2's.

---

## 7. Rendered examples (§3.8 required result: "examples of different prompt designs")

Verbatim output of `build_prompt` for merge prediction on a small illustrative
diff (same PR, four strategies — note what changes and what stays fixed). Review
-comment prompts follow the same four-way structure with the `<review>` contract.

**Zero-shot** (system: terse assistant):
```
Task: MERGE PREDICTION. Decide whether the pull request below was ultimately
MERGED into the microsoft/vscode project or CLOSED without merging ...
PR title: Return None for missing cache keys instead of raising
Pull request content:
--- src/util/cache.py
@@ ... @@  (the diff)
Give your decision directly.
DECISION: MERGE (or) DECISION: CLOSE / CONFIDENCE: <0.0–1.0>
```

**Chain-of-Thought** (system adds "Reason step by step"): same task+diff, then —
```
Reason through these steps explicitly before you decide:
1. Scope & risk ...   2. Completeness & correctness ...
3. Alignment with intent ...   4. Maturity signals ...
Then weigh these factors and give a calibrated decision. A typical mature
repository merges most PRs, so treat CLOSE as the claim that needs stronger evidence.
DECISION / CONFIDENCE (same contract)
```

**Role-based** (persona in the *system* message):
```
SYSTEM: You are a senior maintainer of the microsoft/vscode project with commit
access -- the person who decides whether a pull request is merged ... you gate
merges on correctness, adequate testing, disciplined scope, and conventions ...
USER: (same task+diff) Apply your own merge criteria as the maintainer ...
DECISION / CONFIDENCE (same contract)
```

**Few-shot** (K balanced labelled demonstrations from the *train* split prepended):
```
Here are 2 labelled examples of past pull requests and whether each was merged
or closed. Infer the pattern ..., then classify the final one.
### Example 1  ... DECISION: MERGE
### Example 2  ... DECISION: CLOSE
### Pull request to classify  ... (the query diff)
DECISION / CONFIDENCE (same contract)
```

The full renders are reproducible from the library; the point visible even in the
excerpts is that the **task and the output contract are byte-identical** across
strategies, and only the reasoning scaffold moves.

---

## 8. How this pre-loads Experiment 3's reflection questions (§3.9)

The design is deliberately built so the five official questions have
evidence-backed answers once Steps 16–17 produce numbers:

1. **Why does code context affect LLM inference?** — the context axis is a first
   -class, isolated variable (Step 15); §3.8(6) compares the four contexts with
   everything else fixed.
2. **Advantages of few-shot over zero-shot?** — few-shot is defined here as
   *analogy + calibration + style transfer* (Sections 3.1–3.2), not "zero-shot
   plus examples"; the metric gap between the two cells measures exactly that.
3. **Which context contributes most to merge prediction?** — answerable because
   contexts are compared under a fixed strategy on the aligned test set.
4. **Why do prompt designs yield different comment quality?** — because the four
   strategies optimise for different things (raw prior vs. analogy vs.
   analyze-then-write vs. a maintainer's rubric); Section 3.2 is the mechanism,
   the BLEU/ROUGE + qualitative side-by-side (Step 17) is the evidence.
5. **LLM vs. the ML model — advantages/disadvantages?** — enabled by the
   identical test set (Section 6); the LLM needs no training and generates
   comments (which SVM/RF cannot), but has inference cost/latency and no
   calibrated probability unless prompted for one (hence `CONFIDENCE:`).

---

## 9. Module map and handoff

| File | Responsibility |
|---|---|
| `src/llm/prompts/schema.py` | `Task`, `Strategy` enums; `PromptContext`, `FewShotExample`, `RenderedPrompt` — the shared vocabulary, no logic |
| `src/llm/prompts/templates.py` | the 8 (task × strategy) builders + `build_prompt` dispatch; the output contracts |
| `src/llm/prompts/few_shot.py` | train-split-only example pools; leakage guarantee; balance & human-comment curation; placeholder context renderer |
| `src/llm/prompts/parsing.py` | `parse_merge_prediction`, `parse_review_comments` — the duals of the contracts |
| `src/llm/prompts/__init__.py` | public API re-exports |

**Handoff to Steps 15–17:**
- **Step 15** supplies the real `context_builder` (four context kinds) and the
  Gemini provider. It should pass its builder as `render_context` to `select_*`
  and feed `RenderedPrompt.messages()` to the provider.
- **Step 16** loops `contexts × strategies × tasks` over the 224 aligned test
  PRs, caching each raw response by `RenderedPrompt.label` + context kind + PR id.
- **Step 17** parses with this module's parsers and scores: accuracy/precision/
  recall/F1 (merge, vs. Exp-2), BLEU/ROUGE + a qualitative side-by-side
  (comments), and inference time.

Not done here, by design: context construction (Step 15), any API call (Step 16),
scoring/metrics (Step 17), and Experiment 4's advanced prompts — self-reflection
and multi-turn (guide §4.4.3), which will extend this same library in Step 19.
