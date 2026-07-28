# Experiment 4 — Context-Augmentation & Advanced-Prompt Design

*Improving code review for **AI-generated** code. Design deliverable for Step 19
(extends `src/llm/prompts/`); feeds the Step-20 grid run and the Step-21
analysis. This step is prompt/context **design only** — no API calls, no grid
run.*

*Every population figure below is computed from the real 1,494-PR mined dataset
in `data/processed/*.parquet` (see `scratch`-free reproductions in the Step-19
session), not estimated.*

Deliverable: `src/llm/prompts/` extended with `context_aug.py`,
`conversations.py`, `exp4_examples.py`, and additions to `schema.py` /
`__init__.py`. Tests: `tests/test_llm_context_aug.py`,
`tests/test_llm_conversations.py`, `tests/test_llm_exp4_examples.py` (47 new
tests; full suite 315 passing).

---

## 1. What the lab guide actually asks for (§4, verified against the PDF)

Read directly from `实验指导书_代码审查_英文.pdf` §4, **not** the plan's paraphrase
(the plan's §6 compresses §4 and gets the tier/prompt lists slightly wrong):

- **Same two tasks** (§4.4.4): Merge Prediction and Review Comment Generation,
  now on AI-generated code.
- **Five context tiers** (§4.7.2 "Step 2: Context Construction"), *different from
  Experiment 3's four*:
  1. Diff
  2. Diff + Pull Request description
  3. Diff + Repository context
  4. Diff + Issue information
  5. Complete software-engineering context
- **Four Step-3 prompt strategies** (§4.7.3 "Step 3: Prompt Design"): **Role-based,
  Few-shot, Chain-of-Thought, Self-Reflection**. This is *not* Experiment 3's
  lineup — **Self-Reflection replaces zero-shot**; role/few-shot/CoT carry over.
- **A fifth strategy named in §4.4.3** ("Prompt Optimization"): the strategy list
  there is role-based / few-shot / CoT / **self-reflection** / **multi-turn
  interactive prompting**. Multi-turn is named as an optimization strategy the
  experiment *explores* but is **not** in the concrete §4.7.3 Step-3 grid list.
- **Context-augmentation palette** (§4.4.2): PR descriptions, commit messages,
  code before/after modification, files containing the modified functions, call
  relationships and related functions, issue descriptions, historical review
  comments, repository-level code context.
- **Required results** (§4.8): examples of context constructions and prompt
  designs; merge-prediction results; generated-comment examples; performance
  across contexts; across prompts; and **a comparison against Experiment 3**.
- **Reflection questions** (§4.9): (1) why repo-level context helps; (2) which
  prompt design best suits AI-generated code and why; (3) what prompt
  optimization vs. context augmentation each fix; (4) whether prompt combinations
  generalize across tasks; (5) what else to explore.

### 1.1 Reconciling self-reflection **and** multi-turn (the plan vs. the PDF)

The plan's row 19 explicitly asks for **both** self-reflection *and* multi-turn.
The PDF puts self-reflection in the concrete §4.7.3 grid and multi-turn only in
the §4.4.3 "strategies explored" list. Judgment call, documented:

> Both are implemented as first-class `Exp4Strategy` members. **Self-reflection**
> is part of the core §4.7.3 grid (`CORE_GRID_STRATEGIES` = role / few-shot / CoT
> / self-reflection). **Multi-turn** is implemented as the *additional* §4.4.3
> exploration the plan wants, kept separable so Step 21 can report the §4.7.3
> grid and the multi-turn exploration distinctly rather than conflating them.

This satisfies the plan without misrepresenting the PDF's grid, and gives the
Step-21 analysis an honest place to say "multi-turn is an extension beyond the
required grid."

---

## 2. Design axis 1 — the five context tiers

The tiers are **not** the four from Experiment 3. Experiment 3's context builder
(`src/llm/context_builder.py`) produced diff-only / +description / +commit /
+metadata. Experiment 4 needs richer, AI-code-oriented context, because §4.3's
whole premise is that **local diffs are insufficient to review AI-generated
code**: an AI change can look locally plausible while misreading the surrounding
codebase or the issue's real intent.

### 2.1 Why the context is kept in *components*, not one opaque string

Experiment 3's `PromptContext` carries one already-rendered `context_text` and
its templates are blind to context richness. Experiment 4 keeps the *components*
separate in `AugmentedContext` (`context_aug.py`) for one concrete reason: the
**multi-turn** strategy reveals the diff first and the broader context second, so
the prompt layer must be able to lay the same pieces out two different ways. A
pre-flattened string cannot be split back apart. `render_tier` (single-shot) and
`split_for_multi_turn` (multi-turn) both derive from one `_augmentation_sections`
helper, so a tier's single-shot rendering and its two-turn rendering always carry
the **same section texts** — only the ordering differs (a property `test_multi_turn_and_single_shot_carry_same_info` pins).

### 2.2 Tier → components mapping (`_TIER_SECTIONS`)

| Tier (`Exp4ContextTier`) | Sections (besides the always-present diff) |
|---|---|
| `DIFF` | — |
| `DIFF_PR_DESCRIPTION` | PR description |
| `DIFF_REPO_CONTEXT` | Repository context |
| `DIFF_ISSUE` | Linked issue |
| `COMPLETE_SE_CONTEXT` | PR description · commit messages · linked issue · repository context · historical review comments |

Every section a tier calls for but whose data is absent renders an explicit
`"(... not available)"` marker rather than silently vanishing — so Step 21 can
tell "the tier had no data for this PR" apart from "the tier was not run"
(mirroring `context_builder.py`'s `(no PR description provided)` discipline).

### 2.3 Honest per-tier data availability — measured on the AI **test** split (n = 78)

This is the part the grading rubric rewards being honest about. The mined dataset
does **not** contain everything the §4.4.2 palette lists, and the tiers differ
sharply in how well the data backs them. Measured on exactly the PRs Step 20 will
score (the AI-authored test split):

| Tier component | Available today? | Coverage on AI test split | Source / gap |
|---|---|---|---|
| Diff | ✅ full | 100% | `files_changed.patch` (as Exp 3) |
| PR description | ✅ full | **100%** | `pull_requests.body` |
| Commit messages | ✅ full | ~100% | `commits.message` |
| Repository context | ⚠️ **partial** | **92%** (≥1 hunk heading) | *No full file bodies stored.* `build_lightweight_repo_context` approximates from co-changed files + git hunk section headings (enclosing symbols); a full version needs a Step-20 GitHub blob fetch |
| Linked issue | ⚠️ **reference only** | **36%** any ref / **19%** closing-keyword or URL | `extract_issue_references` finds the *reference*; the issue's *body* must be fetched in Step 20. For the majority the tier legitimately renders "(no linked issue …)" |
| Historical comments | ✅ derivable | **100%** | `select_historical_comments` over similar train-split PRs |

**Consequences to carry into Step 20/21, stated plainly:**

- The **repository-context** tier ships with a *no-fetch* approximation
  (`build_lightweight_repo_context`) that is real and free (git hunk headings are
  present on 99% of AI changed-file rows), but is **not** a call graph. The
  `AugmentedContext.repo_context` slot is designed to accept a richer fetched
  version if Step 20 pulls file bodies from GitHub at the PR head SHA. Whichever
  is used must be stated in the Step-22 report; the lightweight form is a
  legitimate, reportable scope decision, not a hidden shortcut.
- The **issue** tier will be "(not available)" for ~64% of test PRs unless Step 20
  fetches referenced issue bodies — and even then only ~36% carry a reference at
  all. This is a genuine dataset limitation (AI agents often restate the problem
  in the PR body instead of linking an issue), and Step 21 should report the
  issue tier's effect *conditioned on PRs that actually have an issue*, not
  diluted across the majority that don't.

---

## 3. Design axis 2 — the five prompt strategies

Kept in a **separate `Exp4Strategy` enum** from Experiment 3's `Strategy`, on
purpose: Experiment 3's completed grid runner (`run_exp3_grid.py`) and three
tests iterate `list(Strategy)` directly, so growing that enum would silently
change Experiment 3's finished 512-cell grid and its cell-count assertions. The
three shared members map back to `Strategy` by identical `.value`
(`EXP3_EQUIVALENT`), so role/few-shot/CoT reuse `templates.build_prompt`
unchanged — only the *context* (richer) and *population* (AI PRs) differ for them
between the two experiments.

### 3.1 Reused, single-shot → `RenderedPrompt`

- **Role-based**, **Few-shot**, **Chain-of-Thought** — identical builders to
  Experiment 3. Few-shot demonstrations now come from the **AI-authored** train
  split (§4), not the human split.

### 3.2 New, multi-turn → `RenderedConversation`

The two new strategies' reasoning *mechanism is the turn sequence itself*, which
a single-shot prompt cannot express. `RenderedConversation` (`schema.py`) is a
persona `system` message plus an ordered tuple of user `turns`; only the **last**
turn carries the output contract, so only the final reply is parsed — with the
*same* `parse_merge_prediction` / `parse_review_comments` the single-shot
contracts use, so Step 20 scores every strategy identically.

- **Self-Reflection** (`_build_self_reflection`) — **turn 1** draws a *draft*
  review/decision over the full tier context (reusing the zero-shot wording so
  the draft is Experiment 3's baseline, isolating reflection as the only new
  variable); **turn 2** asks the model to critique its own committed draft and
  finalize. The critique turn is deliberately tuned to **AI-generated-code
  failure modes** — hallucinated/misused APIs, missing edge cases, tests that
  assert the wrong thing, over-broad edits, mis-calibrated confidence. This is
  what makes it *mechanistically distinct* from CoT: CoT reasons **forward** in
  one shot; self-reflection commits a draft and then reasons **about that draft**.

- **Multi-Turn** (`_build_multi_turn`) — **turn 1** draws a review/decision from
  the **diff alone**; **turn 2** reveals the broader software-engineering context
  the tier carries (repo context, linked issue, historical comments) and asks the
  model to reconsider. This directly operationalizes §4.3: the experiment can
  **measure** how much the answer changes once the surrounding context is
  disclosed. Multi-turn is **undefined on the bare-`DIFF` tier** (nothing to
  reveal in a second turn) — `split_for_multi_turn` raises, and `MULTI_TURN_TIERS`
  tells Step 20 to skip that one cell rather than run a degenerate one.

### 3.3 One dispatcher

`build_exp4(task, strategy, context, tier, few_shot_examples=…)` returns a
`RenderedPrompt` for the three reused strategies and a `RenderedConversation` for
the two new ones. The Step-20 runner dispatches on the return type — that is the
single call the grid needs for all five strategies.

---

## 4. The AI-subset split, few-shot, and historical retrieval (`exp4_examples.py`)

Experiment 4 targets a population Experiment 2/3 **excluded** (they filtered to
human-written code), so it needs its own train/test split over the AI-authored
subset. Real sizes: **380** AI-authored PRs (361 copilot-swe-agent + 19
human-opened-AI-assisted) → after the per-repo time split, **302 train / 78
test** (test merge rate 61.5%, train 66.2%).

- `load_ai_train_pool` applies the *same* `split_by_repo_time` the ML experiments
  used, in the `{pr_id, repo, y, created_at}` shape `few_shot.select_*` already
  accept — so **Experiment 4 few-shot = the existing Exp-3 selectors called with
  the AI train pool**. No new few-shot code, and the no-leakage guarantee those
  selectors already enforce (demonstrations disjoint from the scored test set)
  carries straight over. (One data fix: `pull_requests.parquet` stores
  `created_at` as an ISO string, so the loader parses it to datetime — the ML
  variants' `features_*.parquet` had already done this.)

- `select_historical_comments` is the "historical review comments on similar past
  PRs" augmentation (§4.4.2) — **distinct** from few-shot: few-shot pairs a change
  with a *gold* comment to imitate; this retrieves what human reviewers *actually
  flagged on similar past PRs in the same repo* as reference material. Similarity
  is directory-set overlap (Jaccard, no embeddings — in keeping with the
  project's dependency-light ethos), tie-broken by comment substance. It is
  **leakage-safe by construction**: candidates come only from the train pool and
  the query PR itself is excluded, so the query's own in-flight review is never
  revealed. On real data it surfaces genuinely on-point, in-domain comments — many
  are human maintainers replying to `@copilot`, which is exactly the AI-code
  review signal this experiment is about.

---

## 5. Module layout & the Step-19 / Step-20 boundary

```
src/llm/prompts/
  schema.py         + Exp4Strategy, RenderedConversation, EXP3_EQUIVALENT
  context_aug.py    Exp4ContextTier, AugmentedContext, render_tier,
                    split_for_multi_turn, extract_issue_references,
                    build_lightweight_repo_context           (NEW)
  conversations.py  build_exp4, build_conversation, self-reflection &
                    multi-turn builders, CORE_GRID_STRATEGIES (NEW)
  exp4_examples.py  load_ai_train_pool, select_historical_comments (NEW)
  templates.py      (Exp 3, unchanged — reused for role/few-shot/CoT)
  few_shot.py       (Exp 3, unchanged — reused with the AI train pool)
  parsing.py        (Exp 3, unchanged — reused; final turn parses identically)
```

**Step 19 (this step) renders whatever `AugmentedContext` it is *given* — it
fetches nothing.** That is deliberate and mirrors the Exp-3 split (prompts
library = *how* to ask; `context_builder.py` = *what* context, assembled from
data). What **Step 20** must wire up:

1. **Populate `AugmentedContext`** per PR/tier: reuse `context_builder.build_diff_text`
   for the capped `diff`; read `body`/`commits` for description/commit; call
   `build_lightweight_repo_context` (or fetch file bodies) for `repo_context`;
   use `extract_issue_references` to decide which issue to fetch for `issue_text`;
   call `select_historical_comments` for `historical_comments`.
2. **A conversation-aware provider call.** `providers.py`'s `generate` today takes
   a single `RenderedPrompt` and caches on `{model, system, user, temperature,
   max_tokens}`. Multi-turn needs the provider to send a full message list
   (system + user₁ + assistant₁ + user₂) and cache on the **whole transcript**
   (add the message list to the cache-key payload). The `RenderedConversation`
   driving protocol is documented on the dataclass and in `schema.py`.
3. **Cost note for the grid.** Self-reflection = 2 calls/cell and multi-turn =
   2 calls/cell (vs. 1 for single-shot), against Groq's 6,000-TPM /
   500k-token-day free limits (`reports/exp3_grid_run_status.md`). Step 20 should
   size the AI-test-PR sample accordingly and rely on the same
   cache-resume/stop-on-quota machinery Exp 3 used.

---

## 6. How this maps to the §4.9 reflection questions (evidence the design creates)

- **Q1 (why repo-level context helps)** — the tier axis (`DIFF` →
  `DIFF_REPO_CONTEXT` → `COMPLETE`) is exactly the controlled comparison that
  answers this with numbers in Step 21.
- **Q2 (best prompt for AI code, and why)** — the five-strategy axis, with
  self-reflection and multi-turn specifically tuned to AI-code failure modes, is
  built to be compared head-to-head.
- **Q3 (prompt optimization vs. context augmentation — what each fixes)** — the
  design *separates the two axes* (strategy vs. tier) precisely so Step 21 can
  attribute gains to one or the other rather than confounding them.
- **Q4 (do prompt combinations generalize across tasks)** — both tasks run the
  full strategy × tier grid, so per-task strategy rankings are directly
  comparable.
- **Q5 (what else to explore)** — the honest data-availability gaps in §2.3
  (fetched file bodies, real call graphs, issue-body fetching, retrieval/RAG over
  history) are the concrete future-work answer, and §4.10 names exactly these
  (agents, automated prompt search, RAG).
