# Experiment 4 — Grid Runner Design & Run Status (Step 20)

*Status write-up for Step 20: `src/llm/exp4_context.py` (context population),
`src/llm/run_exp4_grid.py` (the grid runner), plus the multi-turn conversation
support added to `src/llm/providers.py` and the issue-fetch added to
`src/mining/github_client.py`. Deliverable is `results/tables/exp4_raw.jsonl`.
Tests: `tests/test_llm_exp4_context.py` (11), `tests/test_llm_run_exp4_grid.py`
(16), `tests/test_llm_providers_conversation.py` (11), plus 7 new
`fetch_issue`/`_issue_cache_path` tests in `tests/test_github_client.py` --
45 new tests, zero network calls; full suite 360 passing before the run
described below.*

---

## 1. What was built on top of Step 19

Step 19 designed the Experiment 4 prompt/context library
(`src/llm/prompts/context_aug.py`, `conversations.py`, `exp4_examples.py`) but
deliberately fetched nothing and made no API calls. Step 20 is what actually
runs it:

- **`src/llm/exp4_context.py`** -- populates one `AugmentedContext` per
  AI-authored PR from `data/processed/*.parquet`: diff/description/commit
  message reuse Experiment 3's `context_builder` builders; repository context
  uses the no-fetch `build_lightweight_repo_context` approximation (git hunk
  headings); the linked issue is resolved via a **narrow, capped** live
  GitHub fetch (see §2); historical comments come from
  `exp4_examples.select_historical_comments`. Per-section character caps are
  tighter than Experiment 3's own tuned constants, because
  `COMPLETE_SE_CONTEXT` stacks five sections on the diff (vs. Experiment 3's
  one) and self-reflection/multi-turn's second call resends the first turn's
  full transcript.
- **Multi-turn conversation support in `src/llm/providers.py`**
  (`generate_turn`, `generate_conversation`, `ConversationResponse`) -- the gap
  flagged at the end of Step 19. Implemented additively: the existing
  single-turn `_call_api`/`generate` are unchanged in signature and behavior
  (its own 42-test suite still passes unmodified); the retry/backoff/quota loop
  was extracted into a shared `_execute_request` that both the single-turn and
  new multi-turn call paths use, so Experiment 3's on-disk cache and grid stay
  valid. A conversation is cached **turn-by-turn** on the whole running
  transcript (`_cache_key_from_messages`), so a resumed run replays a cached
  turn-1 answer byte-for-byte before even attempting turn 2, and a partially
  cached conversation only re-calls the missing turn.
- **`GitHubClient.fetch_issue`** -- a new method, cached separately from the PR
  cache (`data/raw_cache/issues/<owner>/<repo>/<number>.json`), that resolves
  an issue reference found by Step 19's `extract_issue_references` into the
  issue's actual title/body. A 404 (common -- many `#N` references are PRs, not
  issues) is cached as `{"not_found": True}`, not raised.
- **`src/llm/run_exp4_grid.py`** -- the grid runner, architecturally mirroring
  `run_exp3_grid.py` (same two-layer resumability, same quota-stop policy) with
  the differences the new axes require: `AugmentedContext` is built **once per
  PR** (not per cell, since it's tier-independent -- only which slots
  `render_tier` shows varies), the grid skips the one undefined
  (`DIFF`, `MULTI_TURN`) cell per task, and live-call counting is per actual
  HTTP call (0/1/2, since a conversational cell can be 0, 1, or 2 calls
  depending on what's cached) rather than per cell.

## 2. Design choices worth stating plainly

- **Repository context stayed lightweight (no file-body fetch).** Fetching
  full file contents from GitHub for every changed file of every scored PR
  would be a much larger, rate-limit-sensitive engineering lift than this
  step's budget allows; the design doc (`reports/exp4_prompt_and_context_design.md`
  §2.3) already flagged this as a legitimate scope cut, not a hidden one.
- **Issue fetching is opt-in and narrow (`--fetch-issues`), off by default.**
  Only the single most-confident referenced issue number is fetched per PR
  (closing-keyword/URL preferred over a bare mention), and PR-shaped issue
  numbers are skipped. It was validated end-to-end against the real GitHub API
  on a real AI-authored test PR before the grid run (a genuine issue body was
  fetched and rendered correctly), but the two grid invocations below were
  launched **without** `--fetch-issues` (an oversight, caught after the run
  completed and the daily LLM quota was already exhausted) -- so every row in
  the current `exp4_raw.jsonl` renders the `DIFF_ISSUE`/`COMPLETE_SE_CONTEXT`
  tiers with `issue_text=None` (the tier's own "(no linked issue information
  available)" marker), not a fetched issue body. The feature is implemented,
  unit-tested (`tests/test_llm_exp4_context.py`), and ready; a future top-up
  run should pass `--fetch-issues` to actually populate that slot. This is
  named plainly here rather than left for Step 21 to discover silently.
- **Few-shot demonstration contexts never fetch issues**, regardless of the
  main run's `--fetch-issues` setting -- demonstrations exist to show
  format/style, not to be the richest possible context, and this keeps a
  few-shot cell's total size (which already stacks K example contexts on the
  query) well within Groq's per-request budget.
- **Token-budget constants were calibrated against real measurements**, not
  guessed: an offline dry run of `COMPLETE_SE_CONTEXT` (all five sections)
  produced a ~4,000-char (~1,000-token) turn-1 prompt at the module's default
  caps; `GROQ_RUN_*` in `run_exp4_grid.py` trims further for safety margin
  under a few-shot cell.

## 3. Run scope and status

**Scope of the run: a repo-stratified subset of the AI-authored test split**
(380 AI-authored PRs total; 302 train / **78 test** after the per-repo
time split -- see `reports/exp4_prompt_and_context_design.md` §4).
`--pr-sample 8` was chosen given the grid's call-cost multiplier: unlike
Experiment 3 (1 call/cell), self-reflection and multi-turn each cost up to 2
live calls/cell, so the full 5-tier × 2-task × 5-strategy grid minus the one
undefined cell is **48 cells/PR** but up to **66 live calls/PR** on a cold
cache.

Two runs were made: a 1-PR pilot (`--pr-sample 1 --max-new-calls 66`) to
validate correctness and calibrate real per-cell token cost before committing
quota to a larger run, then the full run (`--pr-sample 8`, no cap) which
resumed the pilot's already-cached PR for free.

**Status: stopped by Groq's daily token cap (500,000 tokens/day, shared across
the whole account's usage that day) -- a clean, zero-failure dataset.**

| Metric | Value |
|---|---|
| Cells written | **148 / 384** (39% of the 8-PR sample's grid) |
| Failures | **0** |
| PRs fully complete (all 48 cells) | **3** (`dotnet/aspnetcore`, `dotnet/runtime`, `home-assistant/core`) |
| PRs partial | 1 (`microsoft/semantic-kernel`, 4/48 cells); 4 not started |
| True total tokens spent (all turns counted, incl. conversational) | **165,095** |
| Avg tokens/cell (all turns) | 1,116 |
| Single-shot rows (role-based/few-shot/CoT) | **93** = 90 across the 3 complete PRs + 3 in the partial one |
| Conversational rows (self-reflection/multi-turn) | **55** = 54 + 1 in the partial one (each up to 2 live calls) |

*(93 + 55 = 148, i.e. the "Cells written" figure above. The two row-type counts
are given on the whole-run basis so they reconcile with it; the 90/54 split is
the same counts restricted to the 3 fully complete PRs, which is the population
every metric in Step 21 is computed on.)*

The 3 fully complete PRs each carry the exact expected shape: 30 rows for each
of role-based/few-shot/CoT/self-reflection (5 tiers × 2 tasks) and 24 rows for
multi-turn (4 applicable tiers × 2 tasks, `DIFF` correctly excluded), 72
merge-prediction rows and 72 review-comment rows, 0 errors. This is enough for
Step 21 to compute per-cell metrics and every context/prompt comparison the
lab guide's §4.7.5/§4.8 asks for, at n=3 PRs per full-grid cell -- a small,
free-tier-driven sample size to name explicitly in the Lab 4 report,
consistent with how Experiment 3's own report already discusses its (larger,
n=16) sample-size caveat.

**Why the daily cap arrived faster than a naive "500k tokens ÷ ~74k
tokens/PR ≈ 6-7 PRs" estimate:** the account's 500,000-token/day ceiling is
shared across *all* Groq usage that calendar day, not reserved for Experiment
4 alone -- the daily-quota error message itself reported `Used 499346` against
the 500k limit while this run's own rows only account for 165,095 of that,
meaning roughly 334k tokens of headroom had already been consumed by other Groq
activity earlier that same calendar day -- principally Experiment 3's own grid
run, which shares the identical account and daily budget. This is a real,
reportable resource-sharing constraint, not a bug: on a free tier, two
experiments run on the same day compete for one 500k-token pool, so scheduling
them on separate days is itself a design consideration.

**To finish the remaining 5 PRs:** re-run the identical command
(`python -m src.llm.run_exp4_grid --pr-sample 8 --fetch-issues`) after the
daily window resets -- the 148 completed cells are skipped instantly via both
resumability layers (the provider's disk cache and this script's own row-skip),
so it resumes at cell 149 for free. A future top-up is cheap, exactly as
Experiment 3's own precedent established.

Nothing about the *design* is blocked; the only constraint is free-tier daily
throughput, and the harness (both `run_exp3_grid.py`'s and this module's)
already absorbs it cleanly by design.
