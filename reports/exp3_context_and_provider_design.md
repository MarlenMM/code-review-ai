# Experiment 3 — Context Builder & LLM Provider Design

*Design write-up for Step 15: `src/llm/context_builder.py` and
`src/llm/providers.py`. Feeds Step 16 (the context × prompt × task inference
grid). All percentiles below are computed from the real
`data/processed/*.parquet` dataset (population: human-written, closed PRs,
n=1,114 — same population `src/ml/common.py:split_by_repo_time` splits for
Experiment 2/3's shared test set), not estimated.*

Tests: `tests/test_llm_context_builder.py`, `tests/test_llm_providers.py`,
`tests/test_llm_providers_groq.py` (no real network call anywhere).

> **Provider update (Step 16).** This document's §3 originally described a
> Gemini-only provider. During Step 16 the Gemini free tier proved unusable
> from this account/region (billing-gated after every workaround — see
> `reports/exp3_grid_run_status.md`), so the codebase gained a **`GroqProvider`**
> sharing one `CachedChatProvider` base with `GeminiProvider`; **Groq is now the
> provider the grid actually uses**, and §3.7 below records the switch. Nothing
> about the context builder (§2) changed. Additionally, §2.2 now documents a
> whole-context character ceiling added in Step 16 to keep prompts under Groq's
> per-request token limit.

---

## 1. What Step 15 had to decide

Two independent things, per the plan's own framing of this row:

1. **What text goes into each of the four context tiers** the lab guide
   requires (§3.7.2) — `context_builder.py`.
2. **How to call an LLM without ever paying (in quota or wall-clock) for the
   same question twice** — `providers.py`.

Neither makes an API call. Step 15's own scope, and this write-up's, stops at
"the two files exist and are individually correct"; Step 16 is where they are
actually driven against real PRs.

---

## 2. Context builder: four tiers, one shared diff renderer

The four tiers (verified against the actual lab-guide PDF §3.7.2, not just the
plan's paraphrase): **diff-only**, **diff + PR description**, **diff + commit
message**, **diff + other information** (labels / file list / size). All four
share one diff renderer (`build_diff_text`) and differ only in what section is
prepended — the point being that `src/llm/prompts/templates.py` (Step 14)
never has to know which tier produced its input; it only ever sees a
`PromptContext.context_text` string.

### 2.1 No review-process leakage — carried over from Experiment 2

None of the four tiers touches `reviews.parquet`, `review_comments.parquet`,
or `issue_comments.parquet`. This is the same discipline Experiment 2's
feature spec enforces (`reports/exp2_feature_spec.md` §2/§4.5, lab guide
§2.4.3: "you CANNOT use any review comments/discussions as features") — and
arguably matters *more* here, since handing an LLM "this PR already has 3
approvals" while asking it to *predict* the merge outcome would be an almost
literal answer leak, not a subtle statistical one. `ContextSources` is typed
with exactly three fields (`pull_requests`, `files_changed`, `commits`) so this
isn't just a runtime filter that could be forgotten — the review-process
tables are structurally absent from what this module can even read
(`tests/test_llm_context_builder.py::test_context_sources_has_no_review_process_tables`
pins this down).

### 2.2 Truncation caps — measured, not guessed

Diff size in this dataset is extremely heavy-tailed: **median total diff
length per PR is 5,788 characters, but the maximum is ~21.5 million** (a
handful of PRs touch huge generated/vendored files). Truncation is mandatory
for cost/latency/fairness — Gemini Flash's context window is large enough that
token limits alone wouldn't force it, so this is a payload-size and
cross-PR-fairness decision, made explicit rather than left as an accident of
whatever a provider's request-size limit happens to be.

Measured coverage at each candidate cap (exact, not estimated):

| Total-diff cap | % of PRs fitting untruncated |
|---:|---:|
| 8,000 chars | 58.6% |
| 12,000 chars | 66.7% |
| **16,000 chars (chosen: `MAX_DIFF_CHARS`)** | **71.5%** |
| 20,000 chars | 74.9% |
| 24,000 chars | 78.4% |

| Per-file cap | % of individual files fitting untruncated |
|---:|---:|
| 2,000 chars | 50.9% |
| 3,000 chars | 60.3% |
| **4,000 chars (chosen: `MAX_FILE_PATCH_CHARS`)** | **66.6%** |
| 5,000 chars | 71.3% |

Both caps sit at the point where each further +4,000 characters starts buying
only ~3–4 more points of coverage — a defensible stopping point for
diminishing returns, and both are constructor-overridable if Step 16 wants to
retune them. Every truncation leaves an explicit
`"... [N omitted for length]"` / `"... [file diff truncated]"` marker; nothing
is ever silently presented as a complete diff when it isn't. Verified against
the actual worst-case PR in the dataset (21,455,961 raw diff characters): the
rendered context is capped to 16,112 characters with both markers present
(`test_real_data_truncation_cap_holds_on_worst_case_pr`).

Files are assembled **largest-churn-first** (`additions + deletions`
descending), not in `files_changed`'s stored order, so a truncated context
still shows the most substantive changes rather than whatever happened to be
listed first.

**Whole-context ceiling (added in Step 16).** The per-file and total-diff caps
above bound the *diff*, but the description / commit-message / metadata tiers
stack additional text on top (a PR body alone reaches ~4,000 tokens at the 99th
percentile), so a `diff+description` prompt — and especially a *few-shot* prompt
that repeats K example contexts — can still blow past a provider's per-request
token limit. `build_context(..., max_context_chars=N)` adds an optional hard
ceiling on the *assembled* string (default `None` = off, so nothing changed for
existing callers). The Step 16 Groq run sets it to 2,500 characters, with a
`"... [context truncated to fit the LLM token budget]"` marker; see §3.7 for why
this specific number, and `reports/exp3_grid_run_status.md` for the fidelity
trade-off it represents.

### 2.3 Commit messages: all of them, not just the first

`src/features/feature_extraction.py` uses only the *first* commit message
(`t_commit_msg_len_first`) because a fixed-length ML feature vector needs a
single scalar. An LLM context has no such constraint, so
`build_commit_messages_text` concatenates **every** commit message
(numbered, capped at 4,000 chars total). This is affordable in practice:
median total commit-message length per PR is 330 characters, 90th percentile
1,969 — comfortably inside the cap, with the same explicit-truncation
discipline covering the tail (up to 109,573 characters observed on one
100-commit outlier PR).

### 2.4 A real bug this caught: numpy array truthiness

`pull_requests.labels` is a numpy array (a parquet list column), and the
first draft of `build_metadata_text` wrote `pr.get("labels") or []` — which
raises `ValueError: The truth value of an array with more than one element is
ambiguous` the moment a PR has 2+ labels (77.0% of the population has at least
one label). Caught immediately by smoke-testing against a real multi-label PR
before writing the test suite, not by the tests themselves — a reminder that
"tests pass" and "ran against real data" are different guarantees, per this
project's established verification habit. Fixed by checking `is not None`
explicitly instead of truthiness.

### 2.5 The Step 14 ↔ Step 15 seam

`src/llm/prompts/few_shot.py` (Step 14) was written before this module
existed and needed *some* way to render a demonstration's context text; its
`default_render_context` is a deliberately minimal placeholder documented as
provisional. This module now provides the real implementation
(`render_context_text`), built to be a drop-in for the exact callable shape
`few_shot.py`'s selectors expect (`Callable[[str, sources], str]`, called
positionally). The two modules' "sources" bundles are structurally, not
nominally, compatible: `ContextSources` (three tables) and `few_shot.py`'s
`ExampleSources` (those same three, plus `review_comments`) both satisfy the
`HasCoreTables` protocol by attribute shape, so a `few_shot.ExampleSources`
instance can be passed straight into `render_context_text` with no adapter
(`test_render_context_text_accepts_ExampleSources_via_duck_typing` proves
this, not just documents it). Step 16 is expected to bind a specific tier with
`functools.partial(render_context_text, kind=ContextKind.DIFF_METADATA)` when
it wants few-shot examples rendered at a particular context richness —
`few_shot.py` itself is untouched by this (no edits to an already-tested,
already-committed Step 14 module were needed).

---

## 3. Provider: Gemini over raw HTTP, content-addressed cache

### 3.1 No SDK dependency

No Gemini SDK is installed (`requirements.txt` has none), and this project
already made the identical call once: `src/mining/github_client.py` talks to
GitHub with plain `requests` rather than `PyGithub`, despite the lab guide's
own software list recommending it. `generateContent` is one simple JSON
endpoint; a whole SDK dependency (surface area, version pinning, a heavier
thing to mock in tests) buys little over `requests`, which is already a
project dependency. Nothing new was added to `requirements.txt`.

### 3.2 Cache key = content hash of what actually determines the answer

The cache key is `sha256` of the canonical (sorted-key) JSON of exactly
`{model, system, user, temperature, max_output_tokens}`. Deliberately
**excluded** from the key: which PR / task / strategy / context kind produced
the prompt. Those are bookkeeping (`cache_metadata`, stored alongside the
response for Step 16/17's introspection) — the guarantee this buys is exact:
byte-identical wording always hits the cache, regardless of how Step 16
iterates the grid, and an accidental duplicate call in that loop costs zero
extra quota rather than "less." Cache files are sharded by hash prefix
(`data/llm_cache/<xx>/<hash>.json`, gitignored like `data/raw_cache/`) purely
for filesystem hygiene — the full Experiment 3 grid (2 tasks × 4 strategies ×
4 contexts × 224 test PRs) implies up to ~7,168 entries, before Experiment 4
adds more.

### 3.3 Retry / quota handling — a distinguishable exception, not a bare failure

429s retry with exponential backoff, honoring Gemini's own
`RetryInfo.retryDelay` hint when present (mirroring how `github_client.py`
already honors GitHub's `Retry-After` header). If 429s persist past
`max_retries`, `GeminiQuotaExceededError` is raised — a distinct, catchable
type, not a generic `RuntimeError` — because the plan's own risk mitigation
(§10) is "cache every response; if a quota wall is hit mid-run, resume from
cache on a later call," and that only works if Step 16's harness can tell
"stop, we're out of quota for today" apart from "something is actually
broken." A non-retryable 400 (see §3.4) is raised immediately with no
retries, since backoff cannot fix an account/region restriction.

### 3.4 Connectivity to the Gemini API from this environment is unreliable, not resolved

This finding went through three states as evidence accumulated, and the
current one is the one that matters — recorded honestly rather than leaving
the earlier, more optimistic entries standing:

1. **Step 15, first smoke call:** both `generateContent` and the read-only
   `GET /v1beta/models` returned, verbatim, `400 FAILED_PRECONDITION —
   "User location is not supported for the API use."` — an account/region-wide
   gate, not a malformed request.
2. **Retested immediately after (Step 15 follow-up):** two independent live
   calls via the real `GeminiProvider.generate()` both succeeded (`"PONG"`,
   `"ACK"`, real latencies ~1.5-2.0s, cache-hit correctly returned instant on
   a repeat). This was recorded as "resolved-but-not-fully-explained."
3. **Step 16's first real smoke run through `run_exp3_grid.py`** (1 test PR,
   the full 32-cell single-PR grid attempted): **17/17 attempted cells failed**,
   every one with the identical `FAILED_PRECONDITION` message, interleaved with
   real `429` quota errors before the run's own `GeminiQuotaExceededError`
   stop policy kicked in. Two of those 429 responses quoted the actual
   quota metric and limit Google enforced at that moment:

   ```
   Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests,
   limit: 5, model: gemini-3.6-flash
   ```
   ```
   Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests,
   limit: 20, model: gemini-3.6-flash
   ```

**Conclusion: connectivity is intermittent/unreliable from this environment,
not "resolved."** The two isolated successes in step 2 were not representative
of sustained use — under a real grid run moments later, the failure rate was
100%. The root cause (a load-balanced Google front-end that only sometimes
routes this egress to a region/instance that serves the free tier; a very low
and easily-exhausted per-minute quota compounding the picture; or both) was
not identified and cannot be from this side of the API. `GeminiAPIError`
surfaces Google's message verbatim rather than swallowing it, which is exactly
what made this diagnosable rather than looking like a silent hang or a bug in
`run_exp3_grid.py` — the harness worked correctly; it faithfully recorded 17
real, distinct API rejections. **This blocks a real Step 16 run until
resolved** (see the handoff note at the end of this document for what that
resolution might look like) — the harness is ready to run the moment
connectivity is reliable, but running it against a channel with an observed
100% failure rate would only burn quota and time for zero data.

### 3.5 Rate limiting is a placeholder, and the real limit is now known to be much lower

`min_interval_seconds` (default 4.0s, ~15 requests/minute) enforces a floor
between consecutive *live* calls (cache hits are exempt — instant, free).
§3.4's real 429 error bodies are the first concrete evidence of the actual
free-tier ceiling for `gemini-3.6-flash`: quota metric
`generate_content_free_tier_requests` was reported exhausted at **limit: 5**
and, in another window, **limit: 20** — both far below the ~15/minute this
provider currently assumes, meaning `min_interval_seconds=4.0` is very likely
**too aggressive** for whatever the real window (per-minute? per-day? a
rolling quota that recovers unevenly) turns out to be. This default should be
raised substantially (or made adaptive) before a real run is attempted again;
exactly how much is not yet known with confidence, since the two differing
"limit" values suggest more than one quota bucket is in play and neither was
isolated cleanly.

### 3.6 Testability

`_call_api` (the one method that makes a real HTTP call) is isolated
specifically so tests never need network access or a real key: caching-logic
tests monkeypatch `_call_api` directly (mirroring how `GitHubClient`'s tests
monkeypatch `_graphql`/`_rest_get_paginated`), while retry-loop tests drive a
`FakeSession` returning scripted `FakeResponse`s to exercise the actual retry
/ backoff / quota-exception code path. All the provider tests
(`test_llm_providers.py` + `test_llm_providers_groq.py`) run in ~1 second with
zero network egress.

### 3.7 The provider actually used: Groq (added in Step 16)

Gemini (§3.4) never became reliably usable from this account/region — after
linking billing to clear the region gate, every call returned "prepayment
credits are depleted," i.e. the free tier had been swapped for pay-as-you-go
(full story in `reports/exp3_grid_run_status.md`). A funded balance was
declined (the project is free-tier only), and DeepSeek turned out to be
prepaid too (zero balance, `is_available: false`). **Groq's free tier** — an
OpenAI-compatible `chat/completions` API that needs no credit card — was the
first provider that actually served real completions, so it is what the grid
uses.

`GroqProvider` and `GeminiProvider` share one `CachedChatProvider` base: the
disk cache, retry/backoff, rate-limit floor, and the whole `generate` contract
live in the base, and each subclass supplies only its request shape (Groq:
OpenAI `messages`; Gemini: `contents`/`systemInstruction`), auth header,
retry-delay source (Groq: `retry-after` header; Gemini: `RetryInfo` body), and
response parse. Switching providers is `--provider {groq,gemini}` on the grid
runner; the `model` name is part of the cache key, so the two never collide in
a shared `data/llm_cache/`. Provider errors subclass shared
`LLMAPIError`/`LLMQuotaExceededError` so the grid runner's stop/continue logic
is provider-agnostic.

**Groq's binding constraint is 6,000 tokens/minute**, and — crucially — that is
also a hard *per-request* cap: a single request over 6,000 tokens gets a `413`
that no backoff can fix (it exceeds the entire per-minute budget). This is what
motivated §2.2's whole-context ceiling: the Step 16 grid runs with
`max_diff_chars=2500` + `max_context_chars=2500` + few-shot `k=2`, which keeps
even the worst-case few-shot prompt at ~2,400 tokens (measured across a 24-PR ×
32-cell sample), comfortably under 6,000. The trade-off — a materially smaller
diff than Experiment 2's ML pipeline saw — is a real limitation of running on a
free tier, documented in `reports/exp3_grid_run_status.md` and to be named in
the Lab 3 report.

---

## 4. Handoff to Step 16

Step 16's grid runner (`src/llm/run_exp3_grid.py`) exists and is running against
Groq — see `reports/exp3_grid_run_status.md` for its design, the full
provider-selection saga, the tuned token-budget parameters, and live run
status.
