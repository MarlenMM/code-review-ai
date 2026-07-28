"""Experiment 3 grid runner (lab guide §3.7 Steps 2-4; plan Section 9 row 16).

For every (test PR × context kind × task × prompt strategy) cell, builds the
prompt (Steps 14-15) and invokes Gemini through `GeminiProvider.generate`
(Step 15's disk cache), appending one JSON Lines record per cell to
`results/tables/exp3_raw.jsonl`. This module owns *only* the iteration,
resumability, and quota-aware stopping policy around calls that Steps 14-15
already made correct in isolation -- it makes no scoring judgment (that's
Step 17) and designs no prompts or contexts of its own.

Population: the SAME 224 human-written test PRs Experiment 2 evaluates on
(`src.ml.common.split_by_repo_time(load_variant("v1"))[1]`), for **both**
tasks -- so merge prediction is directly comparable to Exp2's SVM/RF
(reflection Q3.9(5)), and review-comment generation draws from PRs that
mostly have real human comments in `review_comments.parquet` for Step 17's
BLEU/ROUGE scoring.

Grid shape: 224 PRs x 4 context kinds x 2 tasks x 4 strategies = 7,168 cells
(32 cells per PR). The free-tier LLM quota makes the full 7,168-cell grid
impractical in one sitting, so `main` supports a stratified subset
(`--pr-sample N`, proportional across the 5 repos) for a smaller-but-
representative run; whatever is written accumulates and resumes across
invocations (plan Section 10's "spread remaining calls across another
session" mitigation), and a later larger `--pr-sample` is a superset that
re-uses every already-recorded cell.
Few-shot demonstrations are selected ONCE per (task, context kind) -- not
resampled per PR -- because the guide frames few-shot as a fixed set of
demonstrations the model conditions on; resampling per PR would also make the
prompt (and therefore the cache key) needlessly different across PRs that
should share the same demonstrations.

Two independent resumability layers (see `src/llm/providers.py`'s own
docstring for the first):

1. **Content-addressed LLM cache** (`GeminiProvider`): re-running this whole
   script is always safe -- any cell whose exact prompt text was already
   answered costs no quota, regardless of whether this script "knows" it was
   already done.
2. **This script's own row-skip**: `load_existing_keys` reads
   `results/tables/exp3_raw.jsonl` and skips any `(pr_id, context_kind,
   strategy, task)` already recorded there -- a faster skip than even
   reconstructing the prompt and hitting the cache-lookup path, and what
   makes a partial output file safe to build on directly rather than only
   through the LLM cache.

Quota-exhaustion policy: a `GeminiQuotaExceededError` on any cell stops the
**entire run immediately**, not just that cell. Once the free-tier daily quota
is genuinely exhausted, every subsequent live call fails identically after
several minutes of futile retry/backoff (`src/llm/providers.py`'s own
`MAX_RETRIES` schedule) -- continuing to iterate the remaining grid would burn
enormous wall-clock time for zero benefit. Whatever was written before the
stop remains valid; a later run (invoked with `python -m
src.llm.run_exp3_grid`) resumes exactly where this one stopped, at zero cost
for every already-recorded cell.

Output rows (JSON Lines, one per cell): `pr_id`, `repo`, `context_kind`,
`task`, `strategy`, `response_text` (`None` on a recorded API failure),
`error` (`None` on success), `from_cache`, `latency_ms`, `model`.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
from functools import partial
from pathlib import Path
from typing import Optional

from src.llm import context_builder
from src.llm.prompts import few_shot, templates
from src.llm.prompts.schema import Strategy, Task
from src.llm.providers import PROVIDERS, LLMAPIError, LLMQuotaExceededError
from src.ml.common import load_variant, split_by_repo_time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

RAW_OUTPUT_PATH = Path("results/tables/exp3_raw.jsonl")
FEW_SHOT_K = 4

# Token-budget defaults for the actual grid run, tuned for Groq's free tier
# (6,000 tokens/minute, which is ALSO a hard per-request cap). These keep every
# prompt -- including a few-shot prompt that stacks K example contexts + the
# query -- comfortably under 6,000 tokens (~4 chars/token): a K=2 few-shot
# prompt is ~2*(2500 ctx + comment) + 2500 query + overhead ~= 8,000 chars
# ~= 2,000 tokens. Fixed here (not left to ad-hoc flags) so every resumed run
# produces byte-identical prompts and the content-addressed cache keeps hitting.
GROQ_RUN_MAX_DIFF_CHARS = 2_500
GROQ_RUN_MAX_CONTEXT_CHARS = 2_500
GROQ_RUN_FEW_SHOT_K = 2


def load_test_pr_ids(variant: str = "v1") -> list[str]:
    """The Experiment-2-aligned test set: lab guide §3.7.1's "test data from
    Experiment [2]" (the guide's own text says "Experiment 3", a
    self-referential typo -- see `reports/exp3_prompt_design.md` §1.1).
    """
    df = load_variant(variant)
    _train, test, _ = split_by_repo_time(df)
    return test["pr_id"].tolist()


def stratified_test_pr_order(variant: str = "v1") -> list[str]:
    """The test PRs in a deterministic, repo-stratified order: round-robin
    across the 5 repos (each repo's own PRs kept in `created_at` order). Taking
    the first N of this order gives a subset proportionally spread across repos
    rather than dominated by whichever repo's test PRs happen to sort first --
    important because merge rates differ a lot by repo (Exp2 spec §1: 62%-85%),
    so a non-stratified prefix would bias the LLM-vs-ML comparison. Nested by
    construction: `first N+k` is a superset of `first N`, so a later larger
    `--pr-sample` re-uses every already-cached cell.
    """
    df = load_variant(variant)
    _train, test, _ = split_by_repo_time(df)
    per_repo = {
        repo: g.sort_values("created_at")["pr_id"].tolist()
        for repo, g in test.groupby("repo")
    }
    # deterministic repo order (by name) for reproducibility
    repos = sorted(per_repo)
    ordered: list[str] = []
    i = 0
    while len(ordered) < len(test):
        for repo in repos:
            if i < len(per_repo[repo]):
                ordered.append(per_repo[repo][i])
        i += 1
    return ordered


def load_existing_keys(path: Path = RAW_OUTPUT_PATH) -> set[tuple[str, str, str, str]]:
    """Every `(pr_id, context_kind, strategy, task)` already recorded in a
    prior (possibly partial) run of this script."""
    if not path.exists():
        return set()
    keys: set[tuple[str, str, str, str]] = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            keys.add((row["pr_id"], row["context_kind"], row["strategy"], row["task"]))
    return keys


def build_few_shot_pools(
    sources, train_pool, *, k: int = FEW_SHOT_K,
    max_diff_chars: int = context_builder.MAX_DIFF_CHARS,
    max_context_chars: int | None = context_builder.MAX_CONTEXT_CHARS,
) -> dict:
    """One set of few-shot demonstrations per `(task.value, context_kind.value)`,
    rendered at that same context kind (and the same diff cap) as the query it
    accompanies, so a demonstration and the query are a fair analogy (see
    `src/llm/context_builder.py`'s and `src/llm/prompts/few_shot.py`'s module
    docstrings for this seam). `sources` is expected to be a
    `few_shot.ExampleSources` (it must have `review_comments` for
    `select_comment_examples`); it also structurally satisfies
    `context_builder.HasCoreTables`, so the same object serves the context
    renderer bound into each pool.
    """
    pools = {}
    for kind in context_builder.ContextKind:
        render = partial(
            context_builder.render_context_text, kind=kind,
            max_diff_chars=max_diff_chars, max_context_chars=max_context_chars,
        )
        pools[(Task.MERGE_PREDICTION.value, kind.value)] = few_shot.select_merge_examples(
            train_pool, sources, k=k, render_context=render
        )
        pools[(Task.REVIEW_COMMENT.value, kind.value)] = few_shot.select_comment_examples(
            train_pool, sources, k=k, render_context=render
        )
    return pools


def run_grid(
    pr_ids: list[str],
    *,
    provider,
    sources,
    few_shot_pools: dict,
    output_path: Path = RAW_OUTPUT_PATH,
    max_new_calls: Optional[int] = None,
    max_diff_chars: int = context_builder.MAX_DIFF_CHARS,
    max_context_chars: int | None = context_builder.MAX_CONTEXT_CHARS,
    cache_only: bool = False,
) -> dict:
    """Iterate the full grid for `pr_ids`, skipping cells already in
    `output_path`, appending each new result immediately (so a killed or
    quota-exhausted run leaves a valid, resumable partial file). `provider`
    only needs a `generate(prompt, cache_metadata=...) -> response` method and
    a `.model` attribute -- tests use a lightweight fake rather than a real
    `GeminiProvider`. Returns a run summary dict; never raises for a quota
    exhaustion or a recorded API error (see module docstring for the stopping
    policy) -- only a genuinely unexpected exception propagates.
    """
    existing = load_existing_keys(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_skipped = n_written = n_new_calls = n_failed = n_cache_misses = 0
    stopped_reason: Optional[str] = None

    cells = itertools.product(pr_ids, context_builder.ContextKind, Task, Strategy)

    with output_path.open("a") as out:
        for pr_id, kind, task, strategy in cells:
            key = (pr_id, kind.value, strategy.value, task.value)
            if key in existing:
                n_skipped += 1
                continue
            if max_new_calls is not None and n_new_calls >= max_new_calls:
                stopped_reason = "max_new_calls reached"
                break

            ctx = context_builder.build_context(
                pr_id, kind, sources,
                max_diff_chars=max_diff_chars, max_context_chars=max_context_chars,
            )
            examples = (
                few_shot_pools[(task.value, kind.value)] if strategy is Strategy.FEW_SHOT else []
            )
            prompt = templates.build_prompt(task, strategy, ctx, examples)

            response = None
            error_msg: Optional[str] = None
            try:
                response = provider.generate(
                    prompt,
                    cache_metadata={
                        "pr_id": pr_id, "context_kind": kind.value,
                        "strategy": strategy.value, "task": task.value,
                    },
                    cache_only=cache_only,
                )
            except LLMQuotaExceededError as e:
                logger.error(
                    "Quota exceeded at pr=%s kind=%s task=%s strategy=%s; stopping run: %s",
                    pr_id, kind.value, task.value, strategy.value, e,
                )
                stopped_reason = f"quota_exceeded: {e}"
                break
            except LLMAPIError as e:
                logger.warning(
                    "API error at pr=%s kind=%s task=%s strategy=%s: %s",
                    pr_id, kind.value, task.value, strategy.value, e,
                )
                error_msg = str(e)
                n_failed += 1

            # cache_only miss: nothing to record, leave the cell for a live run
            if cache_only and response is None and error_msg is None:
                n_cache_misses += 1
                continue

            if response is not None and not response.from_cache:
                n_new_calls += 1

            row = {
                "pr_id": pr_id,
                "repo": ctx.repo,
                "context_kind": kind.value,
                "task": task.value,
                "strategy": strategy.value,
                "response_text": response.text if response is not None else None,
                "error": error_msg,
                "from_cache": response.from_cache if response is not None else None,
                "latency_ms": response.latency_ms if response is not None else None,
                "usage": response.usage if response is not None else None,
                "model": response.model if response is not None else provider.model,
            }
            out.write(json.dumps(row) + "\n")
            out.flush()
            n_written += 1
            existing.add(key)

    total_cells = len(pr_ids) * len(list(context_builder.ContextKind)) * len(list(Task)) * len(list(Strategy))
    return {
        "total_cells": total_cells,
        "n_written_this_run": n_written,
        "n_skipped_already_done": n_skipped,
        "n_new_api_calls": n_new_calls,
        "n_recorded_failures": n_failed,
        "n_cache_misses_skipped": n_cache_misses,
        "stopped_reason": stopped_reason,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider", choices=sorted(PROVIDERS), default="groq",
        help="Which LLM provider to call (default: groq -- the free-tier "
             "provider that actually works from this account/region; see "
             "reports/exp3_grid_run_status.md).",
    )
    parser.add_argument(
        "--model", default=None,
        help="Override the provider's default model (e.g. a larger Groq model).",
    )
    parser.add_argument(
        "--pr-sample", type=int, default=None,
        help="Run a repo-stratified subset of the first N test PRs (proportional "
             "across the 5 repos, deterministic, nested). The recommended way to "
             "bound a run given free-tier quota.",
    )
    parser.add_argument(
        "--pr-limit", type=int, default=None,
        help="Run the first N test PRs in raw split order (NOT stratified). "
             "Prefer --pr-sample; this exists for parity with other scripts.",
    )
    parser.add_argument(
        "--max-new-calls", type=int, default=None,
        help="Stop after this many NEW (non-cached) live API calls this run "
             "(cache hits/skips don't count) -- a hard cap independent of the "
             "PR count, for capping spend/time on a single invocation.",
    )
    parser.add_argument(
        "--max-diff-chars", type=int, default=GROQ_RUN_MAX_DIFF_CHARS,
        help=f"Per-PR diff truncation cap for the LLM context (default "
             f"{GROQ_RUN_MAX_DIFF_CHARS}, tuned for Groq free tier). Lower it to "
             f"fit more calls under a token-per-minute quota.",
    )
    parser.add_argument(
        "--max-context-chars", type=int, default=GROQ_RUN_MAX_CONTEXT_CHARS,
        help=f"Hard ceiling on the whole assembled context (default "
             f"{GROQ_RUN_MAX_CONTEXT_CHARS}). Keeps even a few-shot prompt "
             f"(which stacks K example contexts) under the provider's "
             f"per-request token limit -- Groq free tier rejects any single "
             f"request over 6,000 tokens with a 413.",
    )
    parser.add_argument("--output-path", type=Path, default=RAW_OUTPUT_PATH)
    parser.add_argument(
        "--few-shot-k", type=int, default=GROQ_RUN_FEW_SHOT_K,
        help=f"Number of few-shot demonstrations per (task, context kind) "
             f"(default {GROQ_RUN_FEW_SHOT_K}, tuned so a few-shot prompt fits "
             f"the free-tier per-request token limit).",
    )
    parser.add_argument(
        "--cache-only", action="store_true",
        help="Never make a live API call: emit rows only for cells already in "
             "the response cache, skipping the rest. Use to regenerate the "
             "output file (e.g. after adding a recorded field) for free.",
    )
    args = parser.parse_args()

    if args.pr_sample is not None:
        pr_ids = stratified_test_pr_order()[: args.pr_sample]
    else:
        pr_ids = load_test_pr_ids()
        if args.pr_limit is not None:
            pr_ids = pr_ids[: args.pr_limit]
    logger.info("Running Experiment 3 grid over %d test PR(s) via %s", len(pr_ids), args.provider)

    sources = few_shot.load_sources()
    train_pool = few_shot.load_train_pool()
    few_shot_pools = build_few_shot_pools(
        sources, train_pool, k=args.few_shot_k,
        max_diff_chars=args.max_diff_chars, max_context_chars=args.max_context_chars,
    )

    provider_kwargs = {"model": args.model} if args.model else {}
    provider = PROVIDERS[args.provider](**provider_kwargs)
    summary = run_grid(
        pr_ids,
        provider=provider,
        sources=sources,
        few_shot_pools=few_shot_pools,
        output_path=args.output_path,
        max_new_calls=args.max_new_calls,
        max_diff_chars=args.max_diff_chars,
        max_context_chars=args.max_context_chars,
        cache_only=args.cache_only,
    )
    logger.info("Run summary: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
