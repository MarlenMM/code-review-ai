"""Experiment 4 grid runner (lab guide §4.7 Steps 2-4; plan Section 9 row 20).

For every (AI-authored test PR x context tier x task x prompt strategy) cell,
builds the Step-19 prompt/conversation (`src.llm.prompts.build_exp4`) over
Step-20's populated context (`src.llm.exp4_context.build_augmented_context`)
and invokes the LLM through `CachedChatProvider.generate` (single-shot
strategies) or `.generate_conversation` (self-reflection / multi-turn),
appending one JSON Lines record per cell to `results/tables/exp4_raw.jsonl`.
Mirrors `run_exp3_grid.py`'s architecture and resumability discipline exactly
-- this module owns only iteration, resumability, and the quota-aware stopping
policy; it makes no scoring judgment (Step 21) and designs no prompts of its
own (Step 19).

Population: the AI-authored **test** split
(`src.llm.prompts.exp4_examples.load_ai_test_pool`) -- the complement of the
train split few-shot demonstrations and historical-comment retrieval draw
from, so nothing scored here was ever shown to the model as a demonstration.

Grid shape and the one skipped cell
------------------------------------
5 context tiers x 2 tasks x 5 strategies = 50 cells/PR, MINUS the
(`Exp4ContextTier.DIFF`, `Exp4Strategy.MULTI_TURN`) cell for each task (2 more
cells) -- multi-turn has nothing to reveal in a second turn on the bare-diff
tier (`context_aug.split_for_multi_turn` raises `ValueError` there by design;
see `Exp4ContextTier` docs). So **48 cells/PR**, `CELLS_PER_PR` below.

Unlike Experiment 3 (which rebuilds its `PromptContext` inside the innermost
loop, once per (pr_id, context_kind) pair, since the kind genuinely changes
what's built), Experiment 4's `AugmentedContext` does NOT vary by tier -- only
which of its slots `render_tier`/`build_exp4` show for a given tier does. So
`run_grid` builds `AugmentedContext` **once per PR** and reuses it across all
48 cells for that PR, which is both simpler and avoids redundant historical-
comment retrieval / issue fetches per PR.

Call-cost multiplier this run must account for
-------------------------------------------------
Self-reflection and multi-turn each cost **2** live calls per cell (vs. 1 for
role-based/few-shot/CoT) when nothing is cached -- so a full 48-cell/PR run
makes up to 30 (single-shot) + 20 (self-reflection, 10 cells x 2) + 16
(multi-turn, 8 cells x 2) = **66 live calls/PR** on a cold cache, against
Groq's free-tier 6,000 TPM / 500,000 tokens/day limits
(`reports/exp3_grid_run_status.md`). `--pr-sample` is the primary lever for
staying inside that budget; `results/tables/exp4_grid_run_status.md` (if
written) documents what a real run actually consumed.

Two independent resumability layers, and the quota-exhaustion stopping policy,
are identical in spirit to `run_exp3_grid.py` -- see that module's docstring;
not repeated here.

Output rows (JSON Lines, one per cell): `pr_id`, `repo`, `context_kind` (the
tier value -- named to match Experiment 3's column for any shared scoring
utility), `task`, `strategy`, `response_text` (the FINAL turn's text for a
conversational strategy), `turn_texts` (all turn replies in order, `None` for
single-shot cells -- kept for Step 21's qualitative "draft -> critique ->
final" / "diff-only -> reconsidered" analysis), `is_conversational`, `error`,
`from_cache` (for a conversation: `True` only if EVERY turn was cached),
`latency_ms` (summed across turns for a conversation), `usage` (the final
turn's usage), `turn_usages`, `model`.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
from pathlib import Path
from typing import Optional

from src.llm import exp4_context
from src.llm.prompts import context_aug, few_shot
from src.llm.prompts.context_aug import Exp4ContextTier
from src.llm.prompts.conversations import build_exp4
from src.llm.prompts.exp4_examples import load_ai_test_pool, load_ai_train_pool
from src.llm.prompts.schema import Exp4Strategy, RenderedConversation, Task
from src.llm.providers import PROVIDERS, LLMAPIError, LLMQuotaExceededError
from src.mining.github_client import GitHubClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

RAW_OUTPUT_PATH = Path("results/tables/exp4_raw.jsonl")
FEW_SHOT_K = 2
HISTORY_K = 2

# The one cell skipped for every PR/task (see module docstring).
_SKIPPED_TIER_STRATEGY = (Exp4ContextTier.DIFF, Exp4Strategy.MULTI_TURN)
CELLS_PER_PR = (
    len(list(Exp4ContextTier)) * len(list(Task)) * len(list(Exp4Strategy))
    - len(list(Task))  # one skipped (tier, strategy) pair, for each of the 2 tasks
)

# Token-budget defaults for the actual grid run, tuned for Groq's free tier.
# Tighter than Experiment 3's own tuned constants (`run_exp3_grid.py`'s
# GROQ_RUN_*) because Experiment 4's COMPLETE_SE_CONTEXT tier stacks FIVE
# sections on the diff (vs. Experiment 3's one), and self-reflection/
# multi-turn's second call resends the first turn's full transcript -- so
# per-section headroom has to be tighter for even the richest tier's SECOND
# turn to stay comfortably under Groq's 6,000-token hard per-request cap.
# Measured real sizes (offline, no live calls) that calibrated these:
# COMPLETE_SE_CONTEXT turn 1 ~= 4,000 chars (~1,000 tokens) at the module
# defaults; these RUN constants trim that further for safety margin under a
# few-shot cell (which additionally stacks K example contexts at the same tier).
GROQ_RUN_MAX_DIFF_CHARS = 900
GROQ_RUN_MAX_FILE_CHARS = 400
GROQ_RUN_MAX_SECTION_CHARS = 400
GROQ_RUN_FEW_SHOT_K = 1
GROQ_RUN_HISTORY_K = 1
GROQ_RUN_MAX_OUTPUT_TOKENS = 500


def stratified_ai_test_pr_order(data_dir: Path = exp4_context.DATA_DIR) -> list[str]:
    """The AI-authored test PRs in a deterministic, repo-stratified order:
    round-robin across repos (each repo's own PRs kept in `created_at` order).
    Mirrors `run_exp3_grid.py:stratified_test_pr_order` -- see there for why
    (proportional repo spread; nested, so a larger `--pr-sample` reuses every
    already-cached cell)."""
    test = load_ai_test_pool(data_dir=data_dir)
    per_repo = {
        repo: g.sort_values("created_at")["pr_id"].tolist()
        for repo, g in test.groupby("repo")
    }
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
    sources: few_shot.ExampleSources,
    train_pool,
    *,
    k: int = FEW_SHOT_K,
    max_diff_chars: int = exp4_context.MAX_DIFF_CHARS,
    max_file_chars: int = exp4_context.MAX_FILE_PATCH_CHARS,
    max_section_chars: int = exp4_context.MAX_SECTION_CHARS,
    history_k: int = exp4_context.HISTORY_K,
    max_history_comment_chars: int = exp4_context.MAX_HISTORY_COMMENT_CHARS,
) -> dict:
    """One set of few-shot demonstrations per `(task.value, tier.value)`,
    rendered at that same tier as the query it accompanies (same rationale as
    `run_exp3_grid.py`'s pools). Demonstration contexts are ALWAYS built with
    `github_client=None` (no issue fetch) regardless of whether the main run
    fetches issues for query PRs -- a deliberate scope cut: demonstrations
    exist to show format/label/style, not to be the richest possible context,
    and skipping their issue fetch keeps a few-shot cell's total size (which
    already stacks K example contexts on top of the query) well within budget.
    """
    pools: dict[tuple[str, str], list] = {}
    for tier in Exp4ContextTier:
        def render(pr_id: str, srcs, _tier: Exp4ContextTier = tier) -> str:
            ctx = exp4_context.build_augmented_context(
                pr_id, srcs, train_pool,
                max_diff_chars=max_diff_chars, max_file_chars=max_file_chars,
                max_section_chars=max_section_chars, history_k=history_k,
                max_history_comment_chars=max_history_comment_chars,
                github_client=None,
            )
            return context_aug.render_tier(ctx, _tier).context_text

        pools[(Task.MERGE_PREDICTION.value, tier.value)] = few_shot.select_merge_examples(
            train_pool, sources, k=k, render_context=render
        )
        pools[(Task.REVIEW_COMMENT.value, tier.value)] = few_shot.select_comment_examples(
            train_pool, sources, k=k, render_context=render
        )
    return pools


def _grid_cells():
    """Every (tier, task, strategy) triple EXCEPT the one undefined multi-turn
    cell (see module docstring)."""
    for tier, task, strategy in itertools.product(Exp4ContextTier, Task, Exp4Strategy):
        if (tier, strategy) == _SKIPPED_TIER_STRATEGY:
            continue
        yield tier, task, strategy


def run_grid(
    pr_ids: list[str],
    *,
    provider,
    sources: few_shot.ExampleSources,
    train_pool,
    few_shot_pools: dict,
    output_path: Path = RAW_OUTPUT_PATH,
    max_new_calls: Optional[int] = None,
    max_output_tokens: int = GROQ_RUN_MAX_OUTPUT_TOKENS,
    github_client=None,
    context_kwargs: Optional[dict] = None,
    cache_only: bool = False,
) -> dict:
    """Iterate the full grid for `pr_ids`, skipping cells already in
    `output_path`, appending each new result immediately. `provider` needs
    `.generate(prompt, ...)`, `.generate_conversation(conversation, ...)`, and
    a `.model` attribute -- tests use a lightweight fake. `github_client`, if
    given, enables the (narrow, capped) issue-body fetch
    `exp4_context.build_augmented_context` supports; `None` skips it.
    `max_new_calls` counts actual live HTTP calls (not cells) -- a
    conversational cell can spend 0, 1, or 2 depending on what's cached, so the
    cap may overshoot by up to one extra call, the same best-effort spirit as
    Experiment 3's own cap.

    Never raises for a quota exhaustion or a recorded API error (see module
    docstring for the stopping policy) -- only a genuinely unexpected exception
    propagates. Returns a run summary dict.
    """
    context_kwargs = context_kwargs or {}
    existing = load_existing_keys(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_skipped = n_written = n_new_calls = n_failed = n_cache_misses = 0
    stopped_reason: Optional[str] = None

    with output_path.open("a") as out:
        for pr_id in pr_ids:
            if stopped_reason is not None:
                break

            aug_ctx = exp4_context.build_augmented_context(
                pr_id, sources, train_pool, github_client=github_client, **context_kwargs
            )

            for tier, task, strategy in _grid_cells():
                key = (pr_id, tier.value, strategy.value, task.value)
                if key in existing:
                    n_skipped += 1
                    continue
                if max_new_calls is not None and n_new_calls >= max_new_calls:
                    stopped_reason = "max_new_calls reached"
                    break

                examples = (
                    few_shot_pools[(task.value, tier.value)]
                    if strategy is Exp4Strategy.FEW_SHOT else []
                )
                rendered = build_exp4(task, strategy, aug_ctx, tier, few_shot_examples=examples)
                is_conv = isinstance(rendered, RenderedConversation)
                cache_metadata = {
                    "pr_id": pr_id, "context_kind": tier.value,
                    "strategy": strategy.value, "task": task.value,
                }

                response = None
                error_msg: Optional[str] = None
                try:
                    if is_conv:
                        response = provider.generate_conversation(
                            rendered, max_output_tokens=max_output_tokens,
                            cache_metadata=cache_metadata, cache_only=cache_only,
                        )
                    else:
                        response = provider.generate(
                            rendered, max_output_tokens=max_output_tokens,
                            cache_metadata=cache_metadata, cache_only=cache_only,
                        )
                except LLMQuotaExceededError as e:
                    logger.error(
                        "Quota exceeded at pr=%s tier=%s task=%s strategy=%s; stopping run: %s",
                        pr_id, tier.value, task.value, strategy.value, e,
                    )
                    stopped_reason = f"quota_exceeded: {e}"
                    break
                except LLMAPIError as e:
                    logger.warning(
                        "API error at pr=%s tier=%s task=%s strategy=%s: %s",
                        pr_id, tier.value, task.value, strategy.value, e,
                    )
                    error_msg = str(e)
                    n_failed += 1

                if cache_only and response is None and error_msg is None:
                    n_cache_misses += 1
                    continue

                row = _response_row(
                    pr_id=pr_id, repo=aug_ctx.repo, tier=tier, task=task, strategy=strategy,
                    response=response, error_msg=error_msg, is_conv=is_conv,
                    provider_model=provider.model,
                )
                n_new_calls += row.pop("_n_new_live_calls")
                out.write(json.dumps(row) + "\n")
                out.flush()
                n_written += 1
                existing.add(key)

    return {
        "total_cells": len(pr_ids) * CELLS_PER_PR,
        "n_written_this_run": n_written,
        "n_skipped_already_done": n_skipped,
        "n_new_api_calls": n_new_calls,
        "n_recorded_failures": n_failed,
        "n_cache_misses_skipped": n_cache_misses,
        "stopped_reason": stopped_reason,
    }


def _response_row(
    *, pr_id, repo, tier, task, strategy, response, error_msg, is_conv, provider_model,
) -> dict:
    """Shape one output row uniformly across the two response types
    (`LLMResponse` for single-shot, `ConversationResponse` for conversational)
    -- includes the internal `_n_new_live_calls` bookkeeping key, popped by the
    caller before writing (kept out of the persisted row)."""
    if response is None:
        return {
            "pr_id": pr_id, "repo": repo, "context_kind": tier.value,
            "task": task.value, "strategy": strategy.value,
            "response_text": None, "turn_texts": None, "is_conversational": is_conv,
            "error": error_msg, "from_cache": None, "latency_ms": None,
            "usage": None, "turn_usages": None, "model": provider_model,
            "_n_new_live_calls": 0,
        }
    if is_conv:
        return {
            "pr_id": pr_id, "repo": repo, "context_kind": tier.value,
            "task": task.value, "strategy": strategy.value,
            "response_text": response.final_text,
            "turn_texts": [t.text for t in response.turns],
            "is_conversational": True,
            "error": error_msg,
            "from_cache": response.all_from_cache,
            "latency_ms": response.total_latency_ms,
            "usage": response.turns[-1].usage,
            "turn_usages": [t.usage for t in response.turns],
            "model": response.turns[-1].model,
            "_n_new_live_calls": sum(1 for t in response.turns if not t.from_cache),
        }
    return {
        "pr_id": pr_id, "repo": repo, "context_kind": tier.value,
        "task": task.value, "strategy": strategy.value,
        "response_text": response.text, "turn_texts": None, "is_conversational": False,
        "error": error_msg, "from_cache": response.from_cache,
        "latency_ms": response.latency_ms, "usage": response.usage, "turn_usages": None,
        "model": response.model,
        "_n_new_live_calls": 0 if response.from_cache else 1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider", choices=sorted(PROVIDERS), default="groq",
        help="Which LLM provider to call (default: groq -- the free-tier "
             "provider that actually works from this account/region; see "
             "reports/exp3_grid_run_status.md).",
    )
    parser.add_argument("--model", default=None, help="Override the provider's default model.")
    parser.add_argument(
        "--pr-sample", type=int, default=None,
        help="Run a repo-stratified subset of the first N AI-authored test PRs "
             "(proportional across repos, deterministic, nested). The "
             "recommended way to bound a run given free-tier quota and the "
             "self-reflection/multi-turn call-cost multiplier.",
    )
    parser.add_argument(
        "--pr-limit", type=int, default=None,
        help="Run the first N test PRs in raw split order (NOT stratified). Prefer --pr-sample.",
    )
    parser.add_argument(
        "--max-new-calls", type=int, default=None,
        help="Stop after this many NEW (non-cached) live API calls this run "
             "(counts individual HTTP calls, not cells -- a conversational "
             "cell can spend up to 2).",
    )
    parser.add_argument("--max-diff-chars", type=int, default=GROQ_RUN_MAX_DIFF_CHARS)
    parser.add_argument("--max-file-chars", type=int, default=GROQ_RUN_MAX_FILE_CHARS)
    parser.add_argument(
        "--max-section-chars", type=int, default=GROQ_RUN_MAX_SECTION_CHARS,
        help="Per-augmentation-section cap (description/commit/repo-context/"
             "issue/history) -- Experiment 4's analogue of Experiment 3's "
             "whole-context cap, but per-section since tiers stack sections "
             "rather than one block.",
    )
    parser.add_argument("--few-shot-k", type=int, default=GROQ_RUN_FEW_SHOT_K)
    parser.add_argument("--history-k", type=int, default=GROQ_RUN_HISTORY_K)
    parser.add_argument("--max-output-tokens", type=int, default=GROQ_RUN_MAX_OUTPUT_TOKENS)
    parser.add_argument("--output-path", type=Path, default=RAW_OUTPUT_PATH)
    parser.add_argument(
        "--fetch-issues", action="store_true",
        help="Fetch the referenced issue's body via the GitHub API for the "
             "DIFF_ISSUE / COMPLETE_SE_CONTEXT tiers (requires GITHUB_TOKEN). "
             "Off by default: the majority of AI PRs carry no closing-keyword "
             "issue reference (measured ~19%% of the AI test split), so this "
             "is a narrow, capped opt-in, not a bulk fetch.",
    )
    parser.add_argument(
        "--cache-only", action="store_true",
        help="Never make a live API call: emit rows only for cells already in "
             "the response cache, skipping the rest.",
    )
    args = parser.parse_args()

    if args.pr_sample is not None:
        pr_ids = stratified_ai_test_pr_order()[: args.pr_sample]
    else:
        pr_ids = load_ai_test_pool()["pr_id"].tolist()
        if args.pr_limit is not None:
            pr_ids = pr_ids[: args.pr_limit]
    logger.info("Running Experiment 4 grid over %d AI-authored test PR(s) via %s", len(pr_ids), args.provider)

    sources = few_shot.load_sources()
    train_pool = load_ai_train_pool()
    few_shot_pools = build_few_shot_pools(
        sources, train_pool, k=args.few_shot_k,
        max_diff_chars=args.max_diff_chars, max_file_chars=args.max_file_chars,
        max_section_chars=args.max_section_chars, history_k=args.history_k,
    )

    github_client = GitHubClient() if args.fetch_issues else None

    provider_kwargs = {"model": args.model} if args.model else {}
    provider = PROVIDERS[args.provider](**provider_kwargs)
    summary = run_grid(
        pr_ids,
        provider=provider,
        sources=sources,
        train_pool=train_pool,
        few_shot_pools=few_shot_pools,
        output_path=args.output_path,
        max_new_calls=args.max_new_calls,
        max_output_tokens=args.max_output_tokens,
        github_client=github_client,
        context_kwargs={
            "max_diff_chars": args.max_diff_chars,
            "max_file_chars": args.max_file_chars,
            "max_section_chars": args.max_section_chars,
            "history_k": args.history_k,
        },
        cache_only=args.cache_only,
    )
    logger.info("Run summary: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
