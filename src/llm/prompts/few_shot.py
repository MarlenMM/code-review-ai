"""Few-shot example selection for Experiment 3 -- drawn ONLY from the
Experiment-2 train split.

Why this module is where the leakage guarantee lives
----------------------------------------------------
Experiment 3's merge-prediction task is scored against the *same* held-out test
set as Experiment 2 (plan §5.4; lab guide §3.7.1), so that the LLM-vs-ML
comparison is apples-to-apples. That test set is exactly what
`src/ml/common.py:split_by_repo_time` puts in the *test* slice. A few-shot
prompt that demonstrated a test PR -- or worse, revealed its merge outcome --
would be showing the model the answer it is about to be graded on. So every
demonstration here comes from the *train* slice of that identical split, and
`select_*` is written so its output is provably disjoint from the test set
(the shared tests assert this).

The same reasoning applies to review-comment generation: BLEU/ROUGE (Step 17)
compares generated comments to the real human comments on the *test* PRs, so
the human comments used as few-shot demonstrations must come from *train* PRs.

What this module does NOT do
----------------------------
It does not build the rich code context -- that is Step 15's `context_builder`.
Each example carries a `context_text` produced by a `render_context` callable
the caller supplies; in the real grid (Step 16) that callable is the context
builder at the *same* context kind as the query, so a demonstration matches the
query in format. Until Step 15 exists (and in tests), `default_render_context`
provides a deliberately minimal diff+title rendering so this module is
self-contained and runnable today.

Selection is deterministic (seeded) so the grid is reproducible and every LLM
response caches under a stable key.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.llm.prompts.schema import FewShotExample
from src.mining.ai_detection import AI_CODING_AGENT_LOGINS, AI_REVIEW_BOT_LOGINS
from src.ml.common import DATA_DIR, load_variant, split_by_repo_time

DEFAULT_K = 4
DEFAULT_SEED = 42

# Diffs are big; a demonstration only needs to convey shape/style, and long
# demonstrations crowd out the query and inflate token cost. Cap the rendered
# example diff (the real context builder governs the query's own budget).
_MAX_EXAMPLE_DIFF_CHARS = 1500
_MAX_EXAMPLE_COMMENT_CHARS = 600

# Review-comment authors to exclude when mining *human* demonstration comments:
# the AI coding-agent and AI review-bot logins (imported from the Experiment-1
# detector so the two never drift), plus non-AI automation that leaves PR
# comments. We want the gold style to be a human maintainer's, not a bot's.
_NON_HUMAN_COMMENT_AUTHORS = (
    AI_CODING_AGENT_LOGINS
    | AI_REVIEW_BOT_LOGINS
    | {"github-actions", "codecov", "codecov-commenter", "azure-pipelines"}
)
_MIN_HUMAN_COMMENT_LEN = 40


@dataclass
class ExampleSources:
    """The raw tables `select_*` reads to render demonstrations and to find
    human review comments. Bundled so callers/tests can inject small synthetic
    frames instead of the full parquet dataset."""

    pull_requests: pd.DataFrame
    files_changed: pd.DataFrame
    commits: pd.DataFrame
    review_comments: pd.DataFrame


def load_sources(data_dir: Path = DATA_DIR) -> ExampleSources:
    return ExampleSources(
        pull_requests=pd.read_parquet(data_dir / "pull_requests.parquet"),
        files_changed=pd.read_parquet(data_dir / "files_changed.parquet"),
        commits=pd.read_parquet(data_dir / "commits.parquet"),
        review_comments=pd.read_parquet(data_dir / "review_comments.parquet"),
    )


def load_train_pool(variant: str = "v1", data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """The Experiment-2 **train** split -- the only legal source of few-shot
    demonstrations. Its complement (`split_by_repo_time(...)[1]`) is exactly the
    test set Experiment 3 is scored on, so selecting from here is what makes the
    no-leakage guarantee hold. Returns rows with at least `pr_id`, `repo`, `y`,
    `created_at`."""
    df = load_variant(variant, data_dir=data_dir)
    train, _test, _cutoffs = split_by_repo_time(df)
    return train


# --------------------------------------------------------------------------- #
# Default (placeholder) context renderer -- superseded by Step 15's builder
# --------------------------------------------------------------------------- #

def default_render_context(pr_id: str, sources: ExampleSources) -> str:
    """Minimal, self-contained rendering of a PR's context: title + a truncated
    concatenation of its file diffs. Intentionally basic -- Step 15's
    `context_builder` replaces it in the real grid (and is passed in via
    `render_context`), but this keeps `few_shot.py` importable and testable
    before Step 15 lands, and gives sensible demonstrations either way."""
    pr_rows = sources.pull_requests.loc[sources.pull_requests["id"] == pr_id]
    title = ""
    if not pr_rows.empty:
        title = str(pr_rows.iloc[0].get("title") or "")

    files = sources.files_changed.loc[sources.files_changed["pr_id"] == pr_id]
    diff_parts = []
    for _, row in files.iterrows():
        patch = row.get("patch")
        if isinstance(patch, str) and patch:
            diff_parts.append(f"--- {row['filename']}\n{patch}")
    diff = "\n".join(diff_parts)
    if len(diff) > _MAX_EXAMPLE_DIFF_CHARS:
        diff = diff[:_MAX_EXAMPLE_DIFF_CHARS] + "\n... [diff truncated]"

    header = f"Title: {title}\n" if title else ""
    return f"{header}Diff:\n{diff}" if diff else f"{header}(no file diff available)"


# --------------------------------------------------------------------------- #
# Merge-prediction examples
# --------------------------------------------------------------------------- #

def select_merge_examples(
    train_pool: pd.DataFrame,
    sources: ExampleSources,
    *,
    k: int = DEFAULT_K,
    balanced: bool = True,
    seed: int = DEFAULT_SEED,
    render_context: Callable[[str, ExampleSources], str] = default_render_context,
) -> list[FewShotExample]:
    """Pick `k` merge-prediction demonstrations from `train_pool` (train split).

    `balanced=True` (default) draws an equal-as-possible number of MERGED and
    CLOSED PRs and interleaves them, rather than sampling the raw ~72%-merged
    base rate. Rationale (discussed in `reports/exp3_prompt_design.md`): a
    few-shot set exists to *demonstrate the decision boundary*, and an all-MERGE
    demonstration set teaches the model only the majority prior it already has.
    Set `balanced=False` to demonstrate the true base rate instead -- the two
    are a genuine design trade-off, not a right/wrong choice, which is why it is
    a parameter.

    Deterministic given `seed`. Presentation order interleaves the classes so
    neither dominates the start of the prompt.
    """
    pool = train_pool.dropna(subset=["y"])
    if balanced:
        merged = pool[pool["y"] == 1]
        closed = pool[pool["y"] == 0]
        n_merge = (k + 1) // 2
        n_close = k // 2
        merge_sel = _sample(merged, n_merge, seed)
        close_sel = _sample(closed, n_close, seed)
        chosen = _interleave(merge_sel, close_sel)
    else:
        chosen = _sample(pool, k, seed).to_dict("records")

    examples = []
    for row in chosen:
        pr_id = row["pr_id"]
        label = "MERGE" if int(row["y"]) == 1 else "CLOSE"
        examples.append(FewShotExample(
            pr_id=pr_id,
            repo=str(row.get("repo", "")),
            context_text=render_context(pr_id, sources),
            label=label,
        ))
    return examples


# --------------------------------------------------------------------------- #
# Review-comment examples
# --------------------------------------------------------------------------- #

def select_comment_examples(
    train_pool: pd.DataFrame,
    sources: ExampleSources,
    *,
    k: int = DEFAULT_K,
    seed: int = DEFAULT_SEED,
    min_comment_len: int = _MIN_HUMAN_COMMENT_LEN,
    render_context: Callable[[str, ExampleSources], str] = default_render_context,
) -> list[FewShotExample]:
    """Pick `k` review-comment demonstrations from `train_pool` (train split):
    train PRs that received a substantive *human* review comment, paired with
    that comment as the gold target.

    'Human' = author not in `_NON_HUMAN_COMMENT_AUTHORS` (the Experiment-1 AI
    allowlists + non-AI automation). For each qualifying PR the single longest
    qualifying comment (capped) is used, as a representative substantive example
    rather than a terse "LGTM". Deterministic given `seed`.
    """
    train_ids = set(train_pool["pr_id"])
    repo_by_id = dict(zip(train_pool["pr_id"], train_pool.get("repo", pd.Series(dtype=str))))

    rc = sources.review_comments.copy()
    rc["author_login"] = rc["author_login"].fillna("")
    rc["body"] = rc["body"].fillna("")
    rc = rc[rc["pr_id"].isin(train_ids)]
    rc = rc[~rc["author_login"].str.lower().isin({a.lower() for a in _NON_HUMAN_COMMENT_AUTHORS})]
    rc = rc[~rc["author_login"].str.contains(r"\[bot\]|-bot$|^bot-", case=False, regex=True)]
    rc = rc[rc["body"].str.len() >= min_comment_len]

    if rc.empty:
        return []

    # one representative (longest) comment per PR
    rc = rc.assign(_len=rc["body"].str.len())
    best = rc.sort_values("_len", ascending=False).drop_duplicates(subset=["pr_id"], keep="first")

    # deterministic choice of which PRs to demonstrate
    chosen = _sample(best, k, seed).to_dict("records")

    examples = []
    for row in chosen:
        pr_id = row["pr_id"]
        comment = str(row["body"]).strip()
        if len(comment) > _MAX_EXAMPLE_COMMENT_CHARS:
            comment = comment[:_MAX_EXAMPLE_COMMENT_CHARS].rstrip() + " ..."
        examples.append(FewShotExample(
            pr_id=pr_id,
            repo=str(repo_by_id.get(pr_id, "")),
            context_text=render_context(pr_id, sources),
            target_comment=comment,
        ))
    return examples


# --------------------------------------------------------------------------- #
# Deterministic sampling helpers
# --------------------------------------------------------------------------- #

def _sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Deterministic sample of up to `n` rows (pandas `sample` is seeded and
    reproducible). Returns all rows if fewer than `n` are available."""
    if n <= 0 or df.empty:
        return df.iloc[0:0]
    if len(df) <= n:
        return df
    return df.sample(n=n, random_state=seed)


def _interleave(a_df: pd.DataFrame, b_df: pd.DataFrame) -> list[dict]:
    """Interleave two frames' rows (a[0], b[0], a[1], b[1], ...) so neither
    class clusters at the start of a balanced few-shot prompt."""
    a = a_df.to_dict("records")
    b = b_df.to_dict("records")
    out = []
    for i in range(max(len(a), len(b))):
        if i < len(a):
            out.append(a[i])
        if i < len(b):
            out.append(b[i])
    return out
