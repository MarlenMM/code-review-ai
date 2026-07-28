"""Experiment 4 example / context selection over the **AI-authored** PR subset.

Experiment 4 re-targets the whole pipeline at AI-generated PRs
(`is_ai_authored=True`), a population Experiment 2/3 deliberately *excluded*
(they filtered to human-written code). So Exp 4 needs its own train/test split
over that subset, and this module provides it -- reusing the same per-repo,
time-based splitter (`src/ml/common.py:split_by_repo_time`) the ML experiments
used, so the leakage discipline is identical: demonstrations and retrieved
context come only from *older* (train) PRs, never the recent (test) PRs the grid
is scored on.

Two things live here:

1. ``load_ai_train_pool`` -- the AI-authored train split, in the
   `{pr_id, repo, y, created_at}` shape `few_shot.select_merge_examples` /
   `select_comment_examples` already accept. So Experiment 4 few-shot prompts
   are just those existing selectors called with this pool -- no new few-shot
   code, and the no-leakage guarantee the Exp 3 selectors already enforce
   carries straight over.

2. ``select_historical_comments`` -- the "historical review comments on similar
   past PRs" augmentation (§4.4.2), which feeds `AugmentedContext.historical_comments`.
   This is distinct from few-shot: few-shot pairs a change with a *gold* comment
   to imitate; this retrieves what human reviewers *actually flagged on similar
   past PRs in the same repo*, as reference material for the model's own review.
   It never touches the query PR's own review (that would leak the answer) and
   draws only from the train pool.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.llm.prompts.few_shot import (
    _MIN_HUMAN_COMMENT_LEN,
    _NON_HUMAN_COMMENT_AUTHORS,
    ExampleSources,
)
from src.ml.common import DATA_DIR, TRAIN_FRACTION, split_by_repo_time

DEFAULT_HISTORY_K = 3
DEFAULT_SEED = 42
_MAX_HISTORY_COMMENT_CHARS = 500


def _load_ai_pool(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """The full AI-authored, closed-PR population (train + test, unsplit), in
    the `{pr_id, repo, y, created_at}` shape `split_by_repo_time` and
    `few_shot.select_*` expect. Population: `is_ai_authored == True` and a
    definite merge label (`is_merged` not null) -- 380 PRs in the mined
    dataset (361 agent-authored + 19 human-opened-AI-assisted)."""
    pr = pd.read_parquet(data_dir / "pull_requests.parquet")
    ai = pr[(pr["is_ai_authored"] == True) & pr["is_merged"].notna()].copy()  # noqa: E712
    return pd.DataFrame({
        "pr_id": ai["id"].values,
        "repo": ai["repo"].values,
        "y": ai["is_merged"].astype(int).values,
        # `pull_requests.parquet` keeps `created_at` as an ISO string, whereas
        # `split_by_repo_time` (built for the ML variants) calls `.isoformat()`
        # on it -- so parse to datetime here, matching `features_*.parquet`.
        "created_at": pd.to_datetime(ai["created_at"].values, utc=True),
    })


def load_ai_train_pool(
    data_dir: Path = DATA_DIR, train_fraction: float = TRAIN_FRACTION
) -> pd.DataFrame:
    """The **train** split of the AI-authored, closed PRs -- the only legal
    source of Experiment 4 few-shot demonstrations and historical-comment
    context. Split per-repo by time (older = train), so the test complement
    (`load_ai_test_pool`) is exactly the recent AI PRs Step 20 scores --
    keeping few-shot demonstrations provably disjoint from the evaluation set.
    """
    train, _test, _cutoffs = split_by_repo_time(_load_ai_pool(data_dir), train_fraction=train_fraction)
    return train


def load_ai_test_pool(
    data_dir: Path = DATA_DIR, train_fraction: float = TRAIN_FRACTION
) -> pd.DataFrame:
    """The **test** split of the AI-authored, closed PRs -- the exact
    complement of `load_ai_train_pool` (same underlying pool, same split) and
    the population `run_exp4_grid.py`'s grid scores on."""
    _train, test, _cutoffs = split_by_repo_time(_load_ai_pool(data_dir), train_fraction=train_fraction)
    return test


def _changed_dirs(pr_id: str, files_changed: pd.DataFrame) -> set[str]:
    """The set of directories a PR's changed files live in -- the cheap
    similarity signal for `select_historical_comments` (no embeddings, in
    keeping with the project's dependency-light ethos)."""
    files = files_changed.loc[files_changed["pr_id"] == pr_id, "filename"]
    dirs: set[str] = set()
    for name in files:
        name = str(name)
        dirs.add(name.rsplit("/", 1)[0] if "/" in name else "(repo root)")
    return dirs


def _is_human_comment(author_login: str) -> bool:
    a = (author_login or "").lower()
    if a in {x.lower() for x in _NON_HUMAN_COMMENT_AUTHORS}:
        return False
    return not ("[bot]" in a or a.endswith("-bot") or a.startswith("bot-"))


def select_historical_comments(
    query_pr_id: str,
    train_pool: pd.DataFrame,
    sources: ExampleSources,
    *,
    k: int = DEFAULT_HISTORY_K,
    min_comment_len: int = _MIN_HUMAN_COMMENT_LEN,
    max_comment_chars: int = _MAX_HISTORY_COMMENT_CHARS,
) -> str:
    """Retrieve up to `k` real human review comments left on *similar past PRs*
    (same repo, most directory overlap with the query PR), rendered as the text
    block `AugmentedContext.historical_comments` carries.

    Leakage-safe by construction: candidates come only from `train_pool` and the
    query PR itself is excluded, so this never reveals the query's own in-flight
    review. Similarity is directory-set overlap (Jaccard), tie-broken by comment
    length (a substantive comment over a terse "LGTM"). Returns "" when the repo
    has no qualifying historical human comment -- Step 20 then renders the
    tier's honest "(no historical review comments available)" marker.
    """
    pr_rows = sources.pull_requests.loc[sources.pull_requests["id"] == query_pr_id]
    if pr_rows.empty:
        return ""
    query_repo = str(pr_rows.iloc[0].get("repo") or "")
    query_dirs = _changed_dirs(query_pr_id, sources.files_changed)

    # candidate PRs: same repo, in the train pool, not the query itself
    candidates = train_pool[(train_pool["repo"] == query_repo)
                            & (train_pool["pr_id"] != query_pr_id)]
    if candidates.empty:
        return ""
    candidate_ids = set(candidates["pr_id"])

    # substantive human review comments on those candidates
    rc = sources.review_comments.copy()
    rc["author_login"] = rc["author_login"].fillna("")
    rc["body"] = rc["body"].fillna("")
    rc = rc[rc["pr_id"].isin(candidate_ids)]
    rc = rc[rc["body"].str.len() >= min_comment_len]
    rc = rc[rc["author_login"].apply(_is_human_comment)]
    if rc.empty:
        return ""

    # one representative (longest) human comment per candidate PR
    rc = rc.assign(_len=rc["body"].str.len())
    best = rc.sort_values("_len", ascending=False).drop_duplicates(subset=["pr_id"], keep="first")

    # score each candidate PR by directory overlap with the query
    def similarity(pr_id: str) -> float:
        d = _changed_dirs(pr_id, sources.files_changed)
        if not d and not query_dirs:
            return 0.0
        inter = len(d & query_dirs)
        union = len(d | query_dirs) or 1
        return inter / union

    best = best.assign(_sim=best["pr_id"].apply(similarity))
    # rank by similarity first, then by comment substance; deterministic order
    best = best.sort_values(["_sim", "_len"], ascending=[False, False]).head(k)

    titles = dict(zip(sources.pull_requests["id"], sources.pull_requests["title"]))
    blocks: list[str] = []
    for row in best.itertuples(index=False):
        pr_id = row.pr_id
        title = str(titles.get(pr_id, "") or "").strip() or "(no title)"
        comment = str(row.body).strip()
        if len(comment) > max_comment_chars:
            comment = comment[:max_comment_chars].rstrip() + " ..."
        blocks.append(
            f'On a past PR in {query_repo} ("{title}"), a reviewer commented:\n'
            f'"{comment}"'
        )
    return "\n\n".join(blocks)
