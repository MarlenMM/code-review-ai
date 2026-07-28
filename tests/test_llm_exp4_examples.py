"""Tests for Experiment 4 example / context selection over the AI-authored
subset (`src/llm/prompts/exp4_examples.py`).

Two guarantees matter here and are checked on both synthetic frames (fast) and,
when the mined dataset is present, end-to-end:

* the AI train pool is the *older* per-repo slice, disjoint from the recent AI
  PRs Step 20 scores -- the same no-leakage discipline the ML experiments used;
* `select_historical_comments` retrieves only human comments on *other* train
  PRs in the same repo -- never the query PR's own in-flight review.
"""

from pathlib import Path

import pandas as pd
import pytest

from src.llm.prompts.exp4_examples import (
    load_ai_train_pool,
    select_historical_comments,
)
from src.llm.prompts.few_shot import ExampleSources

DATA_DIR = Path("data/processed")
HAS_REAL_DATA = (DATA_DIR / "pull_requests.parquet").exists()


# --------------------------------------------------------------------------- #
# load_ai_train_pool (synthetic parquet)
# --------------------------------------------------------------------------- #

def _write_pulls(tmp_path: Path, n_ai: int = 20, n_human: int = 10) -> Path:
    rows = []
    # AI-authored PRs across two repos, spread over time
    for i in range(n_ai):
        rows.append({
            "id": f"ai{i}", "repo": "r/a" if i % 2 == 0 else "r/b",
            "created_at": f"2026-05-{(i % 27) + 1:02d}T00:00:00Z",
            "is_merged": bool(i % 3), "is_ai_authored": True,
        })
    # human PRs that must be filtered out
    for j in range(n_human):
        rows.append({
            "id": f"hu{j}", "repo": "r/a",
            "created_at": f"2026-05-{(j % 27) + 1:02d}T00:00:00Z",
            "is_merged": True, "is_ai_authored": False,
        })
    d = tmp_path / "processed"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(d / "pull_requests.parquet")
    return d


def test_ai_train_pool_shape_and_filtering(tmp_path):
    data_dir = _write_pulls(tmp_path)
    pool = load_ai_train_pool(data_dir=data_dir)
    assert set(pool.columns) >= {"pr_id", "repo", "y", "created_at"}
    # only AI-authored PRs survive
    assert pool["pr_id"].str.startswith("ai").all()
    assert not pool["pr_id"].str.startswith("hu").any()
    # y is the 0/1 merge label
    assert set(pool["y"].unique()) <= {0, 1}


def test_ai_train_pool_is_older_slice_and_disjoint_from_test(tmp_path):
    from src.ml.common import split_by_repo_time

    data_dir = _write_pulls(tmp_path)
    train = load_ai_train_pool(data_dir=data_dir)

    pr = pd.read_parquet(data_dir / "pull_requests.parquet")
    ai = pr[pr.is_ai_authored].copy()
    pool = pd.DataFrame({
        "pr_id": ai.id.values, "repo": ai.repo.values,
        "y": ai.is_merged.astype(int).values,
        "created_at": pd.to_datetime(ai.created_at.values, utc=True),
    })
    _tr, test, _ = split_by_repo_time(pool)
    assert set(train["pr_id"]).isdisjoint(set(test["pr_id"]))
    # per repo, every train PR is no newer than the earliest test PR
    for repo, g_test in test.groupby("repo"):
        g_train = train[train["repo"] == repo]
        if len(g_train) and len(g_test):
            assert g_train["created_at"].max() <= g_test["created_at"].min()


# --------------------------------------------------------------------------- #
# select_historical_comments (synthetic sources)
# --------------------------------------------------------------------------- #

def _sources() -> ExampleSources:
    pulls = pd.DataFrame([
        {"id": "q", "repo": "r/a", "title": "Query PR"},
        {"id": "t1", "repo": "r/a", "title": "Similar past PR"},   # same dir as q
        {"id": "t2", "repo": "r/a", "title": "Other dir PR"},      # different dir
        {"id": "t3", "repo": "r/b", "title": "Different repo PR"}, # different repo
        {"id": "t4", "repo": "r/a", "title": "Bot-reviewed PR"},
    ])
    files = pd.DataFrame([
        {"pr_id": "q", "filename": "src/core/a.py"},
        {"pr_id": "t1", "filename": "src/core/b.py"},   # same dir -> high overlap
        {"pr_id": "t2", "filename": "docs/readme.md"},  # different dir
        {"pr_id": "t3", "filename": "src/core/c.py"},
        {"pr_id": "t4", "filename": "src/core/d.py"},
    ])
    commits = pd.DataFrame(columns=["pr_id", "message"])
    review_comments = pd.DataFrame([
        {"pr_id": "t1", "author_login": "alice",
         "body": "This needs a guard for the empty-input case before indexing."},
        {"pr_id": "t2", "author_login": "bob",
         "body": "Consider documenting the new configuration option you added here."},
        {"pr_id": "t3", "author_login": "carol",
         "body": "This comment is on a different repo and must never be selected."},
        {"pr_id": "t4", "author_login": "copilot-pull-request-reviewer",
         "body": "An AI review bot comment that must be excluded from history."},
        {"pr_id": "q", "author_login": "dave",
         "body": "The query PR's OWN review must never leak into its context."},
    ])
    return ExampleSources(pulls, files, commits, review_comments)


def _train_pool() -> pd.DataFrame:
    # note: 'q' (the query) is intentionally NOT in the train pool; t1..t4 are
    return pd.DataFrame([
        {"pr_id": "t1", "repo": "r/a"},
        {"pr_id": "t2", "repo": "r/a"},
        {"pr_id": "t3", "repo": "r/b"},
        {"pr_id": "t4", "repo": "r/a"},
    ])


def test_historical_comments_same_repo_and_excludes_query_own_review():
    txt = select_historical_comments("q", _train_pool(), _sources(), k=5)
    assert "guard for the empty-input case" in txt     # t1, same repo
    # never the query PR's own review, a different repo, or a bot
    assert "OWN review must never leak" not in txt
    assert "different repo" not in txt
    assert "AI review bot" not in txt


def test_historical_comments_ranks_by_directory_similarity():
    # t1 shares src/core with the query; t2 is docs/. Most-similar first.
    txt = select_historical_comments("q", _train_pool(), _sources(), k=1)
    assert "guard for the empty-input case" in txt      # t1 wins
    assert "documenting the new configuration" not in txt


def test_historical_comments_only_from_train_pool():
    # a train pool that omits t1 must not surface t1's comment even though it is
    # the most similar -- retrieval is confined to the pool (leakage boundary)
    pool = pd.DataFrame([{"pr_id": "t2", "repo": "r/a"}])
    txt = select_historical_comments("q", pool, _sources(), k=5)
    assert "guard for the empty-input case" not in txt
    assert "documenting the new configuration" in txt


def test_historical_comments_empty_when_no_candidates():
    empty_pool = pd.DataFrame(columns=["pr_id", "repo"])
    assert select_historical_comments("q", empty_pool, _sources(), k=5) == ""


def test_historical_comments_empty_for_unknown_query():
    assert select_historical_comments("nope", _train_pool(), _sources(), k=5) == ""


# --------------------------------------------------------------------------- #
# Real-data checks (skipped if the dataset isn't present)
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(not HAS_REAL_DATA, reason="mined dataset not present")
def test_real_ai_train_pool_is_ai_only_and_disjoint_from_test():
    from src.ml.common import split_by_repo_time

    train = load_ai_train_pool()
    assert len(train) > 0

    pr = pd.read_parquet(DATA_DIR / "pull_requests.parquet")
    ai = pr[(pr.is_ai_authored == True) & pr.is_merged.notna()].copy()  # noqa: E712
    pool = pd.DataFrame({
        "pr_id": ai.id.values, "repo": ai.repo.values,
        "y": ai.is_merged.astype(int).values,
        "created_at": pd.to_datetime(ai.created_at.values, utc=True),
    })
    _tr, test, _ = split_by_repo_time(pool)
    assert set(train["pr_id"]).isdisjoint(set(test["pr_id"]))
    # every train pr_id is a genuinely AI-authored PR
    ai_ids = set(ai["id"])
    assert set(train["pr_id"]) <= ai_ids


@pytest.mark.skipif(not HAS_REAL_DATA, reason="mined dataset not present")
def test_real_historical_comments_are_leakage_safe():
    from src.llm.prompts.few_shot import load_sources

    train = load_ai_train_pool()
    sources = load_sources()

    # find one test AI PR that yields historical context, and verify safety
    pr = pd.read_parquet(DATA_DIR / "pull_requests.parquet")
    from src.ml.common import split_by_repo_time
    ai = pr[(pr.is_ai_authored == True) & pr.is_merged.notna()].copy()  # noqa: E712
    pool = pd.DataFrame({
        "pr_id": ai.id.values, "repo": ai.repo.values,
        "y": ai.is_merged.astype(int).values,
        "created_at": pd.to_datetime(ai.created_at.values, utc=True),
    })
    _tr, test, _ = split_by_repo_time(pool)

    checked = 0
    for pid in test["pr_id"].tolist():
        txt = select_historical_comments(pid, train, sources, k=3)
        if not txt:
            continue
        # the query PR's own comments must not appear verbatim
        own = sources.review_comments.loc[
            sources.review_comments["pr_id"] == pid, "body"
        ].fillna("").tolist()
        for body in own:
            if len(body) >= 40:
                assert body[:60] not in txt
        checked += 1
        if checked >= 5:
            break
    assert checked > 0, "expected at least one test PR with historical context"
