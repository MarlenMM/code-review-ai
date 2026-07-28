"""Tests for few-shot example selection (`src/llm/prompts/few_shot.py`).

The headline property is the *no-leakage guarantee*: every demonstration must
come from the Experiment-2 train split and therefore be disjoint from the test
set Experiment 3 is scored on. That is checked both on synthetic frames (fast,
hermetic) and, when the real dataset is present, end-to-end against the actual
`split_by_repo_time` output -- because the guarantee is only meaningful against
the real split the grid will use.
"""

from pathlib import Path

import pandas as pd
import pytest

from src.llm.prompts.few_shot import (
    ExampleSources,
    default_render_context,
    select_comment_examples,
    select_merge_examples,
)

DATA_DIR = Path("data/processed")
HAS_REAL_DATA = (DATA_DIR / "features_v1.parquet").exists()


# --------------------------------------------------------------------------- #
# Synthetic fixtures
# --------------------------------------------------------------------------- #

def _sources():
    pulls = pd.DataFrame([
        {"id": f"p{i}", "title": f"Title {i}"} for i in range(12)
    ])
    files = pd.DataFrame([
        {"pr_id": f"p{i}", "filename": "a.py", "status": "modified",
         "additions": 1, "deletions": 0, "patch": f"@@ diff for p{i} @@"}
        for i in range(12)
    ])
    commits = pd.DataFrame([
        {"pr_id": f"p{i}", "sha": "s", "message": "msg", "author_name": "n",
         "author_email": "e", "author_login": "l", "additions": 1, "deletions": 0}
        for i in range(12)
    ])
    review_comments = pd.DataFrame([
        # human, substantive -> eligible
        {"pr_id": "p0", "author_login": "alice", "body": "Please add a guard clause for the empty case here."},
        {"pr_id": "p1", "author_login": "bob", "body": "This allocation looks unnecessary; can it be hoisted out of the loop?"},
        {"pr_id": "p2", "author_login": "carol", "body": "Rename this to something descriptive; single letters are hard to follow."},
        {"pr_id": "p3", "author_login": "dan", "body": "Missing test coverage for the error path introduced in this change."},
        # bot / automation -> excluded
        {"pr_id": "p4", "author_login": "copilot-pull-request-reviewer", "body": "AI review comment that is quite long and detailed."},
        {"pr_id": "p5", "author_login": "github-actions", "body": "Automated status comment that should never be a demonstration."},
        # too short -> excluded
        {"pr_id": "p6", "author_login": "erin", "body": "LGTM"},
    ])
    return ExampleSources(pulls, files, commits, review_comments)


def _train_pool():
    # 12 PRs, 8 merged / 4 closed
    rows = []
    for i in range(12):
        rows.append({"pr_id": f"p{i}", "repo": "r/x", "y": 1 if i % 3 != 0 else 0})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Merge examples
# --------------------------------------------------------------------------- #

def test_merge_examples_count_and_labels():
    ex = select_merge_examples(_train_pool(), _sources(), k=4)
    assert len(ex) == 4
    assert all(e.label in {"MERGE", "CLOSE"} for e in ex)
    assert all(e.target_comment is None for e in ex)
    # context rendered
    assert all(e.context_text for e in ex)


def test_merge_examples_balanced_covers_both_classes():
    ex = select_merge_examples(_train_pool(), _sources(), k=4, balanced=True)
    labels = [e.label for e in ex]
    assert "MERGE" in labels and "CLOSE" in labels
    # equal-as-possible split for even k
    assert labels.count("MERGE") == 2 and labels.count("CLOSE") == 2


def test_merge_examples_unbalanced_follows_base_rate():
    # base rate here is 8/12 merged; an unbalanced draw should skew merged
    ex = select_merge_examples(_train_pool(), _sources(), k=6, balanced=False, seed=1)
    labels = [e.label for e in ex]
    assert labels.count("MERGE") >= labels.count("CLOSE")


def test_merge_examples_deterministic():
    a = select_merge_examples(_train_pool(), _sources(), k=4, seed=7)
    b = select_merge_examples(_train_pool(), _sources(), k=4, seed=7)
    assert [e.pr_id for e in a] == [e.pr_id for e in b]


def test_merge_examples_seed_changes_selection():
    a = select_merge_examples(_train_pool(), _sources(), k=4, seed=1)
    b = select_merge_examples(_train_pool(), _sources(), k=4, seed=999)
    # not a hard guarantee for tiny pools, but should differ for these
    assert [e.pr_id for e in a] != [e.pr_id for e in b]


def test_merge_examples_only_from_pool():
    pool = _train_pool()
    ex = select_merge_examples(pool, _sources(), k=6)
    assert set(e.pr_id for e in ex) <= set(pool["pr_id"])


# --------------------------------------------------------------------------- #
# Comment examples
# --------------------------------------------------------------------------- #

def test_comment_examples_exclude_bots_and_short():
    ex = select_comment_examples(_train_pool(), _sources(), k=10)
    ids = {e.pr_id for e in ex}
    # eligible human, substantive comments were on p0..p3
    assert ids <= {"p0", "p1", "p2", "p3"}
    # bots (p4,p5) and the short "LGTM" (p6) excluded
    assert ids.isdisjoint({"p4", "p5", "p6"})
    assert all(e.target_comment and e.label is None for e in ex)


def test_comment_examples_deterministic():
    a = select_comment_examples(_train_pool(), _sources(), k=3, seed=5)
    b = select_comment_examples(_train_pool(), _sources(), k=3, seed=5)
    assert [e.pr_id for e in a] == [e.pr_id for e in b]


def test_comment_examples_none_available_returns_empty():
    src = _sources()
    src.review_comments = src.review_comments.iloc[0:0]
    assert select_comment_examples(_train_pool(), src, k=4) == []


# --------------------------------------------------------------------------- #
# default_render_context
# --------------------------------------------------------------------------- #

def test_default_render_context_includes_title_and_diff():
    text = default_render_context("p0", _sources())
    assert "Title 0" in text and "diff for p0" in text


def test_default_render_context_truncates_long_diff():
    src = _sources()
    src.files_changed = pd.DataFrame([{
        "pr_id": "pX", "filename": "big.py", "status": "modified",
        "additions": 1, "deletions": 0, "patch": "x" * 5000,
    }])
    src.pull_requests = pd.DataFrame([{"id": "pX", "title": "Big"}])
    text = default_render_context("pX", src)
    assert "[diff truncated]" in text
    assert len(text) < 5000


# --------------------------------------------------------------------------- #
# Real-data leakage guarantee (skipped if the dataset isn't present)
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(not HAS_REAL_DATA, reason="mined dataset not present")
def test_real_data_examples_disjoint_from_exp2_test_set():
    """The property that matters: demonstrations never touch the Experiment-3
    test set (== Experiment-2 test split)."""
    from src.llm.prompts.few_shot import load_sources, load_train_pool
    from src.ml.common import load_variant, split_by_repo_time

    _train, test, _ = split_by_repo_time(load_variant("v1"))
    test_ids = set(test["pr_id"])

    pool = load_train_pool()
    sources = load_sources()

    merge_ex = select_merge_examples(pool, sources, k=6)
    comment_ex = select_comment_examples(pool, sources, k=6)

    assert merge_ex, "expected some merge examples from real data"
    assert comment_ex, "expected some comment examples from real data"
    for e in merge_ex + comment_ex:
        assert e.pr_id in set(pool["pr_id"])   # from the train pool
        assert e.pr_id not in test_ids          # never from the scored test set


@pytest.mark.skipif(not HAS_REAL_DATA, reason="mined dataset not present")
def test_real_data_comment_examples_are_human_authored():
    from src.llm.prompts.few_shot import _NON_HUMAN_COMMENT_AUTHORS, load_sources, load_train_pool

    pool = load_train_pool()
    sources = load_sources()
    ex = select_comment_examples(pool, sources, k=8)

    rc = sources.review_comments
    banned = {a.lower() for a in _NON_HUMAN_COMMENT_AUTHORS}
    for e in ex:
        authors = rc.loc[rc["pr_id"] == e.pr_id, "author_login"].fillna("").str.lower()
        # the chosen comment's author is human (not in the AI/automation ban list)
        assert e.target_comment
        assert any(a not in banned for a in authors)
