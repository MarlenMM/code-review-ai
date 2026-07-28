"""Tests for Experiment 4 context population (`src/llm/exp4_context.py`).

`build_augmented_context` is the Step-20 counterpart to `context_builder.py`'s
`build_context`: these tests check it assembles every `AugmentedContext` slot
correctly from synthetic parquet-shaped frames, truncates per the module's
caps, and calls out to a `github_client` for the issue slot only when given
one -- never silently, and never in a way that can abort context building for
the whole PR if the fetch fails or the reference doesn't resolve to a real
issue.
"""

from pathlib import Path

import pandas as pd
import pytest

from src.llm.exp4_context import build_augmented_context, load_sources
from src.llm.prompts.few_shot import ExampleSources

DATA_DIR = Path("data/processed")
HAS_REAL_DATA = (DATA_DIR / "pull_requests.parquet").exists()


def _sources() -> ExampleSources:
    pulls = pd.DataFrame([
        {
            "id": "q", "repo": "acme/widgets", "owner": "acme", "repo_name": "widgets",
            "title": "Fix spacing", "body": "Fixes #42. Please review the spacing change.",
        },
        {"id": "t1", "repo": "acme/widgets", "owner": "acme", "repo_name": "widgets", "title": "Past PR"},
    ])
    files = pd.DataFrame([
        {"pr_id": "q", "filename": "src/core/a.py", "status": "modified",
         "additions": 3, "deletions": 1, "patch": "@@ -1,2 +1,3 @@ def f\n+x\n"},
        {"pr_id": "q", "filename": "src/core/b.py", "status": "modified",
         "additions": 1, "deletions": 0, "patch": "@@ -1 +1 @@ class C\n+y\n"},
        {"pr_id": "t1", "filename": "src/core/c.py", "status": "modified",
         "additions": 1, "deletions": 0, "patch": "@@ -1 +1 @@\n+z\n"},
    ])
    commits = pd.DataFrame([
        {"pr_id": "q", "sha": "s1", "message": "fix spacing bug"},
        {"pr_id": "t1", "sha": "s2", "message": "past fix"},
    ])
    review_comments = pd.DataFrame([
        {"pr_id": "t1", "author_login": "alice",
         "body": "This needs a guard for the empty-input case before indexing."},
    ])
    return ExampleSources(pulls, files, commits, review_comments)


def _train_pool() -> pd.DataFrame:
    return pd.DataFrame([{"pr_id": "t1", "repo": "acme/widgets"}])


class FakeGitHubClient:
    """A stand-in for `GitHubClient` that never touches the network -- records
    calls and returns scripted `fetch_issue` records."""

    def __init__(self, record):
        self.record = record
        self.calls: list[tuple[str, str, int]] = []

    def fetch_issue(self, owner, repo, number):
        self.calls.append((owner, repo, number))
        return self.record


# --------------------------------------------------------------------------- #
# Component assembly (no github_client)
# --------------------------------------------------------------------------- #

def test_build_augmented_context_assembles_all_offline_slots():
    ctx = build_augmented_context("q", _sources(), _train_pool(), github_client=None)
    assert "def f" in ctx.diff
    assert "Fixes #42" in ctx.pr_description
    assert "fix spacing bug" in ctx.commit_message
    assert "src/core/a.py" in ctx.repo_context and "class C" in ctx.repo_context
    assert ctx.issue_text is None          # no github_client given
    assert "empty-input case" in ctx.historical_comments
    assert ctx.repo == "acme/widgets" and ctx.title == "Fix spacing"


def test_build_augmented_context_truncates_long_description():
    sources = _sources()
    sources.pull_requests.loc[sources.pull_requests["id"] == "q", "body"] = "x" * 5000
    ctx = build_augmented_context("q", sources, _train_pool(), max_section_chars=100)
    assert len(ctx.pr_description) <= 100 + len(" ... [truncated for length]")
    assert "[truncated for length]" in ctx.pr_description


def test_build_augmented_context_missing_pr_description_is_none():
    sources = _sources()
    sources.pull_requests.loc[sources.pull_requests["id"] == "q", "body"] = ""
    ctx = build_augmented_context("q", sources, _train_pool())
    assert ctx.pr_description is None


def test_build_augmented_context_no_repo_context_when_no_files():
    sources = _sources()
    sources.files_changed = sources.files_changed[sources.files_changed["pr_id"] != "q"]
    ctx = build_augmented_context("q", sources, _train_pool())
    assert ctx.repo_context is None
    assert ctx.diff == "(no file diff available)"


# --------------------------------------------------------------------------- #
# Issue resolution
# --------------------------------------------------------------------------- #

def test_issue_text_fetched_when_client_given():
    client = FakeGitHubClient({
        "not_found": False, "number": 42, "title": "Spacing bug",
        "body": "The spacing is off by one.", "is_pull_request": False,
    })
    ctx = build_augmented_context("q", _sources(), _train_pool(), github_client=client)
    assert ctx.issue_text is not None
    assert "Issue #42" in ctx.issue_text
    assert "Spacing bug" in ctx.issue_text
    assert client.calls == [("acme", "widgets", 42)]


def test_issue_text_none_when_reference_is_a_pull_request():
    client = FakeGitHubClient({"not_found": False, "number": 42, "title": "t", "body": "b", "is_pull_request": True})
    ctx = build_augmented_context("q", _sources(), _train_pool(), github_client=client)
    assert ctx.issue_text is None


def test_issue_text_none_when_not_found():
    client = FakeGitHubClient({"not_found": True})
    ctx = build_augmented_context("q", _sources(), _train_pool(), github_client=client)
    assert ctx.issue_text is None


def test_issue_text_none_when_no_reference_in_body():
    sources = _sources()
    sources.pull_requests.loc[sources.pull_requests["id"] == "q", "body"] = "No issue mentioned here."
    client = FakeGitHubClient({"not_found": False, "number": 1, "title": "t", "body": "b", "is_pull_request": False})
    ctx = build_augmented_context("q", sources, _train_pool(), github_client=client)
    assert ctx.issue_text is None
    assert client.calls == []          # never called -- no reference to resolve


def test_issue_fetch_failure_does_not_abort_context_building():
    class ExplodingClient:
        def fetch_issue(self, owner, repo, number):
            raise RuntimeError("simulated network failure")

    ctx = build_augmented_context("q", _sources(), _train_pool(), github_client=ExplodingClient())
    assert ctx.issue_text is None          # degrades gracefully
    assert ctx.diff                        # the rest of the context still built


def test_issue_text_truncated_to_section_cap():
    client = FakeGitHubClient({
        "not_found": False, "number": 42, "title": "t", "body": "y" * 2000, "is_pull_request": False,
    })
    ctx = build_augmented_context("q", _sources(), _train_pool(), github_client=client, max_section_chars=50)
    assert len(ctx.issue_text) <= 50 + len(" ... [truncated for length]")


# --------------------------------------------------------------------------- #
# Real-data smoke test (skipped if the mined dataset isn't present)
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(not HAS_REAL_DATA, reason="mined dataset not present")
def test_real_data_builds_context_for_an_ai_test_pr():
    from src.llm.prompts.exp4_examples import load_ai_train_pool
    from src.ml.common import split_by_repo_time

    sources = load_sources()
    train = load_ai_train_pool()

    pr = sources.pull_requests
    ai = pr[(pr["is_ai_authored"] == True) & pr["is_merged"].notna()].copy()  # noqa: E712
    pool = pd.DataFrame({
        "pr_id": ai["id"].values, "repo": ai["repo"].values,
        "y": ai["is_merged"].astype(int).values,
        "created_at": pd.to_datetime(ai["created_at"].values, utc=True),
    })
    _tr, test, _ = split_by_repo_time(pool)
    pr_id = test["pr_id"].iloc[0]

    ctx = build_augmented_context(pr_id, sources, train, github_client=None)
    assert ctx.diff
    assert ctx.repo is not None
    # the assembled COMPLETE tier stays well under Groq's per-request budget
    from src.llm.prompts.context_aug import Exp4ContextTier, render_tier
    rendered = render_tier(ctx, Exp4ContextTier.COMPLETE_SE_CONTEXT)
    assert len(rendered.context_text) < 6_000     # ~1,500 tokens, comfortable headroom
