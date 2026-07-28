"""Tests for `src/llm/context_builder.py`: the four context-kind renderers,
truncation behavior, and the structural-typing seam with
`src/llm/prompts/few_shot.py`'s `ExampleSources`.
"""

from dataclasses import fields
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.llm.context_builder import (
    ContextKind,
    ContextSources,
    MAX_DIFF_CHARS,
    build_commit_messages_text,
    build_context,
    build_diff_text,
    build_metadata_text,
    render_context_text,
)

DATA_DIR = Path("data/processed")
HAS_REAL_DATA = (DATA_DIR / "pull_requests.parquet").exists()


def _sources(**overrides):
    pull_requests = pd.DataFrame([{
        "id": "p1", "repo": "acme/widgets", "title": "Fix off-by-one",
        "body": "Fixes the loop boundary.", "labels": np.array(["bug", "backend"]),
        "additions": 10, "deletions": 2,
    }])
    files_changed = pd.DataFrame([
        {"pr_id": "p1", "filename": "a.py", "status": "modified",
         "additions": 8, "deletions": 1, "patch": "@@ -1,2 +1,2 @@\n-old\n+new"},
        {"pr_id": "p1", "filename": "b.py", "status": "added",
         "additions": 2, "deletions": 1, "patch": "@@ -0,0 +1,2 @@\n+x\n+y"},
    ])
    commits = pd.DataFrame([
        {"pr_id": "p1", "sha": "s1", "message": "Fix loop boundary", "author_name": "a",
         "author_email": "a@x.com", "author_login": "alice", "additions": 8, "deletions": 1},
        {"pr_id": "p1", "sha": "s2", "message": "Add test", "author_name": "a",
         "author_email": "a@x.com", "author_login": "alice", "additions": 2, "deletions": 1},
    ])
    base = {"pull_requests": pull_requests, "files_changed": files_changed, "commits": commits}
    base.update(overrides)
    return ContextSources(**base)


# --------------------------------------------------------------------------- #
# build_diff_text
# --------------------------------------------------------------------------- #

def test_build_diff_text_includes_all_files():
    text = build_diff_text("p1", _sources())
    assert "a.py" in text and "b.py" in text
    assert "+new" in text and "+x" in text


def test_build_diff_text_orders_largest_churn_first():
    # a.py has churn 9 (8+1), b.py has churn 3 (2+1) -> a.py should come first
    text = build_diff_text("p1", _sources())
    assert text.index("a.py") < text.index("b.py")


def test_build_diff_text_no_files_returns_placeholder():
    sources = _sources(files_changed=pd.DataFrame(
        columns=["pr_id", "filename", "status", "additions", "deletions", "patch"]
    ))
    assert build_diff_text("p1", sources) == "(no file diff available)"


def test_build_diff_text_missing_patch_still_shows_header():
    files = pd.DataFrame([{
        "pr_id": "p1", "filename": "binary.png", "status": "modified",
        "additions": 0, "deletions": 0, "patch": None,
    }])
    sources = _sources(files_changed=files)
    text = build_diff_text("p1", sources)
    assert "binary.png" in text and "no textual diff" in text


def test_build_diff_text_per_file_truncation():
    files = pd.DataFrame([{
        "pr_id": "p1", "filename": "big.py", "status": "modified",
        "additions": 1000, "deletions": 0, "patch": "x" * 10_000,
    }])
    sources = _sources(files_changed=files)
    text = build_diff_text("p1", sources, max_file_chars=100)
    assert "file diff truncated" in text
    assert len(text) < 10_000


def test_build_diff_text_total_cap_omits_smaller_files():
    files = pd.DataFrame([
        {"pr_id": "p1", "filename": f"f{i}.py", "status": "modified",
         "additions": 100 - i, "deletions": 0, "patch": "y" * 500}
        for i in range(10)
    ])
    sources = _sources(files_changed=files)
    text = build_diff_text("p1", sources, max_total_chars=1200, max_file_chars=500)
    assert "omitted for length" in text
    assert len(text) < 1200 + 200  # cap + reasonable marker overhead


# --------------------------------------------------------------------------- #
# build_commit_messages_text
# --------------------------------------------------------------------------- #

def test_build_commit_messages_numbered_and_ordered():
    text = build_commit_messages_text("p1", _sources())
    assert text.startswith("1. Fix loop boundary")
    assert "2. Add test" in text


def test_build_commit_messages_no_commits():
    sources = _sources(commits=pd.DataFrame(
        columns=["pr_id", "sha", "message", "author_name", "author_email",
                 "author_login", "additions", "deletions"]
    ))
    assert build_commit_messages_text("p1", sources) == "(no commit messages available)"


def test_build_commit_messages_skips_empty_message():
    commits = pd.DataFrame([
        {"pr_id": "p1", "sha": "s1", "message": "", "author_name": "a", "author_email": "a",
         "author_login": "a", "additions": 0, "deletions": 0},
        {"pr_id": "p1", "sha": "s2", "message": "a real message", "author_name": "a", "author_email": "a",
         "author_login": "a", "additions": 0, "deletions": 0},
    ])
    sources = _sources(commits=commits)
    text = build_commit_messages_text("p1", sources)
    assert text.startswith("2.")  # empty first commit skipped, numbering preserved


def test_build_commit_messages_truncates_and_reports_omission():
    commits = pd.DataFrame([
        {"pr_id": "p1", "sha": "s1", "message": "z" * 30, "author_name": "a", "author_email": "a",
         "author_login": "a", "additions": 0, "deletions": 0},
        {"pr_id": "p1", "sha": "s2", "message": "y" * 500, "author_name": "a", "author_email": "a",
         "author_login": "a", "additions": 0, "deletions": 0},
    ])
    sources = _sources(commits=commits)
    text = build_commit_messages_text("p1", sources, max_chars=50)
    assert text.startswith("1.")       # the short first message fits
    assert "omitted for length" in text  # the long second message does not


# --------------------------------------------------------------------------- #
# build_metadata_text
# --------------------------------------------------------------------------- #

def test_build_metadata_text_includes_labels_and_files():
    text = build_metadata_text("p1", _sources())
    assert "bug" in text and "backend" in text
    assert "a.py" in text and "b.py" in text
    assert "2 file(s) changed" in text


def test_build_metadata_text_no_labels():
    pull_requests = pd.DataFrame([{
        "id": "p1", "repo": "acme/widgets", "title": "T", "body": "",
        "labels": np.array([]), "additions": 1, "deletions": 0,
    }])
    sources = _sources(pull_requests=pull_requests)
    text = build_metadata_text("p1", sources)
    assert "(none)" in text


def test_build_metadata_text_caps_file_list():
    files = pd.DataFrame([
        {"pr_id": "p1", "filename": f"f{i}.py", "status": "modified",
         "additions": 1, "deletions": 0, "patch": "x"}
        for i in range(5)
    ])
    sources = _sources(files_changed=files)
    text = build_metadata_text("p1", sources, max_files_listed=3)
    assert "and 2 more file(s)" in text


# --------------------------------------------------------------------------- #
# build_context: assembly per kind
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("kind", list(ContextKind))
def test_build_context_returns_populated_prompt_context(kind):
    ctx = build_context("p1", kind, _sources())
    assert ctx.repo == "acme/widgets"
    assert ctx.title == "Fix off-by-one"
    assert ctx.context_kind == kind.value
    assert "a.py" in ctx.context_text  # the diff is always present


def test_diff_only_has_no_description_or_metadata_leakage():
    ctx = build_context("p1", ContextKind.DIFF_ONLY, _sources())
    assert "PR description" not in ctx.context_text
    assert "Labels" not in ctx.context_text
    assert "Commit messages" not in ctx.context_text


def test_diff_description_includes_body():
    ctx = build_context("p1", ContextKind.DIFF_DESCRIPTION, _sources())
    assert "Fixes the loop boundary." in ctx.context_text


def test_diff_description_empty_body_is_explicit():
    pull_requests = pd.DataFrame([{
        "id": "p1", "repo": "acme/widgets", "title": "T", "body": "",
        "labels": np.array([]), "additions": 1, "deletions": 0,
    }])
    sources = _sources(pull_requests=pull_requests)
    ctx = build_context("p1", ContextKind.DIFF_DESCRIPTION, sources)
    assert "no PR description provided" in ctx.context_text


def test_diff_commit_message_includes_commits():
    ctx = build_context("p1", ContextKind.DIFF_COMMIT_MESSAGE, _sources())
    assert "Fix loop boundary" in ctx.context_text
    assert "Add test" in ctx.context_text


def test_diff_metadata_includes_labels_and_files():
    ctx = build_context("p1", ContextKind.DIFF_METADATA, _sources())
    assert "bug" in ctx.context_text
    assert "b.py" in ctx.context_text


def test_unknown_pr_id_raises_keyerror():
    with pytest.raises(KeyError):
        build_context("does-not-exist", ContextKind.DIFF_ONLY, _sources())


def test_max_context_chars_caps_whole_assembled_context():
    # a long PR body would blow past a per-request token limit even with a
    # capped diff; max_context_chars is the whole-context safety net
    long_body = "x" * 20_000
    pull_requests = pd.DataFrame([{
        "id": "p1", "repo": "acme/widgets", "title": "T", "body": long_body,
        "labels": np.array(["bug"]), "additions": 1, "deletions": 0,
    }])
    sources = _sources(pull_requests=pull_requests)

    uncapped = build_context("p1", ContextKind.DIFF_DESCRIPTION, sources)
    assert len(uncapped.context_text) > 10_000  # body dominates, uncapped

    capped = build_context("p1", ContextKind.DIFF_DESCRIPTION, sources, max_context_chars=2500)
    assert len(capped.context_text) < 2500 + 100  # cap + marker overhead
    assert "context truncated" in capped.context_text


def test_max_context_chars_none_is_a_noop():
    sources = _sources()
    a = build_context("p1", ContextKind.DIFF_ONLY, sources, max_context_chars=None)
    b = build_context("p1", ContextKind.DIFF_ONLY, sources)  # default is None
    assert a.context_text == b.context_text
    assert "context truncated" not in a.context_text


# --------------------------------------------------------------------------- #
# No review-process leakage: ContextSources structurally cannot carry it
# --------------------------------------------------------------------------- #

def test_context_sources_has_no_review_process_tables():
    field_names = {f.name for f in fields(ContextSources)}
    assert field_names == {"pull_requests", "files_changed", "commits"}
    assert "reviews" not in field_names
    assert "review_comments" not in field_names
    assert "issue_comments" not in field_names


# --------------------------------------------------------------------------- #
# render_context_text: the few_shot.py-compatible callable
# --------------------------------------------------------------------------- #

def test_render_context_text_defaults_to_diff_only():
    sources = _sources()
    assert render_context_text("p1", sources) == build_context(
        "p1", ContextKind.DIFF_ONLY, sources
    ).context_text


def test_render_context_text_bindable_to_a_specific_kind():
    sources = _sources()
    render = partial(render_context_text, kind=ContextKind.DIFF_METADATA)
    text = render("p1", sources)  # exactly the (pr_id, sources) shape few_shot expects
    assert "Labels" in text


def test_render_context_text_accepts_ExampleSources_via_duck_typing():
    """few_shot.ExampleSources has an extra `review_comments` field beyond
    what context_builder needs -- confirms the documented structural-typing
    seam between the two modules actually works, not just in the docstring."""
    from src.llm.prompts.few_shot import ExampleSources

    sources = _sources()
    example_sources = ExampleSources(
        pull_requests=sources.pull_requests,
        files_changed=sources.files_changed,
        commits=sources.commits,
        review_comments=pd.DataFrame(columns=["pr_id", "author_login", "body"]),
    )
    text = render_context_text("p1", example_sources)
    assert "a.py" in text


# --------------------------------------------------------------------------- #
# Real-data smoke tests (skipped if the mined dataset isn't present)
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(not HAS_REAL_DATA, reason="mined dataset not present")
def test_real_data_all_kinds_render_for_sample_test_prs():
    from src.llm.context_builder import load_sources
    from src.ml.common import load_variant, split_by_repo_time

    sources = load_sources()
    _train, test, _ = split_by_repo_time(load_variant("v1"))
    sample = test["pr_id"].head(15)

    for pr_id in sample:
        for kind in ContextKind:
            ctx = build_context(pr_id, kind, sources)
            assert ctx.context_text
            assert ctx.context_kind == kind.value


@pytest.mark.skipif(not HAS_REAL_DATA, reason="mined dataset not present")
def test_real_data_truncation_cap_holds_on_worst_case_pr():
    from src.llm.context_builder import load_sources

    sources = load_sources()
    fc = sources.files_changed
    per_pr_len = fc.assign(l=fc["patch"].fillna("").str.len()).groupby("pr_id")["l"].sum()
    biggest_pr = per_pr_len.idxmax()
    assert per_pr_len.max() > MAX_DIFF_CHARS  # the outlier really is bigger than the cap

    ctx = build_context(biggest_pr, ContextKind.DIFF_ONLY, sources)
    # rendered text is bounded (cap + small header/marker overhead), never the raw size
    assert len(ctx.context_text) < MAX_DIFF_CHARS + 500
    assert "omitted for length" in ctx.context_text
