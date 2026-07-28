import numpy as np
import pandas as pd

from src.mining.analyze_exp1 import (
    build_summary,
    make_all_figures,
    summarize_ai_reviewer_presence,
    summarize_ai_vs_human,
    summarize_labels,
    summarize_merge_rate_by_authorship,
    summarize_merge_status,
    summarize_pr_length,
    summarize_review_comment_counts,
    summarize_reviewer_counts,
)


def _pr_df(rows):
    """Build a minimal pull_requests-shaped DataFrame from short dicts."""
    defaults = {
        "additions": 0, "deletions": 0, "labels": np.array([]),
        "is_ai_authored": False, "is_ai_reviewed": False,
    }
    full_rows = [{**defaults, **r} for r in rows]
    return pd.DataFrame(full_rows)


def test_summarize_merge_status_counts_and_rate():
    pr = _pr_df([
        {"id": "1", "repo": "a/x", "is_merged": True},
        {"id": "2", "repo": "a/x", "is_merged": True},
        {"id": "3", "repo": "a/x", "is_merged": False},
        {"id": "4", "repo": "b/y", "is_merged": True},
    ])
    summary = summarize_merge_status(pr)
    assert summary["pooled"] == {"merged": 3, "not_merged": 1, "total": 4, "merge_rate": 0.75}
    assert summary["per_repo"]["a/x"] == {"merged": 2, "not_merged": 1, "total": 3, "merge_rate": round(2 / 3, 4)}
    assert summary["per_repo"]["b/y"] == {"merged": 1, "not_merged": 0, "total": 1, "merge_rate": 1.0}
    # every value must be a native Python type (JSON-serializable without a fallback)
    for v in summary["pooled"].values():
        assert type(v) in (int, float)


def test_summarize_review_comment_counts_fills_zero_for_prs_with_no_comments():
    pr = _pr_df([
        {"id": "1", "repo": "a/x", "is_merged": True},
        {"id": "2", "repo": "a/x", "is_merged": True},
    ])
    review_comments = pd.DataFrame([{"pr_id": "1"}, {"pr_id": "1"}, {"pr_id": "1"}])
    summary, per_pr = summarize_review_comment_counts(pr, review_comments)
    counts = dict(zip(per_pr["id"], per_pr["n_review_comments"]))
    assert counts == {"1": 3, "2": 0}
    assert summary["pooled"]["max"] == 3.0
    assert summary["pooled"]["zero_count"] == 1


def test_summarize_reviewer_counts_deduplicates_repeat_reviewers():
    pr = _pr_df([{"id": "1", "repo": "a/x", "is_merged": True}])
    reviews = pd.DataFrame([
        {"pr_id": "1", "reviewer_login": "alice"},
        {"pr_id": "1", "reviewer_login": "alice"},  # same reviewer, second review
        {"pr_id": "1", "reviewer_login": "bob"},
    ])
    summary, per_pr = summarize_reviewer_counts(pr, reviews)
    assert per_pr.loc[per_pr["id"] == "1", "n_reviewers"].iloc[0] == 2


def test_summarize_pr_length_sums_additions_and_deletions():
    pr = _pr_df([
        {"id": "1", "repo": "a/x", "is_merged": True, "additions": 10, "deletions": 5},
        {"id": "2", "repo": "a/x", "is_merged": True, "additions": 0, "deletions": 0},
    ])
    summary, per_pr = summarize_pr_length(pr)
    assert list(per_pr["pr_length"]) == [15, 0]
    assert summary["pooled"]["max"] == 15.0
    assert summary["pooled"]["zero_count"] == 1


def test_summarize_labels_counts_and_excludes_empty():
    pr = _pr_df([
        {"id": "1", "repo": "a/x", "is_merged": True, "labels": np.array(["bug", "p1"])},
        {"id": "2", "repo": "a/x", "is_merged": True, "labels": np.array(["bug"])},
        {"id": "3", "repo": "b/y", "is_merged": True, "labels": np.array([])},
    ])
    summary = summarize_labels(pr, top_n=5, top_n_per_repo=5)
    pooled = {d["label"]: d["count"] for d in summary["pooled_top"]}
    assert pooled == {"bug": 2, "p1": 1}
    assert summary["total_labeled_prs"] == 2
    assert summary["total_prs"] == 3
    assert "b/y" not in summary["per_repo_top"] or summary["per_repo_top"]["b/y"] == []


def test_summarize_ai_vs_human_counts():
    pr = _pr_df([
        {"id": "1", "repo": "a/x", "is_merged": True, "is_ai_authored": True},
        {"id": "2", "repo": "a/x", "is_merged": True, "is_ai_authored": False},
        {"id": "3", "repo": "a/x", "is_merged": True, "is_ai_authored": False},
    ])
    summary = summarize_ai_vs_human(pr)
    assert summary["pooled"] == {"ai_authored": 1, "human_authored": 2, "total": 3, "ai_share": round(1 / 3, 4)}


def test_summarize_ai_reviewer_presence():
    pr = _pr_df([
        {"id": "1", "repo": "a/x", "is_merged": True, "is_ai_reviewed": True},
        {"id": "2", "repo": "a/x", "is_merged": True, "is_ai_reviewed": False},
    ])
    summary = summarize_ai_reviewer_presence(pr)
    assert summary["pooled"] == {"ai_reviewed": 1, "not_ai_reviewed": 1, "total": 2, "share": 0.5}


def test_summarize_merge_rate_by_authorship_handles_empty_group():
    # A repo with zero AI-authored PRs must not crash (division-by-zero guarded).
    pr = _pr_df([
        {"id": "1", "repo": "a/x", "is_merged": True, "is_ai_authored": False},
        {"id": "2", "repo": "a/x", "is_merged": False, "is_ai_authored": False},
    ])
    summary = summarize_merge_rate_by_authorship(pr)
    assert summary["pooled"]["human_authored"] == {"n": 2, "merge_rate": 0.5}
    assert summary["pooled"]["ai_authored"] == {"n": 0, "merge_rate": None}


def test_build_summary_is_fully_json_serializable():
    import json
    pr = _pr_df([
        {"id": "1", "repo": "a/x", "is_merged": True, "is_ai_authored": True, "is_ai_reviewed": True,
         "additions": 5, "deletions": 1, "labels": np.array(["bug"])},
        {"id": "2", "repo": "b/y", "is_merged": False, "is_ai_authored": False, "is_ai_reviewed": False,
         "additions": 0, "deletions": 0, "labels": np.array([])},
    ])
    reviews = pd.DataFrame([{"pr_id": "1", "reviewer_login": "alice"}])
    review_comments = pd.DataFrame([{"pr_id": "1"}])
    data = {"pull_requests": pr, "reviews": reviews, "review_comments": review_comments}

    summary = build_summary(data)
    # must round-trip through json.dumps with NO default=str fallback needed
    encoded = json.dumps(summary)
    assert '"total_prs": 2' in encoded or '"total_prs":2' in encoded
    assert summary["dataset"]["prs_per_repo"] == {"a/x": 1, "b/y": 1}
    assert "class_balance_note" in summary


def test_make_all_figures_smoke_test_produces_nonempty_pngs(tmp_path):
    # Small, deliberately uneven dataset: one repo has zero AI-authored PRs and
    # zero labels, to make sure the plotting code doesn't crash on empty groups.
    pr = _pr_df([
        {"id": "1", "repo": "a/x", "is_merged": True, "is_ai_authored": True, "is_ai_reviewed": True,
         "additions": 50, "deletions": 10, "labels": np.array(["bug"])},
        {"id": "2", "repo": "a/x", "is_merged": False, "is_ai_authored": False, "is_ai_reviewed": False,
         "additions": 3, "deletions": 1, "labels": np.array([])},
        {"id": "3", "repo": "b/y", "is_merged": True, "is_ai_authored": False, "is_ai_reviewed": False,
         "additions": 0, "deletions": 0, "labels": np.array([])},
    ])
    reviews = pd.DataFrame([{"pr_id": "1", "reviewer_login": "copilot-pull-request-reviewer"}])
    review_comments = pd.DataFrame([{"pr_id": "1"}])
    data = {"pull_requests": pr, "reviews": reviews, "review_comments": review_comments}

    paths = make_all_figures(data, figures_dir=tmp_path)
    assert len(paths) == 8
    for p in paths:
        assert p.exists()
        assert p.stat().st_size > 0
