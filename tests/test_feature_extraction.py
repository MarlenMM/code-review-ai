import math

import pandas as pd
import pytest

from src.features.feature_extraction import (
    M_COLUMNS,
    P_COLUMNS,
    S_COLUMNS,
    T_COLUMNS,
    V0_FEATURE_COLUMNS,
    V1_FEATURE_COLUMNS,
    V2_FEATURE_COLUMNS,
    build_author_history_features,
    build_modification_features,
    build_process_features,
    build_textual_features,
    load_structure_features,
    select_population,
)


# --------------------------------------------------------------------------- #
# Feature-set contract (protects against silent drift between the three variants)
# --------------------------------------------------------------------------- #

def test_v1_is_exactly_m_s_t():
    assert set(V1_FEATURE_COLUMNS) == set(M_COLUMNS) | set(S_COLUMNS) | set(T_COLUMNS)


def test_v2_drops_growing_features_and_adds_author_history():
    assert "m_num_commits" not in V2_FEATURE_COLUMNS
    assert "t_commit_msg_len_total" not in V2_FEATURE_COLUMNS
    assert "author_prior_merge_rate" in V2_FEATURE_COLUMNS
    assert "author_prior_pr_count" in V2_FEATURE_COLUMNS
    # everything else from V1 survives into V2
    untouched = set(V1_FEATURE_COLUMNS) - {"m_num_commits", "t_commit_msg_len_total"}
    assert untouched.issubset(set(V2_FEATURE_COLUMNS))


def test_v0_is_v1_plus_forbidden_process_features():
    assert set(V1_FEATURE_COLUMNS).issubset(set(V0_FEATURE_COLUMNS))
    assert set(P_COLUMNS).issubset(set(V0_FEATURE_COLUMNS))


def test_forbidden_columns_never_leak_into_v1_or_v2():
    forbidden = set(P_COLUMNS)
    assert forbidden.isdisjoint(V1_FEATURE_COLUMNS)
    assert forbidden.isdisjoint(V2_FEATURE_COLUMNS)


# --------------------------------------------------------------------------- #
# select_population
# --------------------------------------------------------------------------- #

def _pr_fixture():
    return pd.DataFrame([
        {"id": "PR1", "repo": "r/a", "author_login": "alice", "created_at": "2026-01-01T00:00:00Z",
         "is_merged": True, "is_ai_authored": False, "state": "MERGED", "labels": [],
         "title": "Fix bug", "body": "Fixes #1", "additions": 10, "deletions": 2, "changed_files": 1,
         "is_ai_reviewed": False, "n_ai_reviewers": 0},
        {"id": "PR2", "repo": "r/a", "author_login": "bob", "created_at": "2026-01-02T00:00:00Z",
         "is_merged": False, "is_ai_authored": False, "state": "CLOSED", "labels": ["wontfix"],
         "title": "WIP", "body": "", "additions": 5, "deletions": 0, "changed_files": 1,
         "is_ai_reviewed": True, "n_ai_reviewers": 1},
        {"id": "PR3", "repo": "r/a", "author_login": "copilot-swe-agent", "created_at": "2026-01-03T00:00:00Z",
         "is_merged": True, "is_ai_authored": True, "state": "MERGED", "labels": [],
         "title": "AI PR", "body": "auto", "additions": 1, "deletions": 1, "changed_files": 1,
         "is_ai_reviewed": False, "n_ai_reviewers": 0},
    ])


def test_select_population_excludes_ai_authored():
    pop = select_population(_pr_fixture())
    assert set(pop["pr_id"]) == {"PR1", "PR2"}
    assert "PR3" not in set(pop["pr_id"])


def test_select_population_renames_and_types():
    pop = select_population(_pr_fixture())
    assert "y" in pop.columns and "is_merged" not in pop.columns
    assert pd.api.types.is_datetime64_any_dtype(pop["created_at"])
    row1 = pop.set_index("pr_id").loc["PR1"]
    assert bool(row1["y"]) is True


# --------------------------------------------------------------------------- #
# build_modification_features
# --------------------------------------------------------------------------- #

def test_build_modification_features_basic_counts():
    pr = _pr_fixture()
    fc = pd.DataFrame([
        {"pr_id": "PR1", "filename": "a.py", "status": "added",
         "patch": "@@ -0,0 +1,3 @@\n+x\n+y\n+z\n"},
        {"pr_id": "PR1", "filename": "b.py", "status": "modified",
         "patch": "@@ -1,1 +1,1 @@\n-old\n+new\n@@ -5,1 +5,1 @@\n-old2\n+new2\n"},
        {"pr_id": "PR2", "filename": "c.py", "status": "removed", "patch": None},
    ])
    cm = pd.DataFrame([
        {"pr_id": "PR1", "message": "commit one"},
        {"pr_id": "PR1", "message": "commit two"},
    ])
    m = build_modification_features(pd.Series(["PR1", "PR2"]), pr, fc, cm).set_index("pr_id")

    assert m.loc["PR1", "m_additions"] == 10
    assert m.loc["PR1", "m_deletions"] == 2
    assert m.loc["PR1", "m_churn"] == 12
    assert m.loc["PR1", "m_net_change"] == 8
    assert m.loc["PR1", "m_num_hunks"] == 3  # 1 in a.py + 2 in b.py
    assert m.loc["PR1", "m_files_added"] == 1
    assert m.loc["PR1", "m_files_modified"] == 1
    assert m.loc["PR1", "m_num_commits"] == 2

    # PR2 has no files_changed patch data and no commits -> zero-filled, not NaN
    assert m.loc["PR2", "m_num_hunks"] == 0
    assert m.loc["PR2", "m_num_commits"] == 0
    assert m.loc["PR2", "m_files_removed"] == 1


def test_build_modification_features_avoids_div_by_zero():
    pr = pd.DataFrame([{"id": "PR1", "additions": 0, "deletions": 0, "changed_files": 0}])
    fc = pd.DataFrame(columns=["pr_id", "filename", "status", "patch"])
    cm = pd.DataFrame(columns=["pr_id", "message"])
    m = build_modification_features(pd.Series(["PR1"]), pr, fc, cm).set_index("pr_id")
    assert m.loc["PR1", "m_avg_file_churn"] == 0
    assert m.loc["PR1", "m_frac_added"] == 0  # 0 / (0 + 1)


# --------------------------------------------------------------------------- #
# build_textual_features
# --------------------------------------------------------------------------- #

def test_build_textual_features():
    pr = pd.DataFrame([
        {"id": "PR1", "title": "Fix the bug", "body": "This fixes issue number one"},
        {"id": "PR2", "title": "x", "body": None},
    ])
    cm = pd.DataFrame([
        {"pr_id": "PR1", "message": "short"},
        {"pr_id": "PR1", "message": "a much longer commit message here"},
    ])
    t = build_textual_features(pd.Series(["PR1", "PR2"]), pr, cm).set_index("pr_id")

    assert t.loc["PR1", "t_title_wordcount"] == 3
    assert t.loc["PR1", "t_body_wordcount"] == 5
    assert t.loc["PR1", "t_body_is_empty"] == 0
    assert t.loc["PR1", "t_commit_msg_len_first"] == len("short")
    assert t.loc["PR1", "t_commit_msg_len_total"] == len("short") + len("a much longer commit message here")

    # PR2 has a null body and zero commits
    assert t.loc["PR2", "t_body_is_empty"] == 1
    assert t.loc["PR2", "t_body_len"] == 0
    assert t.loc["PR2", "t_commit_msg_len_first"] == 0
    assert t.loc["PR2", "t_commit_msg_len_mean"] == 0


# --------------------------------------------------------------------------- #
# load_structure_features
# --------------------------------------------------------------------------- #

def test_load_structure_features_fills_missing_prs():
    ast_cfg = pd.DataFrame([
        {"pr_id": "PR1", "s_ast_node_count": 10, "s_ast_max_depth": 3, "s_cfg_node_count": 2,
         "s_cfg_edge_count": 1, "s_cyclomatic_proxy": 1, "s_num_functions_touched": 1,
         "s_has_parseable_code": 1},
    ])
    s = load_structure_features(pd.Series(["PR1", "PR2"]), ast_cfg).set_index("pr_id")
    assert s.loc["PR1", "s_has_parseable_code"] == 1
    # PR2 absent from the Step 9 cache entirely -> zero-filled, not dropped
    assert s.loc["PR2", "s_has_parseable_code"] == 0
    assert s.loc["PR2", "s_ast_node_count"] == 0


# --------------------------------------------------------------------------- #
# build_process_features
# --------------------------------------------------------------------------- #

def test_build_process_features():
    pr = pd.DataFrame([
        {"id": "PR1", "labels": ["a", "b"], "is_ai_reviewed": True, "n_ai_reviewers": 2},
        {"id": "PR2", "labels": [], "is_ai_reviewed": False, "n_ai_reviewers": 0},
    ])
    rv = pd.DataFrame([
        {"pr_id": "PR1", "reviewer_login": "x", "state": "APPROVED"},
        {"pr_id": "PR1", "reviewer_login": "y", "state": "CHANGES_REQUESTED"},
        {"pr_id": "PR1", "reviewer_login": "x", "state": "COMMENTED"},
    ])
    rc = pd.DataFrame([{"pr_id": "PR1", "comment_id": "c1"}])
    ic = pd.DataFrame(columns=["pr_id", "comment_id"])

    p = build_process_features(pd.Series(["PR1", "PR2"]), pr, rv, rc, ic).set_index("pr_id")
    assert p.loc["PR1", "num_labels"] == 2
    assert p.loc["PR1", "p_num_reviews"] == 3
    assert p.loc["PR1", "p_num_reviewers"] == 2  # x, y distinct
    assert p.loc["PR1", "p_has_approved"] == 1
    assert p.loc["PR1", "p_has_changes_requested"] == 1
    assert p.loc["PR1", "p_num_review_comments"] == 1
    assert p.loc["PR1", "p_num_issue_comments"] == 0

    # PR2 has no reviews/comments at all -> zero-filled
    assert p.loc["PR2", "p_num_reviews"] == 0
    assert p.loc["PR2", "p_has_approved"] == 0


# --------------------------------------------------------------------------- #
# build_author_history_features
# --------------------------------------------------------------------------- #

def test_author_history_strictly_before_and_same_repo_only():
    population = pd.DataFrame([
        {"pr_id": "PR3", "repo": "r/a", "author_login": "alice",
         "created_at": pd.Timestamp("2026-03-01", tz="UTC")},
    ])
    all_prs = pd.DataFrame([
        {"repo": "r/a", "author_login": "alice", "created_at": "2026-01-01T00:00:00Z", "is_merged": True},
        {"repo": "r/a", "author_login": "alice", "created_at": "2026-02-01T00:00:00Z", "is_merged": False},
        # same author, different repo -- must NOT count
        {"repo": "r/b", "author_login": "alice", "created_at": "2026-01-15T00:00:00Z", "is_merged": False},
        # same repo, but NOT strictly before (equal timestamp) -- must NOT count
        {"repo": "r/a", "author_login": "alice", "created_at": "2026-03-01T00:00:00Z", "is_merged": True},
    ])
    hist = build_author_history_features(population, all_prs).set_index("pr_id")
    assert hist.loc["PR3", "author_prior_pr_count"] == 2
    assert hist.loc["PR3", "author_prior_merge_rate"] == pytest.approx(0.5)


def test_author_history_cold_start_and_null_author():
    population = pd.DataFrame([
        {"pr_id": "PR1", "repo": "r/a", "author_login": "newcomer",
         "created_at": pd.Timestamp("2026-01-01", tz="UTC")},
        {"pr_id": "PR2", "repo": "r/a", "author_login": None,
         "created_at": pd.Timestamp("2026-01-01", tz="UTC")},
    ])
    all_prs = pd.DataFrame(columns=["repo", "author_login", "created_at", "is_merged"])
    hist = build_author_history_features(population, all_prs).set_index("pr_id")

    assert hist.loc["PR1", "author_prior_pr_count"] == 0
    assert math.isnan(hist.loc["PR1", "author_prior_merge_rate"])
    assert hist.loc["PR2", "author_prior_pr_count"] == 0
    assert math.isnan(hist.loc["PR2", "author_prior_merge_rate"])
