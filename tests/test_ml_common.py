import numpy as np
import pandas as pd
import pytest

from src.ml.common import (
    BOOL_COLUMNS,
    RATIO_COLUMNS,
    SIGNED_COLUMNS,
    build_preprocessor,
    classify_columns,
    split_by_repo_time,
)
from src.features.feature_extraction import (
    V0_FEATURE_COLUMNS,
    V1_FEATURE_COLUMNS,
    V2_FEATURE_COLUMNS,
)


# --------------------------------------------------------------------------- #
# split_by_repo_time
# --------------------------------------------------------------------------- #

def _population_fixture():
    rows = []
    # repo A: 10 PRs, evenly spaced in time
    for i in range(10):
        rows.append({"pr_id": f"A{i}", "repo": "r/a",
                      "created_at": pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(days=i),
                      "y": i % 2})
    # repo B: 20 PRs, evenly spaced in time (different scale, tests proportionality)
    for i in range(20):
        rows.append({"pr_id": f"B{i}", "repo": "r/b",
                      "created_at": pd.Timestamp("2026-03-01", tz="UTC") + pd.Timedelta(days=i),
                      "y": i % 2})
    return pd.DataFrame(rows)


def test_split_by_repo_time_proportional_per_repo():
    df = _population_fixture()
    train, test, cutoffs = split_by_repo_time(df, train_fraction=0.8)

    train_a = train[train.repo == "r/a"]
    test_a = test[test.repo == "r/a"]
    train_b = train[train.repo == "r/b"]
    test_b = test[test.repo == "r/b"]

    assert len(train_a) == 8 and len(test_a) == 2
    assert len(train_b) == 16 and len(test_b) == 4
    assert cutoffs["r/a"]["n_train"] == 8
    assert cutoffs["r/b"]["n_train"] == 16


def test_split_by_repo_time_no_row_overlap_and_respects_order():
    df = _population_fixture()
    train, test, _ = split_by_repo_time(df, train_fraction=0.8)

    assert set(train.pr_id).isdisjoint(set(test.pr_id))
    assert len(train) + len(test) == len(df)

    # every train row for a repo must be strictly older than every test row
    # for that SAME repo (the actual leakage guarantee this split protects)
    for repo in df.repo.unique():
        max_train = train[train.repo == repo].created_at.max()
        min_test = test[test.repo == repo].created_at.min()
        assert max_train < min_test


# --------------------------------------------------------------------------- #
# classify_columns
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("variant_columns", [V0_FEATURE_COLUMNS, V1_FEATURE_COLUMNS, V2_FEATURE_COLUMNS])
def test_classify_columns_partitions_every_column_exactly_once(variant_columns):
    roles = classify_columns(variant_columns)
    all_classified = roles["bool"] + roles["signed"] + roles["ratio"] + roles["log_count"]
    assert sorted(all_classified) == sorted(variant_columns)
    # no column appears in more than one bucket
    assert len(all_classified) == len(set(all_classified))


def test_classify_columns_known_buckets():
    roles = classify_columns(["m_net_change", "m_frac_added", "s_has_parseable_code", "m_additions"])
    assert roles["signed"] == ["m_net_change"]
    assert roles["ratio"] == ["m_frac_added"]
    assert roles["bool"] == ["s_has_parseable_code"]
    assert roles["log_count"] == ["m_additions"]


def test_bool_signed_ratio_column_sets_are_disjoint():
    assert BOOL_COLUMNS.isdisjoint(SIGNED_COLUMNS)
    assert BOOL_COLUMNS.isdisjoint(RATIO_COLUMNS)
    assert SIGNED_COLUMNS.isdisjoint(RATIO_COLUMNS)


# --------------------------------------------------------------------------- #
# build_preprocessor
# --------------------------------------------------------------------------- #

def test_build_preprocessor_handles_all_roles_and_nan():
    cols = ["m_additions", "m_net_change", "author_prior_merge_rate", "s_has_parseable_code"]
    df = pd.DataFrame({
        "m_additions": [0, 5, 100, 3, 7, 2, 9, 1, 4, 6],
        "m_net_change": [-10, 0, 5, -3, 2, -1, 8, -4, 1, 0],
        "author_prior_merge_rate": [0.5, np.nan, 1.0, 0.0, np.nan, 0.5, 0.7, np.nan, 0.2, 0.6],
        "s_has_parseable_code": [1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
    })
    pre = build_preprocessor(cols)
    out = pre.fit_transform(df[cols])

    assert out.shape[0] == len(df)
    assert not np.isnan(out).any()  # NaN in author_prior_merge_rate must be imputed, not propagated


def test_build_preprocessor_rejects_no_negative_values_after_log1p():
    # a log_count column should never produce NaN/-inf even with an
    # (in-practice-impossible, defensively clipped) negative input
    cols = ["m_additions"]
    df = pd.DataFrame({"m_additions": [-5, 0, 10, 3]})
    pre = build_preprocessor(cols)
    out = pre.fit_transform(df[cols])
    assert not np.isnan(out).any()
    assert np.isfinite(out).all()
