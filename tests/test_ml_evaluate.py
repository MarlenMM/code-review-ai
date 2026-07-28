from pathlib import Path

import joblib
import numpy as np
import pytest

from src.ml.common import VARIANT_COLUMNS, load_variant, split_by_repo_time
from src.ml.evaluate import categorize_feature, compute_metrics, rf_feature_importance

MODELS_DIR = Path("results/models")


# --------------------------------------------------------------------------- #
# compute_metrics
# --------------------------------------------------------------------------- #

def test_compute_metrics_known_confusion_matrix():
    # 4 merged (y=1), 4 not-merged (y=0); model gets 3/4 merged right, 2/4 not-merged right
    y_true = [1, 1, 1, 1, 0, 0, 0, 0]
    y_pred = [1, 1, 1, 0, 0, 0, 1, 1]
    y_proba = [0.9, 0.8, 0.7, 0.4, 0.3, 0.2, 0.6, 0.55]

    m = compute_metrics(y_true, y_pred, y_proba)

    assert m["accuracy"] == pytest.approx(5 / 8)
    assert m["recall_merged"] == pytest.approx(3 / 4)     # 3 of 4 actual merged predicted merged
    assert m["recall_not_merged"] == pytest.approx(2 / 4)  # 2 of 4 actual not-merged predicted not-merged
    assert 0.0 <= m["roc_auc"] <= 1.0
    assert set(m.keys()) == {
        "accuracy", "precision_merged", "recall_merged", "f1_merged",
        "precision_not_merged", "recall_not_merged", "f1_not_merged",
        "macro_f1", "roc_auc",
    }


def test_compute_metrics_perfect_predictions():
    y_true = [1, 1, 0, 0]
    y_pred = [1, 1, 0, 0]
    y_proba = [0.9, 0.8, 0.2, 0.1]
    m = compute_metrics(y_true, y_pred, y_proba)
    assert m["accuracy"] == 1.0
    assert m["recall_merged"] == 1.0
    assert m["recall_not_merged"] == 1.0
    assert m["roc_auc"] == 1.0


# --------------------------------------------------------------------------- #
# categorize_feature
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name,expected_prefix", [
    ("m_additions", "M"),
    ("s_ast_node_count", "S"),
    ("t_title_len", "T"),
    ("author_prior_merge_rate", "author history"),
    ("author_prior_pr_count", "author history"),
    ("p_num_reviews", "P"),
    ("num_labels", "P"),
])
def test_categorize_feature(name, expected_prefix):
    assert categorize_feature(name).startswith(expected_prefix)


# --------------------------------------------------------------------------- #
# rf_feature_importance -- integration against the real saved models
# --------------------------------------------------------------------------- #

pytestmark_skip_if_no_models = pytest.mark.skipif(
    not MODELS_DIR.exists() or not any(MODELS_DIR.glob("rf_*.joblib")),
    reason="trained model artifacts not present",
)


@pytestmark_skip_if_no_models
@pytest.mark.parametrize("variant", ["v0", "v1", "v2"])
def test_rf_feature_importance_categories_sum_to_one(variant):
    result = rf_feature_importance(variant, "plain")
    total = sum(result["importance_by_category"].values())
    assert total == pytest.approx(1.0, abs=1e-3)
    assert len(result["top_10_features"]) == 10
    # importances are sorted descending
    values = [f["importance"] for f in result["top_10_features"]]
    assert values == sorted(values, reverse=True)


@pytestmark_skip_if_no_models
def test_v0_process_category_dominates_v0_but_absent_elsewhere():
    """Regression anchor for the central leakage finding: P (forbidden
    process features) should dominate V0's importance and must not exist
    as a category in V1/V2 at all (they never contain those columns)."""
    v0 = rf_feature_importance("v0", "plain")
    v1 = rf_feature_importance("v1", "plain")
    p_key = "P -- process/forbidden (V0 only)"
    assert v0["importance_by_category"].get(p_key, 0) > 0.5
    assert p_key not in v1["importance_by_category"]


@pytestmark_skip_if_no_models
def test_saved_models_predict_on_their_own_test_split():
    """Smoke test that every saved pipeline can actually predict on the
    reconstructed test split without shape/column errors."""
    for variant, feature_columns in VARIANT_COLUMNS.items():
        df = load_variant(variant)
        _, test_df, _ = split_by_repo_time(df)
        X_test = test_df[feature_columns]
        for model_type in ("svm", "rf"):
            for mode in ("plain", "balanced"):
                path = MODELS_DIR / f"{model_type}_{variant}_{mode}.joblib"
                pipeline = joblib.load(path)
                proba = pipeline.predict_proba(X_test)
                assert proba.shape == (len(test_df), 2)
                assert np.all((proba >= 0) & (proba <= 1))
