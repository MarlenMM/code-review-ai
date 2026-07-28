"""Tests for `src/api/ml_model.py`. Uses the real trained
`results/models/rf_v1_balanced.joblib` artifact (Step 11) -- this module's
whole job is loading and calling that exact file, so mocking it away would
not test anything real."""


import pytest

from src.api import ml_model
from src.features.feature_extraction import V1_FEATURE_COLUMNS


@pytest.fixture(autouse=True)
def _clear_model_cache():
    ml_model.load_model.cache_clear()
    yield
    ml_model.load_model.cache_clear()


def _zero_features() -> dict[str, float]:
    return {col: 0.0 for col in V1_FEATURE_COLUMNS}


def test_model_file_exists():
    assert ml_model.MODEL_PATH.exists(), (
        f"{ml_model.MODEL_PATH} missing -- run `python -m src.ml.train_rf` first."
    )


def test_load_model_returns_fitted_pipeline():
    model = ml_model.load_model()
    assert hasattr(model, "predict_proba")
    assert list(model.classes_) == [0, 1]


def test_load_model_is_cached():
    m1 = ml_model.load_model()
    m2 = ml_model.load_model()
    assert m1 is m2


def test_predict_merge_probability_returns_valid_probability():
    proba = ml_model.predict_merge_probability(_zero_features())
    assert isinstance(proba, float)
    assert 0.0 <= proba <= 1.0


def test_predict_merge_probability_varies_with_features():
    low_churn = _zero_features()
    high_churn = dict(low_churn)
    high_churn.update({
        "m_additions": 5000, "m_deletions": 3000, "m_churn": 8000,
        "m_net_change": 2000, "m_changed_files": 40, "m_num_hunks": 200,
    })
    p_low = ml_model.predict_merge_probability(low_churn)
    p_high = ml_model.predict_merge_probability(high_churn)
    assert p_low != p_high


def test_load_model_raises_if_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(ml_model, "MODEL_PATH", tmp_path / "does_not_exist.joblib")
    with pytest.raises(FileNotFoundError):
        ml_model.load_model()
