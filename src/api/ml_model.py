"""Loads the chosen Experiment 2 model for the FastAPI `/review` endpoint's
fast (ML-only) mode, and turns a V1 feature dict into a merge probability.

Model choice: `rf_v1_balanced.joblib` -- Random Forest, V1 (per-spec)
features, `class_weight='balanced'`. Justified against
`reports/exp2_model_evaluation.md`'s own criteria, not accuracy alone (the
class-imbalance discussion Experiments 1/2 both raise, and plan §10's own
risk note: "class imbalance inflates apparent SVM/RF accuracy"):

* RF vs SVM -- Random Forest "clearly outperforms SVM ... on every metric
  that matters for the minority class" (exp2_model_evaluation.md §2), and
  its `class_weight` behaves as a direct, predictable lever on minority
  recall where SVM's (routed through `CalibratedClassifierCV`) does not
  (§5) -- the more trustworthy choice for a *review aid*, whose entire
  value is in flagging the PRs worth a second look.
* plain vs balanced -- `balanced` lifts not-merged recall from 0.250 to
  0.425 (F1 0.333 -> 0.395) for an explicitly-reported accuracy cost (0.821
  -> 0.768, §2). A tool that always predicts "MERGE" would score higher
  accuracy while catching none of the PRs a reviewer most needs flagged --
  exactly the failure mode plan §10 warns about, so `balanced` is the
  defensible choice here even though it is not the higher-accuracy model.
* V1 vs V2 -- V2 (pre-review-strict + author history) measurably
  outperforms V1 on every matched configuration (§4), but its two
  author-history features need a (repo, author) identity in the mined
  dataset's own PR history; V1 is the variant computable for an arbitrary
  incoming diff (see `src/api/ml_features.py`, `reports/api_design.md`).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from src.features.feature_extraction import V1_FEATURE_COLUMNS

MODEL_PATH = Path("results/models/rf_v1_balanced.joblib")
MODEL_NAME = "rf_v1_balanced"


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH} not found -- run `python -m src.ml.train_rf` (Step 11) first."
        )
    return joblib.load(MODEL_PATH)


def predict_merge_probability(features: dict[str, float]) -> float:
    """P(merged), i.e. `predict_proba`'s column for class `1` -- looked up by
    `model.classes_` rather than assumed to be index 1, since a saved
    `Pipeline`'s class ordering is a property of the fitted estimator, not
    something this caller should hard-code."""
    model = load_model()
    row = pd.DataFrame([{col: features[col] for col in V1_FEATURE_COLUMNS}])
    proba = model.predict_proba(row)[0]
    classes = list(model.classes_)
    return float(proba[classes.index(1)])
