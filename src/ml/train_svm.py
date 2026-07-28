"""Train SVM (RBF) merge-prediction models for Experiment 2 -- Step 11.

For each feature variant (V0 leaky-ablation, V1 per-spec, V2 pre-review-
strict; see `reports/exp2_feature_spec.md` Section 5) and each class-weight
mode (plain, `class_weight='balanced'`), grid-searches `C`/`gamma` via
`StratifiedKFold(5)` on the training split only (see `src/ml/common.py` for
why not `TimeSeriesSplit`), refits the best pipeline (preprocessing + SVC
bundled together so the saved artifact is directly reusable later, e.g. by
the FastAPI backend) on the full training split, and saves it.

This script only trains and saves. Held-out test-set evaluation
(Accuracy/Precision/Recall/F1/ROC-AUC, minority-class recall, feature
importance, V0-vs-V1-vs-V2 comparison) is Step 12's separate job -- not
duplicated here, so the grid-search CV score logged below is a training-time
diagnostic only, not the reported result.

A merge probability (not just a hard label) is needed later for the VSCode
extension's probability gauge (plan Section 7.1). The obvious way,
`SVC(probability=True)`, is deprecated as of scikit-learn 1.9 (removed in
1.11) -- the library's own migration message points to
`CalibratedClassifierCV(SVC(...), ensemble=False)` instead, which is what
this script uses. It calibrates via cross-validated predictions on the
training data (same idea as the old internal Platt-scaling pass, just
via the non-deprecated API), at a similar, non-issue cost at this
dataset's scale (<=891 training rows). One consequence: the SVC's own
hyperparameters sit one level deeper in the pipeline
(`model__estimator__C`/`gamma`, not `model__C`/`gamma`), since
`CalibratedClassifierCV` wraps the SVC as its `estimator`.
"""

from __future__ import annotations

import json
import time

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src.ml.common import (
    INNER_CV,
    MODELS_DIR,
    RANDOM_STATE,
    TABLES_DIR,
    VARIANT_COLUMNS,
    build_preprocessor,
    load_variant,
    split_by_repo_time,
)

PARAM_GRID = {
    "model__estimator__C": [0.1, 1, 10, 100],
    "model__estimator__gamma": ["scale", "auto", 0.01, 0.1],
}


def train_one(feature_columns: list[str], train_df, class_weight):
    preprocessor = build_preprocessor(feature_columns)
    base_svc = SVC(kernel="rbf", class_weight=class_weight, random_state=RANDOM_STATE)
    calibrated = CalibratedClassifierCV(base_svc, ensemble=False)
    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", calibrated),
    ])
    search = GridSearchCV(pipeline, PARAM_GRID, scoring="roc_auc", cv=INNER_CV, n_jobs=-1)
    X = train_df[feature_columns]
    y = train_df["y"]
    t0 = time.time()
    search.fit(X, y)
    elapsed = time.time() - t0
    return search, elapsed


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict = {}

    for variant, feature_columns in VARIANT_COLUMNS.items():
        df = load_variant(variant)
        train_df, test_df, cutoffs = split_by_repo_time(df)
        summary[variant] = {
            "n_train": len(train_df),
            "n_test": len(test_df),
            "split_cutoffs": cutoffs,
            "configs": {},
        }

        for mode, class_weight in (("plain", None), ("balanced", "balanced")):
            search, elapsed = train_one(feature_columns, train_df, class_weight)
            model_path = MODELS_DIR / f"svm_{variant}_{mode}.joblib"
            joblib.dump(search.best_estimator_, model_path)

            summary[variant]["configs"][mode] = {
                "best_params": search.best_params_,
                "best_cv_roc_auc": round(float(search.best_score_), 4),
                "train_seconds": round(elapsed, 1),
                "model_path": str(model_path),
            }
            print(f"[svm/{variant}/{mode}] best_params={search.best_params_} "
                  f"cv_roc_auc={search.best_score_:.4f} ({elapsed:.1f}s)")

    out_path = TABLES_DIR / "exp2_svm_training_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
