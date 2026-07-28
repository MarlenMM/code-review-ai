"""Train Random Forest merge-prediction models for Experiment 2 -- Step 11.

Mirrors `train_svm.py`: same three feature variants (V0/V1/V2), same two
class-weight modes (plain, `class_weight='balanced'`), same per-repo
time-based train/test split and `StratifiedKFold(5)` inner CV from
`src/ml/common.py` -- so the SVM-vs-RF comparison in Step 12 is on
identical rows and identical preprocessing, not incidentally-different
splits. Grid-searches `n_estimators`/`max_depth` per the plan's own wording
("Random Forest ... grid-searched depth/estimators"). Trains and saves only;
held-out evaluation and feature importances are Step 12's job.
"""

from __future__ import annotations

import json
import time

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

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
    "model__n_estimators": [100, 300, 500],
    "model__max_depth": [None, 5, 10, 20],
}


def train_one(feature_columns: list[str], train_df, class_weight):
    preprocessor = build_preprocessor(feature_columns)
    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", RandomForestClassifier(class_weight=class_weight, random_state=RANDOM_STATE,
                                          n_jobs=1)),  # GridSearchCV parallelizes; avoid oversubscription
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
            model_path = MODELS_DIR / f"rf_{variant}_{mode}.joblib"
            joblib.dump(search.best_estimator_, model_path)

            summary[variant]["configs"][mode] = {
                "best_params": search.best_params_,
                "best_cv_roc_auc": round(float(search.best_score_), 4),
                "train_seconds": round(elapsed, 1),
                "model_path": str(model_path),
            }
            print(f"[rf/{variant}/{mode}] best_params={search.best_params_} "
                  f"cv_roc_auc={search.best_score_:.4f} ({elapsed:.1f}s)")

    out_path = TABLES_DIR / "exp2_rf_training_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
