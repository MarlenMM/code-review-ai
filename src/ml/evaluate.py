"""Evaluate the Step 11 SVM/RF models on the held-out test split -- Step 12.

Loads all 12 saved pipelines (`results/models/{svm,rf}_{v0,v1,v2}_{plain,
balanced}.joblib`), reconstructs the exact same per-repo time-based test
split each was trained against (`split_by_repo_time` is deterministic --
calling it again here reproduces the identical rows, no need to persist the
split separately), and reports what the lab guide's §2.8 requires:
Accuracy/Precision/Recall/F1/ROC-AUC (with not-merged recall surfaced
explicitly, since accuracy alone is inflated by the 74/26 imbalance),
Random Forest feature importances aggregated by feature category (grouped
per `reports/exp2_feature_spec.md`'s S/M/T/P taxonomy -- a single top-N
feature list would answer "which feature" but not reflection Q4's actual
question, "which *type* of feature"), and the model comparison the spec's
Section 5 calls for:

- **V1 vs V0**: how much do the forbidden review-process features inflate
  apparent performance? (The concrete demonstration of why lab guide
  §2.4.3 bans them -- Step 11's training-time CV scores already hinted at
  this: V0 scored ~0.95-0.96 ROC-AUC vs V1/V2's 0.72-0.80.)
- **V1 vs V2**: even within the allowed features, how much does
  final-code-state contamination (T1 leakage, spec §3) help vs. a
  stricter submission-time-only view?
- **SVM vs RF**, within each variant and class-weight mode.
- **plain vs `class_weight='balanced'`**: the accuracy/minority-recall
  trade-off spec §6.5 calls for.

Outputs: `results/tables/exp2_metrics.json` (full per-model metrics),
`results/tables/exp2_feature_importance.json` (RF importances, per-feature
and per-category, for all three variants), and two figures under
`results/figures/`.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.ml.common import VARIANT_COLUMNS, load_variant, split_by_repo_time

MODELS_DIR = Path("results/models")
TABLES_DIR = Path("results/tables")
FIGURES_DIR = Path("results/figures")

MODEL_TYPES = ["svm", "rf"]
CLASS_WEIGHT_MODES = ["plain", "balanced"]

# Validated categorical pair (dataviz skill reference palette, slots 1-2):
BLUE = "#2a78d6"
ORANGE = "#eb6834"
PALETTE = ["#2a78d6", "#eb6834", "#3fa15e", "#9b59b6", "#c0392b"]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def compute_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_merged": round(float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "recall_merged": round(float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "f1_merged": round(float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "precision_not_merged": round(float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)), 4),
        "recall_not_merged": round(float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)), 4),
        "f1_not_merged": round(float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4),
    }


def evaluate_all_models() -> dict:
    results = {}
    for variant, feature_columns in VARIANT_COLUMNS.items():
        df = load_variant(variant)
        _, test_df, _ = split_by_repo_time(df)
        X_test = test_df[feature_columns]
        y_test = test_df["y"]

        for model_type in MODEL_TYPES:
            for mode in CLASS_WEIGHT_MODES:
                path = MODELS_DIR / f"{model_type}_{variant}_{mode}.joblib"
                pipeline = joblib.load(path)
                y_pred = pipeline.predict(X_test)
                y_proba = pipeline.predict_proba(X_test)[:, 1]
                key = f"{model_type}_{variant}_{mode}"
                results[key] = {
                    "model_type": model_type,
                    "variant": variant,
                    "class_weight": mode,
                    "n_test": int(len(test_df)),
                    "test_merge_rate": round(float(y_test.mean()), 4),
                    **compute_metrics(y_test, y_pred, y_proba),
                }
    return results


# --------------------------------------------------------------------------- #
# Random Forest feature importance, by feature and by category
# --------------------------------------------------------------------------- #

def categorize_feature(name: str) -> str:
    if name.startswith("m_"):
        return "M -- code modification"
    if name.startswith("s_"):
        return "S -- code structure"
    if name.startswith("t_"):
        return "T -- textual"
    if name.startswith("author_"):
        return "author history (V2 only)"
    if name.startswith("p_") or name == "num_labels":
        return "P -- process/forbidden (V0 only)"
    return "other"


def rf_feature_importance(variant: str, mode: str = "plain") -> dict:
    pipeline = joblib.load(MODELS_DIR / f"rf_{variant}_{mode}.joblib")
    preprocessor = pipeline.named_steps["preprocess"]
    clf = pipeline.named_steps["model"]

    raw_names = preprocessor.get_feature_names_out()
    # strip the ColumnTransformer branch prefix ("log_count__foo" -> "foo")
    feature_names = [n.split("__", 1)[1] if "__" in n else n for n in raw_names]
    importances = clf.feature_importances_

    per_feature = sorted(
        zip(feature_names, importances), key=lambda kv: kv[1], reverse=True
    )
    per_feature_out = [{"feature": f, "importance": round(float(v), 5)} for f, v in per_feature]

    by_category: dict[str, float] = {}
    for f, v in zip(feature_names, importances):
        cat = categorize_feature(f)
        by_category[cat] = by_category.get(cat, 0.0) + float(v)
    by_category = {k: round(v, 4) for k, v in sorted(by_category.items(), key=lambda kv: kv[1], reverse=True)}

    return {
        "variant": variant,
        "class_weight": mode,
        "top_10_features": per_feature_out[:10],
        "importance_by_category": by_category,
    }


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def plot_feature_importance_by_category(importance_by_variant: dict) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    all_cats = sorted({c for imp in importance_by_variant.values() for c in imp["importance_by_category"]})
    variants = list(importance_by_variant.keys())
    x = np.arange(len(all_cats))
    width = 0.8 / len(variants)

    for i, variant in enumerate(variants):
        vals = [importance_by_variant[variant]["importance_by_category"].get(c, 0.0) for c in all_cats]
        ax.bar(x + i * width, vals, width, label=variant.upper(), color=PALETTE[i % len(PALETTE)])

    ax.set_xticks(x + width * (len(variants) - 1) / 2)
    ax.set_xticklabels(all_cats, rotation=20, ha="right")
    ax.set_ylabel("Summed Random Forest importance")
    ax.set_title("RF feature importance by category (plain class-weight)")
    ax.legend(title="Feature set")
    fig.tight_layout()

    out_path = FIGURES_DIR / "exp2_feature_importance_by_category.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_model_comparison(metrics: dict, mode: str = "balanced") -> Path:
    variants = ["v0", "v1", "v2"]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(variants))
    width = 0.35

    for i, model_type in enumerate(MODEL_TYPES):
        vals = [metrics[f"{model_type}_{v}_{mode}"]["roc_auc"] for v in variants]
        ax.bar(x + i * width, vals, width, label=model_type.upper(),
               color=PALETTE[i % len(PALETTE)])

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([v.upper() for v in variants])
    ax.set_ylabel("Test-set ROC-AUC")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"SVM vs RF across feature variants ({mode} class-weight, held-out test)")
    ax.legend(title="Model")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="_chance")
    fig.tight_layout()

    out_path = FIGURES_DIR / "exp2_model_comparison.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    metrics = evaluate_all_models()
    metrics_path = TABLES_DIR / "exp2_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Wrote {metrics_path}")

    metrics_df = pd.DataFrame.from_dict(metrics, orient="index")
    csv_path = TABLES_DIR / "exp2_metrics.csv"
    metrics_df.to_csv(csv_path, index_label="config")
    print(f"Wrote {csv_path}")

    importance = {v: rf_feature_importance(v, "plain") for v in ["v0", "v1", "v2"]}
    importance_path = TABLES_DIR / "exp2_feature_importance.json"
    importance_path.write_text(json.dumps(importance, indent=2))
    print(f"Wrote {importance_path}")

    fig1 = plot_feature_importance_by_category(importance)
    fig2 = plot_model_comparison(metrics, mode="balanced")
    print(f"Wrote {fig1}")
    print(f"Wrote {fig2}")

    # --- printed discussion, for pasting straight into the report --------- #
    print("\n=== V1 (headline) vs V0 (leaky ablation) vs V2 (pre-review-strict) ===")
    for v in ["v0", "v1", "v2"]:
        m = metrics[f"rf_{v}_balanced"]
        print(f"  RF/{v}/balanced: acc={m['accuracy']} roc_auc={m['roc_auc']} "
              f"recall_not_merged={m['recall_not_merged']} f1_not_merged={m['f1_not_merged']}")

    print("\n=== plain vs balanced (accuracy vs minority-recall trade-off) ===")
    for model_type in MODEL_TYPES:
        for v in ["v0", "v1", "v2"]:
            plain = metrics[f"{model_type}_{v}_plain"]
            bal = metrics[f"{model_type}_{v}_balanced"]
            print(f"  {model_type}/{v}: plain acc={plain['accuracy']} recall_not_merged={plain['recall_not_merged']} "
                  f"| balanced acc={bal['accuracy']} recall_not_merged={bal['recall_not_merged']}")

    print("\n=== RF feature importance by category (plain) ===")
    for v in ["v0", "v1", "v2"]:
        print(f"  {v}: {importance[v]['importance_by_category']}")


if __name__ == "__main__":
    main()
