"""Shared data-loading, split, and preprocessing utilities for Experiment 2's
SVM/RF training scripts (`src/ml/train_svm.py`, `train_rf.py`).

Both scripts must use IDENTICAL train/test rows and IDENTICAL preprocessing
for a given variant so the SVM-vs-RF and V0-vs-V1-vs-V2 comparisons (Step 12)
are apples-to-apples -- hence one shared module rather than duplicating
split/preprocessing logic in both training scripts.

Split strategy: per-repo time-based, not a single global cutoff
------------------------------------------------------------------
The spec (`reports/exp2_feature_spec.md` Section 6.4) calls for a time-based
split ("train = older ~80%, test = newest ~20%"). A single GLOBAL cutoff on
`created_at` across all 5 repos was tried first and rejected: Step 4's mining
samples a bounded, recency-biased window per repo (closed PRs, capped, filled
with a recent slice -- plan Section 3.2), not a uniform scrape of each repo's
full multi-year history. That makes a single global 80/20 cutoff conflate
*which repo* with *how recent*: 2026-07-20 is the day most of the 5 repos'
sampled PRs happen to cluster around, so a single global cutoff collapsed
microsoft/semantic-kernel's test slice to 1 PR (of 274) and pushed the test
split's merge rate to 87.9% vs. the training split's 70.8% -- an artifact of
the sampling window, not a meaningful temporal boundary.

Splitting **per repo** instead (sort each repo's own PRs by `created_at`,
oldest ~80% to train, newest ~20% to test, then concatenate across repos)
keeps every repo proportionally represented in both splits while still fully
honoring the actual leakage concern a time-based split exists to prevent: no
test PR should be evaluated using a model that saw a *later* PR from that
same repo during training. Measured result: train merge rate 72.2%, test
82.1% -- a smaller, more plausible residual gap, the kind you'd expect
between two calendar slices of a single project's own history rather than a
sampling artifact.

Inner cross-validation (grid search): StratifiedKFold, not TimeSeriesSplit
----------------------------------------------------------------------------
The outer per-repo split already prevents look-ahead leakage for the
metrics that matter (Step 12's held-out test evaluation). The *training*
split handed to GridSearchCV, however, is a concatenation of five
independently time-sorted per-repo chunks -- it has no single global
chronological order. Running `TimeSeriesSplit` over that concatenation would
silently treat repo-boundary artifacts as if they were temporal folds (worse
than not respecting time order at all, since it looks rigorous without being
so). `StratifiedKFold` (shuffled, fixed seed), which preserves each fold's
merged/not-merged ratio under the dataset's real 74/26 imbalance, is the
more honest and standard choice for hyperparameter selection here; it says
nothing about temporal generalization; that claim is reserved for the
outer split alone.

Preprocessing (spec Section 6): columns are grouped by role, not one blanket
transform, because a blind `log1p` is undefined/wrong for a signed
difference and pointless for an already-bounded ratio:
- **bool** columns pass through untouched (already 0/1).
- **log_count** columns (non-negative, heavy-tailed counts/sizes -- the
  default bucket for anything not otherwise classified) get `log1p`
  (defensively clipped at 0 first; none should ever be negative) then
  `StandardScaler`.
- **signed** columns (currently only `m_net_change`, additions minus
  deletions, which can be negative) get a sign-preserving transform
  (`sign(x) * log1p(|x|)`) then `StandardScaler`.
- **ratio** columns (`m_frac_added`, `author_prior_merge_rate`) are already
  bounded in [0, 1] and are not heavy-tailed in the `log1p` sense; they get
  mean imputation (`author_prior_merge_rate` has real `NaN`s for cold-start
  authors, spec Section 5) then `StandardScaler`, no log transform.

All of this is wrapped in a single sklearn `ColumnTransformer`/`Pipeline` so
it is fit ONLY on whatever rows `GridSearchCV` treats as "train" within a
given fold (spec Section 6.3: fitting on the full set would leak test
distribution) -- enforced by sklearn's pipeline machinery, not hand-rolled.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.features.feature_extraction import (
    V0_FEATURE_COLUMNS,
    V1_FEATURE_COLUMNS,
    V2_FEATURE_COLUMNS,
)

DATA_DIR = Path("data/processed")
MODELS_DIR = Path("results/models")
TABLES_DIR = Path("results/tables")

RANDOM_STATE = 42
TRAIN_FRACTION = 0.8

VARIANT_COLUMNS = {
    "v0": V0_FEATURE_COLUMNS,
    "v1": V1_FEATURE_COLUMNS,
    "v2": V2_FEATURE_COLUMNS,
}

INNER_CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# Column-role classification, by exact name (see module docstring). Anything
# not listed here and not a known bool/signed/ratio column falls through to
# "log_count" -- the correct default for the remaining M/S/T/P columns,
# which are all non-negative counts, sizes, or lengths.
BOOL_COLUMNS = frozenset({
    "s_has_parseable_code", "t_body_is_empty",
    "p_has_changes_requested", "p_has_approved", "p_is_ai_reviewed",
})
SIGNED_COLUMNS = frozenset({"m_net_change"})
RATIO_COLUMNS = frozenset({"m_frac_added", "author_prior_merge_rate"})


def load_variant(name: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return pd.read_parquet(data_dir / f"features_{name}.parquet")


def split_by_repo_time(df: pd.DataFrame, train_fraction: float = TRAIN_FRACTION):
    """Per-repo time-based split -- see module docstring for why not a
    single global cutoff. Returns `(train_df, test_df, cutoffs)` where
    `cutoffs` maps repo -> split diagnostics (sizes + the boundary
    timestamps either side of the cut), for reporting the exact cutoff per
    repo as the spec calls for."""
    train_parts, test_parts, cutoffs = [], [], {}
    for repo, g in df.groupby("repo"):
        g = g.sort_values("created_at")
        cut = int(len(g) * train_fraction)
        train_parts.append(g.iloc[:cut])
        test_parts.append(g.iloc[cut:])
        cutoffs[repo] = {
            "n_train": cut,
            "n_test": len(g) - cut,
            "train_max_created_at": g["created_at"].iloc[cut - 1].isoformat() if cut > 0 else None,
            "test_min_created_at": g["created_at"].iloc[cut].isoformat() if cut < len(g) else None,
        }
    train_df = pd.concat(train_parts).sort_values("created_at").reset_index(drop=True)
    test_df = pd.concat(test_parts).sort_values("created_at").reset_index(drop=True)
    return train_df, test_df, cutoffs


def _signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


def _log1p_clipped(x):
    return np.log1p(np.clip(x, a_min=0, a_max=None))


def classify_columns(feature_columns: list[str]) -> dict[str, list[str]]:
    bool_cols = [c for c in feature_columns if c in BOOL_COLUMNS]
    signed_cols = [c for c in feature_columns if c in SIGNED_COLUMNS]
    ratio_cols = [c for c in feature_columns if c in RATIO_COLUMNS]
    log_cols = [
        c for c in feature_columns
        if c not in BOOL_COLUMNS and c not in SIGNED_COLUMNS and c not in RATIO_COLUMNS
    ]
    return {"bool": bool_cols, "signed": signed_cols, "ratio": ratio_cols, "log_count": log_cols}


def build_preprocessor(feature_columns: list[str]) -> ColumnTransformer:
    """One ColumnTransformer per variant's feature-column list (see
    module docstring for the per-role transforms)."""
    roles = classify_columns(feature_columns)
    transformers = []
    if roles["log_count"]:
        transformers.append((
            "log_count",
            Pipeline([
                ("log1p", FunctionTransformer(_log1p_clipped, feature_names_out="one-to-one")),
                ("scale", StandardScaler()),
            ]),
            roles["log_count"],
        ))
    if roles["signed"]:
        transformers.append((
            "signed",
            Pipeline([
                ("signed_log1p", FunctionTransformer(_signed_log1p, feature_names_out="one-to-one")),
                ("scale", StandardScaler()),
            ]),
            roles["signed"],
        ))
    if roles["ratio"]:
        transformers.append((
            "ratio",
            Pipeline([("impute", SimpleImputer(strategy="mean")), ("scale", StandardScaler())]),
            roles["ratio"],
        ))
    if roles["bool"]:
        transformers.append(("bool", "passthrough", roles["bool"]))
    return ColumnTransformer(transformers)
