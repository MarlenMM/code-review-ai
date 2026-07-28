"""Assemble the Experiment 2 feature matrix (V0/V1/V2) from `data/processed/*.parquet`.

Implements `reports/exp2_feature_spec.md` Sections 4-5: joins the code-
modification (M) and textual (T) features built here with the code-structure
(S) features cached by Step 9 (`src/features/ast_cfg.py`), plus the
process/forbidden (P) features and author-history feature used only by the
ablation variants, and emits the three feature tables the spec calls for:

- **V1 (per-spec)**  -- S ∪ M ∪ T, exactly lab guide §2.7.3 / §2.4.3. The
  primary, compliant, graded feature set.
- **V2 (pre-review-strict)** -- V1 minus the two features that keep growing
  during review (`m_num_commits`, `t_commit_msg_len_total`), plus
  `author_prior_merge_rate`/`author_prior_pr_count` (computed only from PRs
  the same author closed strictly before this PR, see
  `build_author_history_features`).
- **V0 (leaky superset)** -- V1 plus the forbidden process/review-comment
  features (P) plus `num_labels`. **Never used for a graded/compliant
  result** -- exists solely so Step 12 can measure how much these forbidden
  signals inflate accuracy, which is the concrete demonstration of *why*
  lab guide §2.4.3 bans them.

Population: human-written (`is_ai_authored == False`) AND closed
(`state in {MERGED, CLOSED}`) PRs from `pull_requests.parquet` -- every
mined PR is already closed, so the second filter is a documented no-op, not
a silent assumption (spec §1).

Design choices made here that the frozen spec text doesn't pin down exactly:

* **File-status columns.** The spec's `m_files_added/deleted/modified`
  notation is a shorthand; the real `files_changed.status` vocabulary in
  this dataset is `added`/`modified`/`removed`/`renamed` (642 removed, 167
  renamed -- not negligible), so all four get their own count column
  (`m_files_added`, `m_files_modified`, `m_files_removed`,
  `m_files_renamed`) rather than folding `renamed` into `modified` and
  silently losing information.
* **`author_prior_merge_rate` scope (V2).** Scoped to the *same repo* only
  (not global across all 5 mined repos): a reviewer's mental model of a
  contributor's track record is inherently per-project, and pooling across
  unrelated repos would conflate very different review cultures (spec §1
  already shows per-repo merge rates ranging 62.4%-85.1%). Computed over
  the full mined `pull_requests` table (not just the human-written+closed
  population it's attached to), scoped to the author's own PRs with
  `created_at` strictly before the current PR's -- per spec wording. A
  companion `author_prior_pr_count` is added (not in the frozen spec text)
  because a rate computed from 1 prior PR and one from 50 are not
  comparable, and cold-start authors (no prior PRs in that repo, or a null
  `author_login`) get `author_prior_merge_rate = NaN` +
  `author_prior_pr_count = 0` rather than a silently-imputed 0 or global
  mean -- Step 11's preprocessing pipeline is responsible for deciding how
  to impute this (it must be fit on the training split only, so it cannot
  happen here).

What is deliberately NOT done here (left to Step 11, per spec §6): the
`log1p` skew transform, `StandardScaler` fitting, and the time-based
train/test split. All three depend on which rows land in the training
split (a scaler/imputer fit on the full table would leak test-set
distribution into training), so they belong in the training pipeline, not
in this population-agnostic-per-row assembly step. What ships here is the
raw, honestly-missing-marked feature table.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data/processed")
TABLES_DIR = Path("results/tables")

META_COLUMNS = ["pr_id", "y", "created_at", "repo"]

M_COLUMNS = [
    "m_additions", "m_deletions", "m_churn", "m_net_change", "m_changed_files",
    "m_avg_file_churn", "m_frac_added", "m_num_hunks",
    "m_files_added", "m_files_modified", "m_files_removed", "m_files_renamed",
    "m_num_commits",
]
T_COLUMNS = [
    "t_title_len", "t_title_wordcount", "t_body_len", "t_body_wordcount",
    "t_body_is_empty", "t_commit_msg_len_first", "t_commit_msg_len_mean",
    "t_commit_msg_len_total",
]
# S_COLUMNS mirrors src.features.ast_cfg.S_FEATURE_COLUMNS; duplicated (not
# imported) so this module's contract is self-contained and readable without
# cross-referencing ast_cfg.py.
S_COLUMNS = [
    "s_ast_node_count", "s_ast_max_depth", "s_cfg_node_count", "s_cfg_edge_count",
    "s_cyclomatic_proxy", "s_num_functions_touched", "s_has_parseable_code",
]
P_COLUMNS = [
    "p_num_reviews", "p_num_reviewers", "p_num_review_comments",
    "p_num_issue_comments", "p_has_changes_requested", "p_has_approved",
    "p_is_ai_reviewed", "p_n_ai_reviewers", "num_labels",
]
AUTHOR_HISTORY_COLUMNS = ["author_prior_merge_rate", "author_prior_pr_count"]

V1_FEATURE_COLUMNS = M_COLUMNS + S_COLUMNS + T_COLUMNS
V2_DROPPED = ["m_num_commits", "t_commit_msg_len_total"]
V2_FEATURE_COLUMNS = [c for c in V1_FEATURE_COLUMNS if c not in V2_DROPPED] + AUTHOR_HISTORY_COLUMNS
V0_FEATURE_COLUMNS = V1_FEATURE_COLUMNS + P_COLUMNS


# --------------------------------------------------------------------------- #
# Population
# --------------------------------------------------------------------------- #

def select_population(pull_requests: pd.DataFrame) -> pd.DataFrame:
    """Human-written + closed PRs (spec §1). Every mined PR is already
    closed, so that half of the filter is a documented no-op, not an
    unstated assumption."""
    mask = (~pull_requests["is_ai_authored"]) & pull_requests["state"].isin(["MERGED", "CLOSED"])
    pop = pull_requests.loc[mask, [
        "id", "repo", "author_login", "created_at", "is_merged", "labels",
    ]].copy()
    pop = pop.rename(columns={"id": "pr_id", "is_merged": "y"})
    pop["created_at"] = pd.to_datetime(pop["created_at"], utc=True)
    return pop.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# M -- code modification
# --------------------------------------------------------------------------- #

def _count_hunk_markers(patch) -> int:
    if not isinstance(patch, str) or not patch:
        return 0
    return sum(1 for line in patch.split("\n") if line.startswith("@@"))


def build_modification_features(pr_ids: pd.Series, pull_requests: pd.DataFrame,
                                 files_changed: pd.DataFrame, commits: pd.DataFrame) -> pd.DataFrame:
    base = pull_requests.set_index("id").loc[pr_ids, ["additions", "deletions", "changed_files"]]
    base = base.rename(columns={"additions": "m_additions", "deletions": "m_deletions",
                                 "changed_files": "m_changed_files"})
    base.index.name = "pr_id"
    base["m_churn"] = base["m_additions"] + base["m_deletions"]
    base["m_net_change"] = base["m_additions"] - base["m_deletions"]
    base["m_avg_file_churn"] = base["m_churn"] / base["m_changed_files"].clip(lower=1)
    base["m_frac_added"] = base["m_additions"] / (base["m_churn"] + 1)

    fc = files_changed[files_changed["pr_id"].isin(pr_ids)].copy()
    fc["_n_hunks"] = fc["patch"].apply(_count_hunk_markers)
    fc_g = fc.groupby("pr_id")
    hunks = fc_g["_n_hunks"].sum().rename("m_num_hunks")
    status_counts = (
        fc_g["status"].value_counts().unstack(fill_value=0)
        .rename(columns={"added": "m_files_added", "modified": "m_files_modified",
                          "removed": "m_files_removed", "renamed": "m_files_renamed"})
    )
    for col in ("m_files_added", "m_files_modified", "m_files_removed", "m_files_renamed"):
        if col not in status_counts.columns:
            status_counts[col] = 0

    cm = commits[commits["pr_id"].isin(pr_ids)]
    n_commits = cm.groupby("pr_id").size().rename("m_num_commits")

    out = base.join(hunks).join(status_counts[
        ["m_files_added", "m_files_modified", "m_files_removed", "m_files_renamed"]
    ]).join(n_commits)
    out[["m_num_hunks", "m_files_added", "m_files_modified", "m_files_removed",
         "m_files_renamed", "m_num_commits"]] = out[[
        "m_num_hunks", "m_files_added", "m_files_modified", "m_files_removed",
        "m_files_renamed", "m_num_commits",
    ]].fillna(0)
    return out.reset_index()[["pr_id", *M_COLUMNS]]


# --------------------------------------------------------------------------- #
# T -- textual
# --------------------------------------------------------------------------- #

def build_textual_features(pr_ids: pd.Series, pull_requests: pd.DataFrame,
                            commits: pd.DataFrame) -> pd.DataFrame:
    base = pull_requests.set_index("id").loc[pr_ids, ["title", "body"]]
    base.index.name = "pr_id"
    title = base["title"].fillna("")
    body = base["body"].fillna("")

    out = pd.DataFrame(index=base.index)
    out["t_title_len"] = title.str.len()
    out["t_title_wordcount"] = title.str.split().apply(len)
    out["t_body_len"] = body.str.len()
    out["t_body_wordcount"] = body.str.split().apply(len)
    out["t_body_is_empty"] = (out["t_body_len"] == 0).astype(int)

    cm = commits[commits["pr_id"].isin(pr_ids)].copy()
    cm["_msg_len"] = cm["message"].str.len()
    cm_g = cm.groupby("pr_id")["_msg_len"]
    # "first" = first row per pr_id in the commits table's stored/API order
    # (see reports/exp2_ast_cfg_fidelity.md-style caveat: no reliable
    # per-commit timestamp is stored to sort by, see module docstring).
    first = cm_g.first().rename("t_commit_msg_len_first")
    mean_ = cm_g.mean().rename("t_commit_msg_len_mean")
    total = cm_g.sum().rename("t_commit_msg_len_total")

    out = out.join(first).join(mean_).join(total)
    out[["t_commit_msg_len_first", "t_commit_msg_len_mean", "t_commit_msg_len_total"]] = out[[
        "t_commit_msg_len_first", "t_commit_msg_len_mean", "t_commit_msg_len_total",
    ]].fillna(0)
    return out.reset_index()[["pr_id", *T_COLUMNS]]


# --------------------------------------------------------------------------- #
# S -- code structure (Step 9 cache, left-joined + filled)
# --------------------------------------------------------------------------- #

def load_structure_features(pr_ids: pd.Series, ast_cfg_features: pd.DataFrame) -> pd.DataFrame:
    """Left-join Step 9's cache onto the population; PRs absent from it
    (no `files_changed` rows at all -- e.g. a closed PR with 0 changed
    files) get `s_has_parseable_code=0` and all other S columns 0, per
    `src/features/ast_cfg.py`'s own documented join contract, never a
    silently-different default."""
    base = pd.DataFrame({"pr_id": pr_ids})
    merged = base.merge(ast_cfg_features[["pr_id", *S_COLUMNS]], on="pr_id", how="left")
    merged[S_COLUMNS] = merged[S_COLUMNS].fillna(0)
    for col in ("s_has_parseable_code",):
        merged[col] = merged[col].astype(int)
    return merged


# --------------------------------------------------------------------------- #
# P -- process / forbidden (V0 ablation only)
# --------------------------------------------------------------------------- #

def build_process_features(pr_ids: pd.Series, pull_requests: pd.DataFrame,
                            reviews: pd.DataFrame, review_comments: pd.DataFrame,
                            issue_comments: pd.DataFrame) -> pd.DataFrame:
    base = pull_requests.set_index("id").loc[pr_ids, ["labels", "is_ai_reviewed", "n_ai_reviewers"]]
    base.index.name = "pr_id"
    out = pd.DataFrame(index=base.index)
    out["num_labels"] = base["labels"].apply(len)
    out["p_is_ai_reviewed"] = base["is_ai_reviewed"].astype(int)
    out["p_n_ai_reviewers"] = base["n_ai_reviewers"]

    rv = reviews[reviews["pr_id"].isin(pr_ids)]
    rv_g = rv.groupby("pr_id")
    n_reviews = rv_g.size().rename("p_num_reviews")
    n_reviewers = rv_g["reviewer_login"].nunique().rename("p_num_reviewers")
    has_changes_requested = (rv_g["state"].apply(lambda s: (s == "CHANGES_REQUESTED").any())
                              .rename("p_has_changes_requested").astype(int))
    has_approved = (rv_g["state"].apply(lambda s: (s == "APPROVED").any())
                    .rename("p_has_approved").astype(int))

    rc = review_comments[review_comments["pr_id"].isin(pr_ids)]
    n_review_comments = rc.groupby("pr_id").size().rename("p_num_review_comments")

    ic = issue_comments[issue_comments["pr_id"].isin(pr_ids)]
    n_issue_comments = ic.groupby("pr_id").size().rename("p_num_issue_comments")

    out = (out.join(n_reviews).join(n_reviewers).join(has_changes_requested)
           .join(has_approved).join(n_review_comments).join(n_issue_comments))
    count_cols = ["p_num_reviews", "p_num_reviewers", "p_has_changes_requested",
                  "p_has_approved", "p_num_review_comments", "p_num_issue_comments"]
    out[count_cols] = out[count_cols].fillna(0)
    return out.reset_index()[["pr_id", *P_COLUMNS]]


# --------------------------------------------------------------------------- #
# Author-history feature (V2 only)
# --------------------------------------------------------------------------- #

def build_author_history_features(population: pd.DataFrame, all_pull_requests: pd.DataFrame) -> pd.DataFrame:
    """`author_prior_merge_rate`/`author_prior_pr_count` per spec §5 (V2):
    computed from the same author's PRs in the **same repo**, strictly
    `created_at`-before the row in question, over the full mined dataset
    (not just the human-written+closed population -- an author's own past
    PRs are the relevant history regardless of which experiment-2 filter
    they'd individually pass). Cold-start (no such prior PR, or a null
    `author_login`) yields `NaN`/`0`, not an imputed value -- imputation is
    a training-time (train-split-only) decision, made in Step 11."""
    history_src = all_pull_requests[["repo", "author_login", "created_at", "is_merged"]].copy()
    history_src["created_at"] = pd.to_datetime(history_src["created_at"], utc=True)
    history_src = history_src.dropna(subset=["author_login"])

    records = []
    for repo, by_repo in population.groupby("repo"):
        repo_history = history_src[history_src["repo"] == repo]
        for author, by_author in by_repo.groupby("author_login", dropna=False):
            if pd.isna(author):
                for pr_id in by_author["pr_id"]:
                    records.append((pr_id, np.nan, 0))
                continue
            author_history = repo_history[repo_history["author_login"] == author]
            for pr_id, created_at in zip(by_author["pr_id"], by_author["created_at"]):
                prior = author_history[author_history["created_at"] < created_at]
                n = len(prior)
                rate = float(prior["is_merged"].mean()) if n > 0 else np.nan
                records.append((pr_id, rate, n))

    result = pd.DataFrame(records, columns=["pr_id", "author_prior_merge_rate", "author_prior_pr_count"])
    return result


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #

def assemble_feature_tables(data_dir: Path = DATA_DIR) -> dict[str, pd.DataFrame]:
    pull_requests = pd.read_parquet(data_dir / "pull_requests.parquet")
    files_changed = pd.read_parquet(data_dir / "files_changed.parquet")
    commits = pd.read_parquet(data_dir / "commits.parquet")
    reviews = pd.read_parquet(data_dir / "reviews.parquet")
    review_comments = pd.read_parquet(data_dir / "review_comments.parquet")
    issue_comments = pd.read_parquet(data_dir / "issue_comments.parquet")
    ast_cfg_features = pd.read_parquet(data_dir / "ast_cfg_features.parquet")

    population = select_population(pull_requests)
    pr_ids = population["pr_id"]
    logger.info("Experiment 2 population: %d PRs (human-written, closed)", len(population))

    m = build_modification_features(pr_ids, pull_requests, files_changed, commits)
    t = build_textual_features(pr_ids, pull_requests, commits)
    s = load_structure_features(pr_ids, ast_cfg_features)
    p = build_process_features(pr_ids, pull_requests, reviews, review_comments, issue_comments)
    author_hist = build_author_history_features(population, pull_requests)

    meta = population[["pr_id", "y", "created_at", "repo"]].copy()
    meta["y"] = meta["y"].astype(int)

    full = (meta.merge(m, on="pr_id", how="left")
            .merge(s, on="pr_id", how="left")
            .merge(t, on="pr_id", how="left")
            .merge(p, on="pr_id", how="left")
            .merge(author_hist, on="pr_id", how="left"))

    v1 = full[[*META_COLUMNS, *V1_FEATURE_COLUMNS]].copy()
    v2 = full[[*META_COLUMNS, *V2_FEATURE_COLUMNS]].copy()
    v0 = full[[*META_COLUMNS, *V0_FEATURE_COLUMNS]].copy()

    return {"v0": v0, "v1": v1, "v2": v2}


def _summary(tables: dict[str, pd.DataFrame]) -> dict:
    v1 = tables["v1"]
    v2 = tables["v2"]
    summary = {
        "n_prs": int(len(v1)),
        "n_merged": int(v1["y"].sum()),
        "merge_rate": round(float(v1["y"].mean()), 4),
        "by_repo": {
            repo: {"n": int(len(g)), "merge_rate": round(float(g["y"].mean()), 4)}
            for repo, g in v1.groupby("repo")
        },
        "columns": {
            "v0": V0_FEATURE_COLUMNS,
            "v1": V1_FEATURE_COLUMNS,
            "v2": V2_FEATURE_COLUMNS,
        },
        "s_has_parseable_code_rate": round(float(v1["s_has_parseable_code"].mean()), 4),
        "author_prior_pr_count_zero_rate": round(
            float((v2["author_prior_pr_count"] == 0).mean()), 4
        ),
        "author_prior_merge_rate_nan_count": int(v2["author_prior_merge_rate"].isna().sum()),
    }
    return summary


def main() -> None:
    tables = assemble_feature_tables()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        out_path = DATA_DIR / f"features_{name}.parquet"
        df.to_parquet(out_path, index=False)
        logger.info("Wrote %s (%d rows, %d cols)", out_path, len(df), len(df.columns))

    summary = _summary(tables)
    summary_path = TABLES_DIR / "exp2_feature_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("Wrote %s", summary_path)
    logger.info("Summary:\n%s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
