"""Experiment 1 descriptive statistics and visualizations (plan Section 3.5):
merge/non-merge balance, review-comment/reviewer/PR-length distributions,
label distribution, AI-vs-human PR counts, AI-reviewer presence, and merge
rate AI-authored vs. human-authored -- every stat broken out per repo AND
pooled, since a rate computed only in aggregate can hide very different
per-repo behavior (e.g. semantic-kernel has only 23 AI-authored PRs against
vscode's 104, so a per-repo merge-rate comparison is far noisier there and
that needs to be visible, not averaged away).

Figures -> results/figures/*.png
Every number behind a figure -> results/tables/exp1_summary.json
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data/processed")
FIGURES_DIR = Path("results/figures")
TABLES_DIR = Path("results/tables")

POOLED_LABEL = "All repos (pooled)"

# Validated categorical pair (dataviz skill reference palette, slots 1-2):
# adjacent-pair CVD ΔE 9.1 light (>=8 target), normal-vision ΔE 19.6 (>=15 floor).
BLUE = "#2a78d6"     # first/baseline category per chart (Merged, Human-authored)
ORANGE = "#eb6834"   # second category per chart (Not merged, AI-authored)
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": SECONDARY_INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK,
    "axes.titlecolor": INK,
    "grid.color": GRID,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def load_data(data_dir: Path = DATA_DIR) -> dict[str, pd.DataFrame]:
    return {
        "pull_requests": pd.read_parquet(data_dir / "pull_requests.parquet"),
        "reviews": pd.read_parquet(data_dir / "reviews.parquet"),
        "review_comments": pd.read_parquet(data_dir / "review_comments.parquet"),
    }


def repo_order(pr: pd.DataFrame) -> list[str]:
    return sorted(pr["repo"].unique())


# --------------------------------------------------------------------------- #
# numeric summaries -- independent of plotting, so they're directly testable
# --------------------------------------------------------------------------- #

def summarize_merge_status(pr: pd.DataFrame) -> dict[str, Any]:
    def stats(df: pd.DataFrame) -> dict[str, Any]:
        total = len(df)
        merged = int(df["is_merged"].sum())
        not_merged = total - merged
        return {
            "merged": merged, "not_merged": not_merged, "total": total,
            "merge_rate": round(merged / total, 4) if total else None,
        }
    per_repo = {repo: stats(g) for repo, g in pr.groupby("repo")}
    return {"pooled": stats(pr), "per_repo": per_repo}


def _numeric_distribution_stats(values: pd.Series) -> dict[str, Any]:
    values = values.dropna()
    if len(values) == 0:
        return {"n": 0}
    return {
        "n": int(len(values)),
        "mean": round(float(values.mean()), 3),
        "median": float(values.median()),
        "std": round(float(values.std()), 3) if len(values) > 1 else 0.0,
        "min": float(values.min()),
        "max": float(values.max()),
        "p90": float(values.quantile(0.90)),
        "p99": float(values.quantile(0.99)),
        "zero_count": int((values == 0).sum()),
    }


def summarize_review_comment_counts(pr: pd.DataFrame, review_comments: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    counts = review_comments.groupby("pr_id").size()
    per_pr = pr[["id", "repo"]].copy()
    per_pr["n_review_comments"] = per_pr["id"].map(counts).fillna(0).astype(int)
    per_repo = {repo: _numeric_distribution_stats(g["n_review_comments"]) for repo, g in per_pr.groupby("repo")}
    return {"pooled": _numeric_distribution_stats(per_pr["n_review_comments"]), "per_repo": per_repo}, per_pr


def summarize_reviewer_counts(pr: pd.DataFrame, reviews: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    counts = reviews.groupby("pr_id")["reviewer_login"].nunique()
    per_pr = pr[["id", "repo"]].copy()
    per_pr["n_reviewers"] = per_pr["id"].map(counts).fillna(0).astype(int)
    per_repo = {repo: _numeric_distribution_stats(g["n_reviewers"]) for repo, g in per_pr.groupby("repo")}
    return {"pooled": _numeric_distribution_stats(per_pr["n_reviewers"]), "per_repo": per_repo}, per_pr


def summarize_pr_length(pr: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    per_pr = pr[["id", "repo"]].copy()
    per_pr["pr_length"] = (pr["additions"].fillna(0) + pr["deletions"].fillna(0)).astype(int)
    per_repo = {repo: _numeric_distribution_stats(g["pr_length"]) for repo, g in per_pr.groupby("repo")}
    return {"pooled": _numeric_distribution_stats(per_pr["pr_length"]), "per_repo": per_repo}, per_pr


def summarize_labels(pr: pd.DataFrame, top_n: int = 20, top_n_per_repo: int = 8) -> dict[str, Any]:
    exploded = pr[["repo", "labels"]].explode("labels")
    exploded = exploded[exploded["labels"].notna() & (exploded["labels"] != "")]
    pooled_counts = exploded["labels"].value_counts().head(top_n)
    per_repo = {}
    for repo, g in exploded.groupby("repo"):
        top = g["labels"].value_counts().head(top_n_per_repo)
        per_repo[repo] = [{"label": str(k), "count": int(v)} for k, v in top.items()]
    return {
        "pooled_top": [{"label": str(k), "count": int(v)} for k, v in pooled_counts.items()],
        "per_repo_top": per_repo,
        "total_labeled_prs": int((pr["labels"].apply(len) > 0).sum()),
        "total_prs": int(len(pr)),
    }


def summarize_ai_vs_human(pr: pd.DataFrame) -> dict[str, Any]:
    def stats(df: pd.DataFrame) -> dict[str, Any]:
        total = len(df)
        ai = int(df["is_ai_authored"].sum())
        return {"ai_authored": ai, "human_authored": total - ai, "total": total,
                "ai_share": round(ai / total, 4) if total else None}
    per_repo = {repo: stats(g) for repo, g in pr.groupby("repo")}
    return {"pooled": stats(pr), "per_repo": per_repo}


def summarize_ai_reviewer_presence(pr: pd.DataFrame) -> dict[str, Any]:
    def stats(df: pd.DataFrame) -> dict[str, Any]:
        total = len(df)
        rev = int(df["is_ai_reviewed"].sum())
        return {"ai_reviewed": rev, "not_ai_reviewed": total - rev, "total": total,
                "share": round(rev / total, 4) if total else None}
    per_repo = {repo: stats(g) for repo, g in pr.groupby("repo")}
    return {"pooled": stats(pr), "per_repo": per_repo}


def summarize_merge_rate_by_authorship(pr: pd.DataFrame) -> dict[str, Any]:
    def stats(df: pd.DataFrame) -> dict[str, Any]:
        n = len(df)
        return {"n": n, "merge_rate": round(float(df["is_merged"].mean()), 4) if n else None}
    def both(df: pd.DataFrame) -> dict[str, Any]:
        return {"ai_authored": stats(df[df["is_ai_authored"]]), "human_authored": stats(df[~df["is_ai_authored"]])}
    per_repo = {repo: both(g) for repo, g in pr.groupby("repo")}
    return {"pooled": both(pr), "per_repo": per_repo}


# --------------------------------------------------------------------------- #
# plotting -- 2x3 small-multiple grids (5 repos + pooled) for distributions,
# single comparison charts (repos + pooled as x-categories) for AI stats
# --------------------------------------------------------------------------- #

def _grid_axes(suptitle: str, n_panels: int) -> tuple[plt.Figure, list[plt.Axes]]:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    fig.suptitle(suptitle, fontsize=13, color=INK)
    axes = axes.flatten()
    for ax in axes[n_panels:]:
        ax.axis("off")
    return fig, list(axes)


def plot_merge_status(pr: pd.DataFrame, path: Path) -> None:
    groups = repo_order(pr) + [POOLED_LABEL]
    fig, axes = _grid_axes("Merged vs. Not-Merged PRs", len(groups))
    for ax, name in zip(axes, groups):
        df = pr if name == POOLED_LABEL else pr[pr["repo"] == name]
        merged = int(df["is_merged"].sum())
        not_merged = len(df) - merged
        ax.bar(["Merged", "Not merged"], [merged, not_merged], color=[BLUE, ORANGE], width=0.6)
        rate = merged / len(df) if len(df) else 0
        ax.set_title(f"{name}\n(n={len(df)}, {rate:.0%} merged)", fontsize=9, color=INK)
        ax.set_ylabel("PR count")
        ax.grid(axis="y", linewidth=0.6, alpha=0.6)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_value_distribution_grid(
    df: pd.DataFrame, value_col: str, title: str, xlabel: str, path: Path,
    bins: int = 30, clip_percentile: float = 0.98,
) -> None:
    groups = sorted(df["repo"].unique()) + [POOLED_LABEL]
    clip_value = max(df[value_col].quantile(clip_percentile), 1)
    fig, axes = _grid_axes(title, len(groups))
    for ax, name in zip(axes, groups):
        sub = df if name == POOLED_LABEL else df[df["repo"] == name]
        vals = sub[value_col].clip(upper=clip_value)
        ax.hist(vals, bins=bins, color=BLUE, edgecolor="white", linewidth=0.5)
        ax.axvline(clip_value, color=ORANGE, linewidth=1, linestyle="--", alpha=0.8)
        ax.set_yscale("log")
        ax.set_title(f"{name} (n={len(sub)}, median={sub[value_col].median():.0f})", fontsize=9, color=INK)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel("PR count (log scale)", fontsize=8)
        ax.grid(axis="y", linewidth=0.6, alpha=0.6)
        ax.set_axisbelow(True)
    fig.text(
        0.5, 0.005,
        f"Dashed line = {clip_percentile:.0%} percentile clip ({clip_value:.0f}); "
        "the rightmost bin folds in every value at or above it, so it is an aggregate, not a real peak.",
        ha="center", fontsize=8, color=SECONDARY_INK, style="italic",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_label_distribution_grid(pr: pd.DataFrame, path: Path, top_n: int = 8) -> None:
    exploded = pr[["repo", "labels"]].explode("labels")
    exploded = exploded[exploded["labels"].notna() & (exploded["labels"] != "")]
    groups = repo_order(pr) + [POOLED_LABEL]
    fig, axes = _grid_axes("Top PR Labels", len(groups))
    for ax, name in zip(axes, groups):
        sub = exploded if name == POOLED_LABEL else exploded[exploded["repo"] == name]
        counts = sub["labels"].value_counts().head(top_n)
        if len(counts) == 0:
            ax.text(0.5, 0.5, "no labels used", ha="center", va="center", color=MUTED, fontsize=9)
            ax.set_title(name, fontsize=9, color=INK)
            ax.axis("off")
            continue
        ax.barh(counts.index[::-1], counts.values[::-1], color=BLUE, height=0.65)
        ax.set_title(name, fontsize=9, color=INK)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="x", linewidth=0.6, alpha=0.6)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_ai_vs_human_counts(pr: pd.DataFrame, path: Path) -> None:
    groups = repo_order(pr) + [POOLED_LABEL]
    human_counts, ai_counts = [], []
    for name in groups:
        df = pr if name == POOLED_LABEL else pr[pr["repo"] == name]
        ai = int(df["is_ai_authored"].sum())
        ai_counts.append(ai)
        human_counts.append(len(df) - ai)
    x = np.arange(len(groups))
    width = 0.35
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - width / 2, human_counts, width, label="Human-authored", color=BLUE)
    ax.bar(x + width / 2, ai_counts, width, label="AI-authored", color=ORANGE)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("PR count")
    ax.set_title("AI-Authored vs. Human-Authored PR Counts", color=INK)
    ax.legend(frameon=False)
    ax.grid(axis="y", linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_ai_reviewer_presence(pr: pd.DataFrame, path: Path) -> None:
    groups = repo_order(pr) + [POOLED_LABEL]
    shares, ns = [], []
    for name in groups:
        df = pr if name == POOLED_LABEL else pr[pr["repo"] == name]
        shares.append(float(df["is_ai_reviewed"].mean()) if len(df) else 0.0)
        ns.append(len(df))
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(groups))
    bars = ax.bar(x, shares, color=ORANGE, width=0.6)
    for bar, n in zip(bars, ns):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015, f"n={n}",
                ha="center", fontsize=8, color=SECONDARY_INK)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Share of PRs with an AI reviewer")
    ax.set_ylim(0, 1.08)
    ax.set_title("AI-Reviewer Presence per Repo", color=INK)
    ax.grid(axis="y", linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_merge_rate_by_authorship(pr: pd.DataFrame, path: Path) -> None:
    groups = repo_order(pr) + [POOLED_LABEL]
    human_rates, ai_rates, human_ns, ai_ns = [], [], [], []
    for name in groups:
        df = pr if name == POOLED_LABEL else pr[pr["repo"] == name]
        human_df, ai_df = df[~df["is_ai_authored"]], df[df["is_ai_authored"]]
        human_rates.append(float(human_df["is_merged"].mean()) if len(human_df) else 0.0)
        ai_rates.append(float(ai_df["is_merged"].mean()) if len(ai_df) else 0.0)
        human_ns.append(len(human_df))
        ai_ns.append(len(ai_df))
    x = np.arange(len(groups))
    width = 0.35
    fig, ax = plt.subplots(figsize=(11, 6.5))
    bars_h = ax.bar(x - width / 2, human_rates, width, label="Human-authored", color=BLUE)
    bars_a = ax.bar(x + width / 2, ai_rates, width, label="AI-authored", color=ORANGE)
    for bars, ns in ((bars_h, human_ns), (bars_a, ai_ns)):
        for bar, n in zip(bars, ns):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015, f"n={n}",
                    ha="center", fontsize=7, color=SECONDARY_INK)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Merge rate")
    ax.set_ylim(0, 1.12)
    ax.set_title("Merge Rate: AI-Authored vs. Human-Authored\n(small-n repo panels are noisier -- see n= annotations)", fontsize=11, color=INK)
    ax.legend(frameon=False)
    ax.grid(axis="y", linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #

def build_summary(data: dict[str, pd.DataFrame]) -> dict[str, Any]:
    pr, reviews, review_comments = data["pull_requests"], data["reviews"], data["review_comments"]

    merge_status = summarize_merge_status(pr)
    review_comment_summary, _ = summarize_review_comment_counts(pr, review_comments)
    reviewer_summary, _ = summarize_reviewer_counts(pr, reviews)
    pr_length_summary, _ = summarize_pr_length(pr)
    label_summary = summarize_labels(pr)
    ai_vs_human = summarize_ai_vs_human(pr)
    ai_reviewer_presence = summarize_ai_reviewer_presence(pr)
    merge_rate_by_authorship = summarize_merge_rate_by_authorship(pr)

    pooled_merge = merge_status["pooled"]
    class_balance_note = (
        f"Overall merge rate is {pooled_merge['merge_rate']:.1%} "
        f"({pooled_merge['merged']} merged / {pooled_merge['not_merged']} not merged out of "
        f"{pooled_merge['total']}), a moderate class imbalance typical of mature, "
        "actively-maintained repos where most closed PRs eventually get merged. "
        "Experiment 2's training should account for this (class-weighting / "
        "reporting minority-class recall) rather than rely on raw accuracy."
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "total_prs": int(len(pr)),
            "repos": repo_order(pr),
            "prs_per_repo": {k: int(v) for k, v in pr.groupby("repo").size().items()},
        },
        "merge_status": merge_status,
        "class_balance_note": class_balance_note,
        "review_comment_count": review_comment_summary,
        "reviewer_count": reviewer_summary,
        "pr_length": pr_length_summary,
        "labels": label_summary,
        "ai_vs_human": ai_vs_human,
        "ai_reviewer_presence": ai_reviewer_presence,
        "merge_rate_by_authorship": merge_rate_by_authorship,
    }


def make_all_figures(data: dict[str, pd.DataFrame], figures_dir: Path = FIGURES_DIR) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    pr, reviews, review_comments = data["pull_requests"], data["reviews"], data["review_comments"]

    _, review_comment_per_pr = summarize_review_comment_counts(pr, review_comments)
    _, reviewer_per_pr = summarize_reviewer_counts(pr, reviews)
    _, pr_length_per_pr = summarize_pr_length(pr)

    paths = []
    plot_merge_status(pr, figures_dir / "merge_status.png")
    paths.append(figures_dir / "merge_status.png")

    plot_value_distribution_grid(
        review_comment_per_pr, "n_review_comments", "Review-Comment Count Distribution",
        "review comments per PR", figures_dir / "review_comment_count_distribution.png",
    )
    paths.append(figures_dir / "review_comment_count_distribution.png")

    plot_value_distribution_grid(
        reviewer_per_pr, "n_reviewers", "Reviewer Count Distribution",
        "distinct reviewers per PR", figures_dir / "reviewer_count_distribution.png",
    )
    paths.append(figures_dir / "reviewer_count_distribution.png")

    plot_value_distribution_grid(
        pr_length_per_pr, "pr_length", "PR-Length Distribution (additions + deletions)",
        "lines changed", figures_dir / "pr_length_distribution.png",
    )
    paths.append(figures_dir / "pr_length_distribution.png")

    plot_label_distribution_grid(pr, figures_dir / "label_distribution.png")
    paths.append(figures_dir / "label_distribution.png")

    plot_ai_vs_human_counts(pr, figures_dir / "ai_vs_human_counts_per_repo.png")
    paths.append(figures_dir / "ai_vs_human_counts_per_repo.png")

    plot_ai_reviewer_presence(pr, figures_dir / "ai_reviewer_presence_per_repo.png")
    paths.append(figures_dir / "ai_reviewer_presence_per_repo.png")

    plot_merge_rate_by_authorship(pr, figures_dir / "merge_rate_ai_vs_human.png")
    paths.append(figures_dir / "merge_rate_ai_vs_human.png")

    return paths


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    pr = data["pull_requests"]
    logger.info("Loaded %d PRs across %d repos", len(pr), pr["repo"].nunique())

    summary = build_summary(data)
    summary_path = TABLES_DIR / "exp1_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Wrote %s", summary_path)
    logger.info("Class balance: %s", summary["class_balance_note"])

    figure_paths = make_all_figures(data)
    for p in figure_paths:
        logger.info("Wrote %s", p)


if __name__ == "__main__":
    main()
