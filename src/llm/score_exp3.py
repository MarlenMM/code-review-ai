"""Score Experiment 3's raw grid output (lab guide §3.7.5 "Result Analysis";
plan Section 9 row 17).

Reads `results/tables/exp3_raw.jsonl` (produced by `run_exp3_grid.py`) and
computes, per (context kind × prompt strategy) cell and in aggregate:

- **Merge prediction:** Accuracy / Precision / Recall / F1 (both class framings +
  macro), against the same ground-truth merge labels Experiment 2 used, plus a
  direct re-scoring of the Experiment 2 SVM/RF models on the *same* PR subset so
  the ML-vs-LLM comparison is apples-to-apples (guide §3.8(7)).
- **Review-comment generation:** BLEU (sacrebleu) and ROUGE-1/2/L (Google
  `rouge-score`) of the generated review against the real human review comments
  in `review_comments.parquet`. BLEU/ROUGE are a known-weak proxy for review
  quality (plan §5.4), so a qualitative side-by-side of 5-10 generated-vs-real
  comments is emitted alongside the numbers.
- **Inference time** per cell (mean/median latency), for the cost side of the
  context/prompt comparison.

Sample-quality caveats this run is honest about (a consequence of the
free-tier-bounded 16-PR sample -- see `reports/exp3_grid_run_status.md`), and
surfaced in the output JSON's `caveats` block, not hidden:
- the sample is heavily merged-skewed (14:2 here), so merge metrics are
  dominated by the majority class -- the cross-config *comparison* is the
  meaningful signal, not the absolute accuracy;
- only a minority of PRs carry human review comments, so BLEU/ROUGE is computed
  over a small ground-truth set.
Both shrink if the sample is widened further (the scorer re-runs unchanged on a
larger `--pr-sample`).
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import joblib
import matplotlib
import numpy as np
import pandas as pd
from rouge_score import rouge_scorer
from sacrebleu import sentence_bleu
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.features.feature_extraction import V1_FEATURE_COLUMNS
from src.llm.prompts.parsing import parse_merge_prediction, parse_review_comments
from src.ml.common import load_variant
from src.mining.ai_detection import AI_CODING_AGENT_LOGINS, AI_REVIEW_BOT_LOGINS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data/processed")
MODELS_DIR = Path("results/models")
TABLES_DIR = Path("results/tables")
FIGURES_DIR = Path("results/figures")
RAW_PATH = TABLES_DIR / "exp3_raw.jsonl"
OUT_JSON = TABLES_DIR / "exp3_metrics.json"
QUAL_MD = TABLES_DIR / "exp3_qualitative.md"

PALETTE = ["#2a78d6", "#eb6834", "#3fa15e", "#9b59b6", "#c0392b"]

# Exp2 models to re-score on the same PR subset (the compliant V1 variant,
# both plain and class-balanced -- the balanced one is the fair comparison
# under imbalance, matching Exp2's own reporting).
EXP2_MODELS = ["rf_v1_balanced", "rf_v1_plain", "svm_v1_balanced", "svm_v1_plain"]

_NON_HUMAN_COMMENT_AUTHORS = (
    AI_CODING_AGENT_LOGINS | AI_REVIEW_BOT_LOGINS
    | {"github-actions", "codecov", "codecov-commenter", "azure-pipelines"}
)
_MIN_HUMAN_COMMENT_LEN = 12

_ROUGE = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_rows(path: Path = RAW_PATH) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def load_merge_labels(variant: str = "v1") -> dict[str, int]:
    """pr_id -> y (1 = merged), the exact labels Experiment 2 trained/tested on."""
    v = load_variant(variant)
    return dict(zip(v["pr_id"], v["y"].astype(int)))


def _clean_comment(body: str) -> str | None:
    """Normalise a review-comment body for use as BLEU/ROUGE ground truth.
    Drops GitHub ```suggestion blocks that carry no prose and anything too
    short to be a real review remark."""
    if not isinstance(body, str):
        return None
    text = body.strip()
    # a comment that is only a ```suggestion ...``` block has no reviewer prose
    stripped = text.replace("```suggestion", "").replace("```", "").strip()
    if len(stripped) < _MIN_HUMAN_COMMENT_LEN:
        return None
    return text


def load_human_comments() -> dict[str, list[str]]:
    """pr_id -> list of substantive human review-comment bodies (bots excluded
    via Experiment 1's own allowlists, mirroring few_shot.py's curation)."""
    rc = pd.read_parquet(DATA_DIR / "review_comments.parquet")
    rc = rc.copy()
    rc["author_login"] = rc["author_login"].fillna("")
    banned = {a.lower() for a in _NON_HUMAN_COMMENT_AUTHORS}
    rc = rc[~rc["author_login"].str.lower().isin(banned)]
    rc = rc[~rc["author_login"].str.contains(r"\[bot\]", case=False, regex=True)]

    out: dict[str, list[str]] = defaultdict(list)
    for pr_id, body in zip(rc["pr_id"], rc["body"]):
        cleaned = _clean_comment(body)
        if cleaned:
            out[pr_id].append(cleaned)
    return dict(out)


# --------------------------------------------------------------------------- #
# Merge prediction
# --------------------------------------------------------------------------- #

def _binary_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    """Accuracy + per-class and macro P/R/F1 + confusion, positive = merged (1).
    Reports both class framings because under a 14:2 merge skew the minority
    (not-merged) recall is the number that actually distinguishes models."""
    acc = accuracy_score(y_true, y_pred)
    # per-class, labels [0=not_merged, 1=merged]
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average=None, zero_division=0
    )
    macro_f1 = float(precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average="macro", zero_division=0
    )[2])
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "n": len(y_true),
        "n_merged": int(sum(y_true)),
        "n_not_merged": int(len(y_true) - sum(y_true)),
        "accuracy": round(float(acc), 4),
        "merged": {"precision": round(float(p[1]), 4), "recall": round(float(r[1]), 4),
                   "f1": round(float(f[1]), 4)},
        "not_merged": {"precision": round(float(p[0]), 4), "recall": round(float(r[0]), 4),
                       "f1": round(float(f[0]), 4)},
        "macro_f1": round(macro_f1, 4),
        "confusion": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
    }


def score_merge(rows: list[dict], labels: dict[str, int]) -> dict:
    """Per (context, strategy) merge-prediction metrics + a pooled aggregate.
    A pooled cell repeats the same PRs across configs, so it is reported with
    that caveat and the per-config table is the primary artefact."""
    cells: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    abstentions = 0
    all_pairs: list[tuple[int, int]] = []

    for r in rows:
        if r["task"] != "merge_prediction" or r["error"]:
            continue
        pr_id = r["pr_id"]
        if pr_id not in labels:
            continue
        pred = parse_merge_prediction(r["response_text"]).merged
        if pred is None:
            abstentions += 1
            pred_int = 0  # unusable answer counted as a (wrong-for-merged) prediction
        else:
            pred_int = 1 if pred else 0
        y = labels[pr_id]
        cells[(r["context_kind"], r["strategy"])].append((y, pred_int))
        all_pairs.append((y, pred_int))

    per_cell = {}
    for (ctx, strat), pairs in sorted(cells.items()):
        yt = [p[0] for p in pairs]
        yp = [p[1] for p in pairs]
        per_cell[f"{ctx}|{strat}"] = _binary_metrics(yt, yp)

    return {
        "per_cell": per_cell,
        "pooled_over_all_cells": _binary_metrics(
            [p[0] for p in all_pairs], [p[1] for p in all_pairs]
        ) if all_pairs else {},
        "abstentions": abstentions,
        "note": ("pooled_over_all_cells reuses the same PRs across 16 configs, so "
                 "its n is 16x the PR count and its rows are NOT independent; use "
                 "per_cell for model-vs-model and the config comparison."),
    }


# --------------------------------------------------------------------------- #
# Review-comment generation
# --------------------------------------------------------------------------- #

def compute_text_metrics(hypothesis: str, references: list[str]) -> dict:
    """BLEU (sacrebleu, 0-100) + ROUGE-1/2/L F (0-1) of one generated review
    against the PR's human comments. References are concatenated into one
    reference string (the human review, as a whole, is the target); an empty
    hypothesis or reference yields zeros rather than an error."""
    ref = " ".join(references).strip()
    hyp = (hypothesis or "").strip()
    if not ref or not hyp:
        return {"bleu": 0.0, "rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    bleu = sentence_bleu(hyp, [ref]).score
    rouge = _ROUGE.score(ref, hyp)
    return {
        "bleu": round(float(bleu), 3),
        "rouge1": round(float(rouge["rouge1"].fmeasure), 4),
        "rouge2": round(float(rouge["rouge2"].fmeasure), 4),
        "rougeL": round(float(rouge["rougeL"].fmeasure), 4),
    }


def score_reviews(rows: list[dict], human_comments: dict[str, list[str]]) -> dict:
    """Per (context, strategy) BLEU/ROUGE, averaged over the PRs that have human
    review comments to score against."""
    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    prs_with_gt: set[str] = set()

    for r in rows:
        if r["task"] != "review_comment" or r["error"]:
            continue
        pr_id = r["pr_id"]
        refs = human_comments.get(pr_id)
        if not refs:
            continue
        prs_with_gt.add(pr_id)
        hyp = parse_review_comments(r["response_text"]).text
        cells[(r["context_kind"], r["strategy"])].append(compute_text_metrics(hyp, refs))

    per_cell = {}
    for (ctx, strat), scores in sorted(cells.items()):
        per_cell[f"{ctx}|{strat}"] = {
            "n": len(scores),
            "bleu": round(mean(s["bleu"] for s in scores), 3),
            "rouge1": round(mean(s["rouge1"] for s in scores), 4),
            "rouge2": round(mean(s["rouge2"] for s in scores), 4),
            "rougeL": round(mean(s["rougeL"] for s in scores), 4),
        }
    return {"per_cell": per_cell, "n_prs_with_ground_truth": len(prs_with_gt)}


# --------------------------------------------------------------------------- #
# Inference time
# --------------------------------------------------------------------------- #

def _inference_ms(row: dict) -> float | None:
    """The model's own inference time in ms. Prefer the provider's server-side
    figure (Groq `usage.total_time`, in seconds) -- the wall-clock `latency_ms`
    is polluted by client-side rate-limit / 429-retry waiting on the free tier
    (it ran into the tens of seconds), so it is NOT a valid inference time.
    Falls back to `latency_ms` only if no server-side timing is present."""
    usage = row.get("usage") or {}
    total_time = usage.get("total_time")
    if total_time is not None:
        return float(total_time) * 1000.0
    return row.get("latency_ms")


def inference_times(rows: list[dict]) -> dict:
    """Mean/median server-side inference time per (task, context, strategy)."""
    cells: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for r in rows:
        if r["error"]:
            continue
        ms = _inference_ms(r)
        if ms is None:
            continue
        cells[(r["task"], r["context_kind"], r["strategy"])].append(ms)
    out = {}
    for (task, ctx, strat), lats in sorted(cells.items()):
        out[f"{task}|{ctx}|{strat}"] = {
            "n": len(lats),
            "mean_ms": round(mean(lats), 1),
            "median_ms": round(median(lats), 1),
        }
    return out


# --------------------------------------------------------------------------- #
# Experiment 2 comparison (same PR subset)
# --------------------------------------------------------------------------- #

def compare_exp2(pr_ids: list[str], labels: dict[str, int], variant: str = "v1") -> dict:
    """Re-score the saved Experiment 2 models on exactly the PRs Experiment 3
    covered, so the ML-vs-LLM comparison is on identical rows. Returns the same
    metric shape as a merge cell, per model."""
    v = load_variant(variant)
    subset = v[v["pr_id"].isin(pr_ids)].copy()
    if subset.empty:
        return {}
    X = subset[V1_FEATURE_COLUMNS]
    y_true = subset["y"].astype(int).tolist()

    out = {"n_prs": len(subset), "on_pr_ids": "same as Experiment 3 sample"}
    for name in EXP2_MODELS:
        path = MODELS_DIR / f"{name}.joblib"
        if not path.exists():
            continue
        model = joblib.load(path)
        y_pred = [int(p) for p in model.predict(X)]
        out[name] = _binary_metrics(y_true, y_pred)
    return out


# --------------------------------------------------------------------------- #
# Qualitative side-by-side
# --------------------------------------------------------------------------- #

def qualitative_examples(rows: list[dict], human_comments: dict[str, list[str]],
                         *, strategy: str = "role_based", context_kind: str = "diff_metadata",
                         limit: int = 8) -> list[dict]:
    """Generated-vs-real review comments for a fixed (context, strategy) cell,
    for PRs that have a human comment to compare against."""
    examples = []
    for r in rows:
        if (r["task"] != "review_comment" or r["error"]
                or r["strategy"] != strategy or r["context_kind"] != context_kind):
            continue
        refs = human_comments.get(r["pr_id"])
        if not refs:
            continue
        gen = parse_review_comments(r["response_text"])
        examples.append({
            "pr_id": r["pr_id"],
            "repo": r.get("repo"),
            "generated": gen.comments[:4],
            "human": refs[:4],
        })
        if len(examples) >= limit:
            break
    return examples


def _write_qualitative_md(examples: list[dict], strategy: str, context_kind: str,
                          path: Path = QUAL_MD) -> None:
    lines = [
        "# Experiment 3 — Generated vs. Real Review Comments",
        "",
        f"*Cell: context=`{context_kind}`, strategy=`{strategy}`. BLEU/ROUGE are a "
        "weak proxy for review quality (plan §5.4); this side-by-side is the "
        "qualitative complement the lab guide's Step 5 analysis calls for.*",
        "",
    ]
    for i, ex in enumerate(examples, 1):
        lines.append(f"## Example {i} — `{ex['repo']}` ({ex['pr_id']})")
        lines.append("\n**LLM-generated:**")
        lines += [f"- {c}" for c in ex["generated"]] or ["- (none)"]
        lines.append("\n**Real human reviewer:**")
        lines += [f"- {c}" for c in ex["human"]] or ["- (none)"]
        lines.append("")
    path.write_text("\n".join(lines))


# --------------------------------------------------------------------------- #
# Figures (mirrors src/ml/evaluate.py's plotting style/conventions)
# --------------------------------------------------------------------------- #

def plot_merge_accuracy_by_config(merge: dict) -> Path:
    """Grouped bar chart: merge-prediction accuracy per strategy, one group of
    bars per context -- the direct visual of guide §3.8(5)/(6)'s required
    prompt-design and context comparisons."""
    per_cell = merge["per_cell"]
    contexts = sorted({k.split("|")[0] for k in per_cell})
    strategies = sorted({k.split("|")[1] for k in per_cell})

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(contexts))
    width = 0.8 / len(strategies)
    for i, strat in enumerate(strategies):
        vals = [per_cell.get(f"{c}|{strat}", {}).get("accuracy", 0.0) for c in contexts]
        ax.bar(x + i * width, vals, width, label=strat, color=PALETTE[i % len(PALETTE)])

    ax.set_xticks(x + width * (len(strategies) - 1) / 2)
    ax.set_xticklabels(contexts, rotation=15, ha="right")
    ax.set_ylabel("Merge-prediction accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("LLM merge-prediction accuracy by context x prompt strategy")
    ax.legend(title="Strategy", fontsize=8)
    fig.tight_layout()

    out_path = FIGURES_DIR / "exp3_merge_accuracy_by_config.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_inference_time_by_strategy(times: dict) -> Path:
    """Mean server-side inference time per strategy, split by task -- the cost
    axis of the comparison (guide §3.7.5's "inference time")."""
    by_task_strat: dict[tuple[str, str], list[float]] = defaultdict(list)
    for key, v in times.items():
        task, _ctx, strat = key.split("|")
        by_task_strat[(task, strat)].append(v["mean_ms"])

    tasks = sorted({t for t, _ in by_task_strat})
    strategies = sorted({s for _, s in by_task_strat})

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(strategies))
    width = 0.8 / len(tasks)
    for i, task in enumerate(tasks):
        vals = [mean(by_task_strat.get((task, s), [0.0])) for s in strategies]
        ax.bar(x + i * width, vals, width, label=task, color=PALETTE[i % len(PALETTE)])

    ax.set_xticks(x + width * (len(tasks) - 1) / 2)
    ax.set_xticklabels(strategies)
    ax.set_ylabel("Mean server-side inference time (ms)")
    ax.set_title("Inference time by prompt strategy (Groq llama-3.1-8b-instant)")
    ax.legend(title="Task")
    fig.tight_layout()

    out_path = FIGURES_DIR / "exp3_inference_time_by_strategy.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def build_metrics(rows: list[dict]) -> dict:
    labels = load_merge_labels()
    human_comments = load_human_comments()

    pr_ids = sorted({r["pr_id"] for r in rows})
    from collections import Counter
    per_pr = Counter(r["pr_id"] for r in rows)
    complete_prs = [p for p, n in per_pr.items() if n == 32]
    merged = sum(labels.get(p, 0) for p in complete_prs)

    merge = score_merge(rows, labels)
    reviews = score_reviews(rows, human_comments)
    times = inference_times(rows)
    exp2 = compare_exp2(complete_prs, labels)

    return {
        "sample": {
            "n_prs_covered": len(pr_ids),
            "n_prs_complete": len(complete_prs),
            "complete_merged": merged,
            "complete_not_merged": len(complete_prs) - merged,
            "n_rows": len(rows),
            "n_failures": sum(1 for r in rows if r["error"]),
        },
        "caveats": [
            f"Merge sample is skewed {merged}:{len(complete_prs) - merged} "
            "(merged:not-merged); absolute merge metrics are dominated by the "
            "majority class -- read the cross-config comparison, not the raw "
            "accuracy.",
            f"Only {reviews['n_prs_with_ground_truth']} PR(s) carry human review "
            "comments, so BLEU/ROUGE is over a small ground-truth set.",
            "Contexts were capped at 2,500 chars for the free-tier token budget "
            "(see reports/exp3_grid_run_status.md), smaller than Experiment 2's.",
        ],
        "merge_prediction": merge,
        "review_generation": reviews,
        "inference_time": times,
        "exp2_comparison_same_subset": exp2,
    }


def main() -> None:
    rows = load_rows()
    logger.info("Scoring %d rows from %s", len(rows), RAW_PATH)
    metrics = build_metrics(rows)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(metrics, indent=2))
    logger.info("Wrote %s", OUT_JSON)

    human_comments = load_human_comments()
    examples = qualitative_examples(rows, human_comments)
    _write_qualitative_md(examples, "role_based", "diff_metadata")
    logger.info("Wrote %s (%d examples)", QUAL_MD, len(examples))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    p1 = plot_merge_accuracy_by_config(metrics["merge_prediction"])
    p2 = plot_inference_time_by_strategy(metrics["inference_time"])
    logger.info("Wrote %s, %s", p1, p2)

    logger.info("Summary:\n%s", json.dumps(metrics["sample"], indent=2))


if __name__ == "__main__":
    main()
