"""Score Experiment 4's raw grid output (lab guide §4.7.5 "Result Analysis" /
§4.8; plan Section 9 row 21).

Reads `results/tables/exp4_raw.jsonl` (produced by `run_exp4_grid.py`) and
computes, per (context tier × prompt strategy) cell and in the aggregates the
central analysis needs:

- **Merge prediction:** Accuracy / Precision / Recall / F1 (both class framings +
  macro) against the AI-authored PRs' real merge labels (from
  `pull_requests.parquet` -- NOT `features_v1`, which excludes AI-authored PRs by
  Experiment 2's design), plus two aggregates that are the heart of this
  experiment: **by tier** (does richer context help?) and **by strategy** (does
  advanced prompting help?).
- **Review-comment generation:** BLEU/ROUGE against real human review comments
  *where they exist* -- but on this completed sample **no** fully-scored PR
  carries a substantive human comment (a real property of merged agent-authored
  PRs), so BLEU/ROUGE is reported as not-computable and a **generated-comment
  characteristics** proxy (comment count / length by tier and strategy) is
  emitted instead, alongside qualitative turn-by-turn side-by-sides.
- **Inference time** per cell, **conversation-aware**: a self-reflection /
  multi-turn cell sums the server-side time of *both* its API calls, so the cost
  of the advanced strategies is counted honestly.
- **Comparison to Experiment 3** (guide §4.8(7)): LLM-on-AI-code (this
  experiment) vs LLM-on-human-code (`exp3_metrics.json`) on the strategies the
  two experiments share (role-based / few-shot / CoT).

Sample-size honesty (this is the dominant caveat, surfaced in the output JSON's
`caveats`, not buried): the free-tier daily token cap stopped the run at **3
fully-complete PRs** (2 merged / 1 not-merged) -- see
`reports/exp4_grid_run_status.md`. Per-cell merge metrics at n=3 are coarse
(accuracy takes values in {0, ⅓, ⅔, 1}); the *paired, same-PR* comparison across
tiers/strategies is the meaningful signal, not any absolute number. Every
figure and delta here is directional evidence to be topped up when the grid is
(the scorer re-runs unchanged on a larger `exp4_raw.jsonl`).
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.llm.prompts.parsing import parse_merge_prediction, parse_review_comments
# Reuse Experiment 3's scorer helpers verbatim so the two experiments compute
# identical metrics (no drift): the binary-metric bundle, the BLEU/ROUGE
# function, and the human-comment curation (bots excluded via Exp 1's allowlists).
from src.llm.score_exp3 import (
    _binary_metrics,
    compute_text_metrics,
    load_human_comments,
    load_rows,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data/processed")
TABLES_DIR = Path("results/tables")
FIGURES_DIR = Path("results/figures")
RAW_PATH = TABLES_DIR / "exp4_raw.jsonl"
EXP3_METRICS = TABLES_DIR / "exp3_metrics.json"
OUT_JSON = TABLES_DIR / "exp4_metrics.json"
QUAL_MD = TABLES_DIR / "exp4_qualitative.md"

CELLS_PER_PR = 48  # 5 tiers × 2 tasks × 5 strategies − 2 undefined (DIFF×multi_turn)

# Ordered for stable table/figure axes.
TIERS = ["diff", "diff_pr_description", "diff_repo_context", "diff_issue", "complete_se_context"]
STRATEGIES = ["role_based", "few_shot", "cot", "self_reflection", "multi_turn"]
# The strategies Experiment 3 and Experiment 4 have in common (Exp 3 also had
# zero-shot; Exp 4 replaced it with self-reflection and added multi-turn).
SHARED_WITH_EXP3 = ["role_based", "few_shot", "cot"]

PALETTE = ["#2a78d6", "#eb6834", "#3fa15e", "#9b59b6", "#c0392b"]


# --------------------------------------------------------------------------- #
# Labels (AI-authored subset -- the key difference from Experiment 3's loader)
# --------------------------------------------------------------------------- #

def load_ai_merge_labels() -> dict[str, int]:
    """pr_id -> y (1 = merged) for the AI-authored PRs. Read from
    `pull_requests.parquet` because Experiment 2/3's `features_v1.parquet`
    excludes AI-authored PRs by design, so the Experiment-3 label loader would
    return nothing for this experiment's population."""
    pr = pd.read_parquet(DATA_DIR / "pull_requests.parquet")
    ai = pr[(pr["is_ai_authored"] == True) & pr["is_merged"].notna()]  # noqa: E712
    return {pid: int(m) for pid, m in zip(ai["id"], ai["is_merged"])}


# --------------------------------------------------------------------------- #
# Merge prediction
# --------------------------------------------------------------------------- #

def _pairs_metrics(pairs: list[tuple[int, int]]) -> dict:
    return _binary_metrics([p[0] for p in pairs], [p[1] for p in pairs])


def score_merge(rows: list[dict], labels: dict[str, int]) -> dict:
    """Per-(tier, strategy) merge metrics plus the two aggregates the central
    analysis turns on -- **by_tier** (pooled over strategies) and **by_strategy**
    (pooled over tiers) -- and an overall pool. An unparseable answer counts as a
    (wrong-for-merged) prediction, with the abstention tally reported separately."""
    by_cell: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    by_tier: dict[str, list[tuple[int, int]]] = defaultdict(list)
    by_strategy: dict[str, list[tuple[int, int]]] = defaultdict(list)
    all_pairs: list[tuple[int, int]] = []
    abstentions = 0

    for r in rows:
        if r["task"] != "merge_prediction" or r["error"]:
            continue
        pr_id = r["pr_id"]
        if pr_id not in labels:
            continue
        pred = parse_merge_prediction(r["response_text"]).merged
        if pred is None:
            abstentions += 1
            pred_int = 0
        else:
            pred_int = 1 if pred else 0
        pair = (labels[pr_id], pred_int)
        by_cell[(r["context_kind"], r["strategy"])].append(pair)
        by_tier[r["context_kind"]].append(pair)
        by_strategy[r["strategy"]].append(pair)
        all_pairs.append(pair)

    return {
        "per_cell": {f"{t}|{s}": _pairs_metrics(p) for (t, s), p in sorted(by_cell.items())},
        "by_tier": {t: _pairs_metrics(by_tier[t]) for t in TIERS if t in by_tier},
        "by_strategy": {s: _pairs_metrics(by_strategy[s]) for s in STRATEGIES if s in by_strategy},
        "pooled_over_all_cells": _pairs_metrics(all_pairs) if all_pairs else {},
        "abstentions": abstentions,
        "note": ("by_tier/by_strategy pool the SAME few PRs across the other axis, "
                 "so their rows are not independent; they answer 'does this axis "
                 "shift predictions on the same PRs', which is the paired signal "
                 "this experiment is after. per_cell n is tiny (see caveats)."),
    }


# --------------------------------------------------------------------------- #
# Review-comment generation (BLEU/ROUGE where possible; characteristics always)
# --------------------------------------------------------------------------- #

def score_reviews(rows: list[dict], human_comments: dict[str, list[str]]) -> dict:
    """BLEU/ROUGE per (tier, strategy) over PRs that have human comments to score
    against. On this sample that set is empty, so this returns a note and an
    empty `per_cell`; the informative signal is `review_characteristics`."""
    by_cell: dict[tuple[str, str], list[dict]] = defaultdict(list)
    prs_with_gt: set[str] = set()

    for r in rows:
        if r["task"] != "review_comment" or r["error"]:
            continue
        refs = human_comments.get(r["pr_id"])
        if not refs:
            continue
        prs_with_gt.add(r["pr_id"])
        hyp = parse_review_comments(r["response_text"]).text
        by_cell[(r["context_kind"], r["strategy"])].append(compute_text_metrics(hyp, refs))

    per_cell = {}
    for (t, s), scores in sorted(by_cell.items()):
        per_cell[f"{t}|{s}"] = {
            "n": len(scores),
            "bleu": round(mean(x["bleu"] for x in scores), 3),
            "rouge1": round(mean(x["rouge1"] for x in scores), 4),
            "rouge2": round(mean(x["rouge2"] for x in scores), 4),
            "rougeL": round(mean(x["rougeL"] for x in scores), 4),
        }

    note = None
    if not prs_with_gt:
        note = ("No fully-scored PR in this sample carries a substantive human "
                "review comment -- a real property of merged agent-authored PRs "
                "(all 3 complete PRs have 0 human review comments). BLEU/ROUGE is "
                "therefore not computable here; see review_characteristics for the "
                "proxy signal and exp4_qualitative.md for turn-by-turn examples.")
    return {"per_cell": per_cell, "n_prs_with_ground_truth": len(prs_with_gt), "note": note}


def _review_shape(text: str) -> tuple[int, int]:
    """(#comments, total chars) of a parsed generated review."""
    parsed = parse_review_comments(text)
    return len(parsed.comments), len(parsed.text)


def review_characteristics(rows: list[dict]) -> dict:
    """A gold-reference-free proxy for 'does context/prompting change the review':
    mean comment count and length of the *generated* reviews, by tier and by
    strategy. Not a quality measure -- a coverage/verbosity signal to read
    alongside the qualitative examples."""
    by_tier: dict[str, list[tuple[int, int]]] = defaultdict(list)
    by_strategy: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for r in rows:
        if r["task"] != "review_comment" or r["error"] or not r["response_text"]:
            continue
        shape = _review_shape(r["response_text"])
        by_tier[r["context_kind"]].append(shape)
        by_strategy[r["strategy"]].append(shape)

    def summarize(bucket):
        out = {}
        for k, shapes in bucket.items():
            out[k] = {
                "n": len(shapes),
                "mean_n_comments": round(mean(s[0] for s in shapes), 2),
                "mean_chars": round(mean(s[1] for s in shapes), 1),
            }
        return out

    return {
        "by_tier": {t: summarize(by_tier)[t] for t in TIERS if t in by_tier},
        "by_strategy": {s: summarize(by_strategy)[s] for s in STRATEGIES if s in by_strategy},
    }


# --------------------------------------------------------------------------- #
# Inference time (conversation-aware)
# --------------------------------------------------------------------------- #

def _inference_ms_and_calls(row: dict) -> tuple[float | None, int]:
    """(server-side inference time in ms, number of API calls) for one cell.
    A conversational cell (self-reflection / multi-turn) sums the `total_time`
    of *every* turn -- so its cost reflects both live calls, not just the final
    one. Prefers the provider's server-side `total_time` over wall-clock
    `latency_ms` (which the free tier pollutes with rate-limit waiting)."""
    if row.get("is_conversational") and row.get("turn_usages"):
        times = [(u or {}).get("total_time") for u in row["turn_usages"]]
        present = [t for t in times if t is not None]
        if present:
            return sum(present) * 1000.0, len(row["turn_usages"])
        return row.get("latency_ms"), len(row["turn_usages"])
    usage = row.get("usage") or {}
    tt = usage.get("total_time")
    if tt is not None:
        return float(tt) * 1000.0, 1
    return row.get("latency_ms"), 1


def inference_times(rows: list[dict]) -> dict:
    """Mean/median server-side inference time and mean call-count per
    (task, tier, strategy) -- the cost axis, honest about the 2× call cost of
    the conversational strategies."""
    cells: dict[tuple[str, str, str], list[tuple[float, int]]] = defaultdict(list)
    for r in rows:
        if r["error"]:
            continue
        ms, calls = _inference_ms_and_calls(r)
        if ms is None:
            continue
        cells[(r["task"], r["context_kind"], r["strategy"])].append((ms, calls))
    out = {}
    for (task, tier, strat), vals in sorted(cells.items()):
        lats = [v[0] for v in vals]
        out[f"{task}|{tier}|{strat}"] = {
            "n": len(vals),
            "mean_ms": round(mean(lats), 1),
            "median_ms": round(median(lats), 1),
            "mean_calls": round(mean(v[1] for v in vals), 2),
        }
    return out


# --------------------------------------------------------------------------- #
# Comparison to Experiment 3 (LLM on human-written code)
# --------------------------------------------------------------------------- #

def _exp3_by_strategy(exp3: dict) -> dict[str, dict]:
    """Aggregate Experiment 3's per-cell merge metrics to a per-strategy view by
    averaging over its context kinds (each Exp-3 cell shares the same 16 PRs, so
    a mean over the 4 contexts is a fair strategy-level summary)."""
    per_cell = exp3.get("merge_prediction", {}).get("per_cell", {})
    by_strat: dict[str, list[dict]] = defaultdict(list)
    for key, m in per_cell.items():
        _ctx, strat = key.split("|")
        by_strat[strat].append(m)
    out = {}
    for strat, ms in by_strat.items():
        out[strat] = {
            "accuracy": round(mean(m["accuracy"] for m in ms), 4),
            "not_merged_recall": round(mean(m["not_merged"]["recall"] for m in ms), 4),
            "macro_f1": round(mean(m["macro_f1"] for m in ms), 4),
        }
    return out


def compare_exp3(exp4_merge: dict) -> dict:
    """Side-by-side of merge metrics on the strategies both experiments ran
    (role-based / few-shot / CoT): Experiment 3 = LLM on *human-written* code,
    Experiment 4 = LLM on *AI-authored* code. Different populations and samples,
    so this is a qualitative direction check, not a controlled A/B."""
    if not EXP3_METRICS.exists():
        return {"note": "exp3_metrics.json not found; run score_exp3 first."}
    exp3 = json.loads(EXP3_METRICS.read_text())
    exp3_strat = _exp3_by_strategy(exp3)
    exp4_strat = exp4_merge.get("by_strategy", {})

    rows = {}
    for strat in SHARED_WITH_EXP3:
        e3 = exp3_strat.get(strat)
        e4 = exp4_strat.get(strat)
        if not e3 or not e4:
            continue
        rows[strat] = {
            "exp3_human_code": {"accuracy": e3["accuracy"],
                                "not_merged_recall": e3["not_merged_recall"],
                                "macro_f1": e3["macro_f1"]},
            "exp4_ai_code": {"accuracy": e4["accuracy"],
                             "not_merged_recall": e4["not_merged"]["recall"],
                             "macro_f1": e4["macro_f1"]},
            "delta_accuracy": round(e4["accuracy"] - e3["accuracy"], 4),
        }
    return {
        "shared_strategies": SHARED_WITH_EXP3,
        "merge_by_strategy": rows,
        "exp3_sample": exp3.get("sample", {}),
        "exp3_shared_strategy_mean_not_merged_recall": round(
            mean(exp3_strat[s]["not_merged_recall"] for s in SHARED_WITH_EXP3 if s in exp3_strat), 4
        ) if all(s in exp3_strat for s in SHARED_WITH_EXP3) else None,
        "note": ("Exp 3 = LLM on human-written PRs (16 complete, 14:2 merged, base "
                 "rate 0.875); Exp 4 = LLM on AI-authored PRs (3 complete, 2:1 "
                 "merged, base rate 0.67). Populations, sample sizes, and context "
                 "tiers all differ, so read these as directional, not a controlled "
                 "comparison. Two things survive the small n: (1) raw accuracy is "
                 "higher on human code, but in BOTH experiments the baseline "
                 "strategies barely exceed their always-merge base rate, so the "
                 "gap is mostly the different base rates, not discriminative power; "
                 "(2) on human code the shared strategies DO reach the minority "
                 "class occasionally (few-shot 0.25, role-based 0.125, CoT 0.0 "
                 "averaged over contexts), whereas on AI-authored code those same "
                 "three strategies score 0.0 not-merged recall everywhere and only "
                 "Exp 4's NEW strategies (self-reflection 0.6, multi-turn 0.5) "
                 "catch it at all -- consistent with guide 4.3's premise that "
                 "AI-generated code is harder to judge from the diff alone. See "
                 "central_analysis."),
    }


# --------------------------------------------------------------------------- #
# Central analysis -- the plain-number headlines the report quotes
# --------------------------------------------------------------------------- #

def central_analysis(merge: dict, chars: dict) -> dict:
    """Extract the 'does richer context / better prompting help?' headlines as
    flat numbers, so the Lab-4 report and Q&A quote one source of truth."""
    by_tier = merge["by_tier"]
    by_strategy = merge["by_strategy"]

    def acc(d):
        return {k: v["accuracy"] for k, v in d.items()}

    def nmr(d):
        return {k: v["not_merged"]["recall"] for k, v in d.items()}

    # which strategies ever catch the not-merged case at all
    catches_not_merged = [s for s, v in by_strategy.items() if v["not_merged"]["recall"] > 0]

    return {
        "context_effect_on_merge": {
            "accuracy_by_tier": acc(by_tier),
            "not_merged_recall_by_tier": nmr(by_tier),
        },
        "prompt_effect_on_merge": {
            "accuracy_by_strategy": acc(by_strategy),
            "not_merged_recall_by_strategy": nmr(by_strategy),
            "strategies_that_ever_catch_not_merged": catches_not_merged,
        },
        "review_verbosity": {
            "mean_comments_by_tier": {k: v["mean_n_comments"] for k, v in chars["by_tier"].items()},
            "mean_comments_by_strategy": {k: v["mean_n_comments"] for k, v in chars["by_strategy"].items()},
        },
    }


# --------------------------------------------------------------------------- #
# Qualitative: what the advanced prompts actually do (turn-by-turn)
# --------------------------------------------------------------------------- #

def _first(rows, **match):
    for r in rows:
        if all(r.get(k) == v for k, v in match.items()) and not r["error"] and r.get("turn_texts"):
            return r
    return None


def qualitative_turns(rows: list[dict], *, limit: int = 3) -> list[dict]:
    """Turn-by-turn examples that show what the two NEW strategies do that a
    single-shot prompt cannot: self-reflection's draft→critique→final, and
    multi-turn's diff-only→context-revealed→revised. These are the qualitative
    evidence for reflection Q4.9(2) ('which prompt design best suits AI code')."""
    examples = []
    seen = 0
    for r in rows:
        if seen >= limit:
            break
        if (r["strategy"] != "self_reflection" or r["task"] != "review_comment"
                or r["error"] or not r.get("turn_texts") or len(r["turn_texts"]) < 2):
            continue
        examples.append({
            "kind": "self_reflection",
            "pr_id": r["pr_id"], "repo": r.get("repo"), "tier": r["context_kind"],
            "draft": parse_review_comments(r["turn_texts"][0]).comments[:4],
            "final": parse_review_comments(r["turn_texts"][1]).comments[:4],
        })
        seen += 1

    seen = 0
    for r in rows:
        if seen >= limit:
            break
        if (r["strategy"] != "multi_turn" or r["task"] != "review_comment"
                or r["error"] or not r.get("turn_texts") or len(r["turn_texts"]) < 2):
            continue
        examples.append({
            "kind": "multi_turn",
            "pr_id": r["pr_id"], "repo": r.get("repo"), "tier": r["context_kind"],
            "diff_only": parse_review_comments(r["turn_texts"][0]).comments[:4],
            "with_context": parse_review_comments(r["turn_texts"][1]).comments[:4],
        })
        seen += 1
    return examples


def _write_qualitative_md(examples: list[dict], path: Path = QUAL_MD) -> None:
    lines = [
        "# Experiment 4 — What the Advanced Prompts Actually Do (turn-by-turn)",
        "",
        "*No fully-scored PR in the completed sample carries a substantive human "
        "review comment (all 3 complete PRs have none — a real property of merged "
        "agent-authored PRs), so BLEU/ROUGE against human gold is not computable. "
        "Instead, this shows the mechanism the two NEW strategies add: how "
        "self-reflection's critique turn revises its own draft, and how "
        "multi-turn's review changes once the broader context is revealed. This is "
        "the qualitative evidence for §4.9(2).*",
        "",
    ]
    for i, ex in enumerate(examples, 1):
        if ex["kind"] == "self_reflection":
            lines.append(f"## Self-reflection {i} — `{ex['repo']}` · tier=`{ex['tier']}`")
            lines.append("\n**Draft review (turn 1):**")
            lines += [f"- {c}" for c in ex["draft"]] or ["- (none)"]
            lines.append("\n**Final review after self-critique (turn 2):**")
            lines += [f"- {c}" for c in ex["final"]] or ["- (none)"]
        else:
            lines.append(f"## Multi-turn {i} — `{ex['repo']}` · tier=`{ex['tier']}`")
            lines.append("\n**Review from the diff alone (turn 1):**")
            lines += [f"- {c}" for c in ex["diff_only"]] or ["- (none)"]
            lines.append("\n**Revised review after the broader context is revealed (turn 2):**")
            lines += [f"- {c}" for c in ex["with_context"]] or ["- (none)"]
        lines.append("")
    path.write_text("\n".join(lines))


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def plot_merge_accuracy_by_config(merge: dict) -> Path:
    """Grouped bars: merge accuracy per strategy, one group per tier -- the
    §4.8(5)/(6) context×prompt comparison."""
    per_cell = merge["per_cell"]
    tiers = [t for t in TIERS if any(k.startswith(t + "|") for k in per_cell)]
    strategies = [s for s in STRATEGIES if any(k.endswith("|" + s) for k in per_cell)]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(tiers))
    width = 0.8 / len(strategies)
    for i, strat in enumerate(strategies):
        vals = [per_cell.get(f"{t}|{strat}", {}).get("accuracy", np.nan) for t in tiers]
        ax.bar(x + i * width, vals, width, label=strat, color=PALETTE[i % len(PALETTE)])
    ax.set_xticks(x + width * (len(strategies) - 1) / 2)
    ax.set_xticklabels(tiers, rotation=15, ha="right")
    ax.set_ylabel("Merge-prediction accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Exp 4 (AI code): merge accuracy by context tier × prompt strategy  (n=3 PRs — directional)")
    ax.legend(title="Strategy", fontsize=8, ncol=2)
    fig.tight_layout()
    out = FIGURES_DIR / "exp4_merge_accuracy_by_config.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_not_merged_recall_by_strategy(merge: dict, exp3_ref: float | None = None) -> Path:
    """The differentiator the small sample still shows: only the advanced
    strategies catch the not-merged PR. Experiment 3's shared-strategy mean
    not-merged recall (on human code) is drawn as a reference line when known."""
    by_strat = merge["by_strategy"]
    strategies = [s for s in STRATEGIES if s in by_strat]
    vals = [by_strat[s]["not_merged"]["recall"] for s in strategies]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(strategies))]
    ax.bar(strategies, vals, color=colors)
    if exp3_ref is not None:
        ax.axhline(exp3_ref, color="#c0392b", linestyle="--", linewidth=1,
                   label=f"Exp 3 (human code) shared-strategy mean = {exp3_ref:.2f}")
        ax.legend(fontsize=8)
    ax.set_ylabel("Not-merged recall (pooled over tiers)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Exp 4: which prompt strategies catch the not-merged PR  (n=3 — directional)")
    fig.tight_layout()
    out = FIGURES_DIR / "exp4_not_merged_recall_by_strategy.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_inference_time_by_strategy(times: dict) -> Path:
    """Mean server-side inference time per strategy, split by task -- shows the
    ~2× cost of the conversational strategies (self-reflection / multi-turn)."""
    by_task_strat: dict[tuple[str, str], list[float]] = defaultdict(list)
    for key, v in times.items():
        task, _tier, strat = key.split("|")
        by_task_strat[(task, strat)].append(v["mean_ms"])
    tasks = sorted({t for t, _ in by_task_strat})
    strategies = [s for s in STRATEGIES if any(s == st for _, st in by_task_strat)]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(strategies))
    width = 0.8 / max(1, len(tasks))
    for i, task in enumerate(tasks):
        vals = [mean(by_task_strat.get((task, s), [0.0])) for s in strategies]
        ax.bar(x + i * width, vals, width, label=task, color=PALETTE[i % len(PALETTE)])
    ax.set_xticks(x + width * (len(tasks) - 1) / 2)
    ax.set_xticklabels(strategies)
    ax.set_ylabel("Mean server-side inference time (ms)")
    ax.set_title("Exp 4: inference time by strategy (conversational strategies sum both turns)")
    ax.legend(title="Task")
    fig.tight_layout()
    out = FIGURES_DIR / "exp4_inference_time_by_strategy.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def build_metrics(rows: list[dict]) -> dict:
    labels = load_ai_merge_labels()
    human_comments = load_human_comments()

    per_pr = Counter(r["pr_id"] for r in rows)
    complete_prs = [p for p, n in per_pr.items() if n == CELLS_PER_PR]
    merged = sum(labels.get(p, 0) for p in complete_prs)

    merge = score_merge(rows, labels)
    reviews = score_reviews(rows, human_comments)
    chars = review_characteristics(rows)
    times = inference_times(rows)
    exp3_cmp = compare_exp3(merge)
    analysis = central_analysis(merge, chars)

    return {
        "sample": {
            "n_prs_covered": len(per_pr),
            "n_prs_complete": len(complete_prs),
            "complete_merged": merged,
            "complete_not_merged": len(complete_prs) - merged,
            "n_rows": len(rows),
            "n_failures": sum(1 for r in rows if r["error"]),
        },
        "caveats": [
            f"Only {len(complete_prs)} PR(s) are fully complete "
            f"({merged} merged / {len(complete_prs) - merged} not-merged); the "
            "free-tier daily token cap stopped the run (reports/"
            "exp4_grid_run_status.md). At this n, per-cell merge accuracy is "
            "coarse and every number here is DIRECTIONAL -- read the paired "
            "cross-tier/cross-strategy comparison, not absolute values.",
            "No complete PR carries a substantive human review comment, so "
            "BLEU/ROUGE is not computable on this sample; review quality is shown "
            "via generated-comment characteristics + qualitative turn examples.",
            "The DIFF_ISSUE / COMPLETE tiers were run without --fetch-issues, so "
            "their issue slot is empty (reports/exp4_grid_run_status.md); the "
            "issue tier's true effect awaits a top-up run.",
            "Context sections were capped tightly for the free-tier token budget "
            "(run_exp4_grid.py GROQ_RUN_* constants).",
        ],
        "merge_prediction": merge,
        "review_generation": reviews,
        "review_characteristics": chars,
        "inference_time": times,
        "central_analysis": analysis,
        "exp3_comparison": exp3_cmp,
    }


def main() -> None:
    rows = load_rows(RAW_PATH)
    logger.info("Scoring %d rows from %s", len(rows), RAW_PATH)
    metrics = build_metrics(rows)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(metrics, indent=2))
    logger.info("Wrote %s", OUT_JSON)

    examples = qualitative_turns(rows)
    _write_qualitative_md(examples)
    logger.info("Wrote %s (%d turn examples)", QUAL_MD, len(examples))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    p1 = plot_merge_accuracy_by_config(metrics["merge_prediction"])
    p2 = plot_not_merged_recall_by_strategy(
        metrics["merge_prediction"],
        metrics["exp3_comparison"].get("exp3_shared_strategy_mean_not_merged_recall"),
    )
    p3 = plot_inference_time_by_strategy(metrics["inference_time"])
    logger.info("Wrote %s, %s, %s", p1, p2, p3)

    logger.info("Sample:\n%s", json.dumps(metrics["sample"], indent=2))
    logger.info("Central analysis:\n%s", json.dumps(metrics["central_analysis"], indent=2))


if __name__ == "__main__":
    main()
