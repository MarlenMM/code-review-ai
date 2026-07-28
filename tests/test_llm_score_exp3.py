"""Tests for `src/llm/score_exp3.py`: the metric computations that turn the
raw grid output into Experiment 3's scored results. Pure-function tests on
synthetic rows -- no dataset, no network.
"""


from src.llm.score_exp3 import (
    _binary_metrics,
    _clean_comment,
    _inference_ms,
    compute_text_metrics,
    inference_times,
    score_merge,
    score_reviews,
)


# --------------------------------------------------------------------------- #
# _binary_metrics
# --------------------------------------------------------------------------- #

def test_binary_metrics_perfect():
    m = _binary_metrics([1, 1, 0, 0], [1, 1, 0, 0])
    assert m["accuracy"] == 1.0
    assert m["merged"]["recall"] == 1.0
    assert m["not_merged"]["recall"] == 1.0
    assert m["macro_f1"] == 1.0
    assert m["confusion"] == {"tp": 2, "fp": 0, "tn": 2, "fn": 0}


def test_binary_metrics_always_predicts_merged_under_skew():
    # 4 merged, 1 not-merged, model always says merged
    m = _binary_metrics([1, 1, 1, 1, 0], [1, 1, 1, 1, 1])
    assert m["accuracy"] == 0.8               # 4/5 right
    assert m["merged"]["recall"] == 1.0
    assert m["not_merged"]["recall"] == 0.0   # missed the only negative
    assert m["confusion"] == {"tp": 4, "fp": 1, "tn": 0, "fn": 0}


def test_binary_metrics_zero_division_safe():
    # no positives predicted, none present in preds -> no crash, zeros
    m = _binary_metrics([0, 0], [0, 0])
    assert m["merged"]["precision"] == 0.0
    assert m["not_merged"]["recall"] == 1.0


# --------------------------------------------------------------------------- #
# score_merge
# --------------------------------------------------------------------------- #

def _merge_row(pr_id, ctx, strat, text, error=None):
    return {"pr_id": pr_id, "task": "merge_prediction", "context_kind": ctx,
            "strategy": strat, "response_text": text, "error": error}


def test_score_merge_groups_by_cell_and_counts_abstentions():
    labels = {"p1": 1, "p2": 0}
    rows = [
        _merge_row("p1", "diff_only", "zero_shot", "DECISION: MERGE"),
        _merge_row("p2", "diff_only", "zero_shot", "DECISION: CLOSE"),
        _merge_row("p1", "diff_only", "cot", "no decision here"),   # abstention
        _merge_row("p1", "diff_only", "zero_shot", "DECISION: MERGE", error="boom"),  # skipped
    ]
    result = score_merge(rows, labels)
    cell = result["per_cell"]["diff_only|zero_shot"]
    assert cell["n"] == 2 and cell["accuracy"] == 1.0
    assert result["abstentions"] == 1  # the unparseable cot row


def test_score_merge_skips_rows_without_a_label():
    labels = {"p1": 1}
    rows = [_merge_row("p_unknown", "diff_only", "zero_shot", "DECISION: MERGE")]
    result = score_merge(rows, labels)
    assert result["per_cell"] == {}


# --------------------------------------------------------------------------- #
# text metrics
# --------------------------------------------------------------------------- #

def test_compute_text_metrics_identical_text_scores_high():
    m = compute_text_metrics("add a null check here", ["add a null check here"])
    assert m["rouge1"] == 1.0
    assert m["rougeL"] == 1.0
    assert m["bleu"] > 50


def test_compute_text_metrics_disjoint_text_scores_low():
    m = compute_text_metrics("completely unrelated tokens xyz", ["add a null check"])
    assert m["rouge1"] == 0.0
    assert m["bleu"] < 5


def test_compute_text_metrics_empty_is_zero_not_error():
    assert compute_text_metrics("", ["ref"]) == {"bleu": 0.0, "rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    assert compute_text_metrics("hyp", []) == {"bleu": 0.0, "rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}


def test_score_reviews_only_counts_prs_with_ground_truth():
    human = {"p1": ["please add a guard clause"]}
    rows = [
        {"pr_id": "p1", "task": "review_comment", "context_kind": "diff_only",
         "strategy": "zero_shot", "response_text": "<review>\n- add a guard clause\n</review>", "error": None},
        {"pr_id": "p2", "task": "review_comment", "context_kind": "diff_only",
         "strategy": "zero_shot", "response_text": "<review>\n- something\n</review>", "error": None},
    ]
    result = score_reviews(rows, human)
    assert result["n_prs_with_ground_truth"] == 1  # only p1 has a reference
    assert "diff_only|zero_shot" in result["per_cell"]
    assert result["per_cell"]["diff_only|zero_shot"]["n"] == 1


# --------------------------------------------------------------------------- #
# _clean_comment
# --------------------------------------------------------------------------- #

def test_clean_comment_drops_empty_suggestion_block():
    assert _clean_comment("```suggestion\n```") is None
    assert _clean_comment("LGTM") is None  # too short
    assert _clean_comment("Please rename this variable to be descriptive.") is not None


# --------------------------------------------------------------------------- #
# inference time: server-side total_time preferred over polluted wall-clock
# --------------------------------------------------------------------------- #

def test_inference_ms_prefers_server_side_total_time():
    # wall-clock latency_ms is polluted (68s from rate-limit waits); the
    # server-side total_time (0.5s) is the real inference time
    row = {"usage": {"total_time": 0.5}, "latency_ms": 68000.0}
    assert _inference_ms(row) == 500.0


def test_inference_ms_falls_back_to_latency_when_no_usage():
    assert _inference_ms({"usage": None, "latency_ms": 123.0}) == 123.0
    assert _inference_ms({"latency_ms": 42.0}) == 42.0


def test_inference_times_uses_server_side_time():
    rows = [
        {"task": "merge_prediction", "context_kind": "diff_only", "strategy": "zero_shot",
         "error": None, "usage": {"total_time": 0.1}, "latency_ms": 50000.0},
        {"task": "merge_prediction", "context_kind": "diff_only", "strategy": "zero_shot",
         "error": None, "usage": {"total_time": 0.3}, "latency_ms": 60000.0},
    ]
    out = inference_times(rows)
    cell = out["merge_prediction|diff_only|zero_shot"]
    assert cell["mean_ms"] == 200.0  # (100 + 300) / 2, NOT the ~55000ms wall clock
