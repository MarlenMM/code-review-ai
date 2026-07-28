"""Tests for `src/llm/score_exp4.py`: the Experiment-4-specific scoring logic
(by-tier / by-strategy merge aggregates, conversation-aware inference time,
generated-comment characteristics, the central-analysis extraction). Pure
functions on synthetic rows -- no dataset, no network.

The metric primitives themselves (`_binary_metrics`, `compute_text_metrics`)
are Experiment 3's and are covered in `test_llm_score_exp3.py`; these tests only
exercise what Experiment 4 adds on top.
"""

from src.llm.score_exp4 import (
    _inference_ms_and_calls,
    central_analysis,
    inference_times,
    review_characteristics,
    score_merge,
    score_reviews,
)


def _merge_row(pr_id, tier, strat, text, error=None):
    return {"pr_id": pr_id, "task": "merge_prediction", "context_kind": tier,
            "strategy": strat, "response_text": text, "error": error}


# --------------------------------------------------------------------------- #
# score_merge: by_tier / by_strategy aggregates
# --------------------------------------------------------------------------- #

def test_score_merge_aggregates_by_tier_and_strategy():
    labels = {"p1": 1, "p2": 0}
    rows = [
        # tier=diff: role_based right on both PRs; self_reflection catches the negative
        _merge_row("p1", "diff", "role_based", "DECISION: MERGE"),
        _merge_row("p2", "diff", "role_based", "DECISION: MERGE"),          # wrong (p2 not-merged)
        _merge_row("p1", "diff", "self_reflection", "DECISION: MERGE"),
        _merge_row("p2", "diff", "self_reflection", "DECISION: CLOSE"),     # right
    ]
    result = score_merge(rows, labels)

    # by_strategy: role_based misses the negative, self_reflection catches it
    assert result["by_strategy"]["role_based"]["not_merged"]["recall"] == 0.0
    assert result["by_strategy"]["self_reflection"]["not_merged"]["recall"] == 1.0
    # by_tier pools both strategies over the diff tier (4 predictions)
    assert result["by_tier"]["diff"]["n"] == 4
    # pooled over all cells == every prediction
    assert result["pooled_over_all_cells"]["n"] == 4


def test_score_merge_orders_tiers_and_strategies_canonically():
    labels = {"p1": 1}
    rows = [
        _merge_row("p1", "complete_se_context", "multi_turn", "DECISION: MERGE"),
        _merge_row("p1", "diff", "role_based", "DECISION: MERGE"),
    ]
    result = score_merge(rows, labels)
    # TIERS/STRATEGIES canonical order preserved (diff before complete_se_context)
    assert list(result["by_tier"]) == ["diff", "complete_se_context"]
    assert list(result["by_strategy"]) == ["role_based", "multi_turn"]


def test_score_merge_abstention_counts_as_close_prediction():
    labels = {"p1": 1}
    rows = [_merge_row("p1", "diff", "cot", "waffle with no decision")]
    result = score_merge(rows, labels)
    assert result["abstentions"] == 1
    # unparseable -> predicted not-merged (0); p1 is merged -> wrong
    assert result["by_strategy"]["cot"]["merged"]["recall"] == 0.0


# --------------------------------------------------------------------------- #
# conversation-aware inference time
# --------------------------------------------------------------------------- #

def test_inference_ms_sums_all_turns_for_conversational_row():
    row = {
        "is_conversational": True,
        "turn_usages": [{"total_time": 0.3}, {"total_time": 0.5}],
        "usage": {"total_time": 0.5},  # only the final turn's usage -- must NOT be used alone
        "latency_ms": 40000.0,
    }
    ms, calls = _inference_ms_and_calls(row)
    assert ms == 800.0            # 0.3 + 0.5 summed, in ms -- both API calls counted
    assert calls == 2


def test_inference_ms_single_shot_uses_usage_total_time():
    row = {"is_conversational": False, "usage": {"total_time": 0.2}, "latency_ms": 50000.0}
    ms, calls = _inference_ms_and_calls(row)
    assert ms == 200.0
    assert calls == 1


def test_inference_ms_conversational_falls_back_to_latency_without_turn_times():
    row = {"is_conversational": True, "turn_usages": [{}, {}], "latency_ms": 123.0}
    ms, calls = _inference_ms_and_calls(row)
    assert ms == 123.0
    assert calls == 2


def test_inference_times_reports_mean_calls():
    rows = [
        {"task": "merge_prediction", "context_kind": "diff", "strategy": "self_reflection",
         "error": None, "is_conversational": True,
         "turn_usages": [{"total_time": 0.1}, {"total_time": 0.1}], "latency_ms": 9},
        {"task": "merge_prediction", "context_kind": "diff", "strategy": "self_reflection",
         "error": None, "is_conversational": True,
         "turn_usages": [{"total_time": 0.2}, {"total_time": 0.2}], "latency_ms": 9},
    ]
    out = inference_times(rows)
    cell = out["merge_prediction|diff|self_reflection"]
    assert cell["mean_calls"] == 2.0
    assert cell["mean_ms"] == 300.0  # mean of (200, 400)


# --------------------------------------------------------------------------- #
# review characteristics (proxy when no human gold exists)
# --------------------------------------------------------------------------- #

def _review_row(pr_id, tier, strat, text):
    return {"pr_id": pr_id, "task": "review_comment", "context_kind": tier,
            "strategy": strat, "response_text": text, "error": None}


def test_review_characteristics_counts_comments_and_length():
    rows = [
        _review_row("p1", "diff", "role_based", "<review>\n- one\n- two\n- three\n</review>"),
        _review_row("p1", "diff", "cot", "<review>\n- only one\n</review>"),
    ]
    chars = review_characteristics(rows)
    assert chars["by_strategy"]["role_based"]["mean_n_comments"] == 3
    assert chars["by_strategy"]["cot"]["mean_n_comments"] == 1
    # both are on the diff tier -> averaged there
    assert chars["by_tier"]["diff"]["mean_n_comments"] == 2.0  # (3 + 1) / 2


def test_review_characteristics_skips_errors_and_empty():
    rows = [
        {"pr_id": "p1", "task": "review_comment", "context_kind": "diff",
         "strategy": "role_based", "response_text": None, "error": None},
        {"pr_id": "p1", "task": "review_comment", "context_kind": "diff",
         "strategy": "role_based", "response_text": "<review>\n- x\n</review>", "error": "boom"},
    ]
    chars = review_characteristics(rows)
    assert chars["by_tier"] == {} and chars["by_strategy"] == {}


# --------------------------------------------------------------------------- #
# score_reviews: graceful when no human ground truth exists (the real case)
# --------------------------------------------------------------------------- #

def test_score_reviews_no_ground_truth_returns_note_and_empty_cells():
    rows = [_review_row("p1", "diff", "role_based", "<review>\n- something\n</review>")]
    result = score_reviews(rows, human_comments={})  # no PR has gold comments
    assert result["n_prs_with_ground_truth"] == 0
    assert result["per_cell"] == {}
    assert result["note"] is not None and "not computable" in result["note"]


# --------------------------------------------------------------------------- #
# central_analysis: the flat headline numbers the report quotes
# --------------------------------------------------------------------------- #

def test_central_analysis_flags_strategies_that_catch_not_merged():
    labels = {"p1": 1, "p2": 0}
    rows = [
        _merge_row("p1", "diff", "role_based", "DECISION: MERGE"),
        _merge_row("p2", "diff", "role_based", "DECISION: MERGE"),        # never catches negative
        _merge_row("p1", "diff", "multi_turn", "DECISION: MERGE"),
        _merge_row("p2", "diff", "multi_turn", "DECISION: CLOSE"),        # catches negative
    ]
    merge = score_merge(rows, labels)
    chars = review_characteristics([])
    analysis = central_analysis(merge, chars)

    caught = analysis["prompt_effect_on_merge"]["strategies_that_ever_catch_not_merged"]
    assert caught == ["multi_turn"]
    assert analysis["prompt_effect_on_merge"]["not_merged_recall_by_strategy"]["role_based"] == 0.0
    assert analysis["prompt_effect_on_merge"]["not_merged_recall_by_strategy"]["multi_turn"] == 1.0
