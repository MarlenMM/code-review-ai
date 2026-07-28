"""Tests for `src/llm/run_exp4_grid.py`: grid iteration (including the one
skipped multi-turn/DIFF cell), resumability (row-skip), conversational vs.
single-shot dispatch, live-call counting for 2-call conversational cells, and
the quota-aware stop policy -- all against fake providers. No real network
call or API key is used anywhere in this file.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.llm.prompts.context_aug import Exp4ContextTier
from src.llm.prompts.few_shot import ExampleSources
from src.llm.prompts.schema import Exp4Strategy, Task
from src.llm.providers import GroqAPIError, GroqQuotaExceededError
from src.llm.run_exp4_grid import (
    CELLS_PER_PR,
    build_few_shot_pools,
    load_existing_keys,
    run_grid,
)

N_TIERS = len(list(Exp4ContextTier))
N_TASKS = len(list(Task))
N_STRATEGIES = len(list(Exp4Strategy))


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #

class FakeSingleResponse:
    def __init__(self, text="response", from_cache=False, model="fake-model"):
        self.text = text
        self.from_cache = from_cache
        self.latency_ms = 1.0
        self.model = model
        self.finish_reason = "stop"
        self.usage = {"total_tokens": 5}


class FakeTurn:
    def __init__(self, text, from_cache=False, model="fake-model"):
        self.text = text
        self.from_cache = from_cache
        self.latency_ms = 1.0
        self.model = model
        self.usage = {"t": 1}


class FakeConversationResponse:
    def __init__(self, n_turns=2, from_cache=False, model="fake-model"):
        self.turns = tuple(FakeTurn(f"turn-{i}", from_cache=from_cache, model=model) for i in range(n_turns))

    @property
    def final_text(self):
        return self.turns[-1].text

    @property
    def all_from_cache(self):
        return all(t.from_cache for t in self.turns)

    @property
    def any_from_cache(self):
        return any(t.from_cache for t in self.turns)

    @property
    def total_latency_ms(self):
        return sum(t.latency_ms for t in self.turns)


class FakeProvider:
    """A minimal stand-in for a CachedChatProvider: no network, fully
    deterministic, can be told to raise on a specific call index to simulate a
    quota wall or transient error partway through a run. Counts single-shot
    and conversational calls separately so tests can assert dispatch."""

    model = "fake-model"

    def __init__(self, fail_at=None, quota_at=None, all_cached=False, conv_turns=2):
        self.calls: list[tuple[str, dict]] = []
        self.fail_at = fail_at
        self.quota_at = quota_at
        self.all_cached = all_cached
        self.conv_turns = conv_turns
        self._n = 0

    def _next(self):
        idx = self._n
        self._n += 1
        if self.quota_at is not None and idx == self.quota_at:
            raise GroqQuotaExceededError("simulated quota exhaustion")
        if self.fail_at is not None and idx == self.fail_at:
            raise GroqAPIError("simulated api error")
        return idx

    def generate(self, prompt, max_output_tokens=None, cache_metadata=None, cache_only=False):
        idx = self._next()
        self.calls.append((prompt.label, dict(cache_metadata or {})))
        return FakeSingleResponse(text=f"response-{idx}", from_cache=self.all_cached)

    def generate_conversation(self, conversation, max_output_tokens=None, cache_metadata=None, cache_only=False):
        self._next()  # advances the counter and honors fail_at/quota_at; index itself is unused here
        self.calls.append((conversation.label, dict(cache_metadata or {})))
        return FakeConversationResponse(n_turns=self.conv_turns, from_cache=self.all_cached)


def _sources(n_prs=2):
    pull_requests = pd.DataFrame([
        {"id": f"p{i}", "repo": "acme/widgets", "owner": "acme", "repo_name": "widgets",
         "title": f"Title {i}", "body": f"Body {i} references nothing.",
         "labels": np.array(["bug"]), "additions": 5, "deletions": 1}
        for i in range(n_prs)
    ])
    files_changed = pd.DataFrame([
        {"pr_id": f"p{i}", "filename": "a.py", "status": "modified",
         "additions": 5, "deletions": 1, "patch": f"@@ -1 +1 @@\n-old{i}\n+new{i}"}
        for i in range(n_prs)
    ])
    commits = pd.DataFrame([
        {"pr_id": f"p{i}", "sha": f"s{i}", "message": f"Fix {i}"} for i in range(n_prs)
    ])
    review_comments = pd.DataFrame([
        {"pr_id": f"p{i}", "author_login": "bob",
         "body": f"Please double-check the edge case in change {i} carefully."}
        for i in range(n_prs)
    ])
    return ExampleSources(pull_requests, files_changed, commits, review_comments)


def _train_pool(n_prs=2):
    return pd.DataFrame([
        {"pr_id": f"p{i}", "repo": "acme/widgets", "y": i % 2} for i in range(n_prs)
    ])


# --------------------------------------------------------------------------- #
# load_existing_keys
# --------------------------------------------------------------------------- #

def test_load_existing_keys_missing_file_is_empty(tmp_path):
    assert load_existing_keys(tmp_path / "does_not_exist.jsonl") == set()


def test_load_existing_keys_reads_rows(tmp_path):
    path = tmp_path / "raw.jsonl"
    rows = [
        {"pr_id": "p0", "context_kind": "diff", "strategy": "role_based", "task": "merge_prediction"},
        {"pr_id": "p1", "context_kind": "complete_se_context", "strategy": "self_reflection", "task": "review_comment"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert load_existing_keys(path) == {
        ("p0", "diff", "role_based", "merge_prediction"),
        ("p1", "complete_se_context", "self_reflection", "review_comment"),
    }


# --------------------------------------------------------------------------- #
# build_few_shot_pools
# --------------------------------------------------------------------------- #

def test_build_few_shot_pools_covers_every_task_and_tier():
    sources = _sources()
    pools = build_few_shot_pools(sources, _train_pool(), k=1)
    for tier in Exp4ContextTier:
        assert (Task.MERGE_PREDICTION.value, tier.value) in pools
        assert (Task.REVIEW_COMMENT.value, tier.value) in pools
    merge_examples = pools[(Task.MERGE_PREDICTION.value, Exp4ContextTier.DIFF.value)]
    assert all(ex.label in {"MERGE", "CLOSE"} for ex in merge_examples)


# --------------------------------------------------------------------------- #
# run_grid: grid shape (the skipped multi-turn/DIFF cell)
# --------------------------------------------------------------------------- #

def test_cells_per_pr_excludes_multi_turn_on_bare_diff():
    assert CELLS_PER_PR == N_TIERS * N_TASKS * N_STRATEGIES - N_TASKS  # 5*2*5 - 2 = 48


def test_run_grid_never_emits_multi_turn_on_diff_tier(tmp_path):
    sources = _sources()
    pools = build_few_shot_pools(sources, _train_pool(), k=1)
    out_path = tmp_path / "raw.jsonl"

    run_grid(["p0"], provider=FakeProvider(), sources=sources, train_pool=_train_pool(),
             few_shot_pools=pools, output_path=out_path)

    rows = [json.loads(line) for line in out_path.read_text().splitlines()]
    assert len(rows) == CELLS_PER_PR
    bad = [r for r in rows if r["strategy"] == "multi_turn" and r["context_kind"] == "diff"]
    assert bad == []


# --------------------------------------------------------------------------- #
# run_grid: basic coverage, dispatch, and idempotence
# --------------------------------------------------------------------------- #

def test_run_grid_writes_every_cell_for_full_run(tmp_path):
    sources = _sources(n_prs=2)
    train_pool = _train_pool(n_prs=2)
    pools = build_few_shot_pools(sources, train_pool, k=1)
    out_path = tmp_path / "raw.jsonl"

    summary = run_grid(["p0", "p1"], provider=FakeProvider(), sources=sources,
                       train_pool=train_pool, few_shot_pools=pools, output_path=out_path)

    assert summary["total_cells"] == 2 * CELLS_PER_PR
    assert summary["n_written_this_run"] == 2 * CELLS_PER_PR
    assert summary["stopped_reason"] is None

    rows = [json.loads(line) for line in out_path.read_text().splitlines()]
    assert len(rows) == 2 * CELLS_PER_PR
    keys = {(r["pr_id"], r["context_kind"], r["strategy"], r["task"]) for r in rows}
    assert len(keys) == 2 * CELLS_PER_PR


def test_run_grid_dispatches_conversational_vs_single_shot_correctly(tmp_path):
    sources = _sources()
    train_pool = _train_pool()
    pools = build_few_shot_pools(sources, train_pool, k=1)
    out_path = tmp_path / "raw.jsonl"

    run_grid(["p0"], provider=FakeProvider(), sources=sources, train_pool=train_pool,
             few_shot_pools=pools, output_path=out_path)
    rows = [json.loads(line) for line in out_path.read_text().splitlines()]

    conv_rows = [r for r in rows if r["is_conversational"]]
    single_rows = [r for r in rows if not r["is_conversational"]]
    assert {r["strategy"] for r in conv_rows} == {"self_reflection", "multi_turn"}
    assert {r["strategy"] for r in single_rows} == {"role_based", "few_shot", "cot"}
    # conversational rows carry every turn's text; single-shot rows don't
    assert all(r["turn_texts"] is not None and len(r["turn_texts"]) == 2 for r in conv_rows)
    assert all(r["turn_texts"] is None for r in single_rows)


def test_run_grid_is_idempotent_on_rerun(tmp_path):
    sources = _sources()
    train_pool = _train_pool()
    pools = build_few_shot_pools(sources, train_pool, k=1)
    out_path = tmp_path / "raw.jsonl"

    run_grid(["p0"], provider=FakeProvider(), sources=sources, train_pool=train_pool,
             few_shot_pools=pools, output_path=out_path)
    provider2 = FakeProvider()
    summary2 = run_grid(["p0"], provider=provider2, sources=sources, train_pool=train_pool,
                        few_shot_pools=pools, output_path=out_path)

    assert summary2["n_skipped_already_done"] == CELLS_PER_PR
    assert summary2["n_written_this_run"] == 0
    assert len(provider2.calls) == 0
    rows = out_path.read_text().splitlines()
    assert len(rows) == CELLS_PER_PR


# --------------------------------------------------------------------------- #
# run_grid: live-call counting (single-shot = 1, conversational = up to 2)
# --------------------------------------------------------------------------- #

def test_run_grid_counts_two_live_calls_per_uncached_conversational_cell(tmp_path):
    sources = _sources()
    train_pool = _train_pool()
    pools = build_few_shot_pools(sources, train_pool, k=1)
    out_path = tmp_path / "raw.jsonl"

    summary = run_grid(["p0"], provider=FakeProvider(conv_turns=2), sources=sources,
                       train_pool=train_pool, few_shot_pools=pools, output_path=out_path)

    rows = [json.loads(line) for line in out_path.read_text().splitlines()]
    n_conv = sum(1 for r in rows if r["is_conversational"])
    n_single = sum(1 for r in rows if not r["is_conversational"])
    # single-shot: 1 call each; conversational (uncached): 2 calls each
    assert summary["n_new_api_calls"] == n_single * 1 + n_conv * 2


def test_run_grid_all_cached_conversation_counts_zero_new_calls(tmp_path):
    sources = _sources()
    train_pool = _train_pool()
    pools = build_few_shot_pools(sources, train_pool, k=1)
    out_path = tmp_path / "raw.jsonl"

    provider = FakeProvider(all_cached=True)
    summary = run_grid(["p0"], provider=provider, sources=sources, train_pool=train_pool,
                       few_shot_pools=pools, output_path=out_path)
    assert summary["n_new_api_calls"] == 0
    assert summary["n_written_this_run"] == CELLS_PER_PR

    rows = [json.loads(line) for line in out_path.read_text().splitlines()]
    conv_rows = [r for r in rows if r["is_conversational"]]
    assert all(r["from_cache"] is True for r in conv_rows)


# --------------------------------------------------------------------------- #
# run_grid: quota-exhaustion stop policy
# --------------------------------------------------------------------------- #

def test_run_grid_stops_entire_run_on_quota_exceeded(tmp_path):
    sources = _sources(n_prs=2)
    train_pool = _train_pool(n_prs=2)
    pools = build_few_shot_pools(sources, train_pool, k=1)
    out_path = tmp_path / "raw.jsonl"

    provider = FakeProvider(quota_at=5)
    summary = run_grid(["p0", "p1"], provider=provider, sources=sources, train_pool=train_pool,
                       few_shot_pools=pools, output_path=out_path)

    assert summary["stopped_reason"] is not None
    assert summary["stopped_reason"].startswith("quota_exceeded")
    assert summary["n_written_this_run"] == 5
    rows = out_path.read_text().splitlines()
    assert len(rows) == 5


def test_run_grid_quota_stop_leaves_resumable_file(tmp_path):
    sources = _sources()
    train_pool = _train_pool()
    pools = build_few_shot_pools(sources, train_pool, k=1)
    out_path = tmp_path / "raw.jsonl"

    provider1 = FakeProvider(quota_at=3)
    summary1 = run_grid(["p0"], provider=provider1, sources=sources, train_pool=train_pool,
                        few_shot_pools=pools, output_path=out_path)
    assert summary1["n_written_this_run"] == 3
    assert summary1["stopped_reason"].startswith("quota_exceeded")

    provider2 = FakeProvider()
    summary2 = run_grid(["p0"], provider=provider2, sources=sources, train_pool=train_pool,
                        few_shot_pools=pools, output_path=out_path)
    assert summary2["n_skipped_already_done"] == 3
    assert summary2["n_written_this_run"] == CELLS_PER_PR - 3
    assert summary2["stopped_reason"] is None

    rows = out_path.read_text().splitlines()
    assert len(rows) == CELLS_PER_PR


# --------------------------------------------------------------------------- #
# run_grid: recorded (non-quota) API failures don't stop the run
# --------------------------------------------------------------------------- #

def test_run_grid_records_api_error_and_continues(tmp_path):
    sources = _sources(n_prs=2)
    train_pool = _train_pool(n_prs=2)
    pools = build_few_shot_pools(sources, train_pool, k=1)
    out_path = tmp_path / "raw.jsonl"

    provider = FakeProvider(fail_at=0)
    summary = run_grid(["p0", "p1"], provider=provider, sources=sources, train_pool=train_pool,
                       few_shot_pools=pools, output_path=out_path)

    assert summary["stopped_reason"] is None
    assert summary["n_recorded_failures"] == 1
    assert summary["n_written_this_run"] == 2 * CELLS_PER_PR

    rows = [json.loads(line) for line in out_path.read_text().splitlines()]
    failed = [r for r in rows if r["error"] is not None]
    assert len(failed) == 1
    assert failed[0]["response_text"] is None


# --------------------------------------------------------------------------- #
# run_grid: max_new_calls cap
# --------------------------------------------------------------------------- #

def test_run_grid_respects_max_new_calls(tmp_path):
    sources = _sources(n_prs=2)
    train_pool = _train_pool(n_prs=2)
    pools = build_few_shot_pools(sources, train_pool, k=1)
    out_path = tmp_path / "raw.jsonl"

    provider = FakeProvider()
    summary = run_grid(["p0", "p1"], provider=provider, sources=sources, train_pool=train_pool,
                       few_shot_pools=pools, output_path=out_path, max_new_calls=7)

    assert summary["n_new_api_calls"] >= 7  # may overshoot by up to 1 (a 2-call conv cell)
    assert summary["stopped_reason"] == "max_new_calls reached"


# --------------------------------------------------------------------------- #
# run_grid: cache_only
# --------------------------------------------------------------------------- #

def test_run_grid_cache_only_skips_uncached_cells(tmp_path):
    sources = _sources()
    train_pool = _train_pool()
    pools = build_few_shot_pools(sources, train_pool, k=1)
    out_path = tmp_path / "raw.jsonl"

    class HalfCachedProvider:
        model = "fake-model"

        def __init__(self):
            self._n = 0

        def generate(self, prompt, max_output_tokens=None, cache_metadata=None, cache_only=False):
            self._n += 1
            return None if self._n % 2 == 0 else FakeSingleResponse(from_cache=True)

        def generate_conversation(self, conversation, max_output_tokens=None, cache_metadata=None, cache_only=False):
            self._n += 1
            return None if self._n % 2 == 0 else FakeConversationResponse(from_cache=True)

    summary = run_grid(["p0"], provider=HalfCachedProvider(), sources=sources, train_pool=train_pool,
                       few_shot_pools=pools, output_path=out_path, cache_only=True)
    assert summary["n_written_this_run"] == CELLS_PER_PR // 2
    assert summary["n_cache_misses_skipped"] == CELLS_PER_PR // 2
    assert summary["n_recorded_failures"] == 0
    rows = [json.loads(line) for line in out_path.read_text().splitlines()]
    assert all(r["error"] is None for r in rows)


# --------------------------------------------------------------------------- #
# Real-data smoke tests (skipped if the mined dataset isn't present)
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(
    not Path("data/processed/pull_requests.parquet").exists(), reason="mined dataset not present"
)
def test_stratified_order_is_permutation_and_spreads_repos():
    from src.llm.prompts.exp4_examples import load_ai_test_pool
    from src.llm.run_exp4_grid import stratified_ai_test_pr_order

    order = stratified_ai_test_pr_order()
    test = load_ai_test_pool()

    assert set(order) == set(test["pr_id"])
    assert len(order) == len(test) == len(set(order))
    assert order[:5] == order[:10][:5]  # nested prefix

    repo_of = dict(zip(test["pr_id"], test["repo"]))
    first_n_repos = {repo_of[pid] for pid in order[: test["repo"].nunique()]}
    assert len(first_n_repos) == test["repo"].nunique()
