"""Tests for `src/llm/run_exp3_grid.py`: grid iteration, resumability
(row-skip), and the quota-aware stop policy -- all against a fake, in-memory
provider. No real network call or API key is used anywhere in this file.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.llm.context_builder import ContextKind
from src.llm.prompts.few_shot import ExampleSources
from src.llm.prompts.schema import Strategy, Task
from src.llm.providers import GeminiAPIError, GeminiQuotaExceededError
from src.llm.run_exp3_grid import build_few_shot_pools, load_existing_keys, run_grid

N_CONTEXT_KINDS = len(list(ContextKind))
N_TASKS = len(list(Task))
N_STRATEGIES = len(list(Strategy))
CELLS_PER_PR = N_CONTEXT_KINDS * N_TASKS * N_STRATEGIES  # 4 * 2 * 4 = 32


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #

@dataclass
class FakeResponse:
    text: str
    model: str
    from_cache: bool
    latency_ms: float
    usage: dict | None = None


class FakeProvider:
    """A minimal stand-in for GeminiProvider.generate: no network, fully
    deterministic, and can be told to raise on a specific call index to
    simulate a quota wall or a transient API error partway through a run."""

    def __init__(self, model="fake-model", fail_at=None, quota_at=None):
        self.model = model
        self.calls: list[tuple[str, dict]] = []
        self.fail_at = fail_at
        self.quota_at = quota_at
        self._n = 0

    def generate(self, prompt, cache_metadata=None, cache_only=False):
        idx = self._n
        self._n += 1
        self.calls.append((prompt.label, dict(cache_metadata or {})))
        if self.quota_at is not None and idx == self.quota_at:
            raise GeminiQuotaExceededError("simulated quota exhaustion")
        if self.fail_at is not None and idx == self.fail_at:
            raise GeminiAPIError("simulated api error")
        return FakeResponse(text=f"response-{idx}", model=self.model, from_cache=False, latency_ms=1.0)


def _sources(n_prs=2):
    pull_requests = pd.DataFrame([
        {"id": f"p{i}", "repo": "acme/widgets", "title": f"Title {i}",
         "body": f"Body {i}", "labels": np.array(["bug"]), "additions": 5, "deletions": 1}
        for i in range(n_prs)
    ])
    files_changed = pd.DataFrame([
        {"pr_id": f"p{i}", "filename": "a.py", "status": "modified",
         "additions": 5, "deletions": 1, "patch": f"@@ -1 +1 @@\n-old{i}\n+new{i}"}
        for i in range(n_prs)
    ])
    commits = pd.DataFrame([
        {"pr_id": f"p{i}", "sha": f"s{i}", "message": f"Fix {i}", "author_name": "a",
         "author_email": "a@x.com", "author_login": "alice", "additions": 5, "deletions": 1}
        for i in range(n_prs)
    ])
    review_comments = pd.DataFrame([
        {"pr_id": f"p{i}", "author_login": "bob",
         "body": f"Please double-check the edge case in change {i}."}
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
        {"pr_id": "p0", "context_kind": "diff_only", "strategy": "zero_shot", "task": "merge_prediction"},
        {"pr_id": "p1", "context_kind": "diff_metadata", "strategy": "cot", "task": "review_comment"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    keys = load_existing_keys(path)
    assert keys == {
        ("p0", "diff_only", "zero_shot", "merge_prediction"),
        ("p1", "diff_metadata", "cot", "review_comment"),
    }


def test_load_existing_keys_skips_blank_lines(tmp_path):
    path = tmp_path / "raw.jsonl"
    row = {"pr_id": "p0", "context_kind": "diff_only", "strategy": "zero_shot", "task": "merge_prediction"}
    path.write_text(json.dumps(row) + "\n\n\n")
    assert len(load_existing_keys(path)) == 1


# --------------------------------------------------------------------------- #
# build_few_shot_pools
# --------------------------------------------------------------------------- #

def test_build_few_shot_pools_covers_every_task_and_kind():
    sources = _sources()
    train_pool = _train_pool()
    pools = build_few_shot_pools(sources, train_pool, k=2)
    for kind in ContextKind:
        assert (Task.MERGE_PREDICTION.value, kind.value) in pools
        assert (Task.REVIEW_COMMENT.value, kind.value) in pools
    merge_examples = pools[(Task.MERGE_PREDICTION.value, ContextKind.DIFF_ONLY.value)]
    assert all(ex.label in {"MERGE", "CLOSE"} for ex in merge_examples)


# --------------------------------------------------------------------------- #
# run_grid: basic coverage and idempotence
# --------------------------------------------------------------------------- #

def test_run_grid_writes_every_cell_for_full_run(tmp_path):
    sources = _sources(n_prs=2)
    train_pool = _train_pool(n_prs=2)
    pools = build_few_shot_pools(sources, train_pool, k=2)
    provider = FakeProvider()
    out_path = tmp_path / "raw.jsonl"

    summary = run_grid(
        ["p0", "p1"], provider=provider, sources=sources, few_shot_pools=pools, output_path=out_path,
    )

    assert summary["total_cells"] == 2 * CELLS_PER_PR
    assert summary["n_written_this_run"] == 2 * CELLS_PER_PR
    assert summary["n_skipped_already_done"] == 0
    assert summary["n_new_api_calls"] == 2 * CELLS_PER_PR
    assert summary["stopped_reason"] is None

    rows = [json.loads(line) for line in out_path.read_text().splitlines()]
    assert len(rows) == 2 * CELLS_PER_PR
    # every cell of the grid is present exactly once
    keys = {(r["pr_id"], r["context_kind"], r["strategy"], r["task"]) for r in rows}
    assert len(keys) == 2 * CELLS_PER_PR


def test_run_grid_is_idempotent_on_rerun(tmp_path):
    # 2 PRs in the source pool (one merged, one not) so k=1 few-shot selection
    # has a non-empty class to draw from; run_grid below still only iterates
    # a single PR ("p0") -- these tests exercise grid iteration/resumability,
    # not few-shot selection correctness (covered in test_llm_few_shot.py).
    sources = _sources()
    train_pool = _train_pool()
    pools = build_few_shot_pools(sources, train_pool, k=1)
    out_path = tmp_path / "raw.jsonl"

    provider1 = FakeProvider()
    run_grid(["p0"], provider=provider1, sources=sources, few_shot_pools=pools, output_path=out_path)

    provider2 = FakeProvider()
    summary2 = run_grid(["p0"], provider=provider2, sources=sources, few_shot_pools=pools, output_path=out_path)

    assert summary2["n_skipped_already_done"] == CELLS_PER_PR
    assert summary2["n_written_this_run"] == 0
    assert len(provider2.calls) == 0  # no new calls made at all
    rows = out_path.read_text().splitlines()
    assert len(rows) == CELLS_PER_PR  # file wasn't duplicated


def test_run_grid_resumes_partial_output(tmp_path):
    """A pre-existing partial raw.jsonl (as if a prior run stopped early)
    is respected: already-recorded cells are skipped, the rest are filled in."""
    # 2 PRs in the source pool (one merged, one not) so k=1 few-shot selection
    # has a non-empty class to draw from; run_grid below still only iterates
    # a single PR ("p0") -- these tests exercise grid iteration/resumability,
    # not few-shot selection correctness (covered in test_llm_few_shot.py).
    sources = _sources()
    train_pool = _train_pool()
    pools = build_few_shot_pools(sources, train_pool, k=1)
    out_path = tmp_path / "raw.jsonl"

    partial_row = {
        "pr_id": "p0", "repo": "acme/widgets", "context_kind": ContextKind.DIFF_ONLY.value,
        "task": Task.MERGE_PREDICTION.value, "strategy": Strategy.ZERO_SHOT.value,
        "response_text": "DECISION: MERGE", "error": None, "from_cache": False,
        "latency_ms": 5.0, "model": "some-earlier-model",
    }
    out_path.write_text(json.dumps(partial_row) + "\n")

    provider = FakeProvider()
    summary = run_grid(["p0"], provider=provider, sources=sources, few_shot_pools=pools, output_path=out_path)

    assert summary["n_skipped_already_done"] == 1
    assert summary["n_written_this_run"] == CELLS_PER_PR - 1
    rows = [json.loads(line) for line in out_path.read_text().splitlines()]
    assert len(rows) == CELLS_PER_PR
    # the pre-existing row's response is untouched, not overwritten
    kept = [r for r in rows if r["strategy"] == "zero_shot" and r["context_kind"] == "diff_only"]
    assert kept[0]["response_text"] == "DECISION: MERGE"


# --------------------------------------------------------------------------- #
# run_grid: quota-exhaustion stop policy
# --------------------------------------------------------------------------- #

def test_run_grid_stops_entire_run_on_quota_exceeded(tmp_path):
    sources = _sources(n_prs=2)
    train_pool = _train_pool(n_prs=2)
    pools = build_few_shot_pools(sources, train_pool, k=2)
    out_path = tmp_path / "raw.jsonl"

    provider = FakeProvider(quota_at=5)  # the 6th call (index 5) raises quota error
    summary = run_grid(["p0", "p1"], provider=provider, sources=sources, few_shot_pools=pools, output_path=out_path)

    assert summary["stopped_reason"] is not None
    assert summary["stopped_reason"].startswith("quota_exceeded")
    assert summary["n_written_this_run"] == 5  # the 5 successful calls before the stop
    assert summary["n_new_api_calls"] == 5
    rows = out_path.read_text().splitlines()
    assert len(rows) == 5


def test_run_grid_quota_stop_leaves_resumable_file(tmp_path):
    """After a quota stop, re-running with a provider that never fails
    completes the rest of the grid, touching none of the already-written rows."""
    # 2 PRs in the source pool (one merged, one not) so k=1 few-shot selection
    # has a non-empty class to draw from; run_grid below still only iterates
    # a single PR ("p0") -- these tests exercise grid iteration/resumability,
    # not few-shot selection correctness (covered in test_llm_few_shot.py).
    sources = _sources()
    train_pool = _train_pool()
    pools = build_few_shot_pools(sources, train_pool, k=1)
    out_path = tmp_path / "raw.jsonl"

    provider1 = FakeProvider(quota_at=3)
    summary1 = run_grid(["p0"], provider=provider1, sources=sources, few_shot_pools=pools, output_path=out_path)
    assert summary1["n_written_this_run"] == 3
    assert summary1["stopped_reason"].startswith("quota_exceeded")

    provider2 = FakeProvider()
    summary2 = run_grid(["p0"], provider=provider2, sources=sources, few_shot_pools=pools, output_path=out_path)
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
    pools = build_few_shot_pools(sources, train_pool, k=2)
    out_path = tmp_path / "raw.jsonl"

    provider = FakeProvider(fail_at=0)  # first cell fails, rest should succeed
    summary = run_grid(["p0", "p1"], provider=provider, sources=sources, few_shot_pools=pools, output_path=out_path)

    assert summary["stopped_reason"] is None
    assert summary["n_recorded_failures"] == 1
    assert summary["n_written_this_run"] == 2 * CELLS_PER_PR  # the failure is still a written row

    rows = [json.loads(line) for line in out_path.read_text().splitlines()]
    failed = [r for r in rows if r["error"] is not None]
    assert len(failed) == 1
    assert failed[0]["response_text"] is None
    assert failed[0]["from_cache"] is None


# --------------------------------------------------------------------------- #
# run_grid: max_new_calls cap
# --------------------------------------------------------------------------- #

def test_run_grid_respects_max_new_calls(tmp_path):
    sources = _sources(n_prs=2)
    train_pool = _train_pool(n_prs=2)
    pools = build_few_shot_pools(sources, train_pool, k=2)
    out_path = tmp_path / "raw.jsonl"

    provider = FakeProvider()
    summary = run_grid(
        ["p0", "p1"], provider=provider, sources=sources, few_shot_pools=pools,
        output_path=out_path, max_new_calls=7,
    )

    assert summary["n_new_api_calls"] == 7
    assert summary["n_written_this_run"] == 7
    assert summary["stopped_reason"] == "max_new_calls reached"


def test_run_grid_from_cache_responses_do_not_count_as_new_calls(tmp_path):
    # 2 PRs in the source pool (one merged, one not) so k=1 few-shot selection
    # has a non-empty class to draw from; run_grid below still only iterates
    # a single PR ("p0") -- these tests exercise grid iteration/resumability,
    # not few-shot selection correctness (covered in test_llm_few_shot.py).
    sources = _sources()
    train_pool = _train_pool()
    pools = build_few_shot_pools(sources, train_pool, k=1)
    out_path = tmp_path / "raw.jsonl"

    class AllCachedProvider:
        model = "fake-model"

        def generate(self, prompt, cache_metadata=None, cache_only=False):
            return FakeResponse(text="cached answer", model=self.model, from_cache=True, latency_ms=0.0)

    summary = run_grid(
        ["p0"], provider=AllCachedProvider(), sources=sources, few_shot_pools=pools, output_path=out_path,
    )
    assert summary["n_new_api_calls"] == 0
    assert summary["n_written_this_run"] == CELLS_PER_PR


def test_run_grid_cache_only_skips_uncached_cells(tmp_path):
    """cache_only=True: a provider that returns None on a miss produces no rows
    for those cells (they are left for a future live run), not error rows."""
    sources = _sources()
    train_pool = _train_pool()
    pools = build_few_shot_pools(sources, train_pool, k=1)
    out_path = tmp_path / "raw.jsonl"

    class HalfCachedProvider:
        model = "fake-model"

        def __init__(self):
            self._n = 0

        def generate(self, prompt, cache_metadata=None, cache_only=False):
            self._n += 1
            if self._n % 2 == 0:  # half are "cache misses"
                return None
            return FakeResponse(text="cached", model=self.model, from_cache=True,
                                latency_ms=0.0, usage={"total_time": 0.5})

    summary = run_grid(
        ["p0"], provider=HalfCachedProvider(), sources=sources, few_shot_pools=pools,
        output_path=out_path, cache_only=True,
    )
    # only the cached half is written; the misses are skipped, not errored
    assert summary["n_written_this_run"] == CELLS_PER_PR // 2
    assert summary["n_cache_misses_skipped"] == CELLS_PER_PR // 2
    assert summary["n_recorded_failures"] == 0
    rows = [json.loads(line) for line in out_path.read_text().splitlines()]
    assert all(r["error"] is None for r in rows)
    assert all(r["usage"] == {"total_time": 0.5} for r in rows)


# --------------------------------------------------------------------------- #
# Real-data smoke test (skipped if the mined dataset isn't present) --
# validates load_test_pr_ids against the actual Experiment-2 split, no
# network call.
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(
    not Path("data/processed/features_v1.parquet").exists(), reason="mined dataset not present"
)
def test_load_test_pr_ids_matches_exp2_test_split():
    from src.llm.run_exp3_grid import load_test_pr_ids
    from src.ml.common import load_variant, split_by_repo_time

    pr_ids = load_test_pr_ids()
    _train, test, _ = split_by_repo_time(load_variant("v1"))
    assert set(pr_ids) == set(test["pr_id"])
    assert len(pr_ids) == len(test)


@pytest.mark.skipif(
    not Path("data/processed/features_v1.parquet").exists(), reason="mined dataset not present"
)
def test_stratified_order_is_permutation_nested_and_spreads_repos():
    from src.llm.run_exp3_grid import stratified_test_pr_order
    from src.ml.common import load_variant, split_by_repo_time

    order = stratified_test_pr_order()
    _train, test, _ = split_by_repo_time(load_variant("v1"))

    # it's a permutation of the full test set (no PR dropped or duplicated)
    assert set(order) == set(test["pr_id"])
    assert len(order) == len(test) == len(set(order))

    # nested: first N is a prefix of first N+k (so a larger --pr-sample reuses work)
    assert order[:10] == order[:20][:10]

    # a small prefix is spread across multiple repos, not dominated by one
    repo_of = dict(zip(test["pr_id"], test["repo"]))
    first20_repos = {repo_of[pid] for pid in order[:20]}
    n_repos = test["repo"].nunique()
    assert len(first20_repos) == min(n_repos, 20)  # all repos represented early
