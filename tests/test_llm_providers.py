"""Tests for `src/llm/providers.py`'s `GeminiProvider`: disk caching (the
part that guarantees nothing is ever paid for twice), retry/backoff, and the
distinction between transient errors, quota exhaustion, and non-retryable
errors.

No real network call is made anywhere in this file. The one live HTTP call
`GeminiProvider` makes is isolated in `_call_api`, which either gets
monkeypatched directly (to test `generate`'s caching logic) or driven through
a `FakeSession` that returns scripted `FakeResponse`s (to test `_call_api`'s
own retry loop) -- the same two-level approach already used for
`GitHubClient` in `tests/test_github_client.py`.
"""

import pytest

from src.llm.prompts.schema import RenderedPrompt, Strategy, Task
from src.llm.providers import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_TEMPERATURE,
    GeminiAPIError,
    GeminiProvider,
    GeminiQuotaExceededError,
    MAX_BACKOFF_SECONDS,
    _backoff_delay,
    _parse_generate_response,
    _parse_retry_delay,
)
import src.llm.providers as providers_mod


def _prompt(user="user message", system="system message") -> RenderedPrompt:
    return RenderedPrompt(Task.MERGE_PREDICTION, Strategy.ZERO_SHOT, system, user, "contract")


def _candidate_payload(text="DECISION: MERGE\nCONFIDENCE: 0.8", finish_reason="STOP", usage=None):
    payload = {"candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": finish_reason}]}
    if usage is not None:
        payload["usageMetadata"] = usage
    return payload


class FakeResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload
        self.headers: dict = {}

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "json": json})
        return self._responses.pop(0)


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #

def test_provider_requires_api_key(monkeypatch, tmp_path):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="No Gemini API key"):
        GeminiProvider(cache_dir=tmp_path)


def test_provider_accepts_explicit_api_key(tmp_path):
    GeminiProvider(api_key="fake-key-for-test", cache_dir=tmp_path)  # must not raise


# --------------------------------------------------------------------------- #
# Cache key / path
# --------------------------------------------------------------------------- #

def test_cache_key_deterministic(tmp_path):
    provider = GeminiProvider(api_key="fake", cache_dir=tmp_path)
    k1 = provider._cache_key("sys", "user", 0.2, 100)
    k2 = provider._cache_key("sys", "user", 0.2, 100)
    assert k1 == k2


def test_cache_key_changes_with_any_input(tmp_path):
    provider = GeminiProvider(api_key="fake", cache_dir=tmp_path)
    base = provider._cache_key("sys", "user", 0.2, 100)
    assert provider._cache_key("sys2", "user", 0.2, 100) != base
    assert provider._cache_key("sys", "user2", 0.2, 100) != base
    assert provider._cache_key("sys", "user", 0.5, 100) != base
    assert provider._cache_key("sys", "user", 0.2, 200) != base


def test_cache_path_sharded_by_hash_prefix(tmp_path):
    provider = GeminiProvider(api_key="fake", cache_dir=tmp_path)
    key = "abcdef0123456789"
    path = provider._cache_path(key)
    assert path == tmp_path / "ab" / f"{key}.json"
    assert path.parent.exists()


# --------------------------------------------------------------------------- #
# generate(): caching behavior (mocking _call_api, like GitHubClient's tests
# mock _graphql/_rest_get_paginated)
# --------------------------------------------------------------------------- #

def test_generate_cache_miss_then_hit(tmp_path, monkeypatch):
    provider = GeminiProvider(api_key="fake", cache_dir=tmp_path, min_interval_seconds=0)
    calls = {"n": 0}

    def fake_call(system, user, temperature, max_output_tokens):
        calls["n"] += 1
        return _candidate_payload()

    monkeypatch.setattr(provider, "_call_api", fake_call)
    prompt = _prompt()

    r1 = provider.generate(prompt)
    r2 = provider.generate(prompt)

    assert r1.from_cache is False
    assert r2.from_cache is True
    assert r1.text == r2.text == "DECISION: MERGE\nCONFIDENCE: 0.8"
    assert r1.cache_key == r2.cache_key
    assert calls["n"] == 1


def test_generate_cache_key_ignores_metadata(tmp_path, monkeypatch):
    """Different cache_metadata for the textually-identical prompt must still
    hit the same cache entry -- metadata is bookkeeping, not part of the key."""
    provider = GeminiProvider(api_key="fake", cache_dir=tmp_path, min_interval_seconds=0)
    calls = {"n": 0}

    def fake_call(system, user, temperature, max_output_tokens):
        calls["n"] += 1
        return _candidate_payload()

    monkeypatch.setattr(provider, "_call_api", fake_call)
    prompt = _prompt()

    provider.generate(prompt, cache_metadata={"pr_id": "p1"})
    r2 = provider.generate(prompt, cache_metadata={"pr_id": "p2", "task": "different"})

    assert r2.from_cache is True
    assert calls["n"] == 1


def test_generate_different_prompt_different_cache_entry(tmp_path, monkeypatch):
    provider = GeminiProvider(api_key="fake", cache_dir=tmp_path, min_interval_seconds=0)
    calls = {"n": 0}

    def fake_call(system, user, temperature, max_output_tokens):
        calls["n"] += 1
        return _candidate_payload(text=f"response {calls['n']}")

    monkeypatch.setattr(provider, "_call_api", fake_call)

    r1 = provider.generate(_prompt(user="prompt A"))
    r2 = provider.generate(_prompt(user="prompt B"))

    assert r1.cache_key != r2.cache_key
    assert calls["n"] == 2


def test_generate_force_refresh_bypasses_cache(tmp_path, monkeypatch):
    provider = GeminiProvider(api_key="fake", cache_dir=tmp_path, min_interval_seconds=0)
    calls = {"n": 0}

    def fake_call(system, user, temperature, max_output_tokens):
        calls["n"] += 1
        return _candidate_payload()

    monkeypatch.setattr(provider, "_call_api", fake_call)
    prompt = _prompt()

    provider.generate(prompt)
    provider.generate(prompt, force_refresh=True)

    assert calls["n"] == 2


def test_generate_writes_cache_file_with_expected_fields(tmp_path, monkeypatch):
    provider = GeminiProvider(api_key="fake", cache_dir=tmp_path, min_interval_seconds=0)
    monkeypatch.setattr(
        provider, "_call_api",
        lambda system, user, temperature, max_output_tokens: _candidate_payload(usage={"totalTokenCount": 7}),
    )
    prompt = _prompt()
    result = provider.generate(prompt, cache_metadata={"pr_id": "p1", "task": "merge_prediction"})

    path = provider._cache_path(result.cache_key)
    assert path.exists()
    import json
    record = json.loads(path.read_text())
    assert record["response_text"] == result.text
    assert record["model"] == provider.model
    assert record["usage"] == {"totalTokenCount": 7}
    assert record["metadata"] == {"pr_id": "p1", "task": "merge_prediction"}
    assert "cached_at" in record and "latency_ms" in record


def test_generate_corrupt_cache_file_triggers_recall(tmp_path, monkeypatch):
    provider = GeminiProvider(api_key="fake", cache_dir=tmp_path, min_interval_seconds=0)
    prompt = _prompt()
    key = provider._cache_key(prompt.system, prompt.user, DEFAULT_TEMPERATURE, DEFAULT_MAX_OUTPUT_TOKENS)
    provider._cache_path(key).write_text("{not valid json")

    calls = {"n": 0}

    def fake_call(system, user, temperature, max_output_tokens):
        calls["n"] += 1
        return _candidate_payload()

    monkeypatch.setattr(provider, "_call_api", fake_call)
    result = provider.generate(prompt)

    assert calls["n"] == 1
    assert result.from_cache is False


# --------------------------------------------------------------------------- #
# _call_api: retry loop driven through a fake session (no _call_api mocking)
# --------------------------------------------------------------------------- #

def test_call_api_success_first_try(tmp_path):
    session = FakeSession([FakeResponse(200, _candidate_payload())])
    provider = GeminiProvider(api_key="fake", cache_dir=tmp_path, session=session, min_interval_seconds=0)
    raw = provider._call_api("sys", "user", 0.2, 100)
    assert raw["candidates"][0]["content"]["parts"][0]["text"].startswith("DECISION")
    assert len(session.calls) == 1


def test_call_api_retries_on_500_then_succeeds(tmp_path, monkeypatch):
    monkeypatch.setattr(providers_mod.time, "sleep", lambda s: None)
    session = FakeSession([
        FakeResponse(500, {"error": {"status": "INTERNAL", "message": "boom"}}),
        FakeResponse(200, _candidate_payload()),
    ])
    provider = GeminiProvider(api_key="fake", cache_dir=tmp_path, session=session, min_interval_seconds=0)
    raw = provider._call_api("sys", "user", 0.2, 100)
    assert len(session.calls) == 2
    assert raw["candidates"]


def test_call_api_429_exhausts_retries_raises_quota_error(tmp_path, monkeypatch):
    monkeypatch.setattr(providers_mod.time, "sleep", lambda s: None)
    quota_payload = {"error": {"status": "RESOURCE_EXHAUSTED", "message": "quota exceeded"}}
    session = FakeSession([FakeResponse(429, quota_payload) for _ in range(10)])
    provider = GeminiProvider(
        api_key="fake", cache_dir=tmp_path, session=session, min_interval_seconds=0, max_retries=2,
    )
    with pytest.raises(GeminiQuotaExceededError, match="quota"):
        provider._call_api("sys", "user", 0.2, 100)
    assert len(session.calls) == 3  # initial attempt + 2 retries


def test_call_api_429_honors_retry_delay_hint(tmp_path, monkeypatch):
    slept = []
    monkeypatch.setattr(providers_mod.time, "sleep", lambda s: slept.append(s))
    err_payload = {
        "error": {
            "status": "RESOURCE_EXHAUSTED",
            "message": "quota",
            "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "7s"}],
        }
    }
    session = FakeSession([FakeResponse(429, err_payload), FakeResponse(200, _candidate_payload())])
    provider = GeminiProvider(
        api_key="fake", cache_dir=tmp_path, session=session, min_interval_seconds=0, max_retries=3,
    )
    provider._call_api("sys", "user", 0.2, 100)
    assert slept == [7.0]


def test_call_api_non_retryable_400_raises_immediately(tmp_path):
    payload = {"error": {"status": "FAILED_PRECONDITION",
                          "message": "User location is not supported for the API use."}}
    session = FakeSession([FakeResponse(400, payload)])
    provider = GeminiProvider(api_key="fake", cache_dir=tmp_path, session=session, min_interval_seconds=0)
    with pytest.raises(GeminiAPIError, match="location"):
        provider._call_api("sys", "user", 0.2, 100)
    assert len(session.calls) == 1  # no retries for a non-retryable status


# --------------------------------------------------------------------------- #
# Response parsing
# --------------------------------------------------------------------------- #

def test_parse_generate_response_joins_multiple_parts():
    raw = {"candidates": [{"content": {"parts": [{"text": "hello "}, {"text": "world"}]},
                            "finishReason": "STOP"}],
           "usageMetadata": {"totalTokenCount": 5}}
    text, finish_reason, usage = _parse_generate_response(raw)
    assert text == "hello world"
    assert finish_reason == "STOP"
    assert usage == {"totalTokenCount": 5}


def test_parse_generate_response_no_candidates_raises_with_block_reason():
    with pytest.raises(GeminiAPIError, match="SAFETY"):
        _parse_generate_response({"promptFeedback": {"blockReason": "SAFETY"}})


# --------------------------------------------------------------------------- #
# _parse_retry_delay / _backoff_delay
# --------------------------------------------------------------------------- #

def test_parse_retry_delay_extracts_seconds():
    error = {"details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "38s"}]}
    assert _parse_retry_delay(error) == 38.0


def test_parse_retry_delay_absent_returns_none():
    assert _parse_retry_delay({}) is None
    assert _parse_retry_delay({"details": []}) is None


def test_backoff_delay_respects_retry_after():
    assert _backoff_delay(3, retry_after=12.0) == 12.0


def test_backoff_delay_grows_and_caps():
    early = _backoff_delay(0)
    late = _backoff_delay(20)
    assert 0 < early < late <= MAX_BACKOFF_SECONDS


# --------------------------------------------------------------------------- #
# Rate limiting
# --------------------------------------------------------------------------- #

def test_respect_rate_limit_sleeps_for_remaining_interval(tmp_path, monkeypatch):
    provider = GeminiProvider(api_key="fake", cache_dir=tmp_path, min_interval_seconds=5.0)
    monkeypatch.setattr(providers_mod.time, "monotonic", lambda: 100.0)
    slept = []
    monkeypatch.setattr(providers_mod.time, "sleep", lambda s: slept.append(s))

    provider._last_call_at = 98.0  # 2s ago; interval is 5s -> should sleep ~3s
    provider._respect_rate_limit()
    assert slept == pytest.approx([3.0])


def test_respect_rate_limit_no_sleep_once_interval_elapsed(tmp_path, monkeypatch):
    provider = GeminiProvider(api_key="fake", cache_dir=tmp_path, min_interval_seconds=5.0)
    monkeypatch.setattr(providers_mod.time, "monotonic", lambda: 200.0)
    slept = []
    monkeypatch.setattr(providers_mod.time, "sleep", lambda s: slept.append(s))

    provider._last_call_at = 100.0  # 100s ago, well past the interval
    provider._respect_rate_limit()
    assert slept == []


def test_respect_rate_limit_first_call_never_sleeps(tmp_path, monkeypatch):
    provider = GeminiProvider(api_key="fake", cache_dir=tmp_path, min_interval_seconds=5.0)
    slept = []
    monkeypatch.setattr(providers_mod.time, "sleep", lambda s: slept.append(s))
    assert provider._last_call_at is None
    provider._respect_rate_limit()
    assert slept == []
