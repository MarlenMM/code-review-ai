"""Tests for `GroqProvider` and the shared `CachedChatProvider` base in
`src/llm/providers.py`.

`GroqProvider` inherits all of its caching/retry/rate-limit machinery from the
same base class `GeminiProvider` uses (already covered exhaustively in
`tests/test_llm_providers.py`), so this file focuses on what is Groq-specific:
the OpenAI-style request body, response parsing, the `retry-after`-header quota
hint, and the provider-agnostic exception hierarchy the grid runner relies on.
No real network call is made anywhere.
"""

import pytest

from src.llm.prompts.schema import RenderedPrompt, Strategy, Task
from src.llm.providers import (
    GroqAPIError,
    GroqProvider,
    GroqQuotaExceededError,
    GeminiAPIError,
    GeminiQuotaExceededError,
    LLMAPIError,
    LLMQuotaExceededError,
    PROVIDERS,
    _parse_chat_completion,
)
import src.llm.providers as providers_mod


def _prompt(user="user message", system="system message") -> RenderedPrompt:
    return RenderedPrompt(Task.MERGE_PREDICTION, Strategy.ZERO_SHOT, system, user, "contract")


def _chat_payload(text="DECISION: MERGE\nCONFIDENCE: 0.8", finish_reason="stop", usage=None):
    payload = {"choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": finish_reason}]}
    if usage is not None:
        payload["usage"] = usage
    return payload


class FakeResponse:
    def __init__(self, status_code: int, payload: dict, headers: dict | None = None):
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "json": json})
        return self._responses.pop(0)


# --------------------------------------------------------------------------- #
# Construction / registry
# --------------------------------------------------------------------------- #

def test_groq_requires_api_key(monkeypatch, tmp_path):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ValueError, match="No Groq API key"):
        GroqProvider(cache_dir=tmp_path)


def test_groq_accepts_explicit_key(tmp_path):
    p = GroqProvider(api_key="fake-key", cache_dir=tmp_path)
    assert p.provider_name == "Groq"
    assert p.model  # a default model is set


def test_provider_registry_maps_names():
    assert PROVIDERS["groq"] is GroqProvider
    assert set(PROVIDERS) == {"groq", "gemini", "qwen"}


def _provider_flag_default(module) -> str:
    """The `default=` of the `--provider` argument, read out of the runner's
    own source. Both runners build their parser inline in `main()`, so there is
    no parser object to query without executing the CLI -- the AST is the
    honest way to assert on it."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(module.main))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if getattr(node.func, "attr", None) != "add_argument":
            continue
        if not node.args or getattr(node.args[0], "value", None) != "--provider":
            continue
        for kw in node.keywords:
            if kw.arg == "default":
                return kw.value.value
    raise AssertionError(f"no --provider default found in {module.__name__}.main")


@pytest.mark.parametrize("module_name", ["src.llm.run_exp3_grid", "src.llm.run_exp4_grid"])
def test_groq_stays_the_grid_default_so_the_cache_still_replays(module_name):
    """`data/llm_cache/` is keyed on `llama-3.1-8b-instant`, so Labs 3/4 only
    reproduce for free under `--provider groq`. The live path moved to Qwen
    (`src/api/llm_review.py`); the grid default deliberately did NOT, because
    re-running a grid under a different model is a new measurement, not a
    reproduction. This test is the guard on that distinction."""
    import importlib

    module = importlib.import_module(module_name)
    assert _provider_flag_default(module) == "groq"


# --------------------------------------------------------------------------- #
# Request shape: OpenAI-style messages
# --------------------------------------------------------------------------- #

def test_groq_builds_openai_style_body_with_system(tmp_path):
    p = GroqProvider(api_key="fake", cache_dir=tmp_path)
    body = p._build_body("sys instructions", "the user text", 0.2, 128)
    assert body["model"] == p.model
    assert body["messages"] == [
        {"role": "system", "content": "sys instructions"},
        {"role": "user", "content": "the user text"},
    ]
    assert body["temperature"] == 0.2
    assert body["max_tokens"] == 128


def test_groq_omits_empty_system_message(tmp_path):
    p = GroqProvider(api_key="fake", cache_dir=tmp_path)
    body = p._build_body("", "just user", 0.2, 64)
    assert body["messages"] == [{"role": "user", "content": "just user"}]


def test_groq_endpoint_and_auth_header(tmp_path):
    p = GroqProvider(api_key="secret-key", cache_dir=tmp_path)
    assert p._endpoint_url().endswith("/chat/completions")
    assert p._headers()["Authorization"] == "Bearer secret-key"


# --------------------------------------------------------------------------- #
# Response parsing
# --------------------------------------------------------------------------- #

def test_parse_chat_completion_extracts_text_finish_usage():
    raw = _chat_payload(text="hi", finish_reason="stop", usage={"total_tokens": 9})
    text, finish, usage = _parse_chat_completion(raw)
    assert text == "hi"
    assert finish == "stop"
    assert usage == {"total_tokens": 9}


def test_parse_chat_completion_no_choices_raises():
    with pytest.raises(GroqAPIError, match="no choices"):
        _parse_chat_completion({"choices": []})


# --------------------------------------------------------------------------- #
# End-to-end generate() through a FakeSession (real retry loop, no network)
# --------------------------------------------------------------------------- #

def test_groq_generate_success_and_caches(tmp_path):
    session = FakeSession([FakeResponse(200, _chat_payload(text="DECISION: CLOSE"))])
    p = GroqProvider(api_key="fake", cache_dir=tmp_path, session=session, min_interval_seconds=0)
    r1 = p.generate(_prompt())
    assert r1.text == "DECISION: CLOSE"
    assert r1.from_cache is False
    assert len(session.calls) == 1
    # second call served from disk cache -- no new POST
    r2 = p.generate(_prompt())
    assert r2.from_cache is True
    assert len(session.calls) == 1


def test_groq_429_honors_retry_after_header(tmp_path, monkeypatch):
    slept = []
    monkeypatch.setattr(providers_mod.time, "sleep", lambda s: slept.append(s))
    session = FakeSession([
        FakeResponse(429, {"error": {"message": "rate limited", "type": "requests"}}, headers={"retry-after": "13"}),
        FakeResponse(200, _chat_payload()),
    ])
    p = GroqProvider(api_key="fake", cache_dir=tmp_path, session=session, min_interval_seconds=0, max_retries=3)
    p._call_api("sys", "user", 0.2, 100)
    assert slept == [13.0]


def test_groq_429_exhausts_to_quota_error(tmp_path, monkeypatch):
    monkeypatch.setattr(providers_mod.time, "sleep", lambda s: None)
    session = FakeSession([
        FakeResponse(429, {"error": {"message": "quota", "type": "tokens"}}) for _ in range(5)
    ])
    p = GroqProvider(api_key="fake", cache_dir=tmp_path, session=session, min_interval_seconds=0, max_retries=2)
    with pytest.raises(GroqQuotaExceededError):
        p._call_api("sys", "user", 0.2, 100)
    assert len(session.calls) == 3  # initial + 2 retries


def test_groq_daily_quota_429_stops_immediately_without_retry(tmp_path, monkeypatch):
    """A per-DAY 429 can't clear within a retry window, so it must raise the
    quota error on the FIRST hit (no futile ~8-min grind), unlike a per-minute
    429 which retries."""
    monkeypatch.setattr(providers_mod.time, "sleep", lambda s: None)
    daily_msg = ("Rate limit reached ... on tokens per day (TPD): "
                 "Limit 500000, Used 499987, Requested 2913.")
    session = FakeSession([FakeResponse(429, {"error": {"message": daily_msg}}) for _ in range(5)])
    p = GroqProvider(api_key="fake", cache_dir=tmp_path, session=session, min_interval_seconds=0, max_retries=5)
    with pytest.raises(GroqQuotaExceededError, match="daily quota exhausted"):
        p._call_api("sys", "user", 0.2, 100)
    assert len(session.calls) == 1  # NOT retried


def test_groq_non_retryable_400_raises_api_error(tmp_path):
    session = FakeSession([FakeResponse(400, {"error": {"message": "bad request", "type": "invalid_request_error"}})])
    p = GroqProvider(api_key="fake", cache_dir=tmp_path, session=session, min_interval_seconds=0)
    with pytest.raises(GroqAPIError, match="bad request"):
        p._call_api("sys", "user", 0.2, 100)
    assert len(session.calls) == 1


class FlakySession:
    """Raises a connection-level exception on the first `n_fail` posts, then
    returns a scripted success -- models a transient VPN/proxy drop."""

    def __init__(self, n_fail, success_payload):
        self.n_fail = n_fail
        self.success_payload = success_payload
        self.n_calls = 0

    def post(self, url, headers=None, json=None, timeout=None):
        self.n_calls += 1
        if self.n_calls <= self.n_fail:
            raise providers_mod.requests.exceptions.ProxyError("Unable to connect to proxy")
        return FakeResponse(200, self.success_payload)


def test_groq_retries_transient_connection_error_then_succeeds(tmp_path, monkeypatch):
    monkeypatch.setattr(providers_mod.time, "sleep", lambda s: None)
    session = FlakySession(n_fail=2, success_payload=_chat_payload(text="recovered"))
    p = GroqProvider(api_key="fake", cache_dir=tmp_path, session=session, min_interval_seconds=0, max_retries=3)
    raw = p._call_api("sys", "user", 0.2, 100)
    assert raw["choices"][0]["message"]["content"] == "recovered"
    assert session.n_calls == 3  # 2 failures + 1 success


def test_groq_connection_error_past_retries_raises_api_error(tmp_path, monkeypatch):
    monkeypatch.setattr(providers_mod.time, "sleep", lambda s: None)
    session = FlakySession(n_fail=99, success_payload=_chat_payload())
    p = GroqProvider(api_key="fake", cache_dir=tmp_path, session=session, min_interval_seconds=0, max_retries=2)
    # exhausted retries surface as a (recordable, non-fatal) APIError, not a
    # raw ProxyError that would crash the grid run
    with pytest.raises(GroqAPIError, match="connection error"):
        p._call_api("sys", "user", 0.2, 100)
    assert session.n_calls == 3  # initial + 2 retries


# --------------------------------------------------------------------------- #
# Provider-agnostic exception hierarchy (what run_exp3_grid catches)
# --------------------------------------------------------------------------- #

def test_exception_hierarchy_is_provider_agnostic():
    # both providers' errors are catchable via the shared base classes
    assert issubclass(GroqAPIError, LLMAPIError)
    assert issubclass(GroqQuotaExceededError, LLMQuotaExceededError)
    assert issubclass(GroqQuotaExceededError, LLMAPIError)
    assert issubclass(GeminiAPIError, LLMAPIError)
    assert issubclass(GeminiQuotaExceededError, LLMQuotaExceededError)
    # a quota error is a kind of api error (so `except LLMAPIError` catches both)
    assert issubclass(LLMQuotaExceededError, LLMAPIError)


def test_cache_key_differs_by_model_so_providers_never_collide(tmp_path):
    """Groq and Gemini (or two Groq models) can share one cache_dir because the
    model name is part of the content-addressed key."""
    groq_a = GroqProvider(api_key="fake", model="llama-3.1-8b-instant", cache_dir=tmp_path)
    groq_b = GroqProvider(api_key="fake", model="llama-3.3-70b-versatile", cache_dir=tmp_path)
    k_a = groq_a._cache_key("sys", "user", 0.2, 100)
    k_b = groq_b._cache_key("sys", "user", 0.2, 100)
    assert k_a != k_b
