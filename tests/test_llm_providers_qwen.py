"""Tests for `QwenProvider` (Alibaba Cloud DashScope) in `src/llm/providers.py`.

`QwenProvider` inherits all of its caching/retry/rate-limit machinery from the
same `CachedChatProvider` base `GeminiProvider` and `GroqProvider` use (covered
exhaustively in `tests/test_llm_providers.py`), so this file covers only what is
Qwen-specific: key/region resolution, the OpenAI-compatible request body,
DashScope's own 429 vocabulary, and the fact that switching to it cannot
disturb the Groq-keyed Experiment 3/4 cache. No real network call is made
anywhere.
"""

import pytest

from src.llm.prompts.schema import RenderedPrompt, Strategy, Task
from src.llm.providers import (
    PROVIDERS,
    QWEN_BASE_URL,
    QWEN_DEFAULT_MODEL,
    QWEN_INTL_BASE_URL,
    GroqProvider,
    LLMAPIError,
    LLMQuotaExceededError,
    QwenAPIError,
    QwenProvider,
    QwenQuotaExceededError,
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


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """These tests assert on key/region resolution, so a real DashScope setup in
    the developer's own `.env` must not leak in and make them pass for the
    wrong reason. `providers.py` calls `load_dotenv()` at import, so the values
    are already in `os.environ` by now and have to be removed, not just unset."""
    for var in ("DASHSCOPE_API_KEY", "QWEN_API_KEY", "DASHSCOPE_BASE_URL", "QWEN_MODEL"):
        monkeypatch.delenv(var, raising=False)


# --------------------------------------------------------------------------- #
# Key and region resolution
# --------------------------------------------------------------------------- #

def test_qwen_requires_api_key(tmp_path):
    with pytest.raises(ValueError, match="No Qwen/DashScope API key"):
        QwenProvider(cache_dir=tmp_path)


def test_missing_key_error_mentions_the_regional_endpoint(tmp_path):
    """A Singapore key against the Beijing endpoint fails as a 401, which reads
    like a bad key rather than a wrong region -- so the error says so upfront."""
    with pytest.raises(ValueError, match="DASHSCOPE_BASE_URL"):
        QwenProvider(cache_dir=tmp_path)


def test_qwen_reads_dashscope_api_key(monkeypatch, tmp_path):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "from-dashscope-var")
    assert QwenProvider(cache_dir=tmp_path).api_key == "from-dashscope-var"


def test_qwen_accepts_qwen_api_key_as_an_alias(monkeypatch, tmp_path):
    monkeypatch.setenv("QWEN_API_KEY", "from-qwen-var")
    assert QwenProvider(cache_dir=tmp_path).api_key == "from-qwen-var"


def test_dashscope_key_wins_when_both_are_set(monkeypatch, tmp_path):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "canonical")
    monkeypatch.setenv("QWEN_API_KEY", "alias")
    assert QwenProvider(cache_dir=tmp_path).api_key == "canonical"


def test_explicit_key_beats_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "from-env")
    assert QwenProvider(api_key="explicit", cache_dir=tmp_path).api_key == "explicit"


def test_defaults_to_the_beijing_endpoint(tmp_path):
    p = QwenProvider(api_key="fake", cache_dir=tmp_path)
    assert p.base_url == QWEN_BASE_URL
    assert p.model == QWEN_DEFAULT_MODEL


def test_region_is_overridable_without_a_code_change(monkeypatch, tmp_path):
    monkeypatch.setenv("DASHSCOPE_BASE_URL", QWEN_INTL_BASE_URL)
    p = QwenProvider(api_key="fake", cache_dir=tmp_path)
    assert p.base_url == QWEN_INTL_BASE_URL
    assert p._endpoint_url().startswith(QWEN_INTL_BASE_URL)


def test_model_is_overridable_without_a_code_change(monkeypatch, tmp_path):
    """A pinned model id has already been retired upstream once in this
    project's life (Groq's `llama-3.1-8b-instant`); fixing that again should
    not need an edit and a redeploy."""
    monkeypatch.setenv("QWEN_MODEL", "qwen-turbo")
    assert QwenProvider(api_key="fake", cache_dir=tmp_path).model == "qwen-turbo"


def test_explicit_model_beats_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("QWEN_MODEL", "qwen-turbo")
    assert QwenProvider(api_key="fake", model="qwen-max", cache_dir=tmp_path).model == "qwen-max"


def test_provider_registry_includes_qwen():
    assert PROVIDERS["qwen"] is QwenProvider


# --------------------------------------------------------------------------- #
# Request shape: OpenAI-compatible, same as Groq
# --------------------------------------------------------------------------- #

def test_qwen_builds_openai_style_body_with_system(tmp_path):
    p = QwenProvider(api_key="fake", cache_dir=tmp_path)
    body = p._build_body("sys instructions", "the user text", 0.2, 128)
    assert body["model"] == p.model
    assert body["messages"] == [
        {"role": "system", "content": "sys instructions"},
        {"role": "user", "content": "the user text"},
    ]
    assert body["temperature"] == 0.2
    assert body["max_tokens"] == 128


def test_qwen_omits_empty_system_message(tmp_path):
    p = QwenProvider(api_key="fake", cache_dir=tmp_path)
    assert p._build_body("", "just user", 0.2, 64)["messages"] == [
        {"role": "user", "content": "just user"}
    ]


def test_qwen_multi_turn_needs_no_role_translation(tmp_path):
    """Unlike Gemini (which calls the model's own turns "model"), compatible
    mode takes the provider-neutral roles as-is -- the Experiment 4 multi-turn
    transcript passes straight through."""
    p = QwenProvider(api_key="fake", cache_dir=tmp_path)
    history = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "second"},
    ]
    body = p._build_body_multi("sys", history, 0.2, 256)
    assert body["messages"] == [{"role": "system", "content": "sys"}] + history


def test_qwen_endpoint_and_auth_header(tmp_path):
    p = QwenProvider(api_key="secret-key", cache_dir=tmp_path)
    assert p._endpoint_url() == f"{QWEN_BASE_URL}/chat/completions"
    assert p._headers()["Authorization"] == "Bearer secret-key"


# --------------------------------------------------------------------------- #
# Response parsing
# --------------------------------------------------------------------------- #

def test_qwen_parses_a_chat_completion(tmp_path):
    p = QwenProvider(api_key="fake", cache_dir=tmp_path)
    text, finish, usage = p._parse_response(
        _chat_payload(text="hi", finish_reason="stop", usage={"total_tokens": 9})
    )
    assert (text, finish, usage) == ("hi", "stop", {"total_tokens": 9})


def test_an_empty_qwen_response_raises_a_qwen_error_not_a_groq_one(tmp_path):
    """The shared `_parse_chat_completion` used to hard-code `GroqAPIError`; a
    Qwen failure reported as a Groq one would send anyone debugging it to the
    wrong provider."""
    p = QwenProvider(api_key="fake", cache_dir=tmp_path)
    with pytest.raises(QwenAPIError, match="Qwen returned no choices"):
        p._parse_response({"choices": []})


# --------------------------------------------------------------------------- #
# End-to-end generate() through a FakeSession (real retry loop, no network)
# --------------------------------------------------------------------------- #

def test_qwen_generate_success_and_caches(tmp_path):
    session = FakeSession([FakeResponse(200, _chat_payload(text="DECISION: CLOSE"))])
    p = QwenProvider(api_key="fake", cache_dir=tmp_path, session=session, min_interval_seconds=0)
    r1 = p.generate(_prompt())
    assert r1.text == "DECISION: CLOSE"
    assert r1.from_cache is False
    r2 = p.generate(_prompt())
    assert r2.from_cache is True
    assert len(session.calls) == 1


def test_qwen_429_honors_retry_after_header(tmp_path, monkeypatch):
    slept = []
    monkeypatch.setattr(providers_mod.time, "sleep", lambda s: slept.append(s))
    session = FakeSession([
        FakeResponse(429, {"error": {"message": "Throttling.RateQuota", "type": "Throttling"}},
                     headers={"retry-after": "7"}),
        FakeResponse(200, _chat_payload()),
    ])
    p = QwenProvider(api_key="fake", cache_dir=tmp_path, session=session,
                     min_interval_seconds=0, max_retries=3)
    p._call_api("sys", "user", 0.2, 100)
    assert slept == [7.0]


def test_a_per_minute_throttle_is_retried(tmp_path, monkeypatch):
    """DashScope uses 429 for both a transient throttle and a spent allowance.
    This one clears on its own, so it must NOT stop the run."""
    monkeypatch.setattr(providers_mod.time, "sleep", lambda s: None)
    session = FakeSession([
        FakeResponse(429, {"error": {"message": "Requests throttling triggered."}}),
        FakeResponse(200, _chat_payload(text="recovered")),
    ])
    p = QwenProvider(api_key="fake", cache_dir=tmp_path, session=session,
                     min_interval_seconds=0, max_retries=3)
    raw = p._call_api("sys", "user", 0.2, 100)
    assert raw["choices"][0]["message"]["content"] == "recovered"
    assert len(session.calls) == 2


@pytest.mark.parametrize("message", [
    "Allocated quota exceeded, please increase your quota limit",
    "Arrearage: your account is in debt",
    "Insufficient balance for this request",
])
def test_a_spent_allowance_stops_immediately_without_retrying(tmp_path, monkeypatch, message):
    """No amount of waiting inside a run clears a spent allowance, so grinding
    through five backoffs only wastes time and then reports a misleading
    "persisted after N retries"."""
    monkeypatch.setattr(providers_mod.time, "sleep", lambda s: None)
    session = FakeSession([FakeResponse(429, {"error": {"message": message}}) for _ in range(5)])
    p = QwenProvider(api_key="fake", cache_dir=tmp_path, session=session,
                     min_interval_seconds=0, max_retries=5)
    with pytest.raises(QwenQuotaExceededError, match="account quota exhausted"):
        p._call_api("sys", "user", 0.2, 100)
    assert len(session.calls) == 1  # NOT retried


def test_a_daily_cap_still_stops_immediately_via_the_shared_rule(tmp_path, monkeypatch):
    """Qwen's override must extend the base class's per-day rule, not replace
    it."""
    monkeypatch.setattr(providers_mod.time, "sleep", lambda s: None)
    session = FakeSession([
        FakeResponse(429, {"error": {"message": "Limit reached: requests per day (RPD)"}})
        for _ in range(5)
    ])
    p = QwenProvider(api_key="fake", cache_dir=tmp_path, session=session,
                     min_interval_seconds=0, max_retries=5)
    with pytest.raises(QwenQuotaExceededError, match="daily quota exhausted"):
        p._call_api("sys", "user", 0.2, 100)
    assert len(session.calls) == 1


def test_qwen_non_retryable_400_raises_a_qwen_api_error(tmp_path):
    session = FakeSession([
        FakeResponse(400, {"error": {"message": "Model not exist", "type": "invalid_request_error"}})
    ])
    p = QwenProvider(api_key="fake", cache_dir=tmp_path, session=session, min_interval_seconds=0)
    with pytest.raises(QwenAPIError, match="Model not exist"):
        p._call_api("sys", "user", 0.2, 100)
    assert len(session.calls) == 1


# --------------------------------------------------------------------------- #
# The switch must not disturb the Experiment 3/4 record
# --------------------------------------------------------------------------- #

def test_qwen_errors_are_catchable_by_the_provider_agnostic_bases():
    # `run_exp3_grid` / `run_exp4_grid` catch only the base classes.
    assert issubclass(QwenAPIError, LLMAPIError)
    assert issubclass(QwenQuotaExceededError, LLMQuotaExceededError)
    assert issubclass(QwenQuotaExceededError, LLMAPIError)


def test_switching_to_qwen_cannot_collide_with_the_groq_era_cache(tmp_path):
    """`data/llm_cache/` holds ~700 real Groq responses that Labs 3/4's numbers
    came from. The cache key includes the model, so a Qwen run can neither read
    a Groq entry (which would silently mix providers in one result set) nor
    overwrite one (which would destroy the record)."""
    groq = GroqProvider(api_key="fake", model="llama-3.1-8b-instant", cache_dir=tmp_path)
    qwen = QwenProvider(api_key="fake", model="qwen-plus", cache_dir=tmp_path)
    assert groq._cache_key("sys", "user", 0.2, 100) != qwen._cache_key("sys", "user", 0.2, 100)
    assert (groq._cache_key_from_messages("s", [{"role": "user", "content": "u"}], 0.2, 100)
            != qwen._cache_key_from_messages("s", [{"role": "user", "content": "u"}], 0.2, 100))
