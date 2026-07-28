"""LLM provider clients for Experiments 3-4, with every response cached to disk
so nothing is ever paid for (in quota or wall-clock) twice.

Two providers share one base class (`CachedChatProvider`):

- **`GroqProvider`** — the provider actually used for the Experiment 3 grid.
  Groq's free tier serves an OpenAI-compatible `chat/completions` endpoint with
  no credit card required (14,400 requests/day observed on
  `llama-3.1-8b-instant`), which is what made it the working choice after Gemini
  and DeepSeek both turned out to gate real use behind billing from this
  account/region (see `reports/exp3_grid_run_status.md` for the full story).
- **`GeminiProvider`** — the plan's original §5.1 choice, kept because it is
  fully built and tested and may work from a different account/region; it is a
  one-line swap (`--provider gemini`) in the grid runner.

Why raw HTTP instead of an SDK
-------------------------------
No LLM SDK is in `requirements.txt`, and `src/mining/github_client.py` already
set this precedent (plain `requests`, not `PyGithub`). Both providers here hit
a single stable JSON endpoint; `requests` (already a dependency) is enough, and
a shared base keeps the caching/retry logic in exactly one place.

Caching design
--------------
Cache key = `sha256` of the canonical (sorted-key) JSON of everything that
determines the answer: `{model, system, user, temperature, max_output_tokens}`.
Deliberately NOT part of the key: which PR/task/strategy/context produced the
prompt (that's bookkeeping metadata, stored alongside the response). So the
guarantee is exact — identical wording always hits the cache, regardless of how
Step 16 iterates, and an accidental duplicate call costs zero quota. The `model`
IS in the key, so Groq and Gemini responses (or two Groq models) never collide
even sharing one `cache_dir`. Cache files are sharded by the first two hex chars
of the key (`data/llm_cache/<xx>/<hash>.json`, gitignored) for filesystem
hygiene at the ~7,000-entry scale a full grid implies.

Retry / quota handling
-----------------------
A 429 retries with exponential backoff (honoring a provider-supplied delay hint
— Gemini's `RetryInfo.retryDelay`, Groq's `retry-after` header — when present)
up to `max_retries`; still failing after that raises a provider-specific
`*QuotaExceededError` (both subclass `LLMQuotaExceededError`) rather than
retrying forever, so the grid runner can stop cleanly and resume from cache
later (plan §10). Other retryable 5xx codes retry too; a non-retryable 4xx
(e.g. a bad request, or Gemini's region `FAILED_PRECONDITION`) raises the
provider's `*APIError` (both subclass `LLMAPIError`) immediately.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests
from dotenv import load_dotenv

from src.llm.prompts.schema import RenderedConversation, RenderedPrompt

load_dotenv()

logger = logging.getLogger(__name__)

# ---- shared defaults ---------------------------------------------------- #
DEFAULT_CACHE_DIR = Path("data/llm_cache")
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_OUTPUT_TOKENS = 2048
DEFAULT_TIMEOUT_SECONDS = 60.0

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
MAX_RETRIES = 5
BASE_BACKOFF_SECONDS = 2.0
MAX_BACKOFF_SECONDS = 60.0

# ---- Gemini ------------------------------------------------------------- #
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_DEFAULT_MODEL = "gemini-flash-latest"
GEMINI_DEFAULT_MIN_INTERVAL = 4.0
DEFAULT_MODEL = GEMINI_DEFAULT_MODEL  # backwards-compat alias

# ---- Groq --------------------------------------------------------------- #
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_DEFAULT_MODEL = "llama-3.1-8b-instant"
# Groq free tier for 8b-instant advertised ~30 req/min; the token-per-minute
# ceiling (6,000 TPM observed in the rate-limit headers) is the real binding
# constraint and is handled reactively by honoring the `retry-after` header on
# a 429, so this proactive floor is just a light politeness gap.
GROQ_DEFAULT_MIN_INTERVAL = 2.0


# ---- exceptions --------------------------------------------------------- #
class LLMAPIError(RuntimeError):
    """A non-retryable (or retry-exhausted-but-not-quota) provider error."""


class LLMQuotaExceededError(LLMAPIError):
    """429s persisted past `max_retries` — treated as quota exhaustion (not
    transient rate limiting). The caller should stop the run; already-cached
    responses stay valid and a later run resumes for free via the same key."""


class GeminiAPIError(LLMAPIError):
    pass


class GeminiQuotaExceededError(GeminiAPIError, LLMQuotaExceededError):
    pass


class GroqAPIError(LLMAPIError):
    pass


class GroqQuotaExceededError(GroqAPIError, LLMQuotaExceededError):
    pass


@dataclass(frozen=True)
class LLMResponse:
    text: str
    latency_ms: float
    from_cache: bool
    model: str
    finish_reason: Optional[str]
    usage: Optional[dict]
    cache_key: str


@dataclass(frozen=True)
class ConversationResponse:
    """Result of driving a `RenderedConversation` (Experiment 4's self-
    reflection / multi-turn strategies, `src/llm/prompts/schema.py`) to
    completion via `CachedChatProvider.generate_conversation`: one `LLMResponse`
    per user turn, in order, plus the convenience views the Step-20 grid runner
    needs (the final answer to parse, and cache/latency bookkeeping across all
    turns rather than just the last one)."""

    turns: tuple[LLMResponse, ...]

    @property
    def final_text(self) -> str:
        """The last turn's reply -- what `parsing.py` parses (only the final
        turn carries the output contract; see `RenderedConversation`)."""
        return self.turns[-1].text

    @property
    def total_latency_ms(self) -> float:
        return sum(t.latency_ms for t in self.turns)

    @property
    def all_from_cache(self) -> bool:
        return all(t.from_cache for t in self.turns)

    @property
    def any_from_cache(self) -> bool:
        return any(t.from_cache for t in self.turns)


# ---- shared helpers ----------------------------------------------------- #
def _backoff_delay(attempt: int, retry_after: Optional[float] = None) -> float:
    if retry_after is not None:
        return retry_after
    delay = min(MAX_BACKOFF_SECONDS, BASE_BACKOFF_SECONDS * (2 ** attempt))
    return delay * (0.5 + random.random() / 2)


def _parse_retry_delay(error: dict) -> Optional[float]:
    """Gemini's `google.rpc.RetryInfo` detail carries a `retryDelay` like
    `"38s"` — honor it over our own backoff schedule when present."""
    for detail in error.get("details") or []:
        if str(detail.get("@type", "")).endswith("RetryInfo"):
            raw = str(detail.get("retryDelay", ""))
            if raw.endswith("s"):
                try:
                    return float(raw[:-1])
                except ValueError:
                    return None
    return None


def _safe_json(response: requests.Response) -> Optional[dict]:
    try:
        return response.json()
    except ValueError:
        return None


def _parse_generate_response(raw: dict) -> tuple[str, Optional[str], Optional[dict]]:
    """Gemini `generateContent` response -> (text, finish_reason, usage)."""
    candidates = raw.get("candidates") or []
    if not candidates:
        block_reason = (raw.get("promptFeedback") or {}).get("blockReason")
        raise GeminiAPIError(f"Gemini returned no candidates (blockReason={block_reason})")
    candidate = candidates[0]
    parts = (candidate.get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts)
    return text, candidate.get("finishReason"), raw.get("usageMetadata")


def _parse_chat_completion(raw: dict) -> tuple[str, Optional[str], Optional[dict]]:
    """OpenAI-compatible `chat/completions` response (Groq) -> (text,
    finish_reason, usage)."""
    choices = raw.get("choices") or []
    if not choices:
        raise GroqAPIError("Groq returned no choices in the response")
    choice = choices[0]
    text = (choice.get("message") or {}).get("content") or ""
    return text, choice.get("finish_reason"), raw.get("usage")


# ---- base provider ------------------------------------------------------ #
class CachedChatProvider:
    """Shared caching / rate-limit / retry orchestration. Subclasses supply the
    provider-specific request shape and response parsing via the five hooks at
    the bottom; everything else (the exact same disk cache and `generate`
    contract both experiments rely on) lives here once."""

    provider_name = "llm"
    api_error_cls: type[LLMAPIError] = LLMAPIError
    quota_error_cls: type[LLMQuotaExceededError] = LLMQuotaExceededError

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        cache_dir: str | Path,
        session: Optional[requests.Session],
        max_retries: int,
        min_interval_seconds: float,
        timeout: float,
        base_url: str,
    ):
        self.api_key = api_key
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = session or requests.Session()
        self.max_retries = max_retries
        self.min_interval_seconds = min_interval_seconds
        self.timeout = timeout
        self.base_url = base_url
        self._last_call_at: Optional[float] = None

    # ---------- disk cache, content-addressed ----------
    def _cache_key(self, system: str, user: str, temperature: float, max_output_tokens: int) -> str:
        payload = {
            "model": self.model,
            "system": system,
            "user": user,
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        }
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _cache_key_from_messages(
        self, system: str, messages: list[dict[str, str]], temperature: float, max_output_tokens: int
    ) -> str:
        """The multi-turn analogue of `_cache_key`: hashes the WHOLE running
        transcript (not just the latest turn), since two different histories
        that happen to end in the same user message are not the same question.
        Deliberately a separate key function from `_cache_key` -- rather than
        having `_cache_key` special-case a message list -- so every single-turn
        cache entry already on disk from the Experiment 3 grid keeps hashing
        exactly as before and stays valid."""
        payload = {
            "model": self.model,
            "system": system,
            "messages": messages,
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        }
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _cache_path(self, cache_key: str) -> Path:
        shard_dir = self.cache_dir / cache_key[:2]
        shard_dir.mkdir(parents=True, exist_ok=True)
        return shard_dir / f"{cache_key}.json"

    def _read_cache(self, cache_key: str) -> Optional[dict]:
        path = self.cache_dir / cache_key[:2] / f"{cache_key}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            logger.warning("Corrupt LLM cache file %s, will re-call", path)
            return None

    def _write_cache(self, cache_key: str, record: dict) -> None:
        self._cache_path(cache_key).write_text(json.dumps(record, indent=2, ensure_ascii=False))

    # ---------- rate limiting ----------
    def _respect_rate_limit(self) -> None:
        if self._last_call_at is None:
            return
        remaining = self.min_interval_seconds - (time.monotonic() - self._last_call_at)
        if remaining > 0:
            time.sleep(remaining)

    # ---------- the one live HTTP call (retry loop), isolated for tests ----------
    def _call_api(self, system: str, user: str, temperature: float, max_output_tokens: int) -> dict:
        """Single-turn call (Experiment 3's shape): builds a `(system, user)`
        body and executes it. Signature/behavior unchanged from before
        Experiment 4 -- existing tests call this directly."""
        body = self._build_body(system, user, temperature, max_output_tokens)
        return self._execute_request(body)

    def _call_api_multi(
        self, system: str, messages: list[dict[str, str]], temperature: float, max_output_tokens: int
    ) -> dict:
        """Multi-turn call (Experiment 4's self-reflection / multi-turn
        strategies): builds a body from a full running transcript (`messages`,
        provider-neutral `{"role": "user"/"assistant", "content": str}` dicts)
        instead of one `user` string, then executes it via the SAME retry/
        backoff/quota-handling loop `_call_api` uses (`_execute_request`) --
        the two only differ in how the request body is shaped."""
        body = self._build_body_multi(system, messages, temperature, max_output_tokens)
        return self._execute_request(body)

    def _execute_request(self, body: dict[str, Any]) -> dict:
        """The retry loop shared by `_call_api` and `_call_api_multi`: post
        `body` to this provider's endpoint, retrying transient failures and
        distinguishing quota exhaustion from other errors exactly as before --
        extracted here, unchanged in behavior, so both call shapes share one
        implementation of the retry/backoff/quota policy."""
        url = self._endpoint_url()
        headers = self._headers()

        attempt = 0
        while True:
            self._respect_rate_limit()
            try:
                response = self.session.post(url, headers=headers, json=body, timeout=self.timeout)
            except requests.exceptions.RequestException as e:
                # Connection-level failure (proxy/VPN drop, DNS blip, read
                # timeout) -- transient and NOT an HTTP response, so it must be
                # caught here rather than by the status-code branches below.
                # Retrying protects a multi-hour grid run from a momentary
                # network hiccup (a real VPN ProxyError killed an earlier run);
                # only after exhausting retries does it surface as an APIError,
                # which the grid runner records as one failed cell and moves on.
                self._last_call_at = time.monotonic()
                if attempt < self.max_retries:
                    delay = _backoff_delay(attempt)
                    logger.warning(
                        "%s connection error (attempt %d/%d); retrying in %.1fs: %s",
                        self.provider_name, attempt + 1, self.max_retries, delay, e,
                    )
                    time.sleep(delay)
                    attempt += 1
                    continue
                raise self.api_error_cls(
                    f"{self.provider_name} connection error after {attempt} retries: {e}"
                )
            self._last_call_at = time.monotonic()

            if response.status_code < 400:
                return response.json()

            payload = _safe_json(response) or {}
            error = payload.get("error", {}) if isinstance(payload, dict) else {}
            if not isinstance(error, dict):
                error = {}
            status = error.get("status") or error.get("type") or ""
            message = error.get("message") or (json.dumps(payload) if payload else response.text)

            if response.status_code == 429:
                # A *daily* cap (Groq's "tokens per day (TPD)", or any "per day"
                # / RPD limit) cannot clear within any retry window -- it resets
                # only when the 24h rolling window rolls over. Retrying it just
                # burns hours (an earlier run spent ~2.5h grinding ~8-minute
                # waits against an exhausted 500k-token/day cap). So stop the run
                # immediately and cleanly: run_grid catches this, halts, and the
                # partial output resumes for free once quota resets. A *per-minute*
                # (TPM/RPM) 429 is genuinely transient and still retries below.
                lower_msg = message.lower()
                if "per day" in lower_msg or "(tpd)" in lower_msg or "(rpd)" in lower_msg:
                    raise self.quota_error_cls(
                        f"{self.provider_name} daily quota exhausted "
                        f"(not retryable until the daily window resets): {message}"
                    )
                if attempt >= self.max_retries:
                    raise self.quota_error_cls(
                        f"{self.provider_name} 429 persisted after {attempt} retries: {message}"
                    )
                delay = _backoff_delay(attempt, self._extract_retry_delay(response, error))
                logger.warning(
                    "%s 429 (attempt %d/%d); retrying in %.1fs: %s",
                    self.provider_name, attempt + 1, self.max_retries, delay, message,
                )
                time.sleep(delay)
                attempt += 1
                continue

            if response.status_code in RETRYABLE_STATUS_CODES and attempt < self.max_retries:
                delay = _backoff_delay(attempt)
                logger.warning(
                    "%s HTTP %d (attempt %d/%d); retrying in %.1fs: %s",
                    self.provider_name, response.status_code, attempt + 1, self.max_retries, delay, message,
                )
                time.sleep(delay)
                attempt += 1
                continue

            raise self.api_error_cls(
                f"{self.provider_name} API error {response.status_code} ({status}): {message}"
            )

    # ---------- public API ----------
    def generate(
        self,
        prompt: RenderedPrompt,
        *,
        temperature: float = DEFAULT_TEMPERATURE,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        cache_metadata: Optional[dict] = None,
        force_refresh: bool = False,
        cache_only: bool = False,
    ) -> Optional[LLMResponse]:
        """`cache_only=True` returns the cached response if present, else
        `None` -- it never makes a live call. Used to regenerate result files
        from the cache for free (e.g. after adding a field to the recorded
        row), without spending any quota."""
        cache_key = self._cache_key(prompt.system, prompt.user, temperature, max_output_tokens)

        if not force_refresh:
            cached = self._read_cache(cache_key)
            if cached is not None:
                return LLMResponse(
                    text=cached["response_text"],
                    latency_ms=cached["latency_ms"],
                    from_cache=True,
                    model=cached["model"],
                    finish_reason=cached.get("finish_reason"),
                    usage=cached.get("usage"),
                    cache_key=cache_key,
                )

        if cache_only:
            return None

        start = time.monotonic()
        raw = self._call_api(prompt.system, prompt.user, temperature, max_output_tokens)
        latency_ms = (time.monotonic() - start) * 1000
        text, finish_reason, usage = self._parse_response(raw)

        record = {
            "model": self.model,
            "provider": self.provider_name,
            "system": prompt.system,
            "user": prompt.user,
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "response_text": text,
            "finish_reason": finish_reason,
            "usage": usage,
            "latency_ms": latency_ms,
            "cached_at": datetime.now(timezone.utc).isoformat(),
        }
        if cache_metadata:
            record["metadata"] = cache_metadata
        self._write_cache(cache_key, record)

        return LLMResponse(
            text=text,
            latency_ms=latency_ms,
            from_cache=False,
            model=self.model,
            finish_reason=finish_reason,
            usage=usage,
            cache_key=cache_key,
        )

    # ---------- multi-turn (Experiment 4: self-reflection / multi-turn) ----------
    def generate_turn(
        self,
        system: str,
        messages: list[dict[str, str]],
        *,
        temperature: float = DEFAULT_TEMPERATURE,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        cache_metadata: Optional[dict] = None,
        force_refresh: bool = False,
        cache_only: bool = False,
    ) -> Optional[LLMResponse]:
        """Generate ONE assistant reply given a running transcript. `messages`
        is the full prior conversation as provider-neutral `{"role":
        "user"/"assistant", "content": str}` dicts, ENDING in the newest user
        turn to answer -- exactly what `generate_conversation` threads through
        turn by turn. Cached on the whole transcript (`_cache_key_from_messages`),
        so a resumed run replays cached turn-1 answers byte-for-byte before a
        turn-2 call is even attempted. Mirrors `generate`'s contract (including
        `cache_only`'s "return None on a miss, make no live call" semantics)
        but keyed on a message list instead of one `user` string."""
        cache_key = self._cache_key_from_messages(system, messages, temperature, max_output_tokens)

        if not force_refresh:
            cached = self._read_cache(cache_key)
            if cached is not None:
                return LLMResponse(
                    text=cached["response_text"],
                    latency_ms=cached["latency_ms"],
                    from_cache=True,
                    model=cached["model"],
                    finish_reason=cached.get("finish_reason"),
                    usage=cached.get("usage"),
                    cache_key=cache_key,
                )

        if cache_only:
            return None

        start = time.monotonic()
        raw = self._call_api_multi(system, messages, temperature, max_output_tokens)
        latency_ms = (time.monotonic() - start) * 1000
        text, finish_reason, usage = self._parse_response(raw)

        record = {
            "model": self.model,
            "provider": self.provider_name,
            "system": system,
            "messages": messages,
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "response_text": text,
            "finish_reason": finish_reason,
            "usage": usage,
            "latency_ms": latency_ms,
            "cached_at": datetime.now(timezone.utc).isoformat(),
        }
        if cache_metadata:
            record["metadata"] = cache_metadata
        self._write_cache(cache_key, record)

        return LLMResponse(
            text=text, latency_ms=latency_ms, from_cache=False, model=self.model,
            finish_reason=finish_reason, usage=usage, cache_key=cache_key,
        )

    def generate_conversation(
        self,
        conversation: RenderedConversation,
        *,
        temperature: float = DEFAULT_TEMPERATURE,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        cache_metadata: Optional[dict] = None,
        force_refresh: bool = False,
        cache_only: bool = False,
    ) -> Optional[ConversationResponse]:
        """Drive a `RenderedConversation` (Experiment 4's self-reflection /
        multi-turn strategies) to completion: one `generate_turn` call per
        entry in `conversation.turns`, threading each reply into the next
        turn's history exactly per the protocol documented on
        `RenderedConversation`. Returns `None` under `cache_only=True` if ANY
        turn misses the cache -- a conversation is only "available from cache"
        as a whole if every turn is, since a missing early turn's reply is
        needed to even know what the next turn's transcript (and therefore its
        cache key) is without a live call."""
        messages: list[dict[str, str]] = []
        responses: list[LLMResponse] = []
        for turn_text in conversation.turns:
            messages.append({"role": "user", "content": turn_text})
            response = self.generate_turn(
                conversation.system, list(messages),
                temperature=temperature, max_output_tokens=max_output_tokens,
                cache_metadata=cache_metadata, force_refresh=force_refresh, cache_only=cache_only,
            )
            if response is None:
                return None
            responses.append(response)
            messages.append({"role": "assistant", "content": response.text})
        return ConversationResponse(turns=tuple(responses))

    # ---------- provider-specific hooks (subclasses implement) ----------
    def _endpoint_url(self) -> str:
        raise NotImplementedError

    def _headers(self) -> dict[str, str]:
        raise NotImplementedError

    def _build_body(self, system: str, user: str, temperature: float, max_output_tokens: int) -> dict[str, Any]:
        raise NotImplementedError

    def _build_body_multi(
        self, system: str, messages: list[dict[str, str]], temperature: float, max_output_tokens: int
    ) -> dict[str, Any]:
        raise NotImplementedError

    def _extract_retry_delay(self, response: requests.Response, error: dict) -> Optional[float]:
        return None

    def _parse_response(self, raw: dict) -> tuple[str, Optional[str], Optional[dict]]:
        raise NotImplementedError


class GeminiProvider(CachedChatProvider):
    provider_name = "Gemini"
    api_error_cls = GeminiAPIError
    quota_error_cls = GeminiQuotaExceededError

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = GEMINI_DEFAULT_MODEL,
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
        session: Optional[requests.Session] = None,
        max_retries: int = MAX_RETRIES,
        min_interval_seconds: float = GEMINI_DEFAULT_MIN_INTERVAL,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        base_url: str = GEMINI_BASE_URL,
    ):
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "No Gemini API key found. Set GEMINI_API_KEY in .env or pass api_key= explicitly."
            )
        super().__init__(
            api_key=api_key, model=model, cache_dir=cache_dir, session=session,
            max_retries=max_retries, min_interval_seconds=min_interval_seconds,
            timeout=timeout, base_url=base_url,
        )

    def _endpoint_url(self) -> str:
        return f"{self.base_url}/models/{self.model}:generateContent"

    def _headers(self) -> dict[str, str]:
        return {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}

    def _build_body(self, system, user, temperature, max_output_tokens):
        body: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_output_tokens},
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        return body

    def _build_body_multi(self, system, messages, temperature, max_output_tokens):
        # Gemini's `contents` list uses "model" (not "assistant") for the
        # model's own turns; the provider-neutral "assistant" role from
        # `generate_conversation` is mapped here, at the one seam that knows
        # about Gemini's naming.
        contents = [
            {"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
            for m in messages
        ]
        body: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_output_tokens},
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        return body

    def _extract_retry_delay(self, response, error):
        return _parse_retry_delay(error)

    def _parse_response(self, raw):
        return _parse_generate_response(raw)


class GroqProvider(CachedChatProvider):
    provider_name = "Groq"
    api_error_cls = GroqAPIError
    quota_error_cls = GroqQuotaExceededError

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = GROQ_DEFAULT_MODEL,
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
        session: Optional[requests.Session] = None,
        max_retries: int = MAX_RETRIES,
        min_interval_seconds: float = GROQ_DEFAULT_MIN_INTERVAL,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        base_url: str = GROQ_BASE_URL,
    ):
        api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "No Groq API key found. Set GROQ_API_KEY in .env or pass api_key= explicitly."
            )
        super().__init__(
            api_key=api_key, model=model, cache_dir=cache_dir, session=session,
            max_retries=max_retries, min_interval_seconds=min_interval_seconds,
            timeout=timeout, base_url=base_url,
        )

    def _endpoint_url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _build_body(self, system, user, temperature, max_output_tokens):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        return {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_output_tokens,
        }

    def _build_body_multi(self, system, messages, temperature, max_output_tokens):
        # Groq's chat/completions is already OpenAI-style, so the
        # provider-neutral messages ({"role": "user"/"assistant", "content"})
        # need no role translation -- just a system message prepended.
        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)
        return {
            "model": self.model,
            "messages": full_messages,
            "temperature": temperature,
            "max_tokens": max_output_tokens,
        }

    def _extract_retry_delay(self, response, error):
        retry_after = response.headers.get("retry-after")
        if retry_after:
            try:
                return float(retry_after)
            except ValueError:
                return None
        return None

    def _parse_response(self, raw):
        return _parse_chat_completion(raw)


# Provider registry for the grid runner's --provider flag.
PROVIDERS: dict[str, type[CachedChatProvider]] = {
    "groq": GroqProvider,
    "gemini": GeminiProvider,
}
