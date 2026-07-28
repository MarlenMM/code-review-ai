"""Tests for `src/api/llm_review.py`. No real network call is made anywhere
here -- `src/api.llm_review._provider` is monkeypatched to a fake, mirroring
`tests/test_llm_providers_groq.py`'s own no-real-network discipline."""

from __future__ import annotations


from src.api import llm_review
from src.api.diff_parsing import ParsedFile
from src.llm.providers import ConversationResponse, GroqAPIError, GroqQuotaExceededError, LLMResponse


def _response(text: str) -> ConversationResponse:
    turn = LLMResponse(
        text=text, latency_ms=1.0, from_cache=False, model="llama-3.1-8b-instant",
        finish_reason="stop", usage=None, cache_key="k",
    )
    return ConversationResponse(turns=(turn, turn))


class _FakeProvider:
    model = "llama-3.1-8b-instant"

    def __init__(self, response=None, error=None):
        self._response = response
        self._error = error
        self.calls = []

    def generate_conversation(self, conversation, **kwargs):
        self.calls.append((conversation, kwargs))
        if self._error is not None:
            raise self._error
        return self._response


def _files():
    return [
        ParsedFile(
            filename="src/foo.py", status="modified",
            patch="@@ -1,2 +1,2 @@\n def foo():\n-    return 1\n+    return 2\n",
            additions=1, deletions=1,
        ),
    ]


def test_generate_review_comments_success(monkeypatch):
    fake = _FakeProvider(response=_response("<review>\n- looks fine\n- add a test\n</review>"))
    monkeypatch.setattr(llm_review, "_provider", lambda: fake)

    result = llm_review.generate_review_comments(_files(), repo="octocat/demo", title="Fix foo")

    assert result.warning is None
    assert result.comments == ["looks fine", "add a test"]
    assert "multi_turn" in result.config_label
    assert "diff_repo_context" in result.config_label
    assert len(fake.calls) == 1


def test_generate_review_comments_builds_multi_turn_conversation(monkeypatch):
    fake = _FakeProvider(response=_response("<review>\n- ok\n</review>"))
    monkeypatch.setattr(llm_review, "_provider", lambda: fake)

    llm_review.generate_review_comments(_files())

    conversation, kwargs = fake.calls[0]
    assert len(conversation.turns) == 2  # multi-turn: diff-only, then repo context revealed
    assert kwargs["max_output_tokens"] == llm_review.GROQ_RUN_MAX_OUTPUT_TOKENS


def test_generate_review_comments_quota_exceeded(monkeypatch):
    fake = _FakeProvider(error=GroqQuotaExceededError("daily cap hit"))
    monkeypatch.setattr(llm_review, "_provider", lambda: fake)

    result = llm_review.generate_review_comments(_files())

    assert result.comments is None
    assert result.config_label is None
    assert "quota" in result.warning.lower()


def test_generate_review_comments_api_error(monkeypatch):
    fake = _FakeProvider(error=GroqAPIError("500 from Groq"))
    monkeypatch.setattr(llm_review, "_provider", lambda: fake)

    result = llm_review.generate_review_comments(_files())

    assert result.comments is None
    assert "LLM call failed" in result.warning


def test_generate_review_comments_unavailable(monkeypatch):
    def _raise():
        raise ValueError("No Groq API key found.")
    monkeypatch.setattr(llm_review, "_provider", _raise)

    result = llm_review.generate_review_comments(_files())

    assert result.comments is None
    assert "LLM unavailable" in result.warning


def test_render_diff_for_llm_orders_largest_churn_first():
    files = [
        ParsedFile(filename="small.py", status="modified", patch="@@ -1 +1 @@\n-a\n+b\n", additions=1, deletions=1),
        ParsedFile(filename="big.py", status="modified", patch="@@ -1,5 +1,5 @@\n" + "+x\n" * 5, additions=5, deletions=0),
    ]
    text = llm_review._render_diff_for_llm(files, max_total_chars=10_000, max_file_chars=10_000)
    assert text.index("big.py") < text.index("small.py")


def test_render_diff_for_llm_truncates_and_marks_omission():
    files = [ParsedFile(filename=f"f{i}.py", status="modified", patch="@@ -1 +1 @@\n+x\n", additions=1, deletions=0)
             for i in range(5)]
    text = llm_review._render_diff_for_llm(files, max_total_chars=40, max_file_chars=40)
    assert "omitted for length" in text


def test_render_diff_for_llm_empty_files():
    assert llm_review._render_diff_for_llm([], max_total_chars=100, max_file_chars=100) == "(no file diff available)"
