"""Tests for the multi-turn conversation support Experiment 4 adds to
`CachedChatProvider` (`generate_turn`, `generate_conversation`,
`_cache_key_from_messages`, `_build_body_multi`).

The headline properties: turn 2's request replays turn 1's actual reply (not
some placeholder), the whole conversation is resumable turn-by-turn from disk
cache exactly like `generate()` is, and `_call_api`'s single-turn behavior
(exercised exhaustively in `test_llm_providers.py` /
`test_llm_providers_groq.py`) is completely unaffected by the refactor that
introduced `_execute_request`.
"""


from src.llm.prompts.schema import Exp4Strategy, RenderedConversation, Task
from src.llm.providers import ConversationResponse, GeminiProvider, GroqProvider


def _conversation(turns=("draft this", "now critique it"), system="You are a reviewer.") -> RenderedConversation:
    return RenderedConversation(
        task=Task.MERGE_PREDICTION, strategy=Exp4Strategy.SELF_REFLECTION,
        system=system, turns=turns, response_contract="DECISION: ...",
    )


class FakeResponse:
    def __init__(self, payload):
        self.status_code = 200
        self._payload = payload
        self.headers: dict = {}

    def json(self):
        return self._payload


class ScriptedGroqSession:
    """Replies with the next scripted text for each POST, OpenAI-style."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls: list[dict] = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append(json)
        text = self.replies.pop(0)
        return FakeResponse({
            "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"total_tokens": 12},
        })


class ScriptedGeminiSession:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls: list[dict] = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append(json)
        text = self.replies.pop(0)
        return FakeResponse({
            "candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}],
        })


# --------------------------------------------------------------------------- #
# _build_body_multi: role mapping
# --------------------------------------------------------------------------- #

def test_groq_build_body_multi_is_openai_shape(tmp_path):
    p = GroqProvider(api_key="fake", cache_dir=tmp_path)
    body = p._build_body_multi(
        "sys", [{"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"},
               {"role": "user", "content": "u2"}], 0.2, 100,
    )
    assert body["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
    ]


def test_gemini_build_body_multi_maps_assistant_to_model(tmp_path):
    p = GeminiProvider(api_key="fake", cache_dir=tmp_path)
    body = p._build_body_multi(
        "sys", [{"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"}], 0.2, 100,
    )
    assert body["contents"] == [
        {"role": "user", "parts": [{"text": "u1"}]},
        {"role": "model", "parts": [{"text": "a1"}]},
    ]
    assert body["systemInstruction"] == {"parts": [{"text": "sys"}]}


# --------------------------------------------------------------------------- #
# generate_conversation: end-to-end via a scripted session
# --------------------------------------------------------------------------- #

def test_generate_conversation_threads_prior_reply_into_next_turn(tmp_path):
    session = ScriptedGroqSession(["draft answer", "final answer"])
    p = GroqProvider(api_key="fake", cache_dir=tmp_path, session=session, min_interval_seconds=0)

    result = p.generate_conversation(_conversation())

    assert isinstance(result, ConversationResponse)
    assert len(result.turns) == 2
    assert result.final_text == "final answer"
    assert result.all_from_cache is False

    turn2_request = session.calls[1]
    roles_and_content = [(m["role"], m["content"]) for m in turn2_request["messages"]]
    assert roles_and_content == [
        ("system", "You are a reviewer."),
        ("user", "draft this"),
        ("assistant", "draft answer"),   # the model's ACTUAL turn-1 reply, threaded in
        ("user", "now critique it"),
    ]


def test_generate_conversation_gemini_maps_roles_end_to_end(tmp_path):
    session = ScriptedGeminiSession(["draft", "final"])
    p = GeminiProvider(api_key="fake", cache_dir=tmp_path, session=session, min_interval_seconds=0)

    result = p.generate_conversation(_conversation())
    assert result.final_text == "final"
    turn2_request = session.calls[1]
    assert turn2_request["contents"][-2] == {"role": "model", "parts": [{"text": "draft"}]}


def test_generate_conversation_resumes_fully_from_cache(tmp_path):
    session1 = ScriptedGroqSession(["draft answer", "final answer"])
    p1 = GroqProvider(api_key="fake", cache_dir=tmp_path, session=session1, min_interval_seconds=0)
    p1.generate_conversation(_conversation())

    # a second provider instance sharing the cache dir, with a session that
    # would raise if ever called -- the whole conversation must replay from disk
    class ExplodingSession:
        def post(self, *a, **k):
            raise AssertionError("must not make a live call when fully cached")

    p2 = GroqProvider(api_key="fake", cache_dir=tmp_path, session=ExplodingSession(), min_interval_seconds=0)
    result2 = p2.generate_conversation(_conversation())
    assert result2.all_from_cache is True
    assert result2.final_text == "final answer"


def test_generate_conversation_partial_cache_only_calls_live_for_missing_turn(tmp_path):
    # populate turn 1's cache entry only, by running a 1-turn conversation
    session1 = ScriptedGroqSession(["draft answer"])
    p1 = GroqProvider(api_key="fake", cache_dir=tmp_path, session=session1, min_interval_seconds=0)
    one_turn = _conversation(turns=("draft this",))
    p1.generate_conversation(one_turn)
    assert len(session1.calls) == 1

    # now run the full 2-turn conversation: turn 1 must hit cache, turn 2 must
    # be a genuine new live call (same transcript prefix, so the same cache key)
    session2 = ScriptedGroqSession(["final answer"])
    p2 = GroqProvider(api_key="fake", cache_dir=tmp_path, session=session2, min_interval_seconds=0)
    result = p2.generate_conversation(_conversation())
    assert result.turns[0].from_cache is True
    assert result.turns[1].from_cache is False
    assert result.any_from_cache is True
    assert result.all_from_cache is False
    assert len(session2.calls) == 1  # only turn 2 was a live call


def test_generate_conversation_cache_only_none_on_any_miss(tmp_path):
    p = GroqProvider(api_key="fake", cache_dir=tmp_path, session=ScriptedGroqSession([]), min_interval_seconds=0)
    assert p.generate_conversation(_conversation(), cache_only=True) is None


def test_generate_turn_cache_key_depends_on_full_transcript(tmp_path):
    p = GroqProvider(api_key="fake", cache_dir=tmp_path)
    k1 = p._cache_key_from_messages("sys", [{"role": "user", "content": "u1"}], 0.2, 100)
    k2 = p._cache_key_from_messages(
        "sys", [{"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"},
               {"role": "user", "content": "u2"}], 0.2, 100,
    )
    assert k1 != k2


def test_message_cache_key_disjoint_from_single_turn_key(tmp_path):
    """The multi-turn key function must not collide with `_cache_key` --
    otherwise a differently-shaped payload could alias an Experiment 3 cache
    entry (or vice versa)."""
    p = GroqProvider(api_key="fake", cache_dir=tmp_path)
    single = p._cache_key("sys", "hello", 0.2, 100)
    multi = p._cache_key_from_messages("sys", [{"role": "user", "content": "hello"}], 0.2, 100)
    assert single != multi


def test_conversation_response_total_latency_sums_turns(tmp_path):
    session = ScriptedGroqSession(["a", "b"])
    p = GroqProvider(api_key="fake", cache_dir=tmp_path, session=session, min_interval_seconds=0)
    result = p.generate_conversation(_conversation())
    assert result.total_latency_ms == sum(t.latency_ms for t in result.turns)
    assert result.total_latency_ms >= 0


# --------------------------------------------------------------------------- #
# _call_api's single-turn contract is unaffected by the _execute_request
# refactor (a focused regression check; the exhaustive suite lives in
# test_llm_providers.py / test_llm_providers_groq.py)
# --------------------------------------------------------------------------- #

def test_call_api_single_turn_unaffected_by_refactor(tmp_path):
    from tests.test_llm_providers_groq import FakeResponse as GroqFakeResponse
    from tests.test_llm_providers_groq import FakeSession as GroqFakeSession

    session = GroqFakeSession([GroqFakeResponse(200, {
        "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
    })])
    p = GroqProvider(api_key="fake", cache_dir=tmp_path, session=session, min_interval_seconds=0)
    raw = p._call_api("sys", "user", 0.2, 100)
    assert raw["choices"][0]["message"]["content"] == "hi"
