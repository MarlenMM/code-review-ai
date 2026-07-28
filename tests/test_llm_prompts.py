"""Tests for the Experiment 3 prompt templates, dispatch, and output parsers
(`src/llm/prompts/templates.py`, `schema.py`, `parsing.py`).

These assert the *contracts* the downstream harness (Steps 15-17) relies on:
every (task, strategy) cell builds; the four strategies are genuinely distinct
(not identical text); few-shot's example contract is enforced; and each parser
is the faithful dual of its output contract, including the robustness the eval
metrics depend on (abstention on format-less answers, tolerance of markdown).
"""

import itertools

import pytest

from src.llm.prompts import (
    FewShotExample,
    PromptContext,
    Strategy,
    Task,
    build_prompt,
    parse_merge_prediction,
    parse_review_comments,
)

CTX = PromptContext(
    context_text="--- a.py\n@@ -1 +1 @@\n-x=1\n+x = 1\n",
    repo="microsoft/vscode",
    title="Fix spacing",
)

MERGE_EXAMPLES = [
    FewShotExample(pr_id="p1", repo="r", context_text="diff A", label="MERGE"),
    FewShotExample(pr_id="p2", repo="r", context_text="diff B", label="CLOSE"),
]
COMMENT_EXAMPLES = [
    FewShotExample(pr_id="p1", repo="r", context_text="diff A",
                   target_comment="Add a null check here."),
]


# --------------------------------------------------------------------------- #
# build_prompt: every cell builds; strategies are distinct
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("task,strategy", list(itertools.product(Task, Strategy)))
def test_every_task_strategy_cell_builds(task, strategy):
    examples = ()
    if strategy is Strategy.FEW_SHOT:
        examples = MERGE_EXAMPLES if task is Task.MERGE_PREDICTION else COMMENT_EXAMPLES
    prompt = build_prompt(task, strategy, CTX, examples)

    assert prompt.task is task and prompt.strategy is strategy
    assert prompt.user  # user message never empty
    # the code context is always embedded
    assert CTX.context_text.strip() in prompt.user
    # the correct output contract is present
    assert ("DECISION:" in prompt.user) == (task is Task.MERGE_PREDICTION)
    assert ("<review>" in prompt.user) == (task is Task.REVIEW_COMMENT)


@pytest.mark.parametrize("task", list(Task))
def test_strategies_are_genuinely_distinct(task):
    """The four strategies must differ in substance, not just cell identity --
    reflection Q3.9(2)/(4) depend on the prompts actually diverging."""
    def prompt_for(strategy):
        examples = ()
        if strategy is Strategy.FEW_SHOT:
            examples = MERGE_EXAMPLES if task is Task.MERGE_PREDICTION else COMMENT_EXAMPLES
        p = build_prompt(task, strategy, CTX, examples)
        return p.system + "\n" + p.user

    texts = {s: prompt_for(s) for s in Strategy}
    # all four pairwise distinct
    assert len({v for v in texts.values()}) == 4

    # each strategy's defining mechanism actually shows up in its text
    assert "step by step" in texts[Strategy.COT].lower()
    assert "senior maintainer" in texts[Strategy.ROLE_BASED].lower()
    assert "example" in texts[Strategy.FEW_SHOT].lower()
    # zero-shot is the minimal one: no persona, no examples, no reasoning steps
    zs = texts[Strategy.ZERO_SHOT].lower()
    assert "senior maintainer" not in zs
    assert "step by step" not in zs
    assert "example 1" not in zs


def test_role_based_persona_lives_in_system_message():
    p = build_prompt(Task.MERGE_PREDICTION, Strategy.ROLE_BASED, CTX)
    assert "senior maintainer" in p.system.lower()
    # the persona names the project
    assert "microsoft/vscode" in p.system


def test_few_shot_embeds_all_demonstrations():
    p = build_prompt(Task.MERGE_PREDICTION, Strategy.FEW_SHOT, CTX, MERGE_EXAMPLES)
    assert "diff A" in p.user and "diff B" in p.user
    assert "DECISION: MERGE" in p.user and "DECISION: CLOSE" in p.user


def test_comment_few_shot_embeds_gold_comments():
    p = build_prompt(Task.REVIEW_COMMENT, Strategy.FEW_SHOT, CTX, COMMENT_EXAMPLES)
    assert "Add a null check here." in p.user


# --------------------------------------------------------------------------- #
# few-shot contract enforcement
# --------------------------------------------------------------------------- #

def test_few_shot_without_examples_raises():
    with pytest.raises(ValueError, match="requires at least one"):
        build_prompt(Task.MERGE_PREDICTION, Strategy.FEW_SHOT, CTX, [])


@pytest.mark.parametrize("strategy", [Strategy.ZERO_SHOT, Strategy.COT, Strategy.ROLE_BASED])
def test_non_few_shot_with_examples_raises(strategy):
    with pytest.raises(ValueError, match="does not use them"):
        build_prompt(Task.MERGE_PREDICTION, strategy, CTX, MERGE_EXAMPLES)


# --------------------------------------------------------------------------- #
# RenderedPrompt plumbing
# --------------------------------------------------------------------------- #

def test_messages_shape_and_label():
    p = build_prompt(Task.MERGE_PREDICTION, Strategy.COT, CTX)
    msgs = p.messages()
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert p.label == "merge_prediction:cot"


def test_messages_omits_empty_system():
    # an empty system message must not be emitted (some providers reject it)
    p = build_prompt(Task.MERGE_PREDICTION, Strategy.ZERO_SHOT, CTX)
    object.__setattr__(p, "system", "")  # frozen dataclass -> force the edge case
    msgs = p.messages()
    assert [m["role"] for m in msgs] == ["user"]


def test_context_without_repo_uses_generic_project_phrase():
    ctx = PromptContext(context_text="diff", repo=None)
    p = build_prompt(Task.MERGE_PREDICTION, Strategy.ROLE_BASED, ctx)
    assert "large open-source project" in p.system


# --------------------------------------------------------------------------- #
# parse_merge_prediction
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("raw,expected", [
    ("DECISION: MERGE\nCONFIDENCE: 0.9", "MERGE"),
    ("DECISION: CLOSE\nCONFIDENCE: 0.3", "CLOSE"),
    ("reasoning...\nDECISION: **MERGE**\nCONFIDENCE: 0.8", "MERGE"),   # markdown bold
    ("DECISION: merged", "MERGE"),                                     # lowercase alias
    ("DECISION: Rejected", "CLOSE"),                                   # alias
    ("DECISION: Yes", "MERGE"),                                        # weak token OK on decision line
    ("DECISION: No", "CLOSE"),
])
def test_parse_merge_decision_line(raw, expected):
    assert parse_merge_prediction(raw).label == expected


def test_parse_merge_last_decision_wins():
    # a CoT answer may state intermediate leanings; only the final line counts
    raw = "Initially looks like CLOSE.\nDECISION: MERGE\nCONFIDENCE: 0.7"
    assert parse_merge_prediction(raw).label == "MERGE"


def test_parse_merge_fallback_last_line_strong_token():
    raw = "It is complete and tested, therefore it will be merged."
    assert parse_merge_prediction(raw).label == "MERGE"


def test_parse_merge_prose_no_token_abstains():
    # weak "no" in prose must NOT be read as a CLOSE decision
    assert parse_merge_prediction("There is no clear format in this answer.").label is None
    assert parse_merge_prediction("").label is None
    assert parse_merge_prediction("   ").label is None


@pytest.mark.parametrize("raw,expected", [
    ("DECISION: MERGE\nCONFIDENCE: 0.82", 0.82),
    ("DECISION: MERGE\nCONFIDENCE: 70%", 0.70),
    ("DECISION: MERGE\nCONFIDENCE: 1.5", 1.0),   # clamped
    ("DECISION: MERGE", None),
])
def test_parse_merge_confidence(raw, expected):
    assert parse_merge_prediction(raw).confidence == expected


def test_merge_prediction_merged_property():
    assert parse_merge_prediction("DECISION: MERGE").merged is True
    assert parse_merge_prediction("DECISION: CLOSE").merged is False
    assert parse_merge_prediction("no format").merged is None


# --------------------------------------------------------------------------- #
# parse_review_comments
# --------------------------------------------------------------------------- #

def test_parse_review_block_isolated_from_preamble():
    raw = (
        "Analysis: the function lacks a guard.\n"
        "<review>\n"
        "- Missing null check on `config` at line 12.\n"
        "- Add a unit test for the empty-input case.\n"
        "</review>\n"
        "Hope this helps!"
    )
    review = parse_review_comments(raw)
    assert review.comments == [
        "Missing null check on `config` at line 12.",
        "Add a unit test for the empty-input case.",
    ]
    # preamble and trailer excluded from the scored text
    assert "Analysis" not in review.text and "Hope this helps" not in review.text


def test_parse_review_handles_numbered_and_star_bullets():
    raw = "<review>\n1. First issue.\n* Second issue.\n</review>"
    review = parse_review_comments(raw)
    assert review.comments == ["First issue.", "Second issue."]


def test_parse_review_without_sentinels_falls_back_to_whole_text():
    raw = "- Consider renaming `x` to something descriptive."
    review = parse_review_comments(raw)
    assert review.comments == ["Consider renaming `x` to something descriptive."]


def test_parse_review_missing_closing_tag_does_not_leak_open_tag():
    """A real, non-rare failure mode (~19% of Groq llama-3.1-8b-instant Exp3
    responses): the model opens <review> but never closes it. Must NOT surface
    the literal '<review>' string as a bogus first bullet."""
    raw = "<review>\n- Missing null check on `config`.\n- Add a test for the empty case."
    review = parse_review_comments(raw)
    assert review.comments == [
        "Missing null check on `config`.",
        "Add a test for the empty case.",
    ]
    assert "<review>" not in review.comments


def test_parse_review_missing_closing_tag_with_preamble_before_open():
    raw = "Let me analyze this.\n<review>\n- Rename the variable.\n"
    review = parse_review_comments(raw)
    assert review.comments == ["Rename the variable."]
    assert "Let me analyze this." not in review.text


def test_parse_review_empty():
    assert parse_review_comments("").comments == []
    assert parse_review_comments("   ").comments == []
