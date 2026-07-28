"""Tests for Experiment 4 advanced prompts and dispatch
(`src/llm/prompts/conversations.py`, `schema.py`'s Exp4 additions).

These assert the contracts the Step-20 grid depends on: `build_exp4` returns a
single-shot `RenderedPrompt` for the three reused strategies and a two-turn
`RenderedConversation` for self-reflection / multi-turn; each conversational
strategy's *mechanism* actually shows up in its second turn (self-reflection
critiques its own draft; multi-turn withholds then reveals the broader context);
the AI-generated-code framing is present; and only the final turn carries the
output contract the parsers key on.
"""

import itertools

import pytest

from src.llm.prompts import (
    AugmentedContext,
    Exp4ContextTier,
    Exp4Strategy,
    FewShotExample,
    RenderedConversation,
    RenderedPrompt,
    Strategy,
    Task,
    build_exp4,
    build_prompt,
    PromptContext,
)
from src.llm.prompts.conversations import CONVERSATIONAL_STRATEGIES, build_conversation

AUG = AugmentedContext(
    diff="--- a.py (modified)\n@@ -1 +1 @@ def f\n-x=1\n+x = 1\n",
    pr_description="Fix spacing in f().",
    repo_context="File a.py: def f(): return 1",
    issue_text="Issue #9: spacing inconsistent.",
    historical_comments='A reviewer once said: "add a regression test".',
    repo="microsoft/vscode",
    title="Fix spacing",
)

MERGE_EXAMPLES = [
    FewShotExample(pr_id="p1", repo="r", context_text="diff A", label="MERGE"),
    FewShotExample(pr_id="p2", repo="r", context_text="diff B", label="CLOSE"),
]


# --------------------------------------------------------------------------- #
# build_exp4 dispatch: return types
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("strategy", [
    Exp4Strategy.ROLE_BASED, Exp4Strategy.COT,
])
def test_reused_single_shot_strategies_return_rendered_prompt(strategy):
    p = build_exp4(Task.MERGE_PREDICTION, strategy, AUG, Exp4ContextTier.COMPLETE_SE_CONTEXT)
    assert isinstance(p, RenderedPrompt)
    assert p.strategy.value == strategy.value       # mapped to the Exp-3 Strategy
    assert "DECISION:" in p.user


def test_few_shot_passthrough_embeds_examples():
    p = build_exp4(Task.MERGE_PREDICTION, Exp4Strategy.FEW_SHOT, AUG,
                   Exp4ContextTier.DIFF, few_shot_examples=MERGE_EXAMPLES)
    assert isinstance(p, RenderedPrompt)
    assert "diff A" in p.user and "diff B" in p.user


def test_few_shot_without_examples_raises():
    with pytest.raises(ValueError, match="requires at least one"):
        build_exp4(Task.MERGE_PREDICTION, Exp4Strategy.FEW_SHOT, AUG, Exp4ContextTier.DIFF)


@pytest.mark.parametrize("strategy", sorted(CONVERSATIONAL_STRATEGIES, key=lambda s: s.value))
def test_conversational_strategies_return_conversation(strategy):
    tier = Exp4ContextTier.COMPLETE_SE_CONTEXT
    c = build_exp4(Task.REVIEW_COMMENT, strategy, AUG, tier)
    assert isinstance(c, RenderedConversation)
    assert len(c.turns) == 2
    assert c.strategy is strategy
    # only the final turn carries the output contract (what gets parsed)
    assert "<review>" in c.turns[-1]


def test_conversational_strategy_rejects_few_shot_examples():
    with pytest.raises(ValueError, match="does not take few-shot"):
        build_exp4(Task.MERGE_PREDICTION, Exp4Strategy.SELF_REFLECTION, AUG,
                   Exp4ContextTier.DIFF, few_shot_examples=MERGE_EXAMPLES)


def test_every_exp4_cell_builds():
    """Every (task x strategy x applicable-tier) cell must build -- the grid
    Step 20 iterates has no holes except the documented multi-turn@DIFF one."""
    for task, strategy, tier in itertools.product(Task, Exp4Strategy, Exp4ContextTier):
        examples = MERGE_EXAMPLES if (strategy is Exp4Strategy.FEW_SHOT
                                      and task is Task.MERGE_PREDICTION) else ()
        if strategy is Exp4Strategy.FEW_SHOT and task is Task.REVIEW_COMMENT:
            examples = [FewShotExample(pr_id="p", repo="r", context_text="d",
                                       target_comment="Add a null check.")]
        if strategy is Exp4Strategy.MULTI_TURN and tier is Exp4ContextTier.DIFF:
            with pytest.raises(ValueError):
                build_exp4(task, strategy, AUG, tier)
            continue
        out = build_exp4(task, strategy, AUG, tier, few_shot_examples=examples)
        assert isinstance(out, (RenderedPrompt, RenderedConversation))


# --------------------------------------------------------------------------- #
# Self-reflection mechanism
# --------------------------------------------------------------------------- #

def test_self_reflection_turn2_critiques_own_draft():
    c = build_exp4(Task.REVIEW_COMMENT, Exp4Strategy.SELF_REFLECTION, AUG,
                   Exp4ContextTier.DIFF_PR_DESCRIPTION)
    draft, critique = c.turns
    # turn 1 is a normal review request over the full tier context
    assert "Fix spacing" in draft and "<review>" in draft
    # turn 2's mechanism: critique the model's OWN draft, AI-code-tuned
    low = critique.lower()
    assert "critique your own draft" in low
    assert "ai coding agent" in low
    assert "hallucinated" in low            # AI-generated-code failure mode named


def test_self_reflection_merge_reconsiders_decision_and_confidence():
    c = build_exp4(Task.MERGE_PREDICTION, Exp4Strategy.SELF_REFLECTION, AUG,
                   Exp4ContextTier.DIFF)
    critique = c.turns[1].lower()
    assert "confidence" in critique and "reconsidered decision" in critique
    assert "microsoft/vscode" in c.turns[1]  # persona names the project


# --------------------------------------------------------------------------- #
# Multi-turn mechanism (progressive context)
# --------------------------------------------------------------------------- #

def test_multi_turn_withholds_then_reveals_context():
    c = build_exp4(Task.MERGE_PREDICTION, Exp4Strategy.MULTI_TURN, AUG,
                   Exp4ContextTier.DIFF_REPO_CONTEXT)
    turn1, turn2 = c.turns
    # turn 1 sees only the diff; the repo context is withheld
    assert "def f" in turn1
    assert "def f(): return 1" not in turn1
    # turn 2 reveals it and asks to reconsider
    assert "def f(): return 1" in turn2
    assert "reconsidered decision" in turn2.lower()
    assert "ai coding agent" in turn2.lower()


def test_multi_turn_review_reveals_full_se_context():
    c = build_exp4(Task.REVIEW_COMMENT, Exp4Strategy.MULTI_TURN, AUG,
                   Exp4ContextTier.COMPLETE_SE_CONTEXT)
    turn2 = c.turns[1]
    # the broader SE context (issue, repo, history) surfaces only in turn 2
    assert "spacing inconsistent" in turn2
    assert "add a regression test" in turn2
    assert "<review>" in turn2


def test_build_conversation_rejects_single_shot_strategy():
    with pytest.raises(ValueError, match="single-shot"):
        build_conversation(Task.MERGE_PREDICTION, Exp4Strategy.ROLE_BASED, AUG,
                           Exp4ContextTier.DIFF)


# --------------------------------------------------------------------------- #
# RenderedConversation plumbing
# --------------------------------------------------------------------------- #

def test_conversation_label_and_initial_messages():
    c = build_exp4(Task.MERGE_PREDICTION, Exp4Strategy.SELF_REFLECTION, AUG,
                   Exp4ContextTier.DIFF)
    assert c.label == "merge_prediction:self_reflection"
    msgs = c.initial_messages()
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[-1]["content"] == c.turns[0]     # first user turn only
    assert not c.is_single_turn


def test_conversation_requires_at_least_one_turn():
    with pytest.raises(ValueError, match="at least one turn"):
        RenderedConversation(task=Task.MERGE_PREDICTION,
                             strategy=Exp4Strategy.MULTI_TURN,
                             system="s", turns=(), response_contract="c")


def test_from_prompt_wraps_single_shot_as_one_turn():
    p = build_prompt(Task.MERGE_PREDICTION, Strategy.ROLE_BASED,
                     PromptContext(context_text="diff", repo="r"))
    c = RenderedConversation.from_prompt(p)
    assert c.is_single_turn
    assert c.turns == (p.user,)
    assert c.strategy is Exp4Strategy.ROLE_BASED


def test_from_prompt_rejects_zero_shot():
    p = build_prompt(Task.MERGE_PREDICTION, Strategy.ZERO_SHOT,
                     PromptContext(context_text="diff"))
    with pytest.raises(ValueError, match="no Exp4Strategy equivalent"):
        RenderedConversation.from_prompt(p)


def test_initial_messages_omits_empty_system():
    c = RenderedConversation(task=Task.MERGE_PREDICTION,
                             strategy=Exp4Strategy.MULTI_TURN,
                             system="", turns=("hi",), response_contract="c")
    assert [m["role"] for m in c.initial_messages()] == ["user"]
