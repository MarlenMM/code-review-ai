"""Experiment 4 advanced prompts: the self-reflection and multi-turn strategies
(lab guide §4.7.3 + §4.4.3), plus `build_exp4` -- the one dispatcher that turns
any (task, `Exp4Strategy`, `AugmentedContext`, tier) into a runnable prompt.

Where Experiment 3's `templates.py` produces a single-shot `RenderedPrompt`,
the two *new* strategies here produce a `RenderedConversation`, because their
reasoning mechanism is the turn sequence itself (see `schema.RenderedConversation`):

* ``SELF_REFLECTION`` -- turn 1 draws a *draft* review/decision over the full
  tier context; turn 2 asks the model to critique its own draft and finalise.
  The critique turn is deliberately tuned to the ways AI-generated code tends to
  fail (plausible-but-wrong changes, hallucinated/misused APIs, missing edge
  cases, over-broad edits), which is exactly the subject of Experiment 4
  (`is_ai_authored=True`). This is what makes it mechanistically distinct from
  Chain-of-Thought: CoT reasons *forward* in one shot; self-reflection commits a
  draft and then reasons *about that committed draft*.
* ``MULTI_TURN`` -- turn 1 draws a review/decision from the diff *alone*; turn 2
  reveals the broader software-engineering context the tier carries (repo
  context, the linked issue, historical review comments) and asks the model to
  reconsider. This operationalises §4.3's claim that local diffs are often
  insufficient for AI-generated code: the experiment can measure how much the
  answer changes once the surrounding context is disclosed.

The three strategies Experiment 4 *reuses* from Experiment 3 (role-based,
few-shot, CoT) are dispatched straight through to `templates.build_prompt`, so
`build_exp4` is the single call the Step-20 grid needs for all five strategies.
The output contracts are unchanged, so the final reply of every strategy -- a
single-shot answer or a conversation's last turn -- parses with the same
`parsing.py` parsers.
"""

from __future__ import annotations

from collections.abc import Sequence

from src.llm.prompts.context_aug import (
    AugmentedContext,
    Exp4ContextTier,
    render_tier,
    split_for_multi_turn,
)
from src.llm.prompts.schema import (
    EXP3_EQUIVALENT,
    Exp4Strategy,
    FewShotExample,
    PromptContext,
    RenderedConversation,
    RenderedPrompt,
    Strategy,
    Task,
)
from src.llm.prompts.templates import (
    MERGE_OUTPUT_CONTRACT,
    REVIEW_OUTPUT_CONTRACT,
    build_prompt,
)

_OUTPUT_CONTRACT = {
    Task.MERGE_PREDICTION: MERGE_OUTPUT_CONTRACT,
    Task.REVIEW_COMMENT: REVIEW_OUTPUT_CONTRACT,
}

# The strategies that render to a multi-turn RenderedConversation rather than a
# single-shot RenderedPrompt.
CONVERSATIONAL_STRATEGIES: frozenset[Exp4Strategy] = frozenset(
    {Exp4Strategy.SELF_REFLECTION, Exp4Strategy.MULTI_TURN}
)

# The core §4.7.3 Step-3 grid (role-based / few-shot / CoT / self-reflection).
# Multi-turn is the additional strategy §4.4.3 names and the plan's row 19 asks
# for; it is a first-class `Exp4Strategy` member but is called out separately so
# Step 20 can report the §4.7.3 grid and the multi-turn exploration distinctly.
CORE_GRID_STRATEGIES: tuple[Exp4Strategy, ...] = (
    Exp4Strategy.ROLE_BASED,
    Exp4Strategy.FEW_SHOT,
    Exp4Strategy.COT,
    Exp4Strategy.SELF_REFLECTION,
)


def _project_phrase(repo: str | None) -> str:
    return f"the {repo} project" if repo else "a large open-source project"


# --------------------------------------------------------------------------- #
# Self-reflection (critique-then-finalise)
# --------------------------------------------------------------------------- #

def _reflection_turn2(task: Task, repo: str | None) -> str:
    project = _project_phrase(repo)
    contract = _OUTPUT_CONTRACT[task]
    if task is Task.MERGE_PREDICTION:
        return (
            "Now step back and critique your own draft decision above as a "
            "skeptical senior reviewer.\n"
            "This pull request was written by an AI coding agent, and "
            "AI-generated changes are prone to looking plausible while being "
            "subtly wrong: hallucinated or misused APIs, missing edge cases and "
            "error handling, tests that assert the wrong thing, and edits "
            "broader than the task needs. Check specifically:\n"
            "- Did you over- or under-weight the change's risk and completeness?\n"
            "- Is your stated confidence calibrated, or just anchored on the "
            "fact that most PRs merge?\n"
            f"- Would the maintainers of {project} actually have merged this, "
            "given what the diff really shows?\n"
            "Then give your final, reconsidered decision.\n\n"
            f"{contract}"
        )
    return (
        "Now critique your own draft review above before finalising it. "
        "This pull request was written by an AI coding agent, so a second, "
        "skeptical pass is worth it. Check your draft for:\n"
        "- real issues you missed (correctness bugs, missing tests or edge "
        "cases, hallucinated or misused APIs, security or performance problems);\n"
        "- comments that are vague, generic, or not anchored to specific code;\n"
        "- any claim you cannot actually support from the diff -- fix or drop it.\n"
        "Then produce your improved final review.\n\n"
        f"{contract}"
    )


def _build_self_reflection(
    task: Task, context: PromptContext
) -> RenderedConversation:
    # Turn 1 is a plain draft over the FULL tier context -- reusing the
    # zero-shot builder so the draft request stays identical in wording to
    # Experiment 3's baseline, isolating the *reflection* turn as the only new
    # variable. (Self-reflection replaces zero-shot in Exp 4, but zero-shot's
    # neutral request is still the natural way to elicit the draft it critiques.)
    draft = build_prompt(task, Strategy.ZERO_SHOT, context)
    turn2 = _reflection_turn2(task, context.repo)
    return RenderedConversation(
        task=task,
        strategy=Exp4Strategy.SELF_REFLECTION,
        system=draft.system,
        turns=(draft.user, turn2),
        response_contract=_OUTPUT_CONTRACT[task],
        context_kind=context.context_kind,
    )


# --------------------------------------------------------------------------- #
# Multi-turn (progressive context)
# --------------------------------------------------------------------------- #

def _multi_turn_turn2(task: Task, augmentation: str) -> str:
    contract = _OUTPUT_CONTRACT[task]
    preamble = (
        "Here is additional software-engineering context about this change "
        "that was not in the diff alone:\n\n"
        f"{augmentation}\n\n"
        "This pull request was written by an AI coding agent. AI-generated code "
        "often looks correct in isolation yet conflicts with the surrounding "
        "codebase or misreads the change's real intent -- which is only visible "
        "with this broader context."
    )
    if task is Task.MERGE_PREDICTION:
        return (
            f"{preamble}\n"
            "Re-examine your decision in light of it: does the change actually "
            "fit the code it touches and satisfy the intent above? Give your "
            "final, reconsidered decision.\n\n"
            f"{contract}"
        )
    return (
        f"{preamble}\n"
        "Revisit your review using this context: add issues it reveals "
        "(inconsistency with the surrounding code, drift from the issue's "
        "intent, missing integration or tests) and drop any earlier comment it "
        "resolves. Then produce your final review.\n\n"
        f"{contract}"
    )


def _build_multi_turn(
    task: Task, context: AugmentedContext, tier: Exp4ContextTier
) -> RenderedConversation:
    # Turn 1: a review/decision from the diff ALONE (the "local PR view"), built
    # with the same zero-shot wording so only the progressive-disclosure
    # mechanism differs. Turn 2 reveals the tier's broader context.
    base_text, augmentation = split_for_multi_turn(context, tier)
    base_ctx = PromptContext(
        context_text=base_text,
        repo=context.repo,
        title=context.title,
        context_kind=tier.value,
    )
    turn1 = build_prompt(task, Strategy.ZERO_SHOT, base_ctx)
    turn2 = _multi_turn_turn2(task, augmentation)
    return RenderedConversation(
        task=task,
        strategy=Exp4Strategy.MULTI_TURN,
        system=turn1.system,
        turns=(turn1.user, turn2),
        response_contract=_OUTPUT_CONTRACT[task],
        context_kind=tier.value,
    )


def build_conversation(
    task: Task, strategy: Exp4Strategy, context: AugmentedContext, tier: Exp4ContextTier
) -> RenderedConversation:
    """Build the multi-turn prompt for a conversational strategy
    (`SELF_REFLECTION` or `MULTI_TURN`). Raises for the single-shot strategies
    (use `build_exp4`, which routes those to `templates.build_prompt`)."""
    if strategy is Exp4Strategy.SELF_REFLECTION:
        return _build_self_reflection(task, render_tier(context, tier))
    if strategy is Exp4Strategy.MULTI_TURN:
        return _build_multi_turn(task, context, tier)
    raise ValueError(
        f"{strategy} is single-shot, not conversational; call build_exp4 instead."
    )


# --------------------------------------------------------------------------- #
# Unified dispatcher
# --------------------------------------------------------------------------- #

def build_exp4(
    task: Task,
    strategy: Exp4Strategy,
    context: AugmentedContext,
    tier: Exp4ContextTier,
    *,
    few_shot_examples: Sequence[FewShotExample] = (),
) -> RenderedPrompt | RenderedConversation:
    """Materialise one cell of Experiment 4's (task x strategy x tier) grid.

    Returns a single-shot `RenderedPrompt` for the three strategies reused from
    Experiment 3 (role-based / few-shot / CoT), and a `RenderedConversation` for
    the two new ones (self-reflection / multi-turn). The Step-20 runner
    dispatches on the return type: `RenderedPrompt` -> `provider.generate`;
    `RenderedConversation` -> the conversation-aware driver.

    `few_shot_examples` are required iff `strategy is Exp4Strategy.FEW_SHOT`
    (enforced by the underlying `build_prompt`) and must be drawn from the
    AI-authored PRs' *train* split -- see `exp4_examples.load_ai_train_pool`.
    """
    if strategy in EXP3_EQUIVALENT:
        prompt_ctx = render_tier(context, tier)
        return build_prompt(task, EXP3_EQUIVALENT[strategy], prompt_ctx, few_shot_examples)
    if strategy in CONVERSATIONAL_STRATEGIES:
        if few_shot_examples:
            raise ValueError(
                f"{strategy} does not take few-shot examples."
            )
        return build_conversation(task, strategy, context, tier)
    raise ValueError(f"Unknown Exp4Strategy: {strategy!r}")
