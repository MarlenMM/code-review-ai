"""Prompt-template library for the LLM code-review experiments (lab guide §3
and §4).

Experiment 3 (human-written code) -- build a single-shot prompt for one of two
tasks under one of four prompt strategies, over any code context, and parse the
answer back:

    from src.llm.prompts import (
        Task, Strategy, PromptContext, build_prompt,
        parse_merge_prediction, parse_review_comments,
    )

    prompt = build_prompt(Task.MERGE_PREDICTION, Strategy.COT,
                          PromptContext(context_text=diff, repo="microsoft/vscode"))
    result = parse_merge_prediction(response_text)

The four Exp-3 strategies (`Strategy`) are deliberately distinct reasoning
mechanisms -- see `templates.py` and `reports/exp3_prompt_design.md`.

Experiment 4 (AI-generated code) -- extends this with (a) five richer context
tiers (`Exp4ContextTier` + `AugmentedContext`, `context_aug.py`) and (b) two
*new* advanced strategies on top of the three it reuses: self-reflection
(critique-then-finalise) and multi-turn (progressive context), which render to a
multi-message `RenderedConversation`. One dispatcher, `build_exp4`, covers all
five Exp-4 strategies:

    from src.llm.prompts import (
        Task, Exp4Strategy, Exp4ContextTier, AugmentedContext, build_exp4,
    )

    cell = build_exp4(Task.REVIEW_COMMENT, Exp4Strategy.SELF_REFLECTION,
                      AugmentedContext(diff=diff, repo=repo, ...),
                      Exp4ContextTier.COMPLETE_SE_CONTEXT)
    # RenderedPrompt (single-shot strategies) or RenderedConversation (self-
    # reflection / multi-turn) -- the Step-20 runner dispatches on the type.

Few-shot demonstrations come exclusively from a *train* split -- the Exp-2 train
split for Exp 3 (`load_train_pool`), and the AI-authored train split for Exp 4
(`exp4_examples.load_ai_train_pool`) -- keeping the evaluation set clean. See
`reports/exp4_prompt_and_context_design.md` for the Exp-4 design rationale and
the honest per-tier data-availability accounting.

These modules are prompt/context *design* only: no context data-fetching and no
API calls happen here (those are the Exp-3 Step 16 / Exp-4 Step 20 runs).
"""

from __future__ import annotations

from src.llm.prompts.context_aug import (
    AugmentedContext,
    Exp4ContextTier,
    IssueReferences,
    MULTI_TURN_TIERS,
    build_lightweight_repo_context,
    extract_issue_references,
    render_tier,
    split_for_multi_turn,
)
from src.llm.prompts.conversations import (
    CORE_GRID_STRATEGIES,
    build_conversation,
    build_exp4,
)
from src.llm.prompts.exp4_examples import (
    load_ai_test_pool,
    load_ai_train_pool,
    select_historical_comments,
)
from src.llm.prompts.few_shot import (
    ExampleSources,
    default_render_context,
    load_sources,
    load_train_pool,
    select_comment_examples,
    select_merge_examples,
)
from src.llm.prompts.parsing import (
    GeneratedReview,
    MergePrediction,
    parse_merge_prediction,
    parse_review_comments,
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

__all__ = [
    # schema
    "Task",
    "Strategy",
    "Exp4Strategy",
    "EXP3_EQUIVALENT",
    "PromptContext",
    "FewShotExample",
    "RenderedPrompt",
    "RenderedConversation",
    # Experiment 3 templates
    "build_prompt",
    "MERGE_OUTPUT_CONTRACT",
    "REVIEW_OUTPUT_CONTRACT",
    # Experiment 4 context augmentation
    "Exp4ContextTier",
    "AugmentedContext",
    "render_tier",
    "split_for_multi_turn",
    "MULTI_TURN_TIERS",
    "IssueReferences",
    "extract_issue_references",
    "build_lightweight_repo_context",
    # Experiment 4 advanced prompts
    "build_exp4",
    "build_conversation",
    "CORE_GRID_STRATEGIES",
    # Experiment 4 example / context selection
    "load_ai_train_pool",
    "load_ai_test_pool",
    "select_historical_comments",
    # few-shot selection (shared)
    "ExampleSources",
    "load_sources",
    "load_train_pool",
    "select_merge_examples",
    "select_comment_examples",
    "default_render_context",
    # parsing (shared)
    "parse_merge_prediction",
    "parse_review_comments",
    "MergePrediction",
    "GeneratedReview",
]
