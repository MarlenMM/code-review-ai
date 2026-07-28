"""The Experiment 3 prompt templates: 2 tasks × 4 strategies = 8 builders,
composed with any code context via `build_prompt`.

This is the intellectual core of Experiment 3 (lab guide §3.4.3). The four
strategies are written to embody four *different reasoning mechanisms*, not four
wordings of the same request -- because reflection questions §3.9(2) and
§3.9(4) ask specifically why few-shot beats zero-shot and why prompt design
changes comment quality, and those answers only exist if the prompts actually
differ in kind. The mechanism each strategy relies on:

    ZERO_SHOT   no scaffold                -> the model's raw prior (baseline)
    FEW_SHOT    labelled demonstrations    -> analogy + calibration to THIS
                                              dataset's boundary / comment style
    COT         an explicit sub-step order -> serial decomposition of a
                                              judgment a single shot would skip
    ROLE_BASED  a senior-maintainer persona-> shifts the evaluative standard
                                              and merge threshold toward a real
                                              gatekeeper's

The *output contract* is held identical across the four strategies within a
task (a `DECISION:`/`CONFIDENCE:` pair for merge prediction; a
``<review>...</review>`` block for comments) so that Step 17 scores every cell
of the grid the same way and the only thing that varies is the reasoning, not
the parsing. See `reports/exp3_prompt_design.md` for the full rationale and
`src/llm/prompts/parsing.py` for the matching parsers.
"""

from __future__ import annotations

from collections.abc import Sequence

from src.llm.prompts.schema import (
    FewShotExample,
    PromptContext,
    RenderedPrompt,
    Strategy,
    Task,
)

# --------------------------------------------------------------------------- #
# Shared building blocks
# --------------------------------------------------------------------------- #

# The output contracts. Kept as module constants because they are (a) repeated
# verbatim across all four strategies of a task and (b) the exact spec the
# parsers in parsing.py must mirror.
MERGE_OUTPUT_CONTRACT = (
    "End your response with exactly these two lines and nothing after them:\n"
    "DECISION: MERGE        (if you predict the PR was merged)\n"
    "   or\n"
    "DECISION: CLOSE        (if you predict it was closed without merging)\n"
    "CONFIDENCE: <number between 0.0 and 1.0 = your probability that it was merged>"
)

REVIEW_OUTPUT_CONTRACT = (
    "Put your final review inside a single block delimited exactly like this:\n"
    "<review>\n"
    "- <one concrete, actionable comment about specific code>\n"
    "- <another comment>\n"
    "</review>\n"
    "Write one comment per '- ' bullet, refer to specific code, and write "
    "nothing after </review>. If the change looks fine, put a single bullet "
    "saying so."
)

# Terse, neutral system message shared by the two strategies that deliberately
# carry NO persona (zero-shot, few-shot). Role-based replaces this entirely;
# CoT extends it with a "reason first" nudge.
_ASSISTANT_SYSTEM = (
    "You are a code-review assistant. You answer precisely and follow the "
    "required output format exactly."
)


def _project(context: PromptContext) -> str:
    """A natural phrase naming the project, for prompt prose."""
    return f"the {context.repo} project" if context.repo else "a large open-source project"


def _context_block(context: PromptContext) -> str:
    """Render the PR header + code context the templates embed verbatim."""
    lines = []
    if context.title:
        lines.append(f"PR title: {context.title}")
    lines.append("Pull request content:")
    lines.append(context.context_text.strip())
    return "\n".join(lines)


def _require_examples(strategy: Strategy, examples: Sequence[FewShotExample]) -> None:
    """Enforce the few-shot contract: examples are required for (and only for)
    the few-shot strategy. Failing loudly here beats silently sending a
    'few-shot' prompt with zero demonstrations, or letting stray examples ride
    along on a zero-shot call."""
    if strategy is Strategy.FEW_SHOT and not examples:
        raise ValueError("Strategy.FEW_SHOT requires at least one FewShotExample.")
    if strategy is not Strategy.FEW_SHOT and examples:
        raise ValueError(
            f"few_shot_examples were passed for {strategy}, which does not use them."
        )


# --------------------------------------------------------------------------- #
# MERGE PREDICTION
# --------------------------------------------------------------------------- #

_MERGE_FRAME = (
    "Task: MERGE PREDICTION. Decide whether the pull request below was "
    "ultimately MERGED into {project} or CLOSED without merging, judging only "
    "from its content."
)


def _merge_zero_shot(context: PromptContext, _examples) -> RenderedPrompt:
    frame = _MERGE_FRAME.format(project=_project(context))
    user = (
        f"{frame}\n\n"
        f"{_context_block(context)}\n\n"
        f"Give your decision directly.\n\n"
        f"{MERGE_OUTPUT_CONTRACT}"
    )
    return RenderedPrompt(Task.MERGE_PREDICTION, Strategy.ZERO_SHOT,
                          _ASSISTANT_SYSTEM, user, MERGE_OUTPUT_CONTRACT)


def _merge_few_shot(context: PromptContext, examples: Sequence[FewShotExample]) -> RenderedPrompt:
    frame = _MERGE_FRAME.format(project=_project(context))
    shots = []
    for i, ex in enumerate(examples, 1):
        shots.append(
            f"### Example {i}\n"
            f"{ex.context_text.strip()}\n"
            f"DECISION: {ex.label}"
        )
    demos = "\n\n".join(shots)
    user = (
        f"{frame}\n\n"
        f"Here are {len(examples)} labelled examples of past pull requests and "
        f"whether each was merged or closed. Infer the pattern that separates "
        f"merged from closed PRs, then classify the final one.\n\n"
        f"{demos}\n\n"
        f"### Pull request to classify\n"
        f"{_context_block(context)}\n\n"
        f"{MERGE_OUTPUT_CONTRACT}"
    )
    return RenderedPrompt(Task.MERGE_PREDICTION, Strategy.FEW_SHOT,
                          _ASSISTANT_SYSTEM, user, MERGE_OUTPUT_CONTRACT)


def _merge_cot(context: PromptContext, _examples) -> RenderedPrompt:
    frame = _MERGE_FRAME.format(project=_project(context))
    system = (
        _ASSISTANT_SYSTEM
        + " Reason carefully and step by step before committing to an answer."
    )
    user = (
        f"{frame}\n\n"
        f"{_context_block(context)}\n\n"
        f"Reason through these steps explicitly before you decide:\n"
        f"1. Scope & risk: how large is the change (lines, files)? Does it touch "
        f"core or high-risk code?\n"
        f"2. Completeness & correctness: does the diff look self-contained and "
        f"correct? Are there tests, or TODOs, debug leftovers, or obvious "
        f"defects?\n"
        f"3. Alignment with intent: does the change match its stated title and "
        f"description, or is there unexplained scope creep?\n"
        f"4. Maturity signals: any WIP markers, or very large / scattered / "
        f"unfocused churn that suggests an unfinished or contested change?\n"
        f"Then weigh these factors and give a calibrated decision. A typical "
        f"mature repository merges most PRs, so treat CLOSE as the claim that "
        f"needs the stronger evidence.\n\n"
        f"{MERGE_OUTPUT_CONTRACT}"
    )
    return RenderedPrompt(Task.MERGE_PREDICTION, Strategy.COT,
                          system, user, MERGE_OUTPUT_CONTRACT)


def _merge_role_based(context: PromptContext, _examples) -> RenderedPrompt:
    system = (
        f"You are a senior maintainer of {_project(context)} with commit access "
        f"-- the person who decides whether a pull request is merged. You have "
        f"reviewed thousands of PRs. You hold a high bar: you gate merges on "
        f"correctness, adequate testing, disciplined scope, and adherence to "
        f"project conventions, and you are wary of large, risky, or unfocused "
        f"changes. You can judge quickly whether a change is merge-ready."
    )
    frame = _MERGE_FRAME.format(project=_project(context))
    user = (
        f"{frame}\n\n"
        f"{_context_block(context)}\n\n"
        f"Apply your own merge criteria as the maintainer and decide whether you "
        f"would have merged this PR.\n\n"
        f"{MERGE_OUTPUT_CONTRACT}"
    )
    return RenderedPrompt(Task.MERGE_PREDICTION, Strategy.ROLE_BASED,
                          system, user, MERGE_OUTPUT_CONTRACT)


# --------------------------------------------------------------------------- #
# REVIEW COMMENT GENERATION
# --------------------------------------------------------------------------- #

_REVIEW_FRAME = (
    "Task: REVIEW COMMENT GENERATION. Write the code-review comments a reviewer "
    "would leave on the pull request below from {project}."
)


def _review_zero_shot(context: PromptContext, _examples) -> RenderedPrompt:
    frame = _REVIEW_FRAME.format(project=_project(context))
    user = (
        f"{frame}\n\n"
        f"{_context_block(context)}\n\n"
        f"Write your review comments directly.\n\n"
        f"{REVIEW_OUTPUT_CONTRACT}"
    )
    return RenderedPrompt(Task.REVIEW_COMMENT, Strategy.ZERO_SHOT,
                          _ASSISTANT_SYSTEM, user, REVIEW_OUTPUT_CONTRACT)


def _review_few_shot(context: PromptContext, examples: Sequence[FewShotExample]) -> RenderedPrompt:
    frame = _REVIEW_FRAME.format(project=_project(context))
    shots = []
    for i, ex in enumerate(examples, 1):
        shots.append(
            f"### Example {i} -- change under review\n"
            f"{ex.context_text.strip()}\n"
            f"Reviewer comment: {ex.target_comment}"
        )
    demos = "\n\n".join(shots)
    user = (
        f"{frame}\n\n"
        f"Below are {len(examples)} real review comments left by human "
        f"maintainers on past pull requests, each with the change it commented "
        f"on. Match their style -- concise, specific, and focused on what "
        f"actually matters -- when you review the final PR.\n\n"
        f"{demos}\n\n"
        f"### Pull request to review\n"
        f"{_context_block(context)}\n\n"
        f"{REVIEW_OUTPUT_CONTRACT}"
    )
    return RenderedPrompt(Task.REVIEW_COMMENT, Strategy.FEW_SHOT,
                          _ASSISTANT_SYSTEM, user, REVIEW_OUTPUT_CONTRACT)


def _review_cot(context: PromptContext, _examples) -> RenderedPrompt:
    frame = _REVIEW_FRAME.format(project=_project(context))
    system = (
        _ASSISTANT_SYSTEM
        + " Reason carefully and step by step before writing the final review."
    )
    user = (
        f"{frame}\n\n"
        f"{_context_block(context)}\n\n"
        f"Work in three explicit stages before producing the review:\n"
        f"1. Analyze: read the diff and list every concrete observation -- "
        f"possible correctness bugs, missing tests or unhandled edge cases, "
        f"security or performance concerns, error handling, naming, and "
        f"API/design issues.\n"
        f"2. Prioritize: keep only the issues a maintainer would actually raise "
        f"before merging; drop trivial nitpicks.\n"
        f"3. Write: turn each kept issue into one concise, actionable comment "
        f"anchored to the specific code.\n\n"
        f"{REVIEW_OUTPUT_CONTRACT}"
    )
    return RenderedPrompt(Task.REVIEW_COMMENT, Strategy.COT,
                          system, user, REVIEW_OUTPUT_CONTRACT)


def _review_role_based(context: PromptContext, _examples) -> RenderedPrompt:
    system = (
        f"You are a senior maintainer of {_project(context)} performing code "
        f"review. You review with this priority order: (1) correctness, "
        f"(2) security, (3) test coverage, (4) maintainability, (5) style. You "
        f"focus on merge-blocking issues and keep every comment concise, "
        f"specific, and actionable -- never generic praise, and never "
        f"exhaustive linter-style nitpicking."
    )
    frame = _REVIEW_FRAME.format(project=_project(context))
    user = (
        f"{frame}\n\n"
        f"{_context_block(context)}\n\n"
        f"Leave the review comments you would leave as the maintainer, in your "
        f"priority order.\n\n"
        f"{REVIEW_OUTPUT_CONTRACT}"
    )
    return RenderedPrompt(Task.REVIEW_COMMENT, Strategy.ROLE_BASED,
                          system, user, REVIEW_OUTPUT_CONTRACT)


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #

_REGISTRY = {
    (Task.MERGE_PREDICTION, Strategy.ZERO_SHOT): _merge_zero_shot,
    (Task.MERGE_PREDICTION, Strategy.FEW_SHOT): _merge_few_shot,
    (Task.MERGE_PREDICTION, Strategy.COT): _merge_cot,
    (Task.MERGE_PREDICTION, Strategy.ROLE_BASED): _merge_role_based,
    (Task.REVIEW_COMMENT, Strategy.ZERO_SHOT): _review_zero_shot,
    (Task.REVIEW_COMMENT, Strategy.FEW_SHOT): _review_few_shot,
    (Task.REVIEW_COMMENT, Strategy.COT): _review_cot,
    (Task.REVIEW_COMMENT, Strategy.ROLE_BASED): _review_role_based,
}


def build_prompt(
    task: Task,
    strategy: Strategy,
    context: PromptContext,
    few_shot_examples: Sequence[FewShotExample] = (),
) -> RenderedPrompt:
    """Materialise one cell of the task × strategy × context grid.

    `context.context_text` is whatever Step 15's context builder produced for
    this PR and this context kind; `few_shot_examples` are required iff
    `strategy is Strategy.FEW_SHOT` and must have been drawn from the
    Experiment-2 train split by `src/llm/prompts/few_shot.py` (and rendered with
    the SAME context kind as `context`, so the demonstration is a fair analogy).
    """
    if (task, strategy) not in _REGISTRY:
        raise ValueError(f"No template for {task} / {strategy}.")
    _require_examples(strategy, few_shot_examples)
    return _REGISTRY[(task, strategy)](context, few_shot_examples)
