"""Core types for the Experiment 3 prompt-template library (`src/llm/prompts/`).

This module defines the small vocabulary the rest of the library is built on
and, deliberately, *nothing else* -- no prompt text, no data loading, no API
calls. Keeping the enums and dataclasses in one dependency-free module lets the
context builder (Step 15), the inference harness (Step 16), and the scorer
(Step 17) all agree on the same `Task`/`Strategy` labels without importing the
(heavier) template or few-shot logic.

Design boundary between this library and the context builder (Step 15)
----------------------------------------------------------------------
A *prompt* here is a function of three orthogonal things:

    (task, strategy)  ×  a rendered code-context string

The lab guide (§3.7) frames Experiment 3 as a grid of **4 contexts × 4
prompts** for each of **2 tasks**. The two axes are kept strictly separate:

* WHICH context (diff-only / diff+description / diff+commit-message /
  diff+metadata) is Step 15's `context_builder`'s job. It produces the plain
  string carried by `PromptContext.context_text`. This library never decides
  what goes into that string -- so the same four templates compose with any
  context the builder emits, and the grid is one nested loop, not 32 hand-
  written scripts.
* HOW the model is asked to reason about that context (zero-shot / few-shot /
  Chain-of-Thought / role-based) is this library's job.

That separation is why `PromptContext` carries an already-rendered
`context_text` rather than raw diffs/metadata: the templates are intentionally
blind to how rich the context is.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Task(str, Enum):
    """The two code-review tasks the lab guide §3.4.4 defines."""

    MERGE_PREDICTION = "merge_prediction"
    REVIEW_COMMENT = "review_comment"


class Strategy(str, Enum):
    """The four prompt-engineering strategies the lab guide §3.4.3 names.

    These are meant to be *mechanistically* distinct reasoning strategies, not
    reworded versions of one instruction (see `reports/exp3_prompt_design.md`
    for the design rationale):

    - ``ZERO_SHOT``   -- bare instruction, no scaffold: the no-help baseline
      that probes the model's raw prior.
    - ``FEW_SHOT``    -- in-context learning by analogy: labelled
      demonstrations (drawn ONLY from the Experiment-2 train split) calibrate
      the model to this dataset's decision boundary / comment style.
    - ``COT``         -- Chain-of-Thought: an explicit decomposition of the
      judgment into ordered analytical sub-steps before the model commits.
    - ``ROLE_BASED``  -- an expert persona that shifts the *evaluative frame*
      and standards the model applies (a senior maintainer's merge bar).
    """

    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"
    COT = "cot"
    ROLE_BASED = "role_based"


class Exp4Strategy(str, Enum):
    """The prompt-optimisation strategies **Experiment 4** evaluates (lab guide
    §4.7.3 Step 3 + §4.4.3), which deliberately differ from Experiment 3's
    `Strategy` lineup -- self-reflection *replaces* zero-shot, and multi-turn is
    added:

    - ``ROLE_BASED`` / ``FEW_SHOT`` / ``COT`` -- carried over unchanged from
      Experiment 3. The very same `templates.py` builders serve them; the only
      things that change for these three between Exp 3 and Exp 4 are the
      *context* (richer, per §4.4.2) and the *population* (AI-authored PRs).
      They render to a single-shot `RenderedPrompt`, exactly as in Exp 3.
    - ``SELF_REFLECTION`` -- REPLACES ``ZERO_SHOT`` in the concrete §4.7.3
      Step-3 list. The model drafts a review/decision, then critiques its own
      draft (a critique turn tuned to AI-generated-code failure modes) before
      committing a final answer. Renders to a two-turn `RenderedConversation`.
    - ``MULTI_TURN`` -- the additional optimisation strategy §4.4.3 names (and
      the project plan's row 19 asks for explicitly, alongside self-reflection).
      The context is revealed progressively across turns -- the diff first, the
      broader software-engineering context second -- so the model can revise a
      diff-only judgment once it sees what the local diff omitted. This directly
      operationalises §4.3's thesis that "relying solely on local modifications
      within a pull request is frequently insufficient" for AI-generated code.
      Renders to a `RenderedConversation`.

    Kept as a **separate enum** from `Strategy` on purpose: Experiment 3's grid
    runner (`run_exp3_grid.py`) and tests iterate `list(Strategy)` directly, so
    growing that enum would silently change Experiment 3's completed 512-cell
    grid and its cell-count assertions. The three shared members map back to a
    `Strategy` by identical `.value` (see `conversations.EXP3_EQUIVALENT`).
    """

    ROLE_BASED = "role_based"
    FEW_SHOT = "few_shot"
    COT = "cot"
    SELF_REFLECTION = "self_reflection"
    MULTI_TURN = "multi_turn"


@dataclass(frozen=True)
class PromptContext:
    """Everything a template needs about the PR under review, independent of
    *how* the context string was assembled.

    `context_text` is the code context rendered by Step 15's context builder
    (diff, optionally + PR description / commit message / metadata). `repo` and
    `title` are light header fields used to make the prompt read naturally and
    to let the role-based persona name the project; they are optional so the
    library stays usable with a bare diff. `context_kind` is a free-form label
    (e.g. ``"diff_only"``) used only for logging / cache keys / result-table
    columns -- templates never branch on it.
    """

    context_text: str
    repo: str | None = None
    title: str | None = None
    context_kind: str | None = None


@dataclass(frozen=True)
class FewShotExample:
    """One labelled demonstration for a few-shot prompt.

    Every example MUST originate from the Experiment-2 **train** split (see
    `src/llm/prompts/few_shot.py`); its `context_text` is expected to have been
    rendered by the *same* context builder and *same* context kind as the query
    it will accompany, so the demonstration is a fair analogy rather than a
    format mismatch.

    - Merge prediction: `label` is ``"MERGE"`` or ``"CLOSE"``; `target_comment`
      is unused.
    - Review-comment generation: `target_comment` is a real human review
      comment; `label` is unused.
    """

    pr_id: str
    repo: str
    context_text: str
    label: str | None = None
    target_comment: str | None = None


@dataclass(frozen=True)
class RenderedPrompt:
    """A fully-materialised prompt, ready to hand to a provider (Step 15).

    `system` and `user` are the two message parts kept separate so a provider
    can map them onto whatever the API expects (Gemini: ``system_instruction``
    + a user content part; an OpenAI-compatible SDK: two chat messages). The
    role-based persona lives in `system`; the concrete request + context live
    in `user`. `system` may be empty (there is no hard requirement that a
    provider support a system role).

    `response_contract` is the human-readable statement of the required output
    format that the matching parser in `src/llm/prompts/parsing.py` is the dual
    of -- kept on the object so the format the prompt asks for and the format
    the parser expects never drift apart silently.
    """

    task: Task
    strategy: Strategy
    system: str
    user: str
    response_contract: str

    @property
    def label(self) -> str:
        """Stable identifier for this prompt configuration, e.g.
        ``"merge_prediction:cot"`` -- used by Step 16 for cache keys and result
        columns. The context kind is appended by the caller when it varies the
        context axis, keeping this property independent of Step 15."""
        return f"{self.task.value}:{self.strategy.value}"

    def messages(self) -> list[dict[str, str]]:
        """Provider-neutral message list. Omits an empty system message so
        providers that reject a blank system role are not handed one."""
        msgs: list[dict[str, str]] = []
        if self.system:
            msgs.append({"role": "system", "content": self.system})
        msgs.append({"role": "user", "content": self.user})
        return msgs


# Map from an Exp4Strategy value to its Strategy equivalent, for the three
# single-shot strategies Experiment 4 reuses verbatim from Experiment 3. Kept
# next to the two enums so the correspondence is discoverable from the schema
# alone; `conversations.build_exp4` uses it to delegate to `templates.build_prompt`.
EXP3_EQUIVALENT: dict[Exp4Strategy, Strategy] = {
    Exp4Strategy.ROLE_BASED: Strategy.ROLE_BASED,
    Exp4Strategy.FEW_SHOT: Strategy.FEW_SHOT,
    Exp4Strategy.COT: Strategy.COT,
}


@dataclass(frozen=True)
class RenderedConversation:
    """A multi-turn prompt: one persona `system` message plus an ordered tuple
    of user `turns`. This is how Experiment 4's `SELF_REFLECTION` and
    `MULTI_TURN` strategies are represented, because their reasoning *mechanism
    is the sequence itself*: the model must see its own prior reply before the
    next user turn is meaningful (a single-shot prompt cannot express "critique
    the draft you just wrote" or "reconsider now that you've seen more context").

    Driving protocol (implemented by the Step-20 runner against a
    conversation-aware provider call -- see `reports/exp4_prompt_and_context_design.md`):

        history = ([{"role": "system", "content": system}] if system else [])
        for turn in turns:
            history.append({"role": "user", "content": turn})
            reply = provider.chat(history)          # one live/cached call
            history.append({"role": "assistant", "content": reply})
        answer = last assistant reply               # parse with parsing.py

    Only the FINAL turn carries the output contract, so only the final reply is
    parsed (with the same `parse_merge_prediction` / `parse_review_comments`
    the single-shot contract uses); earlier replies are drafts / initial
    reviews. `turns` always has at least one entry, so a single-shot prompt is
    representable as a 1-turn conversation via `from_prompt` -- letting the
    runner use one uniform code path across all five strategies if it prefers.
    """

    task: Task
    strategy: Exp4Strategy
    system: str
    turns: tuple[str, ...]
    response_contract: str
    context_kind: str | None = None

    def __post_init__(self) -> None:
        if not self.turns:
            raise ValueError("RenderedConversation requires at least one turn.")

    @property
    def label(self) -> str:
        """Stable identifier, e.g. ``"review_comment:self_reflection"`` -- used
        by Step 20 for cache keys and result columns (the context tier is
        appended by the caller when it varies the context axis)."""
        return f"{self.task.value}:{self.strategy.value}"

    @property
    def is_single_turn(self) -> bool:
        return len(self.turns) == 1

    def initial_messages(self) -> list[dict[str, str]]:
        """The messages to send for the FIRST turn (system, if non-empty, plus
        turn 0). Subsequent turns are appended by the runner after each reply,
        per the driving protocol above."""
        msgs: list[dict[str, str]] = []
        if self.system:
            msgs.append({"role": "system", "content": self.system})
        msgs.append({"role": "user", "content": self.turns[0]})
        return msgs

    @classmethod
    def from_prompt(cls, prompt: RenderedPrompt) -> "RenderedConversation":
        """Wrap a single-shot `RenderedPrompt` as a 1-turn conversation so a
        runner can treat every strategy uniformly. Only the three strategies
        Experiment 4 reuses (role-based / few-shot / CoT) have an `Exp4Strategy`
        equivalent; wrapping a zero-shot prompt (Exp 3 only) raises."""
        try:
            strategy = Exp4Strategy(prompt.strategy.value)
        except ValueError as exc:  # e.g. Strategy.ZERO_SHOT has no Exp4 analogue
            raise ValueError(
                f"{prompt.strategy} has no Exp4Strategy equivalent; "
                "from_prompt is only for role-based/few-shot/CoT."
            ) from exc
        return cls(
            task=prompt.task,
            strategy=strategy,
            system=prompt.system,
            turns=(prompt.user,),
            response_contract=prompt.response_contract,
        )
