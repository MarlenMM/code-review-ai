"""Deep-mode review-comment generation for the FastAPI `/review` endpoint:
wraps Experiment 4's best-performing config against the live Groq free tier.

Config choice -- `Exp4Strategy.MULTI_TURN` @ `Exp4ContextTier.DIFF_REPO_CONTEXT`,
task `REVIEW_COMMENT` (see `reports/exp4_model_evaluation.md` and
`reports/api_design.md` for the full justification):
* By strategy (§1.3): `multi_turn` and `self_reflection` are the only two
  strategies that ever predict "not merged" correctly on AI-authored code;
  `multi_turn` has the best macro-F1 (0.49) of any strategy without
  `self_reflection`'s accuracy collapse.
* By context tier (§1.2): `diff_repo_context` is the tier that lifts
  not-merged recall (0.0 -> 0.4) and macro-F1 to its peak (0.50); the
  `complete` tier that stacks everything regresses back to 0.0, so more
  context is deliberately NOT used here.
* These two findings are marginal (pooled-over-the-other-axis), not a
  jointly-measured (tier, strategy) cell -- Experiment 4's n=3-PR sample
  is too small to cross them -- so this pairing is the principled choice
  given what was measured, not a further-verified joint optimum.

Repo context is built LOCALLY from the diff's own git hunk headings
(`context_aug.build_lightweight_repo_context`) -- exactly the no-fetch
approximation that function exists for -- so deep mode needs no GitHub call
and works for any diff, not just PRs already in the mined dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from src.api.diff_parsing import ParsedFile
from src.llm.prompts import (
    AugmentedContext,
    Exp4ContextTier,
    Exp4Strategy,
    Task,
    build_exp4,
    build_lightweight_repo_context,
    parse_review_comments,
)
from src.llm.providers import GroqProvider, LLMAPIError, LLMQuotaExceededError
from src.llm.run_exp4_grid import (
    GROQ_RUN_MAX_DIFF_CHARS,
    GROQ_RUN_MAX_FILE_CHARS,
    GROQ_RUN_MAX_OUTPUT_TOKENS,
)

STRATEGY = Exp4Strategy.MULTI_TURN
TIER = Exp4ContextTier.DIFF_REPO_CONTEXT
TASK = Task.REVIEW_COMMENT


@dataclass(frozen=True)
class DeepReviewResult:
    comments: list[str] | None
    config_label: str | None
    warning: str | None


@lru_cache(maxsize=1)
def _provider() -> GroqProvider:
    return GroqProvider()


def _render_diff_for_llm(files: list[ParsedFile], max_total_chars: int, max_file_chars: int) -> str:
    """Same largest-churn-first, capped assembly as
    `context_builder.build_diff_text`, adapted to a `ParsedFile` list
    instead of a `files_changed` DataFrame -- keeps a single request safely
    under Groq's hard per-request token cap (see `run_exp4_grid.py`)."""
    ordered = sorted(files, key=lambda f: f.additions + f.deletions, reverse=True)
    parts: list[str] = []
    total = 0
    n_omitted = 0
    for f in ordered:
        patch = f.patch
        if patch and len(patch) > max_file_chars:
            patch = patch[:max_file_chars] + "\n... [file diff truncated]"
        block = (
            f"--- {f.filename} ({f.status})\n{patch}"
            if patch else f"--- {f.filename} ({f.status}, no textual diff)"
        )
        if total + len(block) > max_total_chars:
            n_omitted += 1
            continue
        parts.append(block)
        total += len(block)

    text = "\n\n".join(parts)
    if n_omitted:
        text += f"\n\n... [{n_omitted} additional changed file(s) omitted for length]"
    return text or "(no file diff available)"


def generate_review_comments(
    files: list[ParsedFile], *, repo: str | None = None, title: str | None = None,
) -> DeepReviewResult:
    """Run the deep-mode LLM pass. Never raises: any provider/network/quota
    failure comes back as a `DeepReviewResult` with `comments=None` and a
    human-readable `warning`, so a Groq outage or an exhausted daily quota
    degrades the `/review` endpoint to fast-mode-only output instead of a
    500 -- the LLM pass is additive, not load-bearing."""
    diff_text = _render_diff_for_llm(files, GROQ_RUN_MAX_DIFF_CHARS, GROQ_RUN_MAX_FILE_CHARS)
    repo_context = build_lightweight_repo_context([
        {"filename": f.filename, "status": f.status, "patch": f.patch} for f in files
    ])
    context = AugmentedContext(diff=diff_text, repo_context=repo_context, repo=repo, title=title)
    conversation = build_exp4(TASK, STRATEGY, context, TIER)

    try:
        provider = _provider()
        response = provider.generate_conversation(
            conversation, max_output_tokens=GROQ_RUN_MAX_OUTPUT_TOKENS
        )
    except LLMQuotaExceededError as exc:
        return DeepReviewResult(None, None, f"Groq quota exhausted: {exc}")
    except LLMAPIError as exc:
        return DeepReviewResult(None, None, f"LLM call failed: {exc}")
    except Exception as exc:  # missing GROQ_API_KEY, network errors, etc.
        return DeepReviewResult(None, None, f"LLM unavailable: {exc}")

    parsed = parse_review_comments(response.final_text)
    label = f"groq/{provider.model} {STRATEGY.value} @ {TIER.value}"
    return DeepReviewResult(parsed.comments, label, None)
