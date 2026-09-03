"""Deep-mode review-comment generation for the FastAPI `/review` endpoint:
wraps Experiment 4's best-performing config against a live LLM provider.

Provider -- **Qwen** (Alibaba Cloud DashScope), not the Groq free tier
Experiments 3/4 were measured on. Groq retired `llama-3.1-8b-instant`, so
every live deep-mode call began returning `404 The model 'llama-3.1-8b-instant'
does not exist or you do not have access to it`. The experiments' recorded
numbers are unaffected -- they are committed artifacts, and a `--provider groq`
grid re-run still replays them for free from `data/llm_cache/` -- but the live
path needed a provider that actually answers.

What that does and does not mean:
* The *config* (strategy, context tier, task, prompt text) is unchanged, so
  this is still Experiment 4's chosen configuration; only the model behind it
  differs.
* The *evidence* for that config is still Experiment 4's, which was measured on
  `llama-3.1-8b-instant`. Whether `multi_turn` @ `diff_repo_context` is also
  the best config for Qwen is untested -- re-running the Exp-4 grid with
  `--provider qwen` is what would answer it. Deep mode's `llm_config` label
  names the model actually used, so a result is never mistaken for one of
  Experiment 4's.

Set `CODE_REVIEW_AI_LLM_PROVIDER` (`qwen` | `groq` | `gemini`) to change it
without a code edit -- a pinned model id has already been retired upstream once
during this project's life, and the fix should not require a redeploy.

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

import os
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
from src.llm.providers import (
    PROVIDERS,
    CachedChatProvider,
    LLMAPIError,
    LLMQuotaExceededError,
)
from src.llm.run_exp4_grid import (
    GROQ_RUN_MAX_DIFF_CHARS,
    GROQ_RUN_MAX_FILE_CHARS,
    GROQ_RUN_MAX_OUTPUT_TOKENS,
)

STRATEGY = Exp4Strategy.MULTI_TURN
TIER = Exp4ContextTier.DIFF_REPO_CONTEXT
TASK = Task.REVIEW_COMMENT

DEFAULT_PROVIDER = "qwen"


@dataclass(frozen=True)
class DeepReviewResult:
    comments: list[str] | None
    config_label: str | None
    warning: str | None


@lru_cache(maxsize=1)
def _provider() -> CachedChatProvider:
    """The live provider, built once per process (the cache is on disk, so a
    single client is enough and avoids re-reading the key per request).

    `lru_cache` means an unset key raises on the *first* deep-mode request and
    is then re-raised cheaply; `generate_review_comments` turns that into an
    `llm_warning`, so a missing key degrades the endpoint rather than 500-ing
    it -- the same treatment as a quota failure."""
    name = os.environ.get("CODE_REVIEW_AI_LLM_PROVIDER", DEFAULT_PROVIDER).strip().lower()
    provider_cls = PROVIDERS.get(name)
    if provider_cls is None:
        raise ValueError(
            f"Unknown CODE_REVIEW_AI_LLM_PROVIDER={name!r}; "
            f"expected one of {sorted(PROVIDERS)}."
        )
    return provider_cls()


def _render_diff_for_llm(files: list[ParsedFile], max_total_chars: int, max_file_chars: int) -> str:
    """Same largest-churn-first, capped assembly as
    `context_builder.build_diff_text`, adapted to a `ParsedFile` list
    instead of a `files_changed` DataFrame.

    The caps are the Groq-era ones from `run_exp4_grid.py` and are deliberately
    kept after the move to Qwen: they are what Experiment 4's prompts were
    measured under, so keeping them keeps deep mode's request identical in
    shape to the grid's. Qwen's context window is far larger, so they are now
    conservative rather than binding -- raising them would be a change to the
    configuration, not a free win, and is untested."""
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
    human-readable `warning`, so a provider outage, an exhausted quota or a
    missing API key degrades the `/review` endpoint to fast-mode-only output
    instead of a 500 -- the LLM pass is additive, not load-bearing.

    The warnings name the provider that actually failed rather than a
    hard-coded one; a message reading "Groq quota exhausted" while the request
    went to Qwen is worse than no message, and this project has already had one
    provider change under it."""
    diff_text = _render_diff_for_llm(files, GROQ_RUN_MAX_DIFF_CHARS, GROQ_RUN_MAX_FILE_CHARS)
    repo_context = build_lightweight_repo_context([
        {"filename": f.filename, "status": f.status, "patch": f.patch} for f in files
    ])
    context = AugmentedContext(diff=diff_text, repo_context=repo_context, repo=repo, title=title)
    conversation = build_exp4(TASK, STRATEGY, context, TIER)

    provider = None
    try:
        provider = _provider()
        response = provider.generate_conversation(
            conversation, max_output_tokens=GROQ_RUN_MAX_OUTPUT_TOKENS
        )
    except LLMQuotaExceededError as exc:
        return DeepReviewResult(None, None, f"{_provider_label(provider)} quota exhausted: {exc}")
    except LLMAPIError as exc:
        return DeepReviewResult(None, None, f"LLM call failed: {exc}")
    except Exception as exc:  # missing API key, bad provider name, network, ...
        return DeepReviewResult(None, None, f"LLM unavailable: {exc}")

    parsed = parse_review_comments(response.final_text)
    label = f"{provider.provider_name.lower()}/{provider.model} {STRATEGY.value} @ {TIER.value}"
    return DeepReviewResult(parsed.comments, label, None)


def _provider_label(provider: CachedChatProvider | None) -> str:
    """Name the provider in a warning even when construction itself failed
    (`provider` is still None at that point)."""
    return provider.provider_name if provider is not None else "LLM provider"
