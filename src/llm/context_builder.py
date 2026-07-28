"""Code-context construction for Experiment 3 (lab guide §3.7.2 "Step 2:
Constructing Code Context"): renders one of four context-richness tiers for a
PR as the `context_text` a `PromptContext` (`src/llm/prompts/schema.py`)
carries into `templates.build_prompt`.

The four tiers, exactly per the guide and the plan's §5.2 paraphrase (verified
against the real PDF, not just the paraphrase):

    DIFF_ONLY            diff alone
    DIFF_DESCRIPTION     diff + the PR's own description (title/body)
    DIFF_COMMIT_MESSAGE  diff + its commit message(s)
    DIFF_METADATA        diff + "other information": labels, file list, size

(The guide's broader §3.4.2 palette -- full file content, historical snippets,
historical review comments -- is *not* built here: §3.7.2's actual required
steps name only these four, and the richer tiers are Experiment 4's territory
per guide §4.4.2 "repository-level code context" / "historical code review
comments".)

This module is deliberately the *only* thing in the Experiment 3 pipeline that
decides what goes into the context string; `src/llm/prompts/templates.py`
(Step 14) only ever sees the rendered `PromptContext.context_text` and never
branches on which tier produced it -- that separation is what lets the same
four prompt strategies serve all four contexts as one nested loop (Step 16),
not sixteen hand-copied scripts.

No leakage of review-process signals
-------------------------------------
None of the four tiers ever touches `reviews.parquet`, `review_comments.parquet`,
or `issue_comments.parquet`. This mirrors Experiment 2's own hard constraint
(`reports/exp2_feature_spec.md` §2/§4.5, lab guide §2.4.3): "num_reviewers",
"has_approved", etc. are only known *after* the review process concludes, and
handing them to an LLM asked to *predict* the merge outcome would be a much more
blatant version of the same leakage the ML feature spec quarantines. The
guide's own §3.4.2 lists "historical review comments" as a *context type* the
course explores -- but only for *other* PRs (as in Experiment 3's few-shot
demonstrations, `src/llm/prompts/few_shot.py`, or Experiment 4's context
augmentation), never this PR's own in-flight review, which is exactly what
would leak the answer.

Truncation caps -- justified against the real dataset, not guessed
--------------------------------------------------------------------
Diff sizes in this dataset are extremely heavy-tailed (population: human-
written, closed PRs, n=1,114): total diff length per PR is a median of 5,788
chars but a max of ~21.5 million (a handful of PRs touch huge generated/vendor
files). Truncation is therefore mandatory for cost/latency/fairness, not
optional -- and Gemini Flash's large context window means the cap is a design
choice about payload size and evenness across PRs, not a hard token-limit
necessity. Measured coverage at each candidate cap (exact figures, not
estimates):

    total-diff cap    8,000 chars -> 58.6% of PRs fit untruncated
                      12,000 chars -> 66.7%
                      16,000 chars -> 71.5%  <- MAX_DIFF_CHARS (chosen)
                      20,000 chars -> 74.9%
                      24,000 chars -> 78.4%

    per-file cap       2,000 chars -> 50.9% of individual files fit untruncated
                       3,000 chars -> 60.3%
                       4,000 chars -> 66.6%  <- MAX_FILE_PATCH_CHARS (chosen)
                       5,000 chars -> 71.3%

16,000/4,000 were picked as the point where coverage growth starts flattening
noticeably (each further +4,000 chars on the total cap buys only ~3-4 more
points of coverage) -- a reasonable place to stop paying for diminishing
returns. Both are constructor-overridable, and every truncation leaves an
explicit `"... [... omitted for length]"` marker rather than silently
presenting a partial diff as complete.

Files are assembled **largest-churn-first** (`additions + deletions`
descending) before the caps are applied, so a truncated context still shows
the most substantive changes rather than whatever happened to be first in
`files_changed`'s stored order.

Commit messages: all of them, not just the first
--------------------------------------------------
`src/features/feature_extraction.py` uses only the *first* commit message as
one of several ML scalar features (`t_commit_msg_len_first`) because a
fixed-length feature vector needs a single number. An LLM context has no such
constraint -- a reviewer reading "diff + commit message" would see every
commit's message, so `build_commit_messages_text` concatenates all of them
(numbered, capped). Real data supports this being affordable: median total
commit-message length per PR is 330 chars, 90th percentile 1,969 -- comfortably
under the 4,000-char cap used here, with the same explicit-truncation-marker
discipline for the tail (up to 109,573 chars observed for one outlier PR with
100 commits).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol

import pandas as pd

from src.llm.prompts.schema import PromptContext

DATA_DIR = Path("data/processed")

MAX_DIFF_CHARS = 16_000
MAX_FILE_PATCH_CHARS = 4_000
MAX_COMMIT_MSG_CHARS = 4_000
MAX_FILE_LIST_ENTRIES = 30

# Optional hard ceiling on the *assembled* context string, applied after all
# sections are joined. `None` = no ceiling (the default, so nothing changes for
# callers that don't set it). The Experiment 3 grid sets this to keep every
# prompt -- including a few-shot prompt that stacks K example contexts -- under
# a provider's per-request token limit (Groq free tier's 6,000 TPM, which is
# also a hard per-request cap: a single request over 6,000 tokens gets a 413
# that no amount of waiting fixes, because it exceeds the entire per-minute
# budget). Capping the *diff* alone is insufficient because the description /
# commit-message / metadata sections stack on top of it (PR bodies alone reach
# ~4,000 tokens at the 99th percentile).
MAX_CONTEXT_CHARS = None


class ContextKind(str, Enum):
    DIFF_ONLY = "diff_only"
    DIFF_DESCRIPTION = "diff_description"
    DIFF_COMMIT_MESSAGE = "diff_commit_message"
    DIFF_METADATA = "diff_metadata"


class HasCoreTables(Protocol):
    """Structural type for what `build_context` needs from a sources bundle:
    anything exposing these three attributes works -- including
    `src/llm/prompts/few_shot.py`'s `ExampleSources` (a superset, with an extra
    `review_comments` field this module never reads), so the same loaded
    sources can serve both few-shot example rendering and query-context
    building without an adapter."""

    pull_requests: pd.DataFrame
    files_changed: pd.DataFrame
    commits: pd.DataFrame


@dataclass
class ContextSources:
    """The three tables `build_context` reads. See `HasCoreTables` -- callers
    that already have a `few_shot.ExampleSources` do not need to construct
    this; it exists for callers (Step 16's main grid) that only need these
    three tables and would rather not load `review_comments.parquet` too."""

    pull_requests: pd.DataFrame
    files_changed: pd.DataFrame
    commits: pd.DataFrame


def load_sources(data_dir: Path = DATA_DIR) -> ContextSources:
    return ContextSources(
        pull_requests=pd.read_parquet(data_dir / "pull_requests.parquet"),
        files_changed=pd.read_parquet(data_dir / "files_changed.parquet"),
        commits=pd.read_parquet(data_dir / "commits.parquet"),
    )


def _pr_row(pr_id: str, sources: HasCoreTables) -> pd.Series:
    rows = sources.pull_requests.loc[sources.pull_requests["id"] == pr_id]
    if rows.empty:
        raise KeyError(f"pr_id {pr_id!r} not found in pull_requests")
    return rows.iloc[0]


# --------------------------------------------------------------------------- #
# Diff (shared by all four context kinds)
# --------------------------------------------------------------------------- #

def build_diff_text(
    pr_id: str,
    sources: HasCoreTables,
    *,
    max_total_chars: int = MAX_DIFF_CHARS,
    max_file_chars: int = MAX_FILE_PATCH_CHARS,
) -> str:
    """Concatenate this PR's file diffs, largest-churn-first, each capped at
    `max_file_chars` and the whole assembly capped at `max_total_chars`.
    Greedy bin-packing: a file that doesn't fit is skipped (not a hard stop),
    so a merely-large file mid-list doesn't preemptively block smaller files
    after it from being shown. A file with no textual patch (binary, or a pure
    rename) still gets a header line so its existence isn't silently dropped.
    """
    files = sources.files_changed.loc[sources.files_changed["pr_id"] == pr_id].copy()
    if files.empty:
        return "(no file diff available)"

    files["_churn"] = files["additions"].fillna(0) + files["deletions"].fillna(0)
    files = files.sort_values("_churn", ascending=False)

    parts: list[str] = []
    total = 0
    n_omitted = 0
    for _, row in files.iterrows():
        patch = row.get("patch")
        patch = patch if isinstance(patch, str) else ""
        if patch:
            truncated = len(patch) > max_file_chars
            if truncated:
                patch = patch[:max_file_chars] + "\n... [file diff truncated]"
            block = f"--- {row['filename']} ({row['status']})\n{patch}"
        else:
            block = f"--- {row['filename']} ({row['status']}, no textual diff)"

        if total + len(block) > max_total_chars:
            n_omitted += 1
            continue
        parts.append(block)
        total += len(block)

    text = "\n\n".join(parts)
    if n_omitted:
        text += f"\n\n... [{n_omitted} additional changed file(s) omitted for length]"
    return text


# --------------------------------------------------------------------------- #
# Per-context-kind sections
# --------------------------------------------------------------------------- #

def build_commit_messages_text(
    pr_id: str, sources: HasCoreTables, *, max_chars: int = MAX_COMMIT_MSG_CHARS
) -> str:
    """All (non-empty) commit messages for this PR, numbered, capped -- see
    module docstring for why this uses every commit rather than just the
    first (as `feature_extraction.py`'s scalar ML feature does)."""
    commits = sources.commits.loc[sources.commits["pr_id"] == pr_id]
    if commits.empty:
        return "(no commit messages available)"

    lines: list[str] = []
    total = 0
    n_omitted = 0
    for i, (_, row) in enumerate(commits.iterrows(), start=1):
        msg = str(row.get("message") or "").strip()
        if not msg:
            continue
        line = f"{i}. {msg}"
        if total + len(line) > max_chars:
            n_omitted += 1
            continue
        lines.append(line)
        total += len(line)

    if not lines:
        return "(no non-empty commit messages)"
    text = "\n".join(lines)
    if n_omitted:
        text += f"\n... [{n_omitted} additional commit message(s) omitted for length]"
    return text


def build_metadata_text(
    pr_id: str, sources: HasCoreTables, *, max_files_listed: int = MAX_FILE_LIST_ENTRIES
) -> str:
    """Labels + file list + size counters -- "other information" per guide
    §3.7.2. Deliberately excludes anything from the review process itself
    (see module docstring's leakage note)."""
    pr = _pr_row(pr_id, sources)
    labels_raw = pr.get("labels")
    # `labels` is stored as a numpy array (parquet list column); `x or []`
    # would raise ValueError on a multi-element array's ambiguous truthiness,
    # so presence is checked explicitly rather than via truthiness.
    labels = list(labels_raw) if labels_raw is not None else []
    files = sources.files_changed.loc[sources.files_changed["pr_id"] == pr_id]

    additions = int(pr.get("additions") or 0)
    deletions = int(pr.get("deletions") or 0)

    lines = [
        f"Labels: {', '.join(labels) if len(labels) else '(none)'}",
        f"Size: {len(files)} file(s) changed, +{additions}/-{deletions} lines",
        "Changed files:",
    ]
    shown = files[["filename", "status"]].head(max_files_listed)
    lines.extend(f"  - {r.filename} ({r.status})" for r in shown.itertuples(index=False))
    if len(files) > max_files_listed:
        lines.append(f"  ... and {len(files) - max_files_listed} more file(s)")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #

def build_context(
    pr_id: str,
    kind: ContextKind,
    sources: HasCoreTables,
    *,
    max_diff_chars: int = MAX_DIFF_CHARS,
    max_file_chars: int = MAX_FILE_PATCH_CHARS,
    max_context_chars: int | None = MAX_CONTEXT_CHARS,
) -> PromptContext:
    """Render the full `PromptContext` for one (PR, context kind) pair -- the
    object `src/llm/prompts/templates.py:build_prompt` consumes.

    Population scope is the caller's responsibility (as with
    `feature_extraction.py`'s `select_population`): this function will happily
    render a context for any `pr_id` present in `pull_requests.parquet`. Step
    16 is expected to call it only for the Experiment-2-aligned test PRs
    (merge prediction) or, for review-comment generation, any PR whose
    generated comments are worth comparing to ground truth.

    `max_context_chars` (default `None`) applies a hard ceiling to the *whole*
    assembled context after the sections are joined -- see the module constant.
    """
    pr = _pr_row(pr_id, sources)
    diff_text = build_diff_text(
        pr_id, sources, max_total_chars=max_diff_chars, max_file_chars=max_file_chars
    )

    if kind is ContextKind.DIFF_ONLY:
        text = f"Diff:\n{diff_text}"
    elif kind is ContextKind.DIFF_DESCRIPTION:
        body = str(pr.get("body") or "").strip()
        description = body if body else "(no PR description provided)"
        text = f"PR description:\n{description}\n\nDiff:\n{diff_text}"
    elif kind is ContextKind.DIFF_COMMIT_MESSAGE:
        commit_text = build_commit_messages_text(pr_id, sources)
        text = f"Commit messages:\n{commit_text}\n\nDiff:\n{diff_text}"
    elif kind is ContextKind.DIFF_METADATA:
        metadata_text = build_metadata_text(pr_id, sources)
        text = f"{metadata_text}\n\nDiff:\n{diff_text}"
    else:
        raise ValueError(f"Unknown ContextKind: {kind!r}")

    text = _cap_context(text, max_context_chars)

    repo = pr.get("repo")
    title = pr.get("title")
    return PromptContext(
        context_text=text,
        repo=str(repo) if repo else None,
        title=str(title) if title else None,
        context_kind=kind.value,
    )


def _cap_context(text: str, max_context_chars: int | None) -> str:
    """Apply the optional whole-context ceiling, leaving an explicit marker so
    a truncated context is never mistaken for a complete one."""
    if max_context_chars is not None and len(text) > max_context_chars:
        return text[:max_context_chars] + "\n... [context truncated to fit the LLM token budget]"
    return text


def render_context_text(
    pr_id: str,
    sources: HasCoreTables,
    *,
    kind: ContextKind = ContextKind.DIFF_ONLY,
    max_diff_chars: int = MAX_DIFF_CHARS,
    max_file_chars: int = MAX_FILE_PATCH_CHARS,
    max_context_chars: int | None = MAX_CONTEXT_CHARS,
) -> str:
    """Plain-text variant matching the `render_context` callable contract
    `src/llm/prompts/few_shot.py`'s `select_merge_examples`/
    `select_comment_examples` expect (`Callable[[str, sources], str]`, called
    with exactly two positional arguments). Defaults to `DIFF_ONLY` -- the tier
    `few_shot.py`'s own placeholder `default_render_context` approximated
    before this module existed. To wire in a different tier for a specific
    cell of Step 16's grid, bind `kind` first:

        from functools import partial
        render = partial(render_context_text, kind=ContextKind.DIFF_METADATA)
        select_merge_examples(train_pool, sources, render_context=render)
    """
    return build_context(
        pr_id, kind, sources, max_diff_chars=max_diff_chars,
        max_file_chars=max_file_chars, max_context_chars=max_context_chars,
    ).context_text
