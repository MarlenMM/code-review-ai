"""Experiment 4 context augmentation: the five software-engineering context
tiers of lab guide §4.7.2, rendered on the *prompt side*.

Experiment 3 built four context tiers in `src/llm/context_builder.py`
(diff-only / +description / +commit / +metadata). Experiment 4's tiers are
different -- richer, and aimed at what AI-generated code needs to be reviewed
well (§4.3: "relying solely on local modifications within a pull request is
frequently insufficient"). Per §4.7.2 they are, in order:

    DIFF                  the diff alone
    DIFF_PR_DESCRIPTION   diff + the PR's own description
    DIFF_REPO_CONTEXT     diff + repository-level context (the file(s) around
                          the change, related/called functions)
    DIFF_ISSUE            diff + the linked issue's text
    COMPLETE_SE_CONTEXT   everything: description + commit messages + issue +
                          repository context + historical review comments on
                          similar past PRs

Why the component fields live here, not one opaque string
---------------------------------------------------------
Experiment 3's `PromptContext` carries a single already-rendered `context_text`
because its templates are blind to how rich the context is. Experiment 4 keeps
the *components* separate in `AugmentedContext` for one reason the multi-turn
strategy forces: multi-turn reveals the diff first and the broader SE context
second, so the prompt layer must be able to lay the pieces out in two different
ways (all-at-once for the single-shot strategies; split for multi-turn). A
pre-flattened string could not be split back apart. `render_tier` and
`split_for_multi_turn` therefore both derive from one `_augmentation_sections`
helper, so a tier's single-shot rendering and its multi-turn (turn 1 + turn 2)
rendering always contain the *same* section texts -- only the ordering differs.

Division of labour with Step 20 (the run)
------------------------------------------
This module renders whatever `AugmentedContext` it is *given*; it does not fetch
anything. Populating the slots is Step 20's data-engineering job, and the slots
differ sharply in how available they are from the mined dataset (verified
against `data/processed/`, not assumed -- see
`reports/exp4_prompt_and_context_design.md` for the exact figures):

* ``diff`` / ``pr_description`` -- fully available today (same sources as Exp 3).
* ``commit_message`` -- fully available (`commits.parquet`).
* ``repo_context`` -- NOT stored (only diff hunks are). `build_lightweight_repo_context`
  assembles a no-fetch approximation from what *is* available (co-changed files,
  git hunk section headings -> enclosing symbols); a full version (fetched file
  bodies, a call graph) is a Step-20 GitHub fetch the slot is designed to accept.
* ``issue_text`` -- the linked issue's *reference* is extractable now
  (`extract_issue_references`; ~29% of AI PRs cite an issue), but the issue's
  own body must be fetched in Step 20.
* ``historical_comments`` -- derivable now from `review_comments.parquet` on
  *similar past train-split PRs* (`exp4_examples.select_historical_comments`).

Every slot that a tier calls for but that is empty renders an explicit
"(... not available)" marker rather than silently vanishing, so Step 20's result
analysis can tell "the tier had no data for this PR" apart from "the tier was
not run" -- mirroring `context_builder.py`'s "(no PR description provided)"
discipline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from src.llm.prompts.schema import PromptContext


class Exp4ContextTier(str, Enum):
    """The five context granularities of lab guide §4.7.2 (Experiment 4)."""

    DIFF = "diff"
    DIFF_PR_DESCRIPTION = "diff_pr_description"
    DIFF_REPO_CONTEXT = "diff_repo_context"
    DIFF_ISSUE = "diff_issue"
    COMPLETE_SE_CONTEXT = "complete_se_context"


@dataclass(frozen=True)
class AugmentedContext:
    """The software-engineering context for one AI-generated PR, kept as
    separate components so the prompt layer can compose them per tier (and
    split them for multi-turn). `diff` is required; every augmentation is
    optional and Step-20-populated. `repo`/`title` are light header fields
    (as in `PromptContext`) so a role persona can name the project.

    The `diff` is expected to arrive already length-capped by the caller
    (Step 20 reuses `context_builder.build_diff_text`'s caps); this module does
    no truncation of its own.
    """

    diff: str
    pr_description: str | None = None
    commit_message: str | None = None
    repo_context: str | None = None
    issue_text: str | None = None
    historical_comments: str | None = None
    repo: str | None = None
    title: str | None = None


# --------------------------------------------------------------------------- #
# Section rendering
# --------------------------------------------------------------------------- #

# Which augmentation sections (beyond the always-present diff) each tier adds,
# and in what order. The value is a list of (heading, attribute, empty_note).
# COMPLETE_SE_CONTEXT stacks every augmentation the experiment defines (§4.4.2).
_TIER_SECTIONS: dict[Exp4ContextTier, list[tuple[str, str, str]]] = {
    Exp4ContextTier.DIFF: [],
    Exp4ContextTier.DIFF_PR_DESCRIPTION: [
        ("PR description", "pr_description", "no PR description provided"),
    ],
    Exp4ContextTier.DIFF_REPO_CONTEXT: [
        ("Repository context", "repo_context", "no repository context available"),
    ],
    Exp4ContextTier.DIFF_ISSUE: [
        ("Linked issue", "issue_text", "no linked issue information available"),
    ],
    Exp4ContextTier.COMPLETE_SE_CONTEXT: [
        ("PR description", "pr_description", "no PR description provided"),
        ("Commit messages", "commit_message", "no commit messages available"),
        ("Linked issue", "issue_text", "no linked issue information available"),
        ("Repository context", "repo_context", "no repository context available"),
        (
            "Historical review comments on similar past PRs",
            "historical_comments",
            "no historical review comments available",
        ),
    ],
}


def _section(heading: str, text: str | None, empty_note: str) -> str:
    body = (text or "").strip()
    if not body:
        body = f"({empty_note})"
    return f"{heading}:\n{body}"


def _augmentation_sections(context: AugmentedContext, tier: Exp4ContextTier) -> list[str]:
    """The rendered non-diff sections a tier calls for (may be empty for DIFF).
    Shared by `render_tier` (single-shot) and `split_for_multi_turn` so the two
    presentations always carry the same section texts."""
    if tier not in _TIER_SECTIONS:
        raise ValueError(f"Unknown Exp4ContextTier: {tier!r}")
    return [
        _section(heading, getattr(context, attr), empty_note)
        for heading, attr, empty_note in _TIER_SECTIONS[tier]
    ]


def _diff_section(context: AugmentedContext) -> str:
    diff = (context.diff or "").strip()
    return f"Diff:\n{diff}" if diff else "Diff:\n(no file diff available)"


def render_tier(context: AugmentedContext, tier: Exp4ContextTier) -> PromptContext:
    """Render one context tier as the single-block `PromptContext` the
    single-shot strategies (role-based / few-shot / CoT / and self-reflection's
    turn 1) consume via `templates.build_prompt`.

    Augmentation sections come first and the diff last -- the same ordering
    Experiment 3's `context_builder` used ("PR description ... Diff ...") so a
    reader hits the framing before the raw change. `context_kind` is set to the
    tier value for logging / cache columns.
    """
    sections = _augmentation_sections(context, tier)
    sections.append(_diff_section(context))
    text = "\n\n".join(sections)
    return PromptContext(
        context_text=text,
        repo=context.repo,
        title=context.title,
        context_kind=tier.value,
    )


def split_for_multi_turn(
    context: AugmentedContext, tier: Exp4ContextTier
) -> tuple[str, str]:
    """Split a tier into (turn-1 text, turn-2 text) for the multi-turn strategy.

    Turn 1 is the diff alone -- the "local PR view" every tier shares. Turn 2 is
    everything the tier adds on top of the diff (the broader SE context),
    revealed only after the model has committed a diff-only judgment. The union
    of the two turns' section texts equals `render_tier(context, tier)`'s
    sections, so multi-turn and single-shot see the same information, just
    disclosed in a different order.

    Raises `ValueError` for `DIFF`: with no augmentation to reveal there is no
    second turn, so multi-turn is undefined on the bare-diff tier and the
    Step-20 grid skips that (tier, MULTI_TURN) cell rather than running a
    degenerate one. See `MULTI_TURN_TIERS`.
    """
    augmentation = _augmentation_sections(context, tier)
    if not augmentation:
        raise ValueError(
            f"multi-turn is undefined for tier {tier.value!r}: it has no "
            "augmentation beyond the diff to reveal in a second turn "
            f"(applicable tiers: {[t.value for t in MULTI_TURN_TIERS]})."
        )
    return _diff_section(context), "\n\n".join(augmentation)


# The tiers multi-turn is meaningful on: every tier except bare DIFF (which has
# nothing to reveal in a second turn). Step 20 iterates this for MULTI_TURN.
MULTI_TURN_TIERS: tuple[Exp4ContextTier, ...] = tuple(
    t for t in Exp4ContextTier if _TIER_SECTIONS[t]
)


# --------------------------------------------------------------------------- #
# Issue-reference extraction (feeds the DIFF_ISSUE / COMPLETE tiers in Step 20)
# --------------------------------------------------------------------------- #

# "Fixes #123", "Closes #123", "resolves gh-123", and full issue URLs. The
# closing-keyword variant is captured separately from a bare "#123" mention
# because a closing keyword is a far stronger signal that the number names the
# issue this PR *addresses* (worth fetching) rather than an incidental cross-ref.
_CLOSING_REF_RE = re.compile(
    r"\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?|implement[sd]?|address(?:e[sd])?)\b"
    r"\s*:?\s+(?:#|gh-)(\d+)",
    re.IGNORECASE,
)
_ISSUE_URL_RE = re.compile(
    r"https?://github\.com/[\w.-]+/[\w.-]+/issues/(\d+)", re.IGNORECASE
)
_BARE_REF_RE = re.compile(r"(?<![\w/])#(\d+)\b")


@dataclass(frozen=True)
class IssueReferences:
    """Issue numbers referenced by a PR, tiered by how confidently the number
    names the issue the PR *addresses* (and is therefore worth fetching for the
    DIFF_ISSUE tier). `closing` = named by a closing keyword or a full issues/N
    URL; `mentioned` = a bare "#N" with no closing keyword. Both are deduped and
    order-preserving; `mentioned` excludes anything already in `closing`."""

    closing: tuple[int, ...]
    mentioned: tuple[int, ...]

    @property
    def best(self) -> tuple[int, ...]:
        """Closing refs if any, else the bare mentions -- the numbers Step 20
        should try to fetch, most-confident first."""
        return self.closing or self.mentioned


def _dedupe_ints(values) -> tuple[int, ...]:
    seen: dict[int, None] = {}
    for v in values:
        seen.setdefault(int(v), None)
    return tuple(seen)


def extract_issue_references(*texts: str | None) -> IssueReferences:
    """Extract issue references from a PR's free text (body, and optionally its
    commit messages -- pass them as extra positional args). Used by Step 20 to
    decide which issue(s) to fetch for the DIFF_ISSUE / COMPLETE tiers.

    This does string extraction only -- it never fetches. It is grounded in the
    real data: of 361 copilot-swe-agent PRs, 105 carry a bare `#N`, 42 a full
    `issues/N` URL, and 35 a "Fixes #N"; the rest carry no reference at all, so
    the DIFF_ISSUE tier will legitimately render "(no linked issue ...)" for
    the majority -- a real coverage limit reported honestly, not hidden.
    """
    blob = "\n".join(t for t in texts if t)
    closing = _dedupe_ints(
        [m.group(1) for m in _CLOSING_REF_RE.finditer(blob)]
        + [m.group(1) for m in _ISSUE_URL_RE.finditer(blob)]
    )
    closing_set = set(closing)
    mentioned = tuple(
        n for n in _dedupe_ints(m.group(1) for m in _BARE_REF_RE.finditer(blob))
        if n not in closing_set
    )
    return IssueReferences(closing=closing, mentioned=mentioned)


# --------------------------------------------------------------------------- #
# Lightweight (no-fetch) repository context
# --------------------------------------------------------------------------- #

# A git unified-diff hunk header: "@@ -a,b +c,d @@ <section heading>". For code
# files the trailing heading is git's best guess at the enclosing function/class
# signature -- a free, already-stored "related code" signal (present on ~99% of
# AI-PR changed-file rows in this dataset). For non-code files it can be
# arbitrary line content, so the rendering labels it "enclosing region (git
# heading)" and does not overclaim it as a call graph.
_HUNK_HEADING_RE = re.compile(
    r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@\s*(.+)$", re.MULTILINE
)


def _enclosing_headings(patch: str, limit: int = 5) -> list[str]:
    seen: dict[str, None] = {}
    for m in _HUNK_HEADING_RE.finditer(patch or ""):
        h = m.group(1).strip()
        if h:
            seen.setdefault(h, None)
        if len(seen) >= limit:
            break
    return list(seen)


def build_lightweight_repo_context(
    files: list[dict], *, max_files: int = 20
) -> str:
    """Assemble a *no-fetch* approximation of repository context from data the
    mining step already stored, for the DIFF_REPO_CONTEXT / COMPLETE tiers when
    Step 20 chooses not to fetch full file bodies.

    `files` is a list of ``{"filename", "status", "patch"}`` dicts (Step 20
    passes the PR's `files_changed` rows). Per file it lists the path, change
    status, and the enclosing regions git names in the hunk headers (usually the
    modified function/class for code). It also groups the changed paths by
    directory, so the model sees which parts of the repository the change spans
    -- a cheap proxy for "the files/relationships around the change" (§4.4.2)
    that needs no extra API calls.

    This is deliberately labelled as an approximation in its own output; the
    full form (fetched surrounding source + an actual call graph) is the
    `AugmentedContext.repo_context` slot Step 20 can populate instead.
    """
    if not files:
        return ""
    files = files[:max_files]

    dirs: dict[str, list[str]] = {}
    lines: list[str] = ["Files touched by this change (repository-level view):"]
    for f in files:
        name = str(f.get("filename", "") or "")
        status = str(f.get("status", "") or "")
        headings = _enclosing_headings(str(f.get("patch", "") or ""))
        lines.append(f"- {name} ({status})")
        if headings:
            lines.append(
                "    enclosing region(s) (git hunk headings): "
                + "; ".join(headings)
            )
        directory = name.rsplit("/", 1)[0] if "/" in name else "(repo root)"
        dirs.setdefault(directory, []).append(name.rsplit("/", 1)[-1])

    if len(dirs) > 1:
        lines.append("")
        lines.append("Directories spanned:")
        for directory, members in dirs.items():
            lines.append(f"- {directory}/  ({len(members)} file(s))")

    lines.append("")
    lines.append(
        "(Approximate: derived from the diff's own file list and git hunk "
        "headings, not from fetched file bodies or a call graph.)"
    )
    return "\n".join(lines)
