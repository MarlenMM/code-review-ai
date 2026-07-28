"""Experiment 4 context *population* (lab guide §4.7.2 "Step 2: Context
Construction"): assembles an `AugmentedContext`
(`src/llm/prompts/context_aug.py`) for one AI-authored PR from
`data/processed/*.parquet`, plus (optionally) a live GitHub issue-body fetch.

This is the Step-20 counterpart to Step 15's `src/llm/context_builder.py`, and
keeps the exact same division of labour that module established for
Experiment 3: the prompt library (`src/llm/prompts/context_aug.py`, Step 19)
renders whatever `AugmentedContext` it is *given* and fetches nothing; this
module is what *builds* one, from data. `src/llm/prompts/context_aug.py`'s own
module docstring calls this division out by name for Experiment 4.

What gets fetched vs. derived, and why (see
`reports/exp4_prompt_and_context_design.md` §2.3 for the measured coverage
these choices produce on the real AI test split):

* ``diff`` / ``pr_description`` / ``commit_message`` -- assembled from already-
  mined parquet data, reusing `context_builder`'s own diff/commit builders so
  Experiment 4's truncation discipline (explicit `"[... omitted]"` markers,
  largest-churn-first file ordering) is identical to Experiment 3's.
* ``repo_context`` -- the *lightweight, no-fetch* approximation
  (`context_aug.build_lightweight_repo_context`): co-changed files + git hunk
  section headings. No file bodies are fetched here; that would be a much
  larger, rate-limit-sensitive engineering lift for a course project already
  tight on both quota and time, and the design doc documents this as a
  legitimate, reported scope choice rather than a hidden shortcut.
* ``issue_text`` -- IS fetched, but narrowly: `extract_issue_references` finds
  candidate issue numbers from the PR body + commit messages, and only the
  most-confident one (a closing keyword or full URL) is fetched via
  `GitHubClient.fetch_issue`, capped and skipped entirely for PRs referencing
  a *pull request* number rather than a real issue (`is_pull_request`). This
  keeps the extra API load small (real measurement: ~19% of the AI test split
  carries a closing-keyword reference) while still closing the gap the design
  doc flagged as Step 20's job to close.
* ``historical_comments`` -- derived via `exp4_examples.select_historical_comments`
  over the AI-authored train split; no fetch needed.

`github_client=None` (the default) skips issue fetching entirely -- every tier
still renders, just with `issue_text=None` (the tier's own "(not available)"
marker). This lets `build_augmented_context` run fully offline (as every test
in this module does) and lets the grid runner opt in explicitly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from src.llm import context_builder
from src.llm.prompts.context_aug import AugmentedContext, build_lightweight_repo_context, extract_issue_references
from src.llm.prompts.exp4_examples import select_historical_comments
from src.llm.prompts.few_shot import ExampleSources

DATA_DIR = Path("data/processed")

# Per-component character caps for Experiment 4's context. Kept smaller than
# Experiment 3's already-tuned Groq-fit constants
# (`context_builder.MAX_DIFF_CHARS` etc.) because the COMPLETE_SE_CONTEXT tier
# stacks FIVE sections on top of the diff (vs. Experiment 3's one), and
# self-reflection/multi-turn's second API call resends the first turn's full
# transcript -- so per-section headroom has to be tighter for the assembled
# context to stay comfortably under Groq's 6,000-token hard per-request cap
# even on the richest tier's second turn. See `reports/exp4_grid_run_status.md`
# for the real per-cell token usage these constants were calibrated against.
MAX_DIFF_CHARS = 1_200
MAX_FILE_PATCH_CHARS = 500
MAX_COMMIT_MSG_CHARS = 500
MAX_SECTION_CHARS = 600
MAX_REPO_CONTEXT_FILES = 8
HISTORY_K = 2
MAX_HISTORY_COMMENT_CHARS = 350


def load_sources(data_dir: Path = DATA_DIR) -> ExampleSources:
    """Same four tables `few_shot.ExampleSources` already bundles -- reused
    directly (not re-declared) since `select_historical_comments` and
    `context_builder`'s diff/commit builders (via `HasCoreTables`) both accept
    this shape already."""
    return ExampleSources(
        pull_requests=pd.read_parquet(data_dir / "pull_requests.parquet"),
        files_changed=pd.read_parquet(data_dir / "files_changed.parquet"),
        commits=pd.read_parquet(data_dir / "commits.parquet"),
        review_comments=pd.read_parquet(data_dir / "review_comments.parquet"),
    )


def _truncate(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) > max_chars:
        return text[:max_chars].rstrip() + " ... [truncated for length]"
    return text


def _pr_row(pr_id: str, sources: ExampleSources) -> pd.Series:
    rows = sources.pull_requests.loc[sources.pull_requests["id"] == pr_id]
    if rows.empty:
        raise KeyError(f"pr_id {pr_id!r} not found in pull_requests")
    return rows.iloc[0]


def _files_for_repo_context(pr_id: str, sources: ExampleSources) -> list[dict]:
    files = sources.files_changed.loc[sources.files_changed["pr_id"] == pr_id]
    files = files.copy()
    files["_churn"] = files["additions"].fillna(0) + files["deletions"].fillna(0)
    files = files.sort_values("_churn", ascending=False)
    return [
        {
            "filename": row.filename,
            "status": row.status,
            "patch": row.patch if isinstance(row.patch, str) else "",
        }
        for row in files.itertuples(index=False)
    ]


def _resolve_issue_text(
    pr_id: str,
    pr: pd.Series,
    sources: ExampleSources,
    github_client,
    max_section_chars: int,
) -> Optional[str]:
    """Fetch the single most-confident referenced issue's text, or `None` if
    there is no reference, no client, or the reference resolves to a PR / a
    missing issue. Never raises on a fetch problem an individual PR might
    have -- an unresolvable reference is exactly the "(no linked issue
    information available)" case `context_aug.render_tier` already handles."""
    if github_client is None:
        return None

    commits = sources.commits.loc[sources.commits["pr_id"] == pr_id, "message"]
    commit_blob = "\n".join(str(m) for m in commits if isinstance(m, str))
    refs = extract_issue_references(str(pr.get("body") or ""), commit_blob)
    if not refs.best:
        return None

    owner = str(pr.get("owner") or "")
    repo_name = str(pr.get("repo_name") or "")
    if not owner or not repo_name:
        return None

    for number in refs.best:
        try:
            record = github_client.fetch_issue(owner, repo_name, number)
        except Exception:
            # A fetch failure for one referenced number (network blip, rate
            # limit, an unexpected 4xx) must not abort context building for
            # the whole PR -- try the next candidate, or fall through to None.
            continue
        if record.get("not_found") or record.get("is_pull_request"):
            continue
        title = str(record.get("title") or "").strip()
        body = str(record.get("body") or "").strip()
        text = f"Issue #{number}: {title}\n\n{body}".strip()
        return _truncate(text, max_section_chars)
    return None


def build_augmented_context(
    pr_id: str,
    sources: ExampleSources,
    train_pool: pd.DataFrame,
    *,
    max_diff_chars: int = MAX_DIFF_CHARS,
    max_file_chars: int = MAX_FILE_PATCH_CHARS,
    max_section_chars: int = MAX_SECTION_CHARS,
    history_k: int = HISTORY_K,
    max_history_comment_chars: int = MAX_HISTORY_COMMENT_CHARS,
    github_client=None,
) -> AugmentedContext:
    """Assemble the full `AugmentedContext` for one AI-authored PR.

    `train_pool` must be the AI-authored **train** split
    (`exp4_examples.load_ai_train_pool`) -- passed through unchanged to
    `select_historical_comments`, keeping the leakage guarantee that module
    documents. `github_client`, if given, enables the (narrow, capped) issue-
    body fetch described in the module docstring; `None` (the default) skips
    it and leaves `issue_text=None`.
    """
    pr = _pr_row(pr_id, sources)

    diff = context_builder.build_diff_text(
        pr_id, sources, max_total_chars=max_diff_chars, max_file_chars=max_file_chars
    )
    pr_description = _truncate(str(pr.get("body") or ""), max_section_chars) or None
    commit_message = context_builder.build_commit_messages_text(
        pr_id, sources, max_chars=max_section_chars
    ) or None

    repo_context = build_lightweight_repo_context(
        _files_for_repo_context(pr_id, sources), max_files=MAX_REPO_CONTEXT_FILES
    )
    repo_context = _truncate(repo_context, max_section_chars) or None

    issue_text = _resolve_issue_text(pr_id, pr, sources, github_client, max_section_chars)

    historical_comments = select_historical_comments(
        pr_id, train_pool, sources, k=history_k, max_comment_chars=max_history_comment_chars,
    ) or None

    repo = pr.get("repo")
    title = pr.get("title")
    return AugmentedContext(
        diff=diff,
        pr_description=pr_description,
        commit_message=commit_message,
        repo_context=repo_context,
        issue_text=issue_text,
        historical_comments=historical_comments,
        repo=str(repo) if repo else None,
        title=str(title) if title else None,
    )
