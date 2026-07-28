"""Compute the Experiment 2 V1 feature vector for a raw diff that has no
`pr_id` -- the FastAPI `/review` endpoint's fast (ML-only) mode.

Mirrors the arithmetic of `src/features/feature_extraction.py`'s
`build_modification_features` / `build_textual_features` and
`src/features/ast_cfg.py`'s per-PR aggregation, but works directly off
`src/api/diff_parsing.py`'s `ParsedFile` list plus optional title/
description/commit-messages -- no parquet lookup, so it works for any diff,
not just PRs already in the mined dataset.

V1 only, deliberately -- not V0 or V2 (see `reports/api_design.md` for the
full discussion): V0's process/`p_*` features (review count, approvals,
label count) don't exist yet for a diff nobody has reviewed, so computing
them isn't a data-availability gap, it's the exact leakage V0 was built to
demonstrate (`reports/exp2_model_evaluation.md` §3). V2's two author-history
features need a (repo, author) identity looked up against the mined
dataset's own PR history, which an arbitrary incoming diff -- from any
contributor, in any repo -- does not reliably carry.

Two honest defaults where a real PR always has an answer but a bare diff
might not:
* `m_num_commits` defaults to 1 (assume a single squashed commit) when no
  `commit_messages` are supplied -- 0 would misrepresent a diff that, by
  definition, came from at least one commit.
* every other missing input (title, description, commit messages) renders
  as empty text, matching the training pipeline's own `fillna(0)` /
  `(no ... provided)` treatment for a PR that's missing that field.
"""

from __future__ import annotations

from src.api.diff_parsing import ParsedFile
from src.features.ast_cfg import FileStructureResult, analyze_file
from src.features.feature_extraction import V1_FEATURE_COLUMNS


def _count_hunks(patch: str) -> int:
    return sum(1 for line in patch.split("\n") if line.startswith("@@"))


def _analyze_file_safe(filename: str, patch: str) -> FileStructureResult:
    """`analyze_file` is normally called through `ast_cfg.analyze_files`,
    which wraps each row in a try/except so one malformed hunk can't crash a
    batch run (see that module). Calling `analyze_file` directly here loses
    that safety net, so it is reinstated at this call site instead."""
    try:
        return analyze_file(filename, patch)
    except Exception:
        return FileStructureResult(language=None)


def compute_v1_features(
    files: list[ParsedFile],
    *,
    title: str = "",
    description: str = "",
    commit_messages: list[str] | None = None,
) -> dict[str, float]:
    """The V1 (per-spec) feature vector for one diff, keyed exactly like
    `V1_FEATURE_COLUMNS` so the result can be handed straight to the trained
    sklearn pipeline as a single-row DataFrame."""
    additions = sum(f.additions for f in files)
    deletions = sum(f.deletions for f in files)
    churn = additions + deletions
    n_files = len(files)

    status_counts = {status: 0 for status in ("added", "modified", "removed", "renamed")}
    for f in files:
        status_counts[f.status] = status_counts.get(f.status, 0) + 1

    num_hunks = sum(_count_hunks(f.patch) for f in files)

    if commit_messages:
        lens = [len(m) for m in commit_messages]
        commit_msg_len_first = float(lens[0])
        commit_msg_len_mean = sum(lens) / len(lens)
        commit_msg_len_total = float(sum(lens))
        num_commits = len(commit_messages)
    else:
        commit_msg_len_first = commit_msg_len_mean = commit_msg_len_total = 0.0
        num_commits = 1

    title = title or ""
    body = description or ""

    ast_node_count = ast_max_depth = cfg_node_count = 0
    cfg_edge_count = cyclomatic_proxy = num_functions_touched = 0
    has_parseable_code = False
    for f in files:
        fr = _analyze_file_safe(f.filename, f.patch)
        if fr.language is not None:
            has_parseable_code = True
        ast_node_count += fr.ast_node_count
        ast_max_depth = max(ast_max_depth, fr.ast_max_depth)
        cfg_node_count += fr.cfg_node_count
        cfg_edge_count += fr.cfg_edge_count
        cyclomatic_proxy += fr.cyclomatic_proxy
        num_functions_touched += fr.num_functions

    features = {
        "m_additions": additions,
        "m_deletions": deletions,
        "m_churn": churn,
        "m_net_change": additions - deletions,
        "m_changed_files": n_files,
        "m_avg_file_churn": churn / max(n_files, 1),
        "m_frac_added": additions / (churn + 1),
        "m_num_hunks": num_hunks,
        "m_files_added": status_counts["added"],
        "m_files_modified": status_counts["modified"],
        "m_files_removed": status_counts["removed"],
        "m_files_renamed": status_counts["renamed"],
        "m_num_commits": num_commits,
        "s_ast_node_count": ast_node_count,
        "s_ast_max_depth": ast_max_depth,
        "s_cfg_node_count": cfg_node_count,
        "s_cfg_edge_count": cfg_edge_count,
        "s_cyclomatic_proxy": cyclomatic_proxy,
        "s_num_functions_touched": num_functions_touched,
        "s_has_parseable_code": int(has_parseable_code),
        "t_title_len": len(title),
        "t_title_wordcount": len(title.split()),
        "t_body_len": len(body),
        "t_body_wordcount": len(body.split()),
        "t_body_is_empty": int(len(body) == 0),
        "t_commit_msg_len_first": commit_msg_len_first,
        "t_commit_msg_len_mean": commit_msg_len_mean,
        "t_commit_msg_len_total": commit_msg_len_total,
    }
    return {col: float(features[col]) for col in V1_FEATURE_COLUMNS}
