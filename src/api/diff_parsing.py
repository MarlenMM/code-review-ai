"""Parse a raw unified diff (e.g. `git diff` output, or a GitHub-style PR
patch) into the same per-file `(filename, status, patch)` shape as the mined
dataset's `files_changed.parquet` rows -- so `src/features/ast_cfg.py` and
`src/api/ml_features.py` can be reused unchanged on a diff that was never
mined and has no `pr_id`.

This is intentionally a pragmatic subset of `git diff`'s format, not a
general-purpose patch parser: it recognizes `diff --git a/X b/Y` file
boundaries, `new file mode` / `deleted file mode` / `rename from` / `rename
to` status markers, and `--- ` / `+++ ` header lines, and treats everything
from the first `@@` hunk marker onward as the file's patch body (matching
GitHub REST's own `files.patch` field, which is hunks only -- no `diff
--git`/`index`/`---`/`+++` preamble). Filenames containing spaces or other
characters git would C-quote are not unquoted -- a known, documented gap
rather than a silent one.

A diff with no `diff --git` boundaries at all (e.g. a single bare unified
patch) is treated as one implicit file, so a minimal `--- a/x\n+++ b/x\n@@
...` snippet still parses.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_DIFF_GIT_RE = re.compile(r"^diff --git a/(.*) b/(.*)$")
_MINUS_HEADER_RE = re.compile(r"^--- (?:a/(.+)|/dev/null)\s*$")
_PLUS_HEADER_RE = re.compile(r"^\+\+\+ (?:b/(.+)|/dev/null)\s*$")

_STATUSES = ("added", "modified", "removed", "renamed")


@dataclass(frozen=True)
class ParsedFile:
    """One changed file, in the same shape `files_changed.parquet` rows have
    (`filename`, `status`, `patch`), plus `additions`/`deletions` counted
    directly from the hunk body (GitHub supplies these as separate columns;
    a raw diff does not, so they are derived here instead of trusted from
    the caller)."""

    filename: str
    status: str  # one of _STATUSES
    patch: str  # hunk-only body, "" if no textual hunks (e.g. pure rename, binary)
    additions: int
    deletions: int


def _strip_timestamp(header_line: str) -> str:
    """`--- a/file\t2024-01-01 00:00:00` -> `--- a/file` -- POSIX `diff -u`
    appends a tab + timestamp that plain `git diff` normally omits, but
    stripping it defensively costs nothing."""
    return header_line.split("\t", 1)[0].rstrip()


def _parse_block(block_lines: list[str]) -> ParsedFile | None:
    status = "modified"
    old_name: str | None = None
    new_name: str | None = None
    hunk_start: int | None = None

    if block_lines and block_lines[0].startswith("diff --git "):
        m = _DIFF_GIT_RE.match(block_lines[0])
        if m:
            old_name, new_name = m.group(1), m.group(2)

    for i, raw_line in enumerate(block_lines):
        line = raw_line
        if line.startswith("new file mode"):
            status = "added"
        elif line.startswith("deleted file mode"):
            status = "removed"
        elif line.startswith("rename from "):
            status = "renamed"
            old_name = old_name or line[len("rename from "):].strip()
        elif line.startswith("rename to "):
            status = "renamed"
            new_name = line[len("rename to "):].strip()
        elif line.startswith("--- "):
            m = _MINUS_HEADER_RE.match(_strip_timestamp(line))
            if m and m.group(1):
                old_name = m.group(1)
        elif line.startswith("+++ "):
            m = _PLUS_HEADER_RE.match(_strip_timestamp(line))
            if m and m.group(1):
                new_name = m.group(1)
        elif line.startswith("@@") and hunk_start is None:
            hunk_start = i

    filename = new_name or old_name
    if not filename:
        return None  # nothing identifiable as a file in this block

    if status == "modified" and new_name is None and old_name is not None:
        # "--- a/x" present with "+++ /dev/null" (or missing) -> a deletion
        # git's own "deleted file mode" line normally already caught this;
        # this is the fallback for a deletion-only patch missing that line.
        status = "removed"

    patch = "\n".join(block_lines[hunk_start:]) if hunk_start is not None else ""
    additions = sum(
        1 for line in block_lines
        if line.startswith("+") and not line.startswith("+++")
    )
    deletions = sum(
        1 for line in block_lines
        if line.startswith("-") and not line.startswith("---")
    )
    return ParsedFile(filename=filename, status=status, patch=patch,
                       additions=additions, deletions=deletions)


def parse_unified_diff(diff_text: str) -> list[ParsedFile]:
    """Split `diff_text` into per-file blocks and parse each one. Returns an
    empty list for empty/whitespace-only input or a diff with no
    identifiable file blocks."""
    if not diff_text or not diff_text.strip():
        return []

    lines = diff_text.splitlines()
    boundaries = [i for i, line in enumerate(lines) if line.startswith("diff --git ")]
    if not boundaries:
        blocks = [(0, len(lines))]
    else:
        boundaries.append(len(lines))
        blocks = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]

    files: list[ParsedFile] = []
    for start, end in blocks:
        parsed = _parse_block(lines[start:end])
        if parsed is not None:
            files.append(parsed)
    return files
