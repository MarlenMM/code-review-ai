/**
 * Parses a unified diff into per-file **new-file line numbers** for the
 * lines it adds -- the position information the extension needs to place a
 * review comment in an editor (click-to-jump, `decorations.ts` highlights).
 *
 * Deliberately NOT the same job as the backend's `src/api/diff_parsing.py`,
 * despite both reading a unified diff: that one reconstructs
 * `(filename, status, patch)` rows to compute ML features and never needs a
 * line number. This one only cares *where in the current working file* each
 * added line lives, so the two are kept separate rather than one being
 * contorted to serve both. (They are also on opposite sides of the wire.)
 *
 * Line numbering follows the `+` side of each `@@ -a,b +c,d @@` header:
 * the cursor starts at `c` and advances on added and context lines but not
 * on removed ones -- which is exactly the numbering an editor uses for the
 * post-change file on disk, so a recorded line number can be handed
 * straight to `vscode.Position`.
 */

export interface DiffLine {
  /** 1-based line number in the *post-change* (current on-disk) file. */
  line: number;
  text: string;
}

/** Retained name for the added-line type (`DiffLine` now also describes
 * context lines, which carry identical fields). */
export type AddedLine = DiffLine;

export interface FileChange {
  /** Repo-relative path, from the diff's `b/` side. */
  path: string;
  addedLines: DiffLine[];
  /**
   * Unchanged lines shown as diff context, with their post-change line
   * numbers. Tracked because review comments very often name the
   * *enclosing* symbol (`the `greet` function...`), which by definition
   * sits on a line the diff did not add -- without these, such a comment
   * can only ever resolve to file-level precision. `anchoring.ts` searches
   * added lines first and treats these as the weaker, secondary source.
   */
  contextLines: DiffLine[];
  /**
   * Best single line to jump to for a file-level comment: the first added
   * line, or the first hunk's start when a file only removes lines.
   */
  firstChangedLine: number;
}

const HUNK_RE = /^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@/;
const DIFF_GIT_RE = /^diff --git a\/(.+) b\/(.+)$/;

export function parseDiffMap(diff: string): FileChange[] {
  if (!diff || !diff.trim()) {
    return [];
  }

  const files: FileChange[] = [];
  let current: FileChange | undefined;
  let cursor = 0;
  let sawHunk = false;

  const finish = () => {
    if (!current) {
      return;
    }
    if (current.firstChangedLine === 0) {
      // A removal-only file: no added lines to anchor to, so fall back to
      // the first hunk's start (where the removal happened).
      current.firstChangedLine = 1;
    }
    files.push(current);
    current = undefined;
  };

  for (const line of diff.split("\n")) {
    const gitMatch = DIFF_GIT_RE.exec(line);
    if (gitMatch) {
      finish();
      current = { path: gitMatch[2], addedLines: [], contextLines: [], firstChangedLine: 0 };
      sawHunk = false;
      continue;
    }

    // `+++ b/path` must be tested before the `+` added-line branch below.
    if (line.startsWith("+++ ")) {
      const path = line.slice(4).split("\t")[0].trim();
      if (path !== "/dev/null") {
        const stripped = path.startsWith("b/") ? path.slice(2) : path;
        if (!current) {
          // A bare patch with no `diff --git` header still names its file.
          current = { path: stripped, addedLines: [], contextLines: [], firstChangedLine: 0 };
          sawHunk = false;
        } else {
          current.path = stripped;
        }
      }
      continue;
    }
    if (line.startsWith("--- ")) {
      continue;
    }

    const hunkMatch = HUNK_RE.exec(line);
    if (hunkMatch) {
      cursor = parseInt(hunkMatch[1], 10);
      sawHunk = true;
      if (current && current.firstChangedLine === 0) {
        current.firstChangedLine = cursor;
      }
      continue;
    }

    if (!current || !sawHunk) {
      continue;
    }

    if (line.startsWith("+")) {
      const text = line.slice(1);
      current.addedLines.push({ line: cursor, text });
      if (current.addedLines.length === 1) {
        current.firstChangedLine = cursor;
      }
      cursor += 1;
    } else if (line.startsWith("-")) {
      // Removed: present in the old file only, so the new-file cursor stays.
    } else if (line.startsWith("\\")) {
      // "\ No newline at end of file" -- metadata, not content.
    } else {
      // Context line (leading space) or a blank trailing line.
      const text = line.startsWith(" ") ? line.slice(1) : line;
      if (text.trim()) {
        current.contextLines.push({ line: cursor, text });
      }
      cursor += 1;
    }
  }

  finish();
  return files;
}
