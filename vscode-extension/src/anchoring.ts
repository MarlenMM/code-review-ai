/**
 * Resolves a free-prose review comment to a `(file, line)` location in the
 * working tree, so the panel can offer click-to-jump and `decorations.ts`
 * can place an inline highlight.
 *
 * WHY THIS EXISTS (the honest version)
 * -------------------------------------
 * The backend returns `review_comments` as a plain `string[]`
 * (`src/api/main.py`) because that is what the Experiment 3/4 output
 * contract produces: the prompts ask for a `<review>` block of prose
 * bullets, and `src/llm/prompts/parsing.py` splits it into lines. No file
 * or line number is requested, so none is returned.
 *
 * The alternative -- changing the prompt contract to demand structured
 * `file:line` output -- was rejected: it would invalidate Experiment 4's
 * already-completed 148-cell grid, its on-disk response cache, and the
 * `parse_review_comments` tests built against that contract, in order to
 * re-run a grid that the shared free-tier Groq quota cannot currently
 * afford to repeat. Re-deriving the location on the *client* from the diff
 * the user just sent costs nothing and touches no completed experiment
 * artifact.
 *
 * So this is a **heuristic**, and its output is deliberately labelled with
 * a `precision` the UI surfaces rather than hides:
 *
 *   'line'          -- the comment named a symbol (or an exact line number)
 *                      that occurs on a specific added line.
 *   'file'          -- the comment named a changed file, but nothing
 *                      narrower; anchored to that file's first change.
 *   'inferred-file' -- the comment named nothing at all, but the diff
 *                      touches exactly ONE file, so that file is the only
 *                      thing it could be about.
 *
 * A comment matching none of the above resolves to `null` and is rendered
 * as plain, non-clickable text -- never guessed into a wrong location,
 * which would be worse than no link at all.
 */

import type { FileChange } from "./diffMap";

export type AnchorPrecision = "line" | "file" | "inferred-file";

export interface CommentLocation {
  path: string;
  /** 1-based, in the post-change file. */
  line: number;
  precision: AnchorPrecision;
}

export interface AnchoredComment {
  text: string;
  location: CommentLocation | null;
}

/** Shortest token worth searching for as a code symbol: 1-2 character
 * tokens (`i`, `x`, `if`) occur in almost any line and would produce
 * confident-looking nonsense. */
const MIN_SYMBOL_LENGTH = 3;

const BACKTICK_RE = /`([^`]+)`/g;
const BARE_FILE_RE = /[\w.\-/]+\.[A-Za-z]{1,6}\b/g;
const EXPLICIT_LINE_RE = /\blines?\s*#?\s*(\d+)/i;

function basename(path: string): string {
  const idx = path.lastIndexOf("/");
  return idx === -1 ? path : path.slice(idx + 1);
}

function extractBacktickTokens(comment: string): string[] {
  const tokens: string[] = [];
  for (const match of comment.matchAll(BACKTICK_RE)) {
    const token = match[1].trim();
    if (token) {
      tokens.push(token);
    }
  }
  return tokens;
}

function extractFileCandidates(comment: string, backtickTokens: string[]): string[] {
  const bare = comment.match(BARE_FILE_RE) ?? [];
  // Longest first: "src/foo.py" is a more specific claim than "foo.py".
  return [...backtickTokens, ...bare].sort((a, b) => b.length - a.length);
}

function matchFile(candidates: string[], files: FileChange[]): { file: FileChange; token: string } | null {
  for (const token of candidates) {
    const cleaned = token.replace(/^\.\//, "").replace(/[),.;:]+$/, "");
    if (!cleaned) {
      continue;
    }
    for (const file of files) {
      if (
        file.path === cleaned ||
        file.path.endsWith(`/${cleaned}`) ||
        basename(file.path) === cleaned
      ) {
        return { file, token };
      }
    }
  }
  return null;
}

function explicitLine(comment: string, file: FileChange): number | null {
  const match = EXPLICIT_LINE_RE.exec(comment);
  if (!match) {
    return null;
  }
  const line = parseInt(match[1], 10);
  // Only trust a stated line number if the diff actually added that line --
  // an 8B model inventing "line 42" for a 4-line diff should not send the
  // user to line 42.
  return file.addedLines.some((l) => l.line === line) ? line : null;
}

/** Where a symbol was found within one file. Added lines are searched
 * before context lines: a comment about a line the PR actually changed is
 * the more likely reading, and preferring it keeps the common case stable.
 * Both yield an exact line number, so both are `precision: 'line'`. */
function findSymbolInFile(symbol: string, file: FileChange): number | null {
  const added = file.addedLines.find((l) => l.text.includes(symbol));
  if (added) {
    return added.line;
  }
  const context = file.contextLines.find((l) => l.text.includes(symbol));
  return context ? context.line : null;
}

function symbolLine(symbols: string[], file: FileChange, excludeToken?: string): number | null {
  for (const symbol of symbols) {
    if (symbol === excludeToken || symbol.length < MIN_SYMBOL_LENGTH) {
      continue;
    }
    const line = findSymbolInFile(symbol, file);
    if (line !== null) {
      return line;
    }
  }
  return null;
}

/** A symbol that appears in exactly one changed file gives a line-precise
 * anchor even when the comment never names the file. Ambiguous symbols
 * (present in several files) are rejected rather than picked arbitrarily. */
function uniqueSymbolAcrossFiles(
  symbols: string[],
  files: FileChange[],
): CommentLocation | null {
  for (const symbol of symbols) {
    if (symbol.length < MIN_SYMBOL_LENGTH) {
      continue;
    }
    const hits = files
      .map((file) => {
        const line = findSymbolInFile(symbol, file);
        return line === null ? null : { path: file.path, line };
      })
      .filter((hit): hit is { path: string; line: number } => hit !== null);

    if (hits.length === 1) {
      return { path: hits[0].path, line: hits[0].line, precision: "line" };
    }
  }
  return null;
}

export function anchorComment(comment: string, files: FileChange[]): CommentLocation | null {
  if (files.length === 0) {
    return null;
  }

  const backtickTokens = extractBacktickTokens(comment);
  const candidates = extractFileCandidates(comment, backtickTokens);
  const matched = matchFile(candidates, files);

  if (matched) {
    const { file, token } = matched;
    const line = explicitLine(comment, file) ?? symbolLine(backtickTokens, file, token);
    if (line !== null) {
      return { path: file.path, line, precision: "line" };
    }
    return { path: file.path, line: file.firstChangedLine, precision: "file" };
  }

  const bySymbol = uniqueSymbolAcrossFiles(backtickTokens, files);
  if (bySymbol) {
    return bySymbol;
  }

  if (files.length === 1) {
    return {
      path: files[0].path,
      line: files[0].firstChangedLine,
      precision: "inferred-file",
    };
  }

  return null;
}

export function anchorComments(comments: string[], files: FileChange[]): AnchoredComment[] {
  return comments.map((text) => ({ text, location: anchorComment(text, files) }));
}
