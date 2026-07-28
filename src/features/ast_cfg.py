"""Code-structure (AST/CFG) feature extraction for Experiment 2.

Implements the "S" (code-structure) columns of the feature contract frozen in
`reports/exp2_feature_spec.md` Section 4.2: `s_ast_node_count`,
`s_ast_max_depth`, `s_cfg_node_count`, `s_cfg_edge_count`,
`s_cyclomatic_proxy`, `s_num_functions_touched`, `s_has_parseable_code`.

Scope limitation, stated up front rather than discovered later
------------------------------------------------------------------
Experiment 1 mined unified-diff *patches* per changed file (`files_changed.
patch`), not full before/after file snapshots -- re-fetching full file
contents for ~27k changed files was judged not worth the extra GitHub API
calls for what this experiment needs. Consequently every AST/CFG here is
built from the **hunk-level post-image text** reconstructed out of each
file's patch (context lines + added lines, with removed lines dropped) --
never the whole file. A hunk is frequently *not* a syntactically complete,
standalone program (e.g. it may be just the body of an `if` block with no
enclosing `def`), so a real, honestly-measured fraction of hunks fail to
parse at all. That failure rate is itself reported (see `fidelity_report`
below and `results/tables/exp2_ast_cfg_fidelity.json` once `main()` has run)
rather than hidden -- it is the direct evidence for the fidelity gap Step 9
was asked to document.

Two genuinely different tools per plan Section 4.2 / lab guide Section 1.7.2
------------------------------------------------------------------------------
* **Python** (semantic-kernel, home-assistant): real AST via the stdlib
  `ast` module, real CFG via `staticfg.CFGBuilder`. Highest fidelity in
  principle, but `staticfg` is unmaintained (last touched ~2021) and its
  `invert()` helper (used by `visit_If`/`visit_While` to label the "false"
  exit edge) has a fallback branch guarded by `type(node) == ast.
  NameConstant` -- an attribute Python's `ast` module has since removed, so
  evaluating that comparison itself raises `AttributeError` (not
  `SyntaxError`) before the intended check ever runs. The fallback fires for
  **any condition that isn't a direct comparison** (`Compare`/invertible
  `BinOp`): `if x:`, `if not x:`, `if len(x):`, `if x and y:` all crash the
  CFG stage, while `if x > 0:` does not. Since bare/boolean/call truthiness
  checks are far more idiomatic in real Python than explicit comparisons,
  this is not a rare edge case -- on the real mined dataset it degrades a
  measured ~27% of otherwise-successfully-parsed Python hunks straight to
  `cfg_node_count = 0` (see `results/tables/exp2_ast_cfg_fidelity.json`).
  AST-derived counts (`ast_node_count`, `s_num_functions_touched`) are
  computed independently and stay valid regardless; we track the two
  outcomes separately (`parsed_ok` vs `cfg_ok`) rather than letting a CFG
  crash masquerade as "no control flow." (staticfg also does not model
  `match` statements as branching -- a `match/case` block reports as one
  opaque block -- understating complexity for PRs using structural pattern
  matching; this does not crash, it just silently undercounts.)

* **TypeScript** (vscode) and **C#** (runtime, aspnetcore): `tree-sitter`
  grammars give a real AST, but there is no turnkey CFG library for either
  in the Python ecosystem, so the CFG is a **lightweight approximation**:
  count `if`/`for`/`for-in`/`for-of`/`while`/`do`/`switch-case`/`catch`
  decision nodes (`D`) in the AST and set `cfg_node_count = D + 1`,
  `cfg_edge_count = 2*D`. This is not a faithful control-flow graph (it
  ignores real nesting/sequencing topology) but it reproduces the standard
  McCabe cyclomatic-complexity value exactly: `E - N + 2 = 2D - (D+1) + 2 =
  D + 1`, matching the true complexity of `D` independent binary decisions.
  Unlike Python's `ast.parse` (all-or-nothing), tree-sitter is
  error-tolerant: it still returns partial, usable structure for a broken
  fragment, only marking the bad region as an `ERROR` node. So "parse rate"
  is measured differently per language family (Python: did `ast.parse`
  succeed at all; TS/C#: did the tree come back with zero `ERROR` nodes) --
  the two numbers are diagnostic, not directly comparable across languages,
  and `fidelity_report` reports them per-language rather than pooled to
  avoid implying a false apples-to-apples ranking.

Public entry points
--------------------
`analyze_file(filename, patch)`      -- structure result for one changed file.
`analyze_files(files_changed_df)`    -- one row per (pr_id, filename).
`aggregate_to_pr_features(file_df)`  -- rolls the file table up to one row
                                         per `pr_id`: the frozen `s_*`
                                         columns plus `diag_*` fidelity
                                         columns.
`build_feature_table(files_changed_df)` -- `analyze_files` +
                                         `aggregate_to_pr_features` in one call.
`fidelity_report(file_df)`           -- per-language coverage/parse-rate
                                         summary for the report.

This module is population-agnostic: it computes structure features for
whatever `pr_id`/`filename`/`patch` rows it is given, over the *entire*
mined `files_changed` table (human- and AI-authored alike), so the same
cache is reusable by Experiment 4 later. Filtering to the Experiment 2
population (human-written, closed) is Step 10's job, by joining on `pr_id`.
"""

from __future__ import annotations

import ast
import json
import logging
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from staticfg import CFGBuilder
import tree_sitter_c_sharp as tscs
import tree_sitter_typescript as tsts
from tree_sitter import Language, Parser

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data/processed")
TABLES_DIR = Path("results/tables")

S_FEATURE_COLUMNS = [
    "s_ast_node_count",
    "s_ast_max_depth",
    "s_cfg_node_count",
    "s_cfg_edge_count",
    "s_cyclomatic_proxy",
    "s_num_functions_touched",
    "s_has_parseable_code",
]


# --------------------------------------------------------------------------- #
# Language classification
# --------------------------------------------------------------------------- #

_PYTHON_EXTS = frozenset({"py"})
_TS_PLAIN_EXTS = frozenset({"ts", "js", "mjs", "cjs"})   # parsed with the "typescript" grammar
_TS_JSX_EXTS = frozenset({"tsx", "jsx"})                  # parsed with the "tsx" grammar (JSX support)
_CSHARP_EXTS = frozenset({"cs"})


def _extension(filename: str) -> str:
    if "." not in filename:
        return ""
    return filename.rsplit(".", 1)[-1].lower()


def classify_extension(filename: str) -> Optional[str]:
    """Return 'python' | 'typescript' | 'csharp' | None (unsupported/no AST)."""
    ext = _extension(filename)
    if ext in _PYTHON_EXTS:
        return "python"
    if ext in _TS_PLAIN_EXTS or ext in _TS_JSX_EXTS:
        return "typescript"
    if ext in _CSHARP_EXTS:
        return "csharp"
    return None


def _ts_parser_kind(filename: str) -> str:
    """Which tree-sitter grammar variant to use for a TypeScript-family file."""
    return "tsx" if _extension(filename) in _TS_JSX_EXTS else "typescript"


# --------------------------------------------------------------------------- #
# Diff patch -> hunk post-image reconstruction
# --------------------------------------------------------------------------- #

def iter_post_image_hunks(patch: Optional[str]):
    """Yield the reconstructed post-change text of each hunk in a unified
    diff `patch` (context lines + added lines, in order; removed lines
    dropped). Each `@@ ... @@` marker starts a new hunk. Yields nothing for
    an empty/missing patch (GitHub omits `patch` for binary or very large
    diffs -- ~8.8% of files_changed rows in the mined dataset)."""
    if not isinstance(patch, str) or not patch:
        return  # covers None, NaN (bool(nan) is True in Python -- `not patch` alone misses it), pd.NA
    current: list[str] = []
    in_hunk = False
    for line in patch.split("\n"):
        if line.startswith("@@"):
            if in_hunk and current:
                yield "\n".join(current)
            current = []
            in_hunk = True
            continue
        if not in_hunk:
            continue
        if line.startswith("-"):
            continue  # removed line: not part of the post-image
        if line.startswith("\\"):
            continue  # "\ No newline at end of file"
        if line.startswith("+") or line.startswith(" "):
            current.append(line[1:])
        elif line == "":
            # `str.split("\n")` always yields one trailing "" for a
            # patch ending in "\n" (the normal case) -- not real diff
            # content, so it must not be appended (it would otherwise
            # corrupt exactly the last line of the last hunk).
            continue
        else:
            # Malformed/unprefixed line (rare); keep it rather than drop content.
            current.append(line)
    if in_hunk and current:
        yield "\n".join(current)


# --------------------------------------------------------------------------- #
# Per-hunk analysis result
# --------------------------------------------------------------------------- #

@dataclass
class HunkResult:
    parsed_ok: bool = False
    cfg_ok: bool = False
    ast_node_count: int = 0
    ast_max_depth: int = 0
    cfg_node_count: int = 0
    cfg_edge_count: int = 0
    cyclomatic_proxy: int = 0
    num_functions: int = 0
    has_error_node: bool = False  # tree-sitter only


# --- Python: real AST (stdlib `ast`) + real CFG (`staticfg`) --------------- #

def _ast_depth(node: ast.AST) -> int:
    children = list(ast.iter_child_nodes(node))
    if not children:
        return 1
    return 1 + max(_ast_depth(c) for c in children)


def _count_cfg_components(cfg) -> int:
    """1 (this scope) + every nested function's own CFG component, recursively."""
    n = 1
    for sub in cfg.functioncfgs.values():
        n += _count_cfg_components(sub)
    return n


def _analyze_python_hunk(src: str) -> HunkResult:
    tree = None
    parsed_src = src
    for candidate in (src, textwrap.dedent(src)):
        try:
            tree = ast.parse(candidate)
            parsed_src = candidate
            break
        except SyntaxError:
            continue
    if tree is None:
        return HunkResult(parsed_ok=False)

    node_count = sum(1 for _ in ast.walk(tree))
    max_depth = _ast_depth(tree)
    num_functions = sum(
        1 for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    )

    result = HunkResult(
        parsed_ok=True,
        ast_node_count=node_count,
        ast_max_depth=max_depth,
        num_functions=num_functions,
    )

    try:
        cfg = CFGBuilder().build_from_src("hunk", parsed_src)
        blocks = list(cfg)
        cfg_nodes = len(blocks)
        cfg_edges = sum(len(b.exits) for b in blocks)
        components = _count_cfg_components(cfg)
        result.cfg_ok = True
        result.cfg_node_count = cfg_nodes
        result.cfg_edge_count = cfg_edges
        result.cyclomatic_proxy = cfg_edges - cfg_nodes + 2 * components
    except Exception:
        # staticfg is unmaintained and known to crash (AttributeError:
        # 'module ast has no attribute NameConstant', see module docstring)
        # on any if/while whose condition isn't a plain comparison; it may
        # also raise on other constructs its stale visitor doesn't handle.
        # AST-derived counts above remain valid regardless.
        result.cfg_ok = False

    return result


# --- TypeScript/C#: tree-sitter AST + lightweight CFG approximation -------- #

_DECISION_NODE_TYPES = {
    "typescript": frozenset({
        "if_statement", "for_statement", "for_in_statement",
        "while_statement", "do_statement",
        "switch_case", "switch_default", "catch_clause",
    }),
    "csharp": frozenset({
        "if_statement", "for_statement", "foreach_statement",
        "while_statement", "do_statement",
        "switch_section", "catch_clause",
    }),
}

_FUNCTION_NODE_TYPES = {
    "typescript": frozenset({
        "function_declaration", "method_definition",
        "arrow_function", "function_expression",
    }),
    "csharp": frozenset({
        "method_declaration", "constructor_declaration",
        "local_function_statement", "lambda_expression",
    }),
}

_PARSERS: dict[str, Parser] = {}


def _get_ts_parser(kind: str) -> Parser:
    if kind not in _PARSERS:
        if kind == "typescript":
            lang = Language(tsts.language_typescript())
        elif kind == "tsx":
            lang = Language(tsts.language_tsx())
        elif kind == "csharp":
            lang = Language(tscs.language())
        else:
            raise ValueError(f"unknown parser kind: {kind}")
        _PARSERS[kind] = Parser(lang)
    return _PARSERS[kind]


@dataclass
class _TsWalkStats:
    node_count: int = 0
    max_depth: int = 0
    decisions: int = 0
    functions: int = 0
    has_error: bool = False


def _walk_ts(node, decision_types: frozenset, function_types: frozenset,
             depth: int, stats: _TsWalkStats) -> None:
    stats.node_count += 1
    if depth > stats.max_depth:
        stats.max_depth = depth
    if node.type in decision_types:
        stats.decisions += 1
    if node.type in function_types:
        stats.functions += 1
    if node.type == "ERROR":
        stats.has_error = True
    for child in node.children:
        _walk_ts(child, decision_types, function_types, depth + 1, stats)


def _analyze_ts_hunk(src: str, family: str, parser_kind: str) -> HunkResult:
    parser = _get_ts_parser(parser_kind)
    tree = parser.parse(src.encode("utf-8", errors="replace"))
    stats = _TsWalkStats()
    _walk_ts(tree.root_node, _DECISION_NODE_TYPES[family], _FUNCTION_NODE_TYPES[family], 1, stats)
    decisions = stats.decisions
    return HunkResult(
        parsed_ok=not stats.has_error,
        cfg_ok=True,  # the approximation is a pure count over the same AST; it cannot "fail"
        ast_node_count=stats.node_count,
        ast_max_depth=stats.max_depth,
        cfg_node_count=decisions + 1,
        cfg_edge_count=2 * decisions,
        cyclomatic_proxy=decisions + 1,
        num_functions=stats.functions,
        has_error_node=stats.has_error,
    )


# --------------------------------------------------------------------------- #
# Per-file analysis (aggregates hunks within one changed file)
# --------------------------------------------------------------------------- #

@dataclass
class FileStructureResult:
    language: Optional[str] = None
    n_hunks: int = 0
    n_hunks_parsed: int = 0
    n_hunks_cfg_ok: int = 0
    ast_node_count: int = 0
    ast_max_depth: int = 0
    cfg_node_count: int = 0
    cfg_edge_count: int = 0
    cyclomatic_proxy: int = 0
    num_functions: int = 0


def analyze_file(filename: str, patch: Optional[str]) -> FileStructureResult:
    """Structure counts for one changed file, summed across its hunks.

    Returns a result with `language=None` and all-zero counts for files in
    a language we don't parse (e.g. .md/.json/.yml) -- distinct from a
    parseable-language file whose patch happens to be empty/omitted, which
    still gets `language` set but zero counts (see `s_has_parseable_code`).
    """
    language = classify_extension(filename)
    result = FileStructureResult(language=language)
    if language is None:
        return result

    if language == "typescript":
        parser_kind = _ts_parser_kind(filename)
    elif language == "csharp":
        parser_kind = "csharp"
    else:
        parser_kind = None

    for hunk_src in iter_post_image_hunks(patch):
        result.n_hunks += 1
        try:
            if language == "python":
                hr = _analyze_python_hunk(hunk_src)
            else:
                hr = _analyze_ts_hunk(hunk_src, language, parser_kind)
        except Exception:
            logger.warning("Unexpected failure analyzing a hunk of %s; scoring as unparsed", filename)
            hr = HunkResult(parsed_ok=False)

        if hr.parsed_ok:
            result.n_hunks_parsed += 1
        if hr.cfg_ok:
            result.n_hunks_cfg_ok += 1
        result.ast_node_count += hr.ast_node_count
        result.ast_max_depth = max(result.ast_max_depth, hr.ast_max_depth)
        result.cfg_node_count += hr.cfg_node_count
        result.cfg_edge_count += hr.cfg_edge_count
        result.cyclomatic_proxy += hr.cyclomatic_proxy
        result.num_functions += hr.num_functions

    return result


# --------------------------------------------------------------------------- #
# files_changed.parquet -> per-file table -> per-PR feature table
# --------------------------------------------------------------------------- #

def analyze_files(files_changed: pd.DataFrame) -> pd.DataFrame:
    """One row per (pr_id, filename): language + structure counts + hunk
    parse/cfg-success diagnostics. The unit `aggregate_to_pr_features` and
    `fidelity_report` are both built on."""
    records = []
    for row in files_changed.itertuples(index=False):
        patch = getattr(row, "patch", None)
        fr = analyze_file(row.filename, patch)
        records.append({
            "pr_id": row.pr_id,
            "filename": row.filename,
            "language": fr.language,
            "n_hunks": fr.n_hunks,
            "n_hunks_parsed": fr.n_hunks_parsed,
            "n_hunks_cfg_ok": fr.n_hunks_cfg_ok,
            "ast_node_count": fr.ast_node_count,
            "ast_max_depth": fr.ast_max_depth,
            "cfg_node_count": fr.cfg_node_count,
            "cfg_edge_count": fr.cfg_edge_count,
            "cyclomatic_proxy": fr.cyclomatic_proxy,
            "num_functions": fr.num_functions,
        })
    columns = [
        "pr_id", "filename", "language", "n_hunks", "n_hunks_parsed",
        "n_hunks_cfg_ok", "ast_node_count", "ast_max_depth", "cfg_node_count",
        "cfg_edge_count", "cyclomatic_proxy", "num_functions",
    ]
    return pd.DataFrame.from_records(records, columns=columns)


def aggregate_to_pr_features(file_table: pd.DataFrame) -> pd.DataFrame:
    """Roll the per-file table up to one row per `pr_id`: the frozen `s_*`
    feature contract (reports/exp2_feature_spec.md Section 4.2) plus
    `diag_*` fidelity columns. PRs absent from `file_table` (no changed-file
    rows at all) are simply absent here -- callers should left-join on the
    full PR population and fill `s_has_parseable_code=0` / zeros for any
    PR that doesn't appear."""
    empty_cols = ["pr_id", *S_FEATURE_COLUMNS,
                  "diag_n_hunks_attempted", "diag_n_hunks_parsed",
                  "diag_n_hunks_cfg_ok", "diag_parse_rate"]
    if file_table.empty:
        return pd.DataFrame(columns=empty_cols)

    work = file_table.copy()
    work["_is_parseable_lang"] = work["language"].notna()

    g = work.groupby("pr_id", sort=False)
    sums = g[[
        "ast_node_count", "cfg_node_count", "cfg_edge_count",
        "cyclomatic_proxy", "num_functions", "n_hunks",
        "n_hunks_parsed", "n_hunks_cfg_ok",
    ]].sum()

    result = pd.DataFrame({
        "s_ast_node_count": sums["ast_node_count"],
        "s_ast_max_depth": g["ast_max_depth"].max(),
        "s_cfg_node_count": sums["cfg_node_count"],
        "s_cfg_edge_count": sums["cfg_edge_count"],
        "s_cyclomatic_proxy": sums["cyclomatic_proxy"],
        "s_num_functions_touched": sums["num_functions"],
        "s_has_parseable_code": g["_is_parseable_lang"].any().astype(int),
        "diag_n_hunks_attempted": sums["n_hunks"],
        "diag_n_hunks_parsed": sums["n_hunks_parsed"],
        "diag_n_hunks_cfg_ok": sums["n_hunks_cfg_ok"],
    })
    result["diag_parse_rate"] = (
        result["diag_n_hunks_parsed"].astype(float)
        / result["diag_n_hunks_attempted"].astype(float).replace(0.0, np.nan)
    )
    return result.reset_index()


def build_feature_table(files_changed: pd.DataFrame) -> pd.DataFrame:
    """`analyze_files` + `aggregate_to_pr_features` in one call."""
    return aggregate_to_pr_features(analyze_files(files_changed))


# --------------------------------------------------------------------------- #
# Fidelity report -- the Step 9 "document the fidelity difference" deliverable
# --------------------------------------------------------------------------- #

def fidelity_report(file_table: pd.DataFrame) -> dict:
    """Per-language coverage and hunk parse-rate summary, computed from the
    real mined dataset. Deliberately per-language rather than pooled: a
    pooled "parse rate" would silently average over the fact that Python's
    `ast.parse` is all-or-nothing while tree-sitter degrades gracefully, so
    the two are not measuring the same failure mode."""
    report: dict = {"n_files_total": int(len(file_table))}
    unsupported = int(file_table["language"].isna().sum()) if len(file_table) else 0
    report["n_files_unsupported_language"] = unsupported

    for lang in ("python", "typescript", "csharp"):
        sub = file_table[file_table["language"] == lang]
        n_files = int(len(sub))
        n_files_with_patch = int((sub["n_hunks"] > 0).sum())
        attempted = int(sub["n_hunks"].sum())
        parsed = int(sub["n_hunks_parsed"].sum())
        entry = {
            "n_files": n_files,
            "n_files_with_patch": n_files_with_patch,
            "n_files_no_patch": n_files - n_files_with_patch,
            "n_hunks_attempted": attempted,
            "n_hunks_parsed_ok": parsed,
            "hunk_parse_rate": round(parsed / attempted, 4) if attempted else None,
        }
        if lang == "python":
            cfg_ok = int(sub["n_hunks_cfg_ok"].sum())
            entry["n_hunks_cfg_ok"] = cfg_ok
            entry["cfg_success_rate_given_parsed"] = (
                round(cfg_ok / parsed, 4) if parsed else None
            )
        report[lang] = entry
    return report


# --------------------------------------------------------------------------- #
# CLI: compute + cache over the real mined dataset
# --------------------------------------------------------------------------- #

def main() -> None:
    files_changed = pd.read_parquet(DATA_DIR / "files_changed.parquet")
    logger.info(
        "Analyzing AST/CFG structure for %d changed-file rows across %d PRs...",
        len(files_changed), files_changed["pr_id"].nunique(),
    )

    t0 = time.time()
    file_table = analyze_files(files_changed)
    pr_table = aggregate_to_pr_features(file_table)
    elapsed = time.time() - t0
    logger.info("Analyzed %d files -> %d PR rows in %.1fs", len(file_table), len(pr_table), elapsed)

    out_path = DATA_DIR / "ast_cfg_features.parquet"
    pr_table.to_parquet(out_path, index=False)
    logger.info("Wrote %s", out_path)

    fidelity = fidelity_report(file_table)
    fidelity["elapsed_seconds"] = round(elapsed, 1)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    fidelity_path = TABLES_DIR / "exp2_ast_cfg_fidelity.json"
    fidelity_path.write_text(json.dumps(fidelity, indent=2))
    logger.info("Wrote %s", fidelity_path)
    logger.info("Fidelity summary:\n%s", json.dumps(fidelity, indent=2))


if __name__ == "__main__":
    main()
