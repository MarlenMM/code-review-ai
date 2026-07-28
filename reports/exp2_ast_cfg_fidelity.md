# AST/CFG Fidelity Write-Up — `src/features/ast_cfg.py` (Step 9)

*Documents the fidelity difference between the Python (real CFG) and TypeScript/C#
(approximate CFG) pipelines, per lab guide §2.4.2/§2.7.2 and plan §4.2. All numbers
below are real, computed by running `python -m src.features.ast_cfg` over the full
1,494-PR / 27,110-file mined dataset — see `results/tables/exp2_ast_cfg_fidelity.json`
and `data/processed/ast_cfg_features.parquet`.*

---

## 1. The scope decision this is all downstream of

Experiment 1 mined unified-diff **patches** per changed file, not full before/after
file snapshots — re-fetching whole files for ~27k changed files was judged not worth
the extra GitHub API calls. So every AST/CFG here is built from the **hunk-level
post-image text** reconstructed out of each file's patch (context lines + added
lines, removed lines dropped) — never the whole file. A hunk is frequently *not* a
syntactically complete, standalone program (e.g. just the body of an `if` with no
enclosing `def`), so a real fraction of hunks fail to parse at all. That failure rate
is the centerpiece of this write-up, not a footnote.

## 2. Coverage: how much of the Experiment 2 population has any code to measure

| | Value |
|---|---|
| Experiment 2 population (human-written, closed) | 1,114 PRs |
| Present in `files_changed` at all | 1,111 (3 PRs have zero changed files — genuinely empty/`changed_files=0` closed PRs, a pre-existing mining-data footnote, not an AST/CFG issue) |
| Has ≥1 file in a supported language (`s_has_parseable_code=1`) | **800 (71.8%)** |
| No supported-language file at all (pure `.md`/`.json`/`.yml`/... diffs) | 28.2% |

This matches the 72% coverage estimate in the Step 8 feature spec, confirming that
estimate held up once the extraction code actually ran.

## 3. Measured parse rates, per language

| Language | Files | Files w/ patch data | Hunks attempted | Hunks parsed OK | **Hunk parse rate** |
|---|---:|---:|---:|---:|---:|
| Python | 1,506 | 1,475 | 3,142 | 1,098 | **35.0%** |
| TypeScript (incl. .tsx/.js) | 7,505 | 6,533 | 10,365 | 6,159 | **59.4%** |
| C# | 6,232 | 5,819 | 14,843 | 5,521 | **37.2%** |

**These three numbers are not measuring the same thing, and must not be read as a
cross-language ranking.** Python's `ast.parse` is all-or-nothing: a hunk either forms
a complete valid module or it raises `SyntaxError`. Tree-sitter (TS/C#) is
*error-tolerant*: it always returns a tree, and "parsed OK" here means "the tree came
back with zero `ERROR` nodes" — it still extracts real, partial structure even for a
hunk it can't fully validate, which our TS/C# counts *use*. So TypeScript's 59.4% does
not mean "TypeScript hunks are structurally more complete than Python's" — it means
tree-sitter's recovery is more forgiving of incomplete fragments than a hard
all-or-nothing parser. C#'s lower rate than TypeScript despite using the same
tree-sitter approach is likely because C# hunks are usually nested inside
class/namespace/method scopes that context lines alone don't reconstruct — a brace-
delimited language is more sensitive to a missing enclosing scope than TS/JS, where
top-level statements and arrow functions are common. All three sitting well under 60%
is the honest headline: **the majority of individual diff hunks, even with 3 lines of
context, are not syntactically self-contained fragments.**

## 4. Python's CFG stage: a real, load-bearing bug in `staticfg`, not an edge case

`staticfg` (the only maintained-ish Python CFG library available, last touched
~2021) builds real control-flow graphs via `ast` — but its `invert()` helper
(`staticfg/builder.py`, used by `visit_If`/`visit_While` to label the CFG's "false"
exit edge) has this fallback:

```python
elif type(node) == ast.NameConstant and node.value in [True, False]:
    inverse_node = ast.NameConstant(value=not node.value)
else:
    inverse_node = ast.UnaryOp(op=ast.Not(), operand=node)
```

`ast.NameConstant` is a pre-3.8 alias that Python's `ast` module has since removed
entirely. **Evaluating `type(node) == ast.NameConstant` itself raises
`AttributeError: module 'ast' has no attribute 'NameConstant'`** — before the intended
check ever runs — for *any* condition that reaches this fallback, i.e. any
`if`/`while` condition that isn't a direct comparison (`Compare`) or an invertible
`BinOp`. Concretely, on Python 3.14 (this project's interpreter):

| Condition | Node type | CFG build |
|---|---|---|
| `if x > 0:` | `Compare` | ✅ works |
| `if x:` | `Name` | ❌ crashes |
| `if not x:` | `UnaryOp` | ❌ crashes |
| `if len(x):` | `Call` | ❌ crashes |
| `if x and y:` | `BoolOp` | ❌ crashes |

Bare/boolean/call truthiness checks (`if x:`, `if not flag:`, `if items:`) are
*more* idiomatic in real Python than explicit comparisons, not less — so this isn't a
rare corner case, it's close to a coin-flip on ordinary code. Measured on the real
dataset: of the 1,098 Python hunks that parsed as valid standalone code, only **802
(73.0%) also got a real CFG** — **27.0% silently lost their CFG counts** to this one
dead code path in an unmaintained dependency, despite being perfectly valid Python.

`ast_cfg.py` tracks this explicitly rather than letting it hide: `parsed_ok` (did
`ast.parse` succeed) and `cfg_ok` (did `staticfg` also succeed) are separate fields,
so `s_ast_node_count`/`s_num_functions_touched` stay correct even when
`s_cfg_node_count`/`s_cfg_edge_count`/`s_cyclomatic_proxy` fall back to 0 for that
hunk. (A second, non-crashing gap: `staticfg` doesn't model `match`/`case` as
branching — a whole `match` block reports as one opaque unbranched block — silently
understating complexity for PRs using structural pattern matching. This doesn't
throw, so it isn't in the 27% above; it's a separate, quieter undercount.)

## 5. TypeScript/C#: a deliberate approximation, not a real CFG

No maintained CFG library exists for TS/C# in the Python ecosystem, so `ast_cfg.py`
counts decision nodes directly from the tree-sitter AST — `if`/`for`/`for-in`/
`for-of`/`while`/`do`/`switch-case`(or `switch-section`)/`catch` — and sets:

```
cfg_node_count = D + 1        cfg_edge_count = 2·D
```

This is **not a faithful graph** (it discards real nesting/sequencing topology — e.g.
it can't distinguish two sequential `if`s from two nested ones), but it reproduces
the standard McCabe cyclomatic-complexity value exactly: `E − N + 2 = 2D − (D+1) + 2
= D + 1`, i.e. the correct complexity for `D` independent binary decisions. Compare
Python's formula (`E − N + 2·components`, using `staticfg`'s real block graph and
recursively counting one component per nested function scope) — both reduce to the
same textbook quantity when they succeed, but only the Python path is a real graph;
the TS/C# path is complexity-equivalent by construction, not topologically faithful.

## 6. Other honest gaps, already visible in the numbers

- **No-patch files.** GitHub omits `patch` for binary/very-large diffs: 31 Python,
  972 TypeScript, and 413 C# files in the mined dataset have no patch text at all —
  these still count toward `s_has_parseable_code` (the *language* is supported) but
  contribute zero to every count (nothing to parse). TypeScript's 972 is the largest
  share (~13% of its files), plausibly from `vscode`'s larger generated/snapshot
  `.ts` fixtures being too large for GitHub's inline diff.
- **Heavy-tailed counts.** Over the 1,111-PR Experiment 2 population,
  `s_ast_node_count` has median 120 but a max of 3.63M (one PR with an enormous
  generated/vendored diff) — confirms the `log1p` transform the Step 8 spec already
  calls for in preprocessing (§6) is necessary, not optional, before feeding SVM/RF.
- **3 Experiment-2 PRs with zero changed files.** Pre-existing in the mined data
  (`changed_files=0`, closed with no diff at all); `s_has_parseable_code` correctly
  resolves to 0 for these once Step 10 left-joins and fills missing `pr_id`s.

## 7. Worked example

Diff hunk (post-image reconstructed from a patch), file `example.py`:

```python
def foo(x):
    if x > 0:
        return x
    return -x

def bar():
    pass
```

`analyze_file` result: `ast_node_count=21`, `ast_max_depth=6`, `num_functions=2`,
`cfg_node_count=5`, `cfg_edge_count=2`, `cyclomatic_proxy=3` (the `if`-without-`else`
in `foo` contributes complexity 2; the branchless `bar` contributes 1; summed via the
recursive per-function component count → `3`). Changing the condition to `if x:`
(same code, only the comparison replaced with a bare-Name truthiness check) still
parses identically but drives `cfg_node_count`/`cfg_edge_count`/`cyclomatic_proxy` to
`0` for that hunk — the exact failure mode in §4, reproduced as a regression test in
`tests/test_ast_cfg.py::test_analyze_file_python_bare_truthy_condition_parses_but_cfg_fails`.

## 8. Artifacts

- `src/features/ast_cfg.py` — the module (public API documented in its own docstring).
- `data/processed/ast_cfg_features.parquet` — one row per PR (1,462 of 1,494 mined
  PRs have ≥1 changed-file row), the frozen `s_*` columns plus `diag_*` fidelity
  columns; population-agnostic, reusable by Experiment 4.
- `results/tables/exp2_ast_cfg_fidelity.json` — the machine-readable version of the
  per-language table in §3, regenerated by `python -m src.features.ast_cfg`.
- `tests/test_ast_cfg.py` — 24 tests, including regression anchors for both bugs this
  write-up depends on (the `staticfg` crash trigger, and the hunk-reconstruction
  trailing-newline fix caught while building this).
