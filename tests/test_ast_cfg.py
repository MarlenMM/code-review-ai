import math

import pandas as pd
import pytest

from src.features.ast_cfg import (
    S_FEATURE_COLUMNS,
    aggregate_to_pr_features,
    analyze_file,
    analyze_files,
    classify_extension,
    fidelity_report,
    iter_post_image_hunks,
)


# --------------------------------------------------------------------------- #
# classify_extension
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("filename,expected", [
    ("foo/bar.py", "python"),
    ("src/App.ts", "typescript"),
    ("src/App.tsx", "typescript"),
    ("src/legacy.js", "typescript"),
    ("src/legacy.mjs", "typescript"),
    ("src/Widget.jsx", "typescript"),
    ("Program.cs", "csharp"),
    ("README.md", None),
    ("data.json", None),
    ("Makefile", None),
])
def test_classify_extension(filename, expected):
    assert classify_extension(filename) == expected


# --------------------------------------------------------------------------- #
# iter_post_image_hunks
# --------------------------------------------------------------------------- #

def test_iter_post_image_hunks_keeps_context_and_added_drops_removed():
    patch = (
        "@@ -1,4 +1,4 @@\n"
        " def foo(x):\n"
        "-    return x\n"
        "+    if x > 0:\n"
        "+        return x\n"
        "     y = 1\n"
    )
    hunks = list(iter_post_image_hunks(patch))
    assert len(hunks) == 1
    assert "return x" not in hunks[0].split("\n")[1]  # the removed line is gone
    assert "if x > 0:" in hunks[0]
    assert "def foo(x):" in hunks[0]  # context line retained
    assert "y = 1" in hunks[0]  # trailing context line retained


def test_iter_post_image_hunks_splits_multiple_hunks():
    patch = (
        "@@ -1,2 +1,2 @@\n"
        "-a\n"
        "+b\n"
        "@@ -10,2 +10,2 @@\n"
        "-c\n"
        "+d\n"
    )
    hunks = list(iter_post_image_hunks(patch))
    assert hunks == ["b", "d"]


@pytest.mark.parametrize("patch", [None, "", float("nan")])
def test_iter_post_image_hunks_handles_missing_patch(patch):
    assert list(iter_post_image_hunks(patch)) == []


# --------------------------------------------------------------------------- #
# analyze_file -- Python (real ast + staticfg CFG)
# --------------------------------------------------------------------------- #

def test_analyze_file_python_valid_snippet():
    patch = (
        "@@ -1,2 +1,7 @@\n"
        " def foo(x):\n"
        "-    return x\n"
        "+    if x > 0:\n"
        "+        return x\n"
        "+    return -x\n"
        "+\n"
        "+def bar():\n"
        "+    pass\n"
    )
    r = analyze_file("mod.py", patch)
    assert r.language == "python"
    assert r.n_hunks == 1
    assert r.n_hunks_parsed == 1
    assert r.n_hunks_cfg_ok == 1
    assert r.ast_node_count > 0
    assert r.num_functions == 2  # foo, bar
    # one if-without-else in foo (complexity 2) + bar with no branches (complexity 1)
    assert r.cyclomatic_proxy == 3


def test_analyze_file_python_unparseable_fragment_scores_zero():
    # a dangling `else:` with no matching `if` is not valid standalone Python,
    # even after dedent -- this is the expected, honestly-reported failure mode
    # for hunk-level (not whole-file) parsing.
    patch = (
        "@@ -1,2 +1,3 @@\n"
        "     else:\n"
        "+        y = 2\n"
        "         return y\n"
    )
    r = analyze_file("frag.py", patch)
    assert r.language == "python"
    assert r.n_hunks == 1
    assert r.n_hunks_parsed == 0
    assert r.ast_node_count == 0
    assert r.cfg_node_count == 0


def test_analyze_file_python_bare_truthy_condition_parses_but_cfg_fails():
    """Regression anchor for the documented staticfg limitation: `invert()`
    (builder.py, used to label an if/while's "false" exit edge) guards its
    fallback with `type(node) == ast.NameConstant`, an attribute Python's
    `ast` module has since removed -- so evaluating that check itself
    raises AttributeError for any condition that isn't a direct comparison
    (Compare/invertible BinOp). `if x:` is exactly such a condition (a bare
    Name), so this hunk parses as valid Python but its CFG build crashes.
    `if x > 0:` (a Compare) does not trigger this and is covered by the
    "valid snippet" test above. If staticfg is ever upgraded/replaced and
    this starts passing where it used to fail, the fidelity write-up's
    numbers need to be revisited too."""
    patch = (
        "@@ -1,2 +1,4 @@\n"
        "+def foo(x):\n"
        "+    if x:\n"
        "+        return x\n"
        "+    return 0\n"
    )
    r = analyze_file("mod.py", patch)
    assert r.language == "python"
    assert r.n_hunks_parsed == 1  # ast.parse succeeds fine on a bare-Name condition
    assert r.n_hunks_cfg_ok == 0  # staticfg's CFG builder crashes on it
    assert r.ast_node_count > 0  # AST-derived counts are unaffected
    assert r.cfg_node_count == 0


# --------------------------------------------------------------------------- #
# analyze_file -- TypeScript / C# (tree-sitter + lightweight CFG approx)
# --------------------------------------------------------------------------- #

def test_analyze_file_typescript_counts_decisions():
    patch = (
        "@@ -1,2 +1,7 @@\n"
        " function foo(x) {\n"
        "-  return x;\n"
        "+  if (x > 0) {\n"
        "+    return x;\n"
        "+  }\n"
        "+  for (const y of [1, 2, 3]) { x += y; }\n"
        "+  return -x;\n"
        " }\n"
    )
    r = analyze_file("foo.ts", patch)
    assert r.language == "typescript"
    assert r.n_hunks_parsed == 1
    assert r.num_functions == 1
    # 2 decisions (if, for-of) -> nodes = D+1, edges = 2D, cyclomatic = D+1
    assert r.cfg_node_count == 3
    assert r.cfg_edge_count == 4
    assert r.cyclomatic_proxy == 3


def test_analyze_file_csharp_counts_decisions():
    patch = (
        "@@ -1,2 +1,6 @@\n"
        " int Bar(int x) {\n"
        "-    return x;\n"
        "+    if (x > 0) { return x; }\n"
        "+    switch (x) { case 1: break; default: break; }\n"
        "+    return -x;\n"
        " }\n"
    )
    r = analyze_file("Foo.cs", patch)
    assert r.language == "csharp"
    assert r.n_hunks_parsed == 1
    # 1 if + 2 switch_section (case, default) = 3 decisions
    assert r.cfg_node_count == 4
    assert r.cfg_edge_count == 6
    assert r.cyclomatic_proxy == 4


def test_analyze_file_unsupported_language_is_a_noop():
    r = analyze_file("README.md", "@@ -1 +1 @@\n-old\n+new\n")
    assert r.language is None
    assert r.n_hunks == 0
    assert r.ast_node_count == 0


# --------------------------------------------------------------------------- #
# PR-level aggregation
# --------------------------------------------------------------------------- #

def _files_changed_fixture():
    ts_patch = (
        "@@ -1,2 +1,4 @@\n"
        " function foo(x) {\n"
        "-  return x;\n"
        "+  if (x > 0) { return x; }\n"
        "+  return -x;\n"
        " }\n"
    )
    cs_patch = (
        "@@ -1,2 +1,4 @@\n"
        " int Bar(int x) {\n"
        "-    return x;\n"
        "+    if (x > 0) { return x; }\n"
        "+    return -x;\n"
        " }\n"
    )
    return pd.DataFrame([
        {"pr_id": "PR1", "filename": "foo.py", "patch": None},       # python, no patch data
        {"pr_id": "PR1", "filename": "bar.md", "patch": "@@ -1 +1 @@\n-a\n+b\n"},  # unsupported
        {"pr_id": "PR2", "filename": "a.ts", "patch": ts_patch},
        {"pr_id": "PR2", "filename": "b.cs", "patch": cs_patch},
        {"pr_id": "PR3", "filename": "only.md", "patch": "@@ -1 +1 @@\n-a\n+b\n"},
    ])


def test_aggregate_to_pr_features_combines_files_and_flags_parseable():
    file_table = analyze_files(_files_changed_fixture())
    pr_table = aggregate_to_pr_features(file_table).set_index("pr_id")

    assert set(S_FEATURE_COLUMNS).issubset(pr_table.columns)

    # PR1: has a .py file (parseable language) even though its patch is null
    assert pr_table.loc["PR1", "s_has_parseable_code"] == 1
    assert pr_table.loc["PR1", "s_ast_node_count"] == 0

    # PR2: sums TS + C# structure counts across both files
    assert pr_table.loc["PR2", "s_has_parseable_code"] == 1
    assert pr_table.loc["PR2", "s_ast_node_count"] > 0
    assert pr_table.loc["PR2", "diag_parse_rate"] == 1.0

    # PR3: only markdown -> no parseable code at all
    assert pr_table.loc["PR3", "s_has_parseable_code"] == 0
    assert pr_table.loc["PR3", "s_ast_node_count"] == 0
    assert math.isnan(pr_table.loc["PR3", "diag_parse_rate"])


def test_aggregate_to_pr_features_empty_input():
    empty = pd.DataFrame(columns=["pr_id", "filename", "patch"])
    file_table = analyze_files(empty)
    pr_table = aggregate_to_pr_features(file_table)
    assert list(pr_table.columns[:1]) == ["pr_id"]
    assert len(pr_table) == 0


# --------------------------------------------------------------------------- #
# fidelity_report
# --------------------------------------------------------------------------- #

def test_fidelity_report_breaks_down_per_language():
    file_table = analyze_files(_files_changed_fixture())
    report = fidelity_report(file_table)

    assert report["n_files_total"] == 5
    assert report["n_files_unsupported_language"] == 2  # the two .md files
    assert report["typescript"]["n_hunks_attempted"] == 1
    assert report["typescript"]["hunk_parse_rate"] == 1.0
    assert report["csharp"]["n_hunks_attempted"] == 1
    assert report["python"]["n_files_no_patch"] == 1
    assert report["python"]["hunk_parse_rate"] is None  # zero hunks attempted
