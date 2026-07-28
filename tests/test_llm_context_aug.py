"""Tests for Experiment 4 context augmentation (`src/llm/prompts/context_aug.py`).

These pin the contracts Step 20 depends on: each of the five §4.7.2 tiers
renders the sections it promises (with an explicit marker when a slot is empty,
never a silent drop); the single-shot rendering and the multi-turn split carry
the *same* information; multi-turn is undefined on the bare-diff tier; and the
issue-reference / lightweight-repo-context helpers behave on real-shaped input.
"""

import pytest

from src.llm.prompts.context_aug import (
    AugmentedContext,
    Exp4ContextTier,
    MULTI_TURN_TIERS,
    build_lightweight_repo_context,
    extract_issue_references,
    render_tier,
    split_for_multi_turn,
)

FULL = AugmentedContext(
    diff="--- a.py (modified)\n@@ -1 +1 @@ def f\n-x=1\n+x = 1\n",
    pr_description="Fix the spacing in f().",
    commit_message="1. fix spacing",
    repo_context="File a.py contains def f(): ...",
    issue_text="Issue #12: inconsistent spacing.",
    historical_comments='On a past PR, a reviewer said: "please add a test".',
    repo="microsoft/vscode",
    title="Fix spacing",
)


# --------------------------------------------------------------------------- #
# render_tier
# --------------------------------------------------------------------------- #

def test_diff_tier_is_diff_only():
    pc = render_tier(FULL, Exp4ContextTier.DIFF)
    assert pc.context_kind == "diff"
    assert "def f" in pc.context_text
    # no augmentation section leaks into the bare-diff tier
    assert "PR description" not in pc.context_text
    assert "Repository context" not in pc.context_text
    assert "Linked issue" not in pc.context_text


@pytest.mark.parametrize("tier,heading,needle", [
    (Exp4ContextTier.DIFF_PR_DESCRIPTION, "PR description", "Fix the spacing"),
    (Exp4ContextTier.DIFF_REPO_CONTEXT, "Repository context", "contains def f"),
    (Exp4ContextTier.DIFF_ISSUE, "Linked issue", "inconsistent spacing"),
])
def test_single_augmentation_tiers(tier, heading, needle):
    pc = render_tier(FULL, tier)
    assert heading in pc.context_text
    assert needle in pc.context_text
    # the diff is always present, and comes last (framing before the raw change)
    assert "Diff:" in pc.context_text
    assert pc.context_text.index(heading) < pc.context_text.index("Diff:")


def test_complete_tier_stacks_every_augmentation():
    pc = render_tier(FULL, Exp4ContextTier.COMPLETE_SE_CONTEXT)
    for heading in ["PR description", "Commit messages", "Linked issue",
                    "Repository context",
                    "Historical review comments on similar past PRs"]:
        assert heading in pc.context_text
    assert "please add a test" in pc.context_text


def test_empty_slot_renders_explicit_marker_not_silent_drop():
    sparse = AugmentedContext(diff="d", repo="r")  # no augmentation slots set
    pc = render_tier(sparse, Exp4ContextTier.DIFF_ISSUE)
    # the tier promised issue info; absent data must be visible, not vanish
    assert "Linked issue" in pc.context_text
    assert "no linked issue information available" in pc.context_text


def test_render_tier_carries_repo_and_title():
    pc = render_tier(FULL, Exp4ContextTier.DIFF_PR_DESCRIPTION)
    assert pc.repo == "microsoft/vscode" and pc.title == "Fix spacing"


def test_unknown_tier_rejected():
    with pytest.raises(ValueError):
        # a value that is not an Exp4ContextTier member
        render_tier(FULL, "not_a_tier")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# split_for_multi_turn
# --------------------------------------------------------------------------- #

def test_multi_turn_split_diff_first_augmentation_second():
    base, aug = split_for_multi_turn(FULL, Exp4ContextTier.DIFF_REPO_CONTEXT)
    assert "Diff:" in base and "def f" in base
    assert "Repository context" in aug and "contains def f" in aug
    # the diff-only turn must NOT contain the withheld augmentation
    assert "Repository context" not in base


def test_multi_turn_and_single_shot_carry_same_info():
    """The union of the two multi-turn turns must equal the single-shot tier's
    sections -- only the ordering differs -- so the two strategies see the same
    context, a property the design write-up claims and Step 21 relies on."""
    tier = Exp4ContextTier.COMPLETE_SE_CONTEXT
    base, aug = split_for_multi_turn(FULL, tier)
    single = render_tier(FULL, tier).context_text
    for needle in ["Fix the spacing", "fix spacing", "inconsistent spacing",
                   "contains def f", "please add a test", "def f"]:
        assert needle in single
        assert needle in (base + "\n" + aug)


def test_multi_turn_undefined_on_bare_diff():
    with pytest.raises(ValueError, match="multi-turn is undefined"):
        split_for_multi_turn(FULL, Exp4ContextTier.DIFF)


def test_multi_turn_tiers_excludes_diff_only():
    assert Exp4ContextTier.DIFF not in MULTI_TURN_TIERS
    assert set(MULTI_TURN_TIERS) == set(Exp4ContextTier) - {Exp4ContextTier.DIFF}


# --------------------------------------------------------------------------- #
# extract_issue_references
# --------------------------------------------------------------------------- #

def test_extract_closing_and_mentioned_refs():
    refs = extract_issue_references(
        "Fixes #42. Related to #7. See https://github.com/o/r/issues/99 too."
    )
    assert 42 in refs.closing and 99 in refs.closing
    assert refs.mentioned == (7,)                 # bare mention, no closing keyword
    assert refs.best == refs.closing              # closing refs preferred


def test_extract_dedupes_and_prefers_closing_over_bare():
    # #42 appears both as a closing ref and a bare mention -> only in `closing`
    refs = extract_issue_references("Closes #42. Also see #42 and #42.")
    assert refs.closing == (42,)
    assert 42 not in refs.mentioned


def test_extract_from_multiple_texts_body_and_commits():
    refs = extract_issue_references("body text no ref", "commit: resolve #5")
    assert 5 in refs.closing


def test_extract_no_refs_returns_empty():
    refs = extract_issue_references("A plain description with no issue numbers.")
    assert refs.closing == () and refs.mentioned == () and refs.best == ()


def test_extract_ignores_none_inputs():
    refs = extract_issue_references(None, "Fixes #3", None)
    assert refs.closing == (3,)


# --------------------------------------------------------------------------- #
# build_lightweight_repo_context
# --------------------------------------------------------------------------- #

def test_lightweight_repo_context_lists_files_and_headings():
    txt = build_lightweight_repo_context([
        {"filename": "src/a/x.ts", "status": "modified",
         "patch": "@@ -1,2 +1,3 @@ class Foo\n+a\n"},
        {"filename": "src/b/y.ts", "status": "added",
         "patch": "@@ -0,0 +1,2 @@ function bar()\n+b\n"},
    ])
    assert "src/a/x.ts (modified)" in txt
    assert "class Foo" in txt and "function bar()" in txt
    assert "Directories spanned" in txt          # >1 directory
    # honest about being an approximation, not a real call graph
    assert "Approximate" in txt


def test_lightweight_repo_context_empty_when_no_files():
    assert build_lightweight_repo_context([]) == ""


def test_lightweight_repo_context_respects_max_files():
    files = [{"filename": f"f{i}.py", "status": "modified", "patch": ""} for i in range(50)]
    txt = build_lightweight_repo_context(files, max_files=5)
    assert txt.count("(modified)") == 5
