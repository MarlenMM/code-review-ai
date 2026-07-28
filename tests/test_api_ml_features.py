"""Tests for `src/api/ml_features.py`."""

from src.api.diff_parsing import parse_unified_diff
from src.api.ml_features import compute_v1_features
from src.features.feature_extraction import V1_FEATURE_COLUMNS

PY_DIFF = """\
diff --git a/src/foo.py b/src/foo.py
index 1234567..89abcde 100644
--- a/src/foo.py
+++ b/src/foo.py
@@ -1,2 +1,3 @@
 def foo():
-    return 1
+    return 2
+    # extra
"""

MD_DIFF = """\
diff --git a/README.md b/README.md
index 1234567..89abcde 100644
--- a/README.md
+++ b/README.md
@@ -1 +1 @@
-old text
+new text
"""


def test_output_keys_match_v1_feature_columns():
    files = parse_unified_diff(PY_DIFF)
    features = compute_v1_features(files)
    assert set(features.keys()) == set(V1_FEATURE_COLUMNS)
    assert all(isinstance(v, float) for v in features.values())


def test_modification_features_from_python_diff():
    files = parse_unified_diff(PY_DIFF)
    features = compute_v1_features(files)
    assert features["m_additions"] == 2
    assert features["m_deletions"] == 1
    assert features["m_churn"] == 3
    assert features["m_net_change"] == 1
    assert features["m_changed_files"] == 1
    assert features["m_files_modified"] == 1
    assert features["m_files_added"] == 0
    assert features["m_num_hunks"] == 1
    # no commit_messages given -> the documented "assume one commit" default
    assert features["m_num_commits"] == 1


def test_structure_features_python_is_parseable():
    files = parse_unified_diff(PY_DIFF)
    features = compute_v1_features(files)
    assert features["s_has_parseable_code"] == 1.0
    assert features["s_ast_node_count"] > 0
    assert features["s_num_functions_touched"] >= 1


def test_structure_features_non_code_file():
    files = parse_unified_diff(MD_DIFF)
    features = compute_v1_features(files)
    assert features["s_has_parseable_code"] == 0.0
    assert features["s_ast_node_count"] == 0.0
    assert features["s_cfg_node_count"] == 0.0


def test_textual_features_from_title_and_description():
    files = parse_unified_diff(PY_DIFF)
    features = compute_v1_features(files, title="Fix foo", description="Fixes the return value.")
    assert features["t_title_len"] == len("Fix foo")
    assert features["t_title_wordcount"] == 2
    assert features["t_body_len"] == len("Fixes the return value.")
    assert features["t_body_is_empty"] == 0.0


def test_textual_features_empty_description_is_marked():
    files = parse_unified_diff(PY_DIFF)
    features = compute_v1_features(files)
    assert features["t_body_is_empty"] == 1.0
    assert features["t_body_len"] == 0.0


def test_commit_messages_populate_commit_features():
    files = parse_unified_diff(PY_DIFF)
    features = compute_v1_features(files, commit_messages=["short", "a longer commit message"])
    assert features["m_num_commits"] == 2
    assert features["t_commit_msg_len_first"] == len("short")
    assert features["t_commit_msg_len_total"] == len("short") + len("a longer commit message")
    expected_mean = (len("short") + len("a longer commit message")) / 2
    assert features["t_commit_msg_len_mean"] == expected_mean


def test_no_commit_messages_defaults_commit_text_features_to_zero():
    files = parse_unified_diff(PY_DIFF)
    features = compute_v1_features(files)
    assert features["t_commit_msg_len_first"] == 0.0
    assert features["t_commit_msg_len_mean"] == 0.0
    assert features["t_commit_msg_len_total"] == 0.0


def test_empty_file_list_yields_zeroed_features():
    features = compute_v1_features([])
    assert features["m_changed_files"] == 0.0
    assert features["m_additions"] == 0.0
    assert features["m_avg_file_churn"] == 0.0
    assert features["m_frac_added"] == 0.0
    assert features["s_has_parseable_code"] == 0.0
    # still assumes one commit even with zero files, per the documented default
    assert features["m_num_commits"] == 1.0
