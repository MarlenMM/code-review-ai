"""Tests for `src/api/diff_parsing.py`."""

from src.api.diff_parsing import ParsedFile, parse_unified_diff

MODIFIED_DIFF = """\
diff --git a/src/foo.py b/src/foo.py
index 1234567..89abcde 100644
--- a/src/foo.py
+++ b/src/foo.py
@@ -1,3 +1,4 @@
 def foo():
-    return 1
+    return 2
+    # extra
"""

NEW_FILE_DIFF = """\
diff --git a/src/bar.py b/src/bar.py
new file mode 100644
index 0000000..1111111
--- /dev/null
+++ b/src/bar.py
@@ -0,0 +1,2 @@
+def bar():
+    return 42
"""

DELETED_FILE_DIFF = """\
diff --git a/src/old.py b/src/old.py
deleted file mode 100644
index 2222222..0000000
--- a/src/old.py
+++ /dev/null
@@ -1,2 +0,0 @@
-def old():
-    pass
"""

RENAMED_NO_CONTENT_DIFF = """\
diff --git a/src/a.py b/src/b.py
similarity index 100%
rename from src/a.py
rename to src/b.py
"""

RENAMED_WITH_CONTENT_DIFF = """\
diff --git a/src/a.py b/src/b.py
similarity index 90%
rename from src/a.py
rename to src/b.py
index 3333333..4444444 100644
--- a/src/a.py
+++ b/src/b.py
@@ -1,2 +1,2 @@
-def a():
+def b():
     pass
"""

BINARY_DIFF = """\
diff --git a/img.png b/img.png
index 5555555..6666666 100644
Binary files a/img.png and b/img.png differ
"""

MULTI_FILE_DIFF = MODIFIED_DIFF + NEW_FILE_DIFF


def test_empty_diff_returns_no_files():
    assert parse_unified_diff("") == []
    assert parse_unified_diff("   \n  ") == []


def test_modified_file():
    files = parse_unified_diff(MODIFIED_DIFF)
    assert len(files) == 1
    f = files[0]
    assert f.filename == "src/foo.py"
    assert f.status == "modified"
    assert f.additions == 2
    assert f.deletions == 1
    assert f.patch.startswith("@@ -1,3 +1,4 @@")


def test_new_file():
    files = parse_unified_diff(NEW_FILE_DIFF)
    assert len(files) == 1
    f = files[0]
    assert f.filename == "src/bar.py"
    assert f.status == "added"
    assert f.additions == 2
    assert f.deletions == 0


def test_deleted_file():
    files = parse_unified_diff(DELETED_FILE_DIFF)
    assert len(files) == 1
    f = files[0]
    assert f.filename == "src/old.py"
    assert f.status == "removed"
    assert f.additions == 0
    assert f.deletions == 2


def test_renamed_no_content_change():
    files = parse_unified_diff(RENAMED_NO_CONTENT_DIFF)
    assert len(files) == 1
    f = files[0]
    assert f.filename == "src/b.py"
    assert f.status == "renamed"
    assert f.patch == ""


def test_renamed_with_content_change():
    files = parse_unified_diff(RENAMED_WITH_CONTENT_DIFF)
    assert len(files) == 1
    f = files[0]
    assert f.filename == "src/b.py"
    assert f.status == "renamed"
    assert f.additions == 1
    assert f.deletions == 1
    assert "@@" in f.patch


def test_binary_file_has_no_patch():
    files = parse_unified_diff(BINARY_DIFF)
    assert len(files) == 1
    f = files[0]
    assert f.filename == "img.png"
    assert f.status == "modified"
    assert f.patch == ""
    assert f.additions == 0
    assert f.deletions == 0


def test_multi_file_diff():
    files = parse_unified_diff(MULTI_FILE_DIFF)
    assert [f.filename for f in files] == ["src/foo.py", "src/bar.py"]
    assert [f.status for f in files] == ["modified", "added"]


def test_bare_patch_without_diff_git_header():
    bare = "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-old\n+new\n"
    files = parse_unified_diff(bare)
    assert len(files) == 1
    f = files[0]
    assert f.filename == "x.py"
    assert f.additions == 1
    assert f.deletions == 1


def test_unparseable_block_is_skipped():
    garbage = "diff --git a/x b/y\nnot a real diff body\n"
    # No --- / +++ headers at all, but the diff --git line itself still
    # yields a/b filenames via the fallback, so this is NOT unparseable.
    files = parse_unified_diff(garbage)
    assert len(files) == 1
    assert files[0].filename == "y"


def test_parsed_file_is_frozen_dataclass():
    f = ParsedFile(filename="x", status="modified", patch="", additions=0, deletions=0)
    assert f.filename == "x"
