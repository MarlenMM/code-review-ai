import assert from "node:assert/strict";
import { test } from "node:test";
import { parseDiffMap } from "../diffMap";

const SIMPLE = `diff --git a/greeter.py b/greeter.py
index d282150..d7cef96 100644
--- a/greeter.py
+++ b/greeter.py
@@ -1,2 +1,4 @@
 def greet(name):
-    return "Hello, " + name
+    if not name:
+        return "Hello, stranger"
+    return f"Hello, {name}!"
`;

test("maps added lines to post-change line numbers", () => {
  const files = parseDiffMap(SIMPLE);
  assert.equal(files.length, 1);
  const file = files[0];
  assert.equal(file.path, "greeter.py");
  assert.deepEqual(
    file.addedLines.map((l) => l.line),
    [2, 3, 4],
  );
  assert.equal(file.addedLines[0].text, "    if not name:");
  assert.equal(file.firstChangedLine, 2);
});

test("a removed line does not advance the new-file cursor", () => {
  // The removed `return "Hello, " + name` sat at old line 2; the three
  // added lines must still be numbered 2,3,4 in the new file, not 3,4,5.
  const file = parseDiffMap(SIMPLE)[0];
  assert.equal(file.addedLines[2].line, 4);
});

test("honours a hunk header's +start offset", () => {
  const diff = `diff --git a/x.py b/x.py
--- a/x.py
+++ b/x.py
@@ -40,3 +40,4 @@ def existing():
 context one
 context two
+added here
 context three
`;
  const file = parseDiffMap(diff)[0];
  assert.deepEqual(file.addedLines, [{ line: 42, text: "added here" }]);
});

test("handles multiple hunks in one file", () => {
  const diff = `diff --git a/x.py b/x.py
--- a/x.py
+++ b/x.py
@@ -1,2 +1,3 @@
 a
+first
 b
@@ -20,2 +21,3 @@
 c
+second
 d
`;
  const file = parseDiffMap(diff)[0];
  assert.deepEqual(
    file.addedLines.map((l) => [l.line, l.text]),
    [
      [2, "first"],
      [22, "second"],
    ],
  );
  assert.equal(file.firstChangedLine, 2);
});

test("handles multiple files", () => {
  const diff = `diff --git a/a.py b/a.py
--- a/a.py
+++ b/a.py
@@ -1 +1,2 @@
 x
+from a
diff --git a/b/nested.ts b/b/nested.ts
--- a/b/nested.ts
+++ b/b/nested.ts
@@ -1 +1,2 @@
 y
+from b
`;
  const files = parseDiffMap(diff);
  assert.deepEqual(files.map((f) => f.path), ["a.py", "b/nested.ts"]);
  assert.equal(files[1].addedLines[0].text, "from b");
});

test("a new file records its added lines from line 1", () => {
  const diff = `diff --git a/new.py b/new.py
new file mode 100644
--- /dev/null
+++ b/new.py
@@ -0,0 +1,2 @@
+def a():
+    pass
`;
  const file = parseDiffMap(diff)[0];
  assert.equal(file.path, "new.py");
  assert.deepEqual(file.addedLines.map((l) => l.line), [1, 2]);
});

test("a removal-only file still yields a usable anchor line", () => {
  const diff = `diff --git a/gone.py b/gone.py
--- a/gone.py
+++ b/gone.py
@@ -5,2 +5,0 @@
-removed one
-removed two
`;
  const file = parseDiffMap(diff)[0];
  assert.equal(file.addedLines.length, 0);
  assert.ok(file.firstChangedLine >= 1);
});

test("`+++` header is not mistaken for an added line", () => {
  const file = parseDiffMap(SIMPLE)[0];
  assert.ok(file.addedLines.every((l) => !l.text.startsWith("+ b/")));
  assert.ok(file.addedLines.every((l) => !l.text.includes("greeter.py")));
});

test("empty or whitespace input yields no files", () => {
  assert.deepEqual(parseDiffMap(""), []);
  assert.deepEqual(parseDiffMap("   \n "), []);
});
