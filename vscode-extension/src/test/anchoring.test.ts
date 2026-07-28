import assert from "node:assert/strict";
import { test } from "node:test";
import { anchorComment, anchorComments } from "../anchoring";
import type { FileChange } from "../diffMap";

const greeter: FileChange = {
  path: "src/greeter.py",
  addedLines: [
    { line: 2, text: "    if not name:" },
    { line: 3, text: '        return "Hello, stranger"' },
    { line: 4, text: '    return f"Hello, {name}!"' },
  ],
  contextLines: [{ line: 1, text: "def greet(name):" }],
  firstChangedLine: 2,
};

const utils: FileChange = {
  path: "src/utils.ts",
  addedLines: [
    { line: 10, text: "export function slugify(input: string) {" },
    { line: 11, text: "  return input.toLowerCase();" },
  ],
  contextLines: [{ line: 9, text: "// string helpers" }],
  firstChangedLine: 10,
};

const TWO_FILES = [greeter, utils];

test("a backticked symbol anchors to the exact added line", () => {
  const loc = anchorComment("The `slugify` helper should handle empty input.", TWO_FILES);
  assert.deepEqual(loc, { path: "src/utils.ts", line: 10, precision: "line" });
});

test("a named file with no narrower hint anchors to its first change", () => {
  const loc = anchorComment("`src/greeter.py` needs a docstring.", TWO_FILES);
  assert.deepEqual(loc, { path: "src/greeter.py", line: 2, precision: "file" });
});

test("a bare (un-backticked) filename is matched too", () => {
  const loc = anchorComment("Consider adding tests for greeter.py.", TWO_FILES);
  assert.equal(loc?.path, "src/greeter.py");
});

test("a basename matches a nested path", () => {
  const loc = anchorComment("`utils.ts` is missing error handling.", TWO_FILES);
  assert.equal(loc?.path, "src/utils.ts");
});

test("a file mention plus a symbol prefers the symbol's line", () => {
  const loc = anchorComment(
    "In `src/utils.ts`, the `toLowerCase` call drops non-ASCII characters.",
    TWO_FILES,
  );
  assert.deepEqual(loc, { path: "src/utils.ts", line: 11, precision: "line" });
});

test("an explicit line number is honoured when the diff really added it", () => {
  const loc = anchorComment("`src/greeter.py` line 3 returns a magic string.", TWO_FILES);
  assert.deepEqual(loc, { path: "src/greeter.py", line: 3, precision: "line" });
});

test("a hallucinated line number is ignored, not followed", () => {
  // Line 999 was never added by this diff; fall back to the file anchor
  // rather than sending the user to a fabricated location.
  const loc = anchorComment("`src/greeter.py` line 999 is wrong.", TWO_FILES);
  assert.deepEqual(loc, { path: "src/greeter.py", line: 2, precision: "file" });
});

test("with exactly one changed file, an unnamed comment is inferred to it", () => {
  const loc = anchorComment("Add a test for the empty-input case.", [greeter]);
  assert.deepEqual(loc, { path: "src/greeter.py", line: 2, precision: "inferred-file" });
});

test("with several changed files, an unnamed comment stays unanchored", () => {
  assert.equal(anchorComment("This change looks reasonable overall.", TWO_FILES), null);
});

test("a symbol present in only one file anchors even without a file mention", () => {
  const loc = anchorComment("`slugify` should be exported from the index.", TWO_FILES);
  assert.deepEqual(loc, { path: "src/utils.ts", line: 10, precision: "line" });
});

test("an ambiguous symbol present in several files is rejected", () => {
  const a: FileChange = {
    path: "a.ts", addedLines: [{ line: 1, text: "const shared = 1;" }],
    contextLines: [], firstChangedLine: 1,
  };
  const b: FileChange = {
    path: "b.ts", addedLines: [{ line: 5, text: "const shared = 2;" }],
    contextLines: [], firstChangedLine: 5,
  };
  assert.equal(anchorComment("`shared` is declared twice.", [a, b]), null);
});

test("a symbol on an unchanged context line still anchors to that line", () => {
  // `greet` is defined on context line 1 -- the diff didn't add it, but a
  // reviewer naming the enclosing function is the common case, and line 1
  // is a better destination than the file's first *changed* line.
  const loc = anchorComment("The `greet` function in `src/greeter.py` needs a docstring.", TWO_FILES);
  assert.deepEqual(loc, { path: "src/greeter.py", line: 1, precision: "line" });
});

test("an added line outranks a context line for the same symbol", () => {
  const file: FileChange = {
    path: "x.py",
    addedLines: [{ line: 20, text: "def helper(): pass" }],
    contextLines: [{ line: 5, text: "# helper lives below" }],
    firstChangedLine: 20,
  };
  const loc = anchorComment("`helper` should be private.", [file]);
  assert.deepEqual(loc, { path: "x.py", line: 20, precision: "line" });
});

test("a context-line symbol unique to one file anchors without a file mention", () => {
  const loc = anchorComment("The `greet` helper should validate its input.", TWO_FILES);
  assert.deepEqual(loc, { path: "src/greeter.py", line: 1, precision: "line" });
});

test("very short backticked tokens are not treated as symbols", () => {
  // "if" appears in greeter.py line 2, but a 2-char token would match
  // almost any line -- too weak to claim a line-precise anchor.
  const loc = anchorComment("`if` statements should be simplified.", TWO_FILES);
  assert.equal(loc, null);
});

test("no changed files means nothing can be anchored", () => {
  assert.equal(anchorComment("`src/greeter.py` is broken.", []), null);
});

test("anchorComments preserves order and text", () => {
  const anchored = anchorComments(
    ["`slugify` needs docs.", "This is a general remark."],
    TWO_FILES,
  );
  assert.equal(anchored.length, 2);
  assert.equal(anchored[0].text, "`slugify` needs docs.");
  assert.equal(anchored[0].location?.path, "src/utils.ts");
  assert.equal(anchored[1].location, null);
});
