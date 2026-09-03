# VSCode Extension — Core Scaffold, Design & Verification (Steps 24–27)

*Deliverable: `vscode-extension/` — a "Review Current Changes" command that
reads the workspace's `git diff`, calls the Step 23 backend, and presents
the result. **Step 24** built the functional core (§1–§4); **Step 25**
added the polish the plan asks for — a merge-probability gauge,
click-to-jump review comments, and inline highlights (§6–§8); **Step 26**
packaged and verified the `.vsix` (§9–§11); **Step 27** was a design pass
over the panel itself — the loading, empty and error states it never had,
a real type and spacing scale, meaning-carrying colour, and the gauge's
calibration marks (§12 onward, with the rules in
`docs/design-constraints.md`). Tests: **66** unit tests (`npm test`) + 7
real end-to-end integration tests (`npm run test:integration`), all
passing.*

---

## 1. What was built

```
vscode-extension/
├── package.json          -- manifest: one command, two settings
├── tsconfig.json
├── .vscode/launch.json    -- F5 "Run Extension" (Extension Development Host)
├── .vscode/tasks.json     -- background `npm run watch` build task
├── .vscodeignore          -- excludes src/, node_modules/, tests from a future .vsix
├── README.md
└── src/
    ├── extension.ts       -- thin glue: registers the command, wires the 3 modules below
    ├── git.ts             -- getWorkingDiff(cwd): runs `git diff`
    ├── apiClient.ts        -- reviewDiff(url, request): POST /review, typed to match src/api/main.py
    ├── panel.ts             -- showReviewPanel(): plain webview, no JS/gauge yet
    └── test/
        ├── git.test.ts / apiClient.test.ts   -- plain-Node unit tests (mocked, no VS Code needed)
        ├── runTest.ts                         -- @vscode/test-electron entry point
        └── suite/{index,extension}.test.ts    -- Mocha suite run INSIDE a real VS Code instance
```

Command: **"Code Review AI: Review Current Changes"**
(`codeReviewAi.reviewCurrentChanges`). Settings: `codeReviewAi.backendUrl`
(default `http://127.0.0.1:8000`) and `codeReviewAi.mode` (`fast`/`deep`,
default `fast`, matching the backend's own default from Step 23).

This is explicitly the **core** scaffold per plan §7.2's build order
("functional core first ... then polish only as time allows") and Section
9 row 24's own scope ("Scaffold `vscode-extension/`... a simple panel").
The probability gauge, click-to-jump comments, and inline decorations are
Step 25's job; the `.vsix` packaging and repo-wide README/demo-GIF polish
are Step 26's.

---

## 2. Design decisions

### 2.1 `git diff`, not `git diff --cached` or a merge-base diff

Plan §7.2 is literal: *"reads the working diff (`git diff`) of the
currently open workspace."* `git.ts` runs exactly `git diff` (working tree
vs. the index) in the first workspace folder. Staged-only diffs or a diff
against a different base (e.g. `main`) are a reasonable future extension,
not scope creep to add now.

### 2.2 Merge probability always shown; review comments only in `deep` mode

Mirrors the backend's own design (`reports/api_design.md` §1): the request
body only ever sets `mode`, nothing else (no title/description/commit
messages) — the command's whole job is reviewing the *current* diff, and
the backend already degrades sensibly (empty title/description) without
them. `panel.ts` renders `review_comments: null` as an explicit "run in
deep mode" hint rather than an empty list, so a `fast`-mode user isn't left
wondering whether the LLM pass silently failed.

### 2.3 A plain, non-interactive webview panel

`showReviewPanel` sets `enableScripts: false` and renders static HTML
(VS Code's own `--vscode-*` CSS variables for native-looking theming, a
colored MERGE/CLOSE badge, a bulleted comment list). No probability gauge
graphic, no click-to-jump-to-line, no `postMessage` interactivity — those
need a webview script and are explicitly Step 25's "polish" scope. Keeping
`enableScripts: false` for now is also the more conservative default (no
script execution surface) until Step 25 actually needs one.

### 2.4 No runtime npm dependencies

`apiClient.ts` uses the extension host's global `fetch` (available since
VS Code's bundled Node 18+) rather than adding `axios`/`node-fetch`;
`git.ts` uses Node's built-in `child_process`. `package.json` declares only
`devDependencies` (TypeScript, test tooling) — nothing needs to ship in a
future `.vsix` beyond the compiled `out/*.js`, which `.vscodeignore`
already reflects (`node_modules/**` excluded).

### 2.5 Two-tier testing, because one tier can't cover this

* **Pure logic** (`git.ts`, `apiClient.ts`) is unit-tested with **zero**
  VS Code dependency: both take an injectable function (`ExecFn`,
  `FetchFn`) defaulting to the real implementation, so tests run under
  plain `node --test` — fast, no VS Code download, matches this project's
  existing Python test style (dependency injection over mocking a global).
* **Wiring** (`extension.ts`: activation, command registration, the real
  webview panel) genuinely needs a real `vscode` API — no amount of
  mocking proves `registerCommand`/`createWebviewPanel` actually work.
  `@vscode/test-electron` launches a real, unmodified VS Code build with
  this extension loaded and runs a Mocha suite *inside* that process
  (`src/test/suite/`) — this is the standard, correct tool for this job,
  and — unlike GUI automation — needs no mouse/keyboard driving at all
  (relevant here: see §3).

---

## 3. Why GUI automation wasn't used

The natural first instinct for "does the extension actually work" is to
open real VS Code and click through it (as was done for the FastAPI
backend with `curl` in Step 23). This was attempted and abandoned for a
concrete reason, not skipped by default:

VS Code is installed on this machine (`/Applications/Visual Studio
Code.app`), and computer-use tooling can see and screenshot it — but
requesting control access showed that IDE/terminal-category apps are only
ever grantable at **"click" tier**: visible and left-clickable, but typing,
key presses, and shortcuts are blocked. Verifying this command requires
opening the Command Palette and *typing* "Review Current Changes" (there's
no menu item), which click-only access cannot do. Retrying the grant
doesn't change the tier — it's a hard rule for this app category, not a
per-request choice.

`@vscode/test-electron` is not a fallback for that gap, it's the *better*
tool anyway: it drives the extension host directly and programmatically
(activate, `executeCommand`, inspect `vscode.window.tabGroups` for the
resulting panel), which is more precise and more standard for extension
testing than asserting on pixels ever would be.

---

## 4. Verification performed

### 4.1 Compile + unit tests

```
cd vscode-extension && npm run compile   # tsc -p ./, zero errors
npm test                                  # node --test out/test/*.test.js
```

8/8 passing: `git.ts` (real stdout passthrough, empty-tree case, stderr
surfaced on failure, message fallback when stderr absent) and
`apiClient.ts` (parses a 200, posts the exact JSON body to `/review`
regardless of trailing slash on the base URL, throws a readable error on a
non-2xx response, wraps a network failure with the backend URL).

### 4.2 Real end-to-end integration test (`npm run test:integration`)

Set up a throwaway git repo (outside this project, in the session
scratchpad) with a real committed file and an uncommitted edit, and started
the real Step 23 backend (`uvicorn src.api.main:app`). Then:

```
CODE_REVIEW_AI_TEST_WORKSPACE=<path to the throwaway repo> \
  npm run test:integration
```

`@vscode/test-electron` downloaded a real VS Code 1.130.0 build (one-time,
~315 MB, cached under `vscode-extension/.vscode-test/` — gitignored) and
launched it with this extension loaded via `--extensionDevelopmentPath`
and the throwaway repo opened as the workspace. **One macOS-specific
hurdle hit and fixed**: the default run failed immediately with `listen
EINVAL ... /1.13-main.sock` — this project's absolute path is long enough
that VS Code's default `--user-data-dir` (nested under it) produces a Unix
domain socket path over macOS's 103-character limit. Fixed by pinning
`--user-data-dir`/`--extensions-dir` to short, fixed `/tmp/...` paths in
`runTest.ts` — a real, documented `@vscode/test-electron`-on-macOS
gotcha, not a bug in the extension itself.

All 3 integration tests passed inside the real extension host:

```
Code Review AI extension (integration)
  ✔ extension activates and registers the command
  ✔ real git diff + real backend produce a valid ReviewResponse (210ms)
  ✔ executing the command opens the results panel (298ms)
3 passing (527ms)
```

* Test 1 confirms `vscode.extensions.getExtension(...)` finds the
  extension, `activate()` succeeds, and `codeReviewAi.reviewCurrentChanges`
  appears in `vscode.commands.getCommands()`.
* Test 2 calls `getWorkingDiff` and `reviewDiff` directly against the real
  throwaway repo and the real running backend — confirming the diff
  actually contained the expected change and the backend returned a valid
  `merge_probability` in `[0, 1]`, a `MERGE`/`CLOSE` prediction, and
  `ml_model: "rf_v1_balanced"`.
* Test 3 executes the real command via
  `vscode.commands.executeCommand("codeReviewAi.reviewCurrentChanges")`
  and polls `vscode.window.tabGroups` until the "Code Review AI: Review
  Results" webview tab appears — proving the full
  command → `git diff` → backend call → panel path works, not just its
  three pieces in isolation.

Both tests that need the backend/a real diff call `this.skip()` gracefully
if the backend isn't reachable or the workspace has no uncommitted
changes, so `npm run test:integration` still means something (and doesn't
just report false failures) when run without that setup.

After verification: the backend process was stopped, the throwaway repo
and `/tmp` VS Code test profile dirs were deleted, and `.vscode-test/` was
added to the root `.gitignore` (it isn't something any earlier step
anticipated, since Step 24 is the first to touch Node tooling at all).

### 4.3 One `.gitignore` fix, unrelated to the extension's own logic

The root `.gitignore`'s `.vscode/` entry (added in Step 1, intended to
ignore *this repo's own* root-level editor settings) is an unanchored
pattern, so it would also have swallowed
`vscode-extension/.vscode/launch.json`/`tasks.json` — files a VS Code
extension project is *supposed* to track so `F5` works out of the box for
anyone who clones the repo. Changed to `/.vscode/` (anchored to the repo
root) so the original intent is preserved without collateral damage to the
extension's own tracked config.

---

## 5. State at the end of Step 24

* No gauge, no click-to-jump, no inline decorations — `panel.ts` was
  intentionally plain text/list HTML (§2.3). **All three are Step 25's job
  and are now done; see §6 onward.**
* No `.vsix` packaging yet — `.vscodeignore` is in place but nothing has
  run `vsce package`; that, plus a real end-to-end test of the *packaged*
  extension and a repo-wide README/demo GIF, is Step 26.

---

# Step 25 — Polish

Plan row 25: *"webview UI with a merge-probability gauge, click-to-jump
review comments, inline highlights."* All three shipped. What follows is
the one genuine design problem this step had to solve, and how it was
verified.

---

## 6. The problem Step 25 actually had to solve

Two of the three asks — click-to-jump and inline highlights — need a
**`(file, line)` per review comment**. The backend doesn't return one.

`ReviewResponse.review_comments` is a plain `string[]`
(`src/api/main.py`), because that is what the Experiment 3/4 output
contract produces: the prompts ask for a `<review>` block of prose
bullets, and `src/llm/prompts/parsing.py` splits it into lines. No file or
line number is ever requested, so none comes back. A comment reads like
*"The `slugify` function's return value is not validated"* — meaningful to
a human, but not a location.

There were two ways forward, and the choice matters:

| Option | Verdict |
|---|---|
| **Change the prompt contract** to demand structured `file:line` output | **Rejected.** It would invalidate Experiment 4's completed 148-cell grid, its on-disk response cache, and the `parse_review_comments` tests written against that contract — in order to re-run a grid the shared free-tier Groq quota can't currently afford to repeat (`reports/exp4_grid_run_status.md`). A UI feature is not worth invalidating a completed experiment. |
| **Re-derive the location client-side** from the diff the user just sent | **Chosen.** Costs nothing, touches no completed experiment artifact, and the diff is already in hand. |

So `src/anchoring.ts` resolves each comment against `src/diffMap.ts`'s
parse of the same diff. Because this is a **heuristic**, its confidence is
reported rather than hidden — every anchor carries a `precision`:

| `precision` | Meaning | UI treatment |
|---|---|---|
| `line` | The comment names a symbol (or a real line number) found in the diff. | Clickable, strong-colored chip, **and** an inline editor highlight. |
| `file` | The comment names a changed file but nothing narrower. | Clickable (jumps to the file's first change), muted chip, **no** highlight. |
| `inferred-file` | The comment names nothing, but the diff touches exactly one file. | Same as `file`. |
| `null` | Nothing matched. | Plain, non-clickable text. No guess is made. |

Two rules keep it honest rather than merely confident:

* **A hallucinated line number is ignored.** If a comment says "line 999"
  but the diff never added line 999, the anchor falls back to file-level
  rather than sending the user to a fabricated location.
* **Only `precision: 'line'` gets an inline highlight.** Navigation is a
  hint; painting a decoration on a specific line is a *claim*. A
  file-level anchor has no business asserting one — so `decorations.ts`
  deliberately skips those.

---

## 7. Verification performed

### 7.1 Unit tests — 49 passing (`npm test`)

Up from 8. The new modules are all `vscode`-free by construction, so they
run under plain `node --test` with no editor, no backend, no network:

* `diffMap.test.ts` (9) — post-change line numbering, including the case
  that actually matters: **a removed line must not advance the new-file
  cursor**. Plus `+start` offsets, multiple hunks, multiple files, new
  files, removal-only files, and `+++` headers not being mistaken for
  added content.
* `anchoring.test.ts` (17) — every precision level and every refusal:
  hallucinated line numbers ignored, ambiguous symbols (present in several
  files) rejected, 1–2 character tokens rejected as too weak, added lines
  outranking context lines.
* `panelHtml.test.ts` (15) — gauge arc proportional to probability (and a
  0% probability rendering a zero-length arc, not a full circle),
  clickable vs. non-clickable markup, and **HTML escaping of
  model-generated text**: this module interpolates LLM output straight
  into a webview document, so an XSS payload in a comment, and a quote
  breaking out of a `data-path` attribute, both have explicit tests.

### 7.2 Integration tests — 7 passing, in a real VS Code (`npm run test:integration`)

Same harness as Step 24 (§4.2), extended to cover the new behaviour:

```
Code Review AI extension (integration)
  ✔ extension activates and registers its commands
  ✔ real git diff + real backend produce a valid ReviewResponse (92ms)
  ✔ executing the command opens the results panel (295ms)
  ✔ the real working diff anchors a comment to a real file and line
  ✔ jumpToComment opens the file and puts the cursor on the right line
  ✔ jumpToComment on a missing file warns instead of throwing
  ✔ clearHighlights runs cleanly
7 passing (501ms)
```

The click-to-jump test is the substantive addition: it executes the real
`codeReviewAi.jumpToComment` command and then asserts on
`vscode.window.activeTextEditor` — that the right document opened and the
cursor landed on the right (0-based) line. `jumpToComment` is a registered
command rather than a closure inside the webview message handler
specifically so this is testable without simulating a browser click.

**Two real bugs the integration run caught**, both fixed:

1. VS Code logged `Menu item references a command 'codeReviewAi.jumpToComment'
   which is not defined in the 'commands' section` — the `menus`
   contribution referenced a command missing from `contributes.commands`.
2. The anchoring integration test failed — and the **test** was wrong, not
   the code. It picked `"if"` as its probe symbol, which `anchoring.ts`
   correctly refuses (a 2-char token matches almost any line), and it
   ignored that a symbol appearing in *both* changed files is rejected as
   ambiguous by design. The test now picks a ≥4-char identifier and names
   the file, and asserts the documented "first added line containing the
   symbol" behaviour.

### 7.3 Against real LLM output — the check that changed the design

Synthetic test strings only prove the heuristic does what I assumed
reviewers write. So one real deep-mode call was made against a throwaway
two-file repo, and the **actual** Groq comments were fed through the
anchoring code.

First run: **3 of 4 resolved.** The two misses were both instructive —
comments about *"the `greet` function"*, where `def greet(name):` is a
**context** line, not an added one. The heuristic only indexed added
lines, so it could not anchor them (one fell back to file precision, one
went unanchored entirely).

That is a realistic and common case — reviewers habitually name the
*enclosing* function, which by definition the diff didn't change. So
`diffMap.ts` now also records context lines, and `anchoring.ts` searches
them as a weaker, secondary source (added lines still win). Re-run:
**4 of 4 resolved**, and each comment now points at the definition it is
actually discussing (`greeter.py:1`, `utils/slug.py:1`) instead of an
arbitrary changed line.

This is the one change in Step 25 that came from evidence rather than
design intent, and it would not have surfaced from unit tests alone.

### 7.4 Visual check of the rendered panel

Because `panelHtml.ts` is `vscode`-free, the real response could be
rendered to a standalone HTML file and opened in a browser — confirming
what no assertion covers: that it *looks* right. Verified visually with
real data (61% MERGE): the gauge arc is proportionally filled and
color-matched to the prediction, each comment carries its `file:line`
chip, the honest footnote ("4 of 4 comment(s) resolved… locations are
inferred from the comment text") renders, and the collapsible feature
table expands to the real 28 values (`m_additions` 4, `m_deletions` 2,
`m_churn` 6 — matching the diff exactly). The click payloads were then
read back from the DOM to confirm all four carry the correct
`data-path`/`data-line` and precision tooltip.

After verification the preview server, throwaway repo, backend process,
and `/tmp` VS Code profile dirs were all removed.

---

## 8. Known limitations after Step 25

* **Anchoring is a text heuristic, and is presented as one.** It cannot
  know what the model *meant*; it matches what the model *wrote* against
  the diff. The `precision` levels and the panel footnote exist so the
  user is never misled about that. A structured-output prompt contract
  would beat it — see §6 for why that is deliberately deferred.
* **Inline highlights are line-level, not range-level.** A comment about a
  multi-line block highlights only the first matching line. Character
  ranges would need the model to say where the block ends, which loops
  back to the same contract question.
* **Decorations follow the diff, not subsequent edits.** If the file is
  edited after a review, highlights keep their original line numbers
  (clamped to the document length). "Clear Review Highlights" exists for
  exactly that; live re-anchoring on edit is out of scope.
* **`fast` mode is what the automated suite exercises** — the integration
  tests use `mode: "fast"` to avoid spending live Groq quota on every run.
  The `deep` path was exercised for real in §7.3 here and in Step 23's own
  verification (`reports/api_design.md` §6.2), and the request shape is
  identical regardless of mode.
* **No `.vsix` packaging yet** — Step 26, done below.

---

# Step 26 — Packaging & final verification

Plan row 26: *"Do an end-to-end test of the extension against the backend,
package a `.vsix`, ... "* Both done; this section covers what verifying a
**packaged** artifact required beyond Steps 24–25's dev-mode integration
suite, and one genuine gap it surfaced.

## 9. Packaging (`npm run package` → `@vscode/vsce`)

Added `@vscode/vsce` as a devDependency, plus the metadata fields it
expects that hadn't been needed yet: `license: "MIT"` (a copy of the root
`LICENSE` is duplicated into `vscode-extension/LICENSE` specifically so it
ships *inside* the `.vsix` — a user installing the packaged extension
doesn't necessarily have the parent repo checked out) and a `repository`
field.

**Honest caveat on `repository`:** no git remote is configured in this
repo (`git remote -v` is empty — this has been worked on entirely locally).
`https://github.com/MarlenMM/code-review-ai` is a best-effort value
inferred from the project plan's own stated GitHub handle, not a URL this
session created, pushed to, or confirmed resolves. Update it (here and in
`README.md`'s clone instructions) once the real repository exists, if the
name differs.

> **Confirmed in Step 28.** The repository now exists at exactly that URL and
> is the configured `origin`, so the guessed value turned out to be correct
> and needs no change. CI runs there on every push (see §15).

`npx vsce package` ran clean, no warnings, and the file manifest is exactly
the intended shipped surface — 8 compiled modules, `package.json`,
`LICENSE.txt`, `readme.md`, nothing else:

```
code-review-ai-0.0.1.vsix (13 files, 21.1 KB)
extension/
├─ LICENSE.txt
├─ package.json
├─ readme.md
└─ out/
   ├─ anchoring.js   ├─ apiClient.js   ├─ decorations.js  ├─ diffMap.js
   ├─ extension.js   ├─ git.js         ├─ panel.js         └─ panelHtml.js
```

No `src/`, no `node_modules/`, no test files, no `.map` files, no stray
`.gitkeep` — `.vscodeignore` (Step 24) worked as designed on the first try.

## 10. Verifying the *packaged* artifact, not just the dev-mode one

Steps 24–25's `npm run test:integration` proves the extension works when
VS Code loads it via `--extensionDevelopmentPath` — i.e. straight from the
compiled `out/` directory. That is not quite the same claim as "the thing a
user installs from the `.vsix` works," and packaging bugs (a stale
`.vscodeignore` rule, a manifest field wrong only in the zipped form) are a
real, distinct failure class. `@vscode/test-electron`'s `runTests()` API is
built specifically around `--extensionDevelopmentPath`, though, so it
cannot itself drive an *installed* (packaged) extension — and, per §3, GUI
automation to click through a real install is unavailable here (click-only
IDE access). So packaging correctness was verified the way the tooling
that *is* available actually allows:

1. **Real, non-GUI installation.** The exact VS Code binary
   `@vscode/test-electron` had already downloaded for Steps 24–25
   (`.vscode-test/vscode-darwin-arm64-1.130.0`) has a `bin/code` CLI. Used
   it directly, headlessly, against a fresh, isolated profile:

   ```
   code --install-extension code-review-ai-0.0.1.vsix \
     --user-data-dir /tmp/crai-pkgtest-userdata \
     --extensions-dir /tmp/crai-pkgtest-extensions
   # -> "Extension 'code-review-ai-0.0.1.vsix' was successfully installed."

   code --list-extensions --show-versions \
     --user-data-dir /tmp/crai-pkgtest-userdata \
     --extensions-dir /tmp/crai-pkgtest-extensions
   # -> marlenmm.code-review-ai@0.0.1
   ```

2. **Byte-for-byte proof the tested code is the shipped code**, closing the
   gap `runTests()` leaves: `diff -rq` between the `out/` VS Code just
   extracted from the `.vsix` into the isolated extensions directory and
   the local `out/` the 7-test integration suite (§4.2/§7.2) had just run
   against, moments earlier. Only difference: the `.js.map` files
   `.vscodeignore` deliberately excludes. **Every `.js` file is identical.**
   That means "activates correctly, calls the real backend, opens the
   panel, supports click-to-jump" — everything the integration suite
   proved — is a true statement about the exact bytes now sitting inside
   the packaged `.vsix`, not an inference from a similar-but-different
   build.

3. **A fresh full re-run of the dev-mode integration suite** immediately
   before packaging, as the final gate: all 7 tests green (§7.2's table),
   confirming nothing regressed between Step 25's last verification and
   packaging.

The isolated `/tmp` profile dirs were deleted after verification; the
built `.vsix` itself is not committed (`*.vsix` is gitignored — regenerate
with `npm run package`).

## 11. What "portfolio-ready" added outside the extension

Covered in `reports/portfolio_readiness.md`, not here, since it's
repo-wide rather than extension-specific: the root `README.md`, `LICENSE`,
`.github/workflows/ci.yml`, and a `ruff` lint pass that fixed 12 real
(if minor) issues across `src/`/`tests/` as part of adding it to CI.

---

# Step 27 — Panel design pass

## 12. The problem: it worked, and it looked generated

Steps 24–26 got the panel to *correct*. They did not get it to *designed*,
and the difference is visible before a single number is read. Audited
against the two checklists that circulate for this ("StyleMole" and the
"make your vibe-coded app not look vibe-coded" list), the panel was hitting
five signals — none fatal on its own, all of them together producing the
look of a first-draft AI output:

| Signal | Where it was | Why it reads as unfinished |
|---|---|---|
| No empty / loading / error state | `extension.ts` used `showInformationMessage` / `showErrorMessage` and never opened a panel | The two most likely *first* runs — a clean tree, and a backend nobody started — produced a toast that vanished and a blank screen |
| Flat typography | six near-identical sizes (`1.05em`, `0.9em`, `0.85em`, `0.85em`, …), each picked in isolation | Heading, caption, table cell and footnote all land within a tenth of an em; the eye has nowhere to rest |
| Ad-hoc spacing | `1.2em 1.5em 2em`, `0.35em`, `1.8em`, `0.45em` … | Everything roughly equidistant from everything else, so nothing groups |
| A coloured strip on every comment | `ol.comments li { border-left: 2px solid … }`, uniform | A status colour applied uniformly stops being a status |
| Emoji standing in for icons | `⚠` in the panel, `💬` in the editor decorations | The most legible "nobody designed this" tell — and functionally worse, since an emoji renders in the platform emoji font's own colour and ignores the user's theme |

## 13. What changed

The rules and the reasoning are in `docs/design-constraints.md`; the
summary of the code changes:

* **Four states, one shell.** `panelHtml.ts` now exports
  `renderPanelHtml`, `renderLoadingHtml`, `renderEmptyHtml` and
  `renderErrorHtml`, all built by one private `documentShell` so they
  cannot drift onto separate stylesheets or separate CSPs. `panel.ts`
  gained `showLoadingPanel` / `showEmptyPanel` / `showErrorPanel` and a
  `rerun` message; `extension.ts` opens the panel at the *start* of a
  review and re-renders it in place.
* **The empty state earns its screen.** It does not say "nothing found";
  it explains that `git.ts` runs plain `git diff` and therefore cannot see
  staged work — which is the trap people actually hit — and gives the
  `git restore --staged .` that undoes it.
* **The error state is typed.** `apiClient.ts` now throws `ReviewError`
  with `kind: "unreachable" | "http"`, because "start the backend" is the
  right advice for a refused connection and the wrong advice for a 422.
  Previously both were a bare `Error` and the caller could only have told
  them apart by pattern-matching the message string.
* **A five-step type scale and a 4px spacing rhythm**, as CSS custom
  properties. Nothing is sized or spaced off-scale.
* **The left bar now means something**: solid accent for a `line`-precise
  anchor, hairline for `file`, dashed for `inferred-file`, none for
  unanchored — the same information the chip colour and the tooltip carry.
* **One icon family, no emoji**: three inline SVGs (16×16, 1.5 stroke,
  `currentColor`) plus `media/review-comment-{dark,light}.svg` as a real
  gutter icon for anchored comments.
* **One animation, and only while something is running**: the loading
  state's indeterminate bar, disabled under `prefers-reduced-motion`. It
  deliberately does *not* animate step-by-step progress through the
  pipeline — the backend reports none, so that would be invented.

### 13.1 The calibration marks — the one detail nothing generates

The gauge gained two marks the arc alone cannot express: a **notch at
50%**, the threshold at which `src/api/main.py` actually flips MERGE to
CLOSE, and a **line at 72%**, the merge rate of the 1,494 mined PRs the
model was trained from (`results/tables/exp1_summary.json` →
`merge_status.pooled.merge_rate` = 0.7195). Both carry an SVG `<title>`, so
they explain themselves on hover instead of needing a legend, and they are
drawn differently on purpose: the threshold is a hard rule cut *through*
the ring, the base rate a reference laid *over* it.

This matters beyond decoration. A bare "61%" reads as healthy. Against the
corpus the model learned from, 61% is a **below-average** change, and the
sentence beside the gauge now says so in those words. It is also the one
element no component library could produce, because it requires knowing
what this particular model was trained on.

## 14. Verification performed

### 14.1 Unit tests — 66 passing (`npm test`)

49 before, 66 after; 17 added, none removed, and every pre-existing
assertion still passes unchanged — the redesign did not require loosening
a single existing test. The new ones cover:

* each new state's *content*, not just that it renders: that the loading
  state names the real pipeline and only promises a Groq call in `deep`
  mode; that the empty state explains the staged-changes trap; that an
  `unreachable` error offers the `uvicorn` command and an `http` error
  explicitly does not.
* **the error state's escaping.** `apiClient.ts` puts the backend's own
  response body into the error message, so `renderErrorHtml` interpolates
  remote text — the same class of untrusted input as the LLM comments, and
  now covered by the same class of test.
* that all four states ship an identical CSP and identical design tokens,
  which is what stops them drifting apart later.
* that no state falls back to an emoji, by regex over the emoji,
  pictograph and dingbat blocks.
* that the thousands separator in "1,494" is locale-independent —
  `toLocaleString` would render "1.494" on a de-DE extension host.

### 14.2 Visual check, in both themes

As in §7.4: `panelHtml.ts` imports nothing from `vscode`, so all four
states were rendered to standalone HTML, served locally, and opened with
VS Code's own Dark Modern **and** Light Modern `--vscode-*` token values
injected. Both gauge marks are legible against the arc and against the
track in each theme (the notch reads as a gap in the ring precisely
because it is drawn in the panel's own background colour), the type scale
produces visible hierarchy at real size, and the comment bars are
distinguishable at 2px.

### 14.3 The README screenshots are real output, re-made

`docs/images/vscode_panel_demo.png` was regenerated rather than left
stale — a redesigned panel behind an old screenshot is a README that
lies. It is a genuine end-to-end run: the real backend on a throwaway
two-file repo, the real `rf_v1_balanced` prediction (**0.606 → 61%
MERGE**, 28 real features), and the real LLM review comments, which
replayed from `data/llm_cache/` — the same recorded Groq response the
original screenshot was made from, so the evidence is real model output
rather than text written to look like it. (Groq was still the live provider
when this was captured; see §14.4.) `panel_states_demo.png` is the
three non-result states, each an actual render of its own function.

### 14.4 A real defect found while doing this

Reproducing a live `deep`-mode call surfaced something unrelated to the
design work and worth recording: Groq now returns **404
`The model 'llama-3.1-8b-instant' does not exist or you do not have access
to it`**. That model id is pinned in `src/llm/providers.py`
(`GROQ_DEFAULT_MODEL`) and is the one every Experiment 3/4 result was
produced with, so live `deep` mode is currently broken against Groq even
though the cache still replays fine.

**Resolved in Step 28** (`reports/api_design.md` §8), and worth recording
how, because the naive fix was the wrong one. The pinned id was doing two
jobs: it was an *experimental constant* (every Lab 3/4 number was measured
on it, and `data/llm_cache/` is keyed on it) and a *live dependency*. Only
the second broke. So the live path moved to Qwen (`qwen-plus`, via
DashScope's OpenAI-compatible endpoint) while the grid runners kept
`--provider groq` as their default — which means a grid re-run still
replays Labs 3/4 from disk, exactly and for free, instead of silently
becoming a new measurement against a different model.

The failure was at least *visible* rather than silent in the meantime —
`llm_review.py` degrades to the fast-mode result plus an `llm_warning`,
which the panel renders in the warning state with a real icon. That path is
still what a missing DashScope key hits today, and the message it now
carries names the variable to set.

One knock-on for this document: `docs/images/vscode_panel_demo.png` (§14.3)
shows real LLM comments replayed from the **Groq-era** cache, which is what
the config label in the screenshot says. It is an accurate record of a real
run; it is not a Qwen result.

