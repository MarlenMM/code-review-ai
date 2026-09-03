# Code Review AI (VS Code extension)

Reviews your current uncommitted changes against the Code Review AI backend
(`src/api/main.py` in the parent repository): a merge-probability prediction
(Experiment 2's trained model) and, in "deep" mode, AI-generated review
comments (Experiment 4's LLM config).

Features:

* A **calibrated merge-probability gauge**: the predicted MERGE/CLOSE
  outcome, marked with the 50% threshold where that call actually flips and
  with the 72% merge rate of the 1,494 PRs the model was trained from — so
  the number is read against something instead of floating free. A
  collapsible breakdown lists all 28 features behind it.
* **Click-to-jump review comments** — each comment is resolved to a
  `file:line` in your diff and clicking it navigates there. The coloured bar
  down the left of each comment says how firmly it was matched.
* **Inline highlights** on the lines comments refer to, with a gutter icon
  and the full comment text on hover.
* **Designed loading, empty and error states** — a review in flight, a
  clean working tree and an unreachable backend each render in the panel
  with what happened and what to do about it, rather than as a notification
  that disappears.

Comment locations are *inferred from the comment text*, because the model
returns prose rather than line numbers — hover a location chip to see how
confidently a given comment was matched, and see
`reports/vscode_extension_design.md` in the parent repository for why it
works this way. The panel's design rules, and the failure each one
prevents, are in `docs/design-constraints.md`.

## Prerequisites

* The backend running locally: from the repo root,
  `source .venv/bin/activate && uvicorn src.api.main:app` (defaults to
  `http://127.0.0.1:8000`, matching this extension's default
  `codeReviewAi.backendUrl` setting).
* Node.js 18+ (for `npm`/`tsc`; VS Code's own extension host bundles a
  compatible Node at runtime).

## Development

```bash
cd vscode-extension
npm install
npm run compile   # or `npm run watch` for incremental builds
npm test          # 66 unit tests: no VS Code, no backend, no network

# End-to-end, inside a real VS Code instance. Needs the backend running;
# point it at a git repo that has uncommitted changes:
CODE_REVIEW_AI_TEST_WORKSPACE=/path/to/a/dirty/repo npm run test:integration
```

Open **this `vscode-extension/` folder** (not the repo root) in VS Code and
press `F5` to launch an Extension Development Host with the extension
loaded (`.vscode/launch.json` runs the `watch` build task first). In the
new window, open a folder that's a git repo with some uncommitted changes,
then run **"Code Review AI: Review Current Changes"** from the Command
Palette.

## Settings

| Setting | Default | Meaning |
|---|---|---|
| `codeReviewAi.backendUrl` | `http://127.0.0.1:8000` | Base URL of the FastAPI backend. |
| `codeReviewAi.mode` | `fast` | `fast`: ML-only probability, instant, free. `deep`: also generates review comments via a live LLM call (real, shared API quota — see `reports/api_design.md`). |

## Commands

| Command | Purpose |
|---|---|
| `Code Review AI: Review Current Changes` | The main entry point. |
| `Code Review AI: Clear Review Highlights` | Removes the inline decorations. |
| `Code Review AI: Jump To Review Comment` | Used by the panel's click handler; hidden from the palette. |

## Design notes / test coverage

Everything that can be tested without VS Code, is — the modules below are
split precisely along that line.

* `src/git.ts` — runs `git diff` (working tree vs. index) in the first
  workspace folder; `child_process.exec` is injected so this is unit-tested
  without a real git repo.
* `src/apiClient.ts` — a thin `POST /review` client matching
  `src/api/main.py`'s Pydantic schema exactly (`reports/api_design.md` §5);
  `fetch` is injected so this is unit-tested without a real backend.
* `src/diffMap.ts` — parses the diff into post-change line numbers for both
  added and context lines. Pure; unit-tested.
* `src/anchoring.ts` — resolves each prose comment to a `file:line`. Pure;
  unit-tested. **Read its header comment before changing it** — the
  precision levels it reports are deliberate.
* `src/panelHtml.ts` — builds the webview HTML for all four states (result,
  loading, empty, error) off one stylesheet and one CSP. Pure and
  `vscode`-free specifically so its HTML escaping of *model-generated text*
  — and of the backend's own response body, which the error state shows —
  is unit-tested. Read `docs/design-constraints.md` before restyling it.
* `src/panel.ts` / `src/decorations.ts` / `src/extension.ts` — the parts
  that genuinely need the editor API: webview lifecycle and CSP, editor
  decorations, and command wiring. Covered by the integration suite that
  runs inside a real VS Code instance (`npm run test:integration`).
* `media/review-comment-{dark,light}.svg` — the gutter icon for an anchored
  comment. Two files because a gutter icon is a real asset and cannot
  reference `--vscode-*` colours, so the contrast is baked per theme.
