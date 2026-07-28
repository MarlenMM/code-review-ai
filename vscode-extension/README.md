# Code Review AI (VS Code extension)

Reviews your current uncommitted changes against the Code Review AI backend
(`src/api/main.py` in the parent repository): a merge-probability prediction
(Experiment 2's trained model) and, in "deep" mode, AI-generated review
comments (Experiment 4's LLM config).

Features:

* A **merge-probability gauge** with the predicted MERGE/CLOSE outcome and a
  collapsible breakdown of all 28 features the model used.
* **Click-to-jump review comments** — each comment is resolved to a
  `file:line` in your diff and clicking it navigates there.
* **Inline highlights** on the lines comments refer to, with the full
  comment text on hover.

Comment locations are *inferred from the comment text*, because the model
returns prose rather than line numbers — hover a location chip to see how
confidently a given comment was matched, and see
`reports/vscode_extension_design.md` in the parent repository for why it
works this way.

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
npm test          # 49 unit tests: no VS Code, no backend, no network

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
* `src/panelHtml.ts` — builds the webview HTML (gauge, comment list,
  feature table). Pure and `vscode`-free specifically so its HTML escaping
  of *model-generated text* is unit-tested.
* `src/panel.ts` / `src/decorations.ts` / `src/extension.ts` — the parts
  that genuinely need the editor API: webview lifecycle and CSP, editor
  decorations, and command wiring. Covered by the integration suite that
  runs inside a real VS Code instance (`npm run test:integration`).
