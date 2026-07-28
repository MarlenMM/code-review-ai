# Experiment 1 — AI-Detection Precision Spot-Check

*Manual validation of the `is_ai_authored` / `is_ai_reviewed` flags produced by
`src/mining/ai_detection.py`, on the mined dataset of 1,494 PRs. This is the
Step 5 deliverable and is written to be dropped directly into the Experiment 1
report's Results section.*

## Method

Of 1,494 mined PRs, **1,180 carry at least one AI flag** (380 `is_ai_authored`,
1,064 `is_ai_reviewed`). Two overlapping manual reviews were run against this
flagged population:

1. **Random sample** — 40 PRs drawn from the flagged union with a fixed seed
   (`pandas.sample(n=40, random_state=42)`, reproducible; audit trail in
   `results/tables/exp1_precision_sample.csv`). Each flag the PR carries was
   assessed by hand against the PR's author login/type, the recorded detection
   `method`/`confidence`/`evidence`, and — for a subset — the raw cached GitHub
   JSON, not just the derived table.
2. **Full census of the weakest tier** — all 19 PRs flagged via the
   lower-confidence secondary signal (`commit_agent_identity`, confidence 0.80)
   were inspected individually against their actual commit trails, since that
   tier is where a false positive is most likely to hide.

## Results

| Review | PRs | Flag assessments | False positives | Precision |
|---|---|---|---|---|
| Random sample (flagged union) | 40 | 49 (12 authored + 37 reviewed) | 0 | **100%** |
| Secondary-signal census | 19 | 19 authored | 0 | **100%** |
| **Combined (distinct PRs)** | **59** | **68** | **0** | **100%** |

**No false positives were found in either review.** With 0 errors in 40 random
draws, the *rule of three* gives a 95% upper bound on the false-positive rate of
≈ 3/40 = 7.5% — i.e. the estimated flag precision is **≥ ~92.5% at 95%
confidence**, with a point estimate of 100% on everything inspected.

### Why precision is this high — the detection is almost entirely structural

The flags reduce to three tiers, and every tier is anchored to a signal GitHub
itself provides rather than a guess:

- **Allowlisted author + `Bot` type (361 PRs, conf 0.95).** Every one of the 361
  has author `copilot-swe-agent`, typed `Bot` by GitHub's GraphQL API —
  unambiguously GitHub's Copilot coding agent. No plausible false positive.
- **Allowlisted reviewer + `Bot` type (1,064 PRs).** Every AI-reviewer flag is
  `copilot-pull-request-reviewer` (1,060) or that plus `coderabbitai` (4), both
  genuine AI review bots typed `Bot`. Spot-checks against raw JSON confirmed the
  reviewer is genuinely present, and — importantly — that co-present *non-AI*
  bots (`vs-code-engineering`, `dependabot`, `home-assistant`) were correctly
  **not** flagged. The allowlist's exclusion of ordinary automation is working.
- **Secondary commit-identity signal (19 PRs, conf 0.80).** All 19 are PRs
  opened by a human (or, in one case, a deleted account) whose commit trail
  contains commits authored by the Copilot coding agent — identified by the
  agent's no-reply commit email (`…+Copilot@users.noreply.github.com`) and/or
  the `copilot-swe-agent[bot]` commit-author name. All 19 were manually
  confirmed genuine.

## Findings worth reporting honestly

**One labeling nuance (not an authorship error).** `microsoft/vscode#320230` is
flagged `is_ai_authored = True` (correct — its commits are entirely Copilot's,
including two "Initial plan" markers) but sub-categorised as
`human_opened_ai_assisted`, whereas its PR author field is actually `null` (a
deleted/ghost account), not a human. The *authorship verdict* is right; only the
*kind* attribution is imprecise for this one edge case. This is a transparent,
reportable limitation of deriving `kind` from the author field when that field
is absent — exactly the sort of edge the `method`/`evidence` audit fields exist
to expose.

**The "human-opened, AI-assisted" gray zone.** The 19 secondary-signal cases are
a genuinely distinct category, not noise: a human opens the PR but the Copilot
coding agent writes some or all of the commits (e.g. `dotnet/runtime#131199`,
`dotnet/aspnetcore#67082`, where every substantive commit is the agent's).
Whether these are "human-opened, AI-assisted" or effectively "AI-authored,
human-supervised" is a definitional judgement; the pipeline records the
authorship as AI while keeping this subset separately labelled so the
distinction is never silently collapsed. They are **5.0% of AI-authored flags**
(19/380) — close to the ~8% anticipated during design.

## Limitations

- This check measures **precision (false positives), not recall.** It says
  nothing about AI-authored/reviewed PRs the allowlist *missed* (e.g. an agent
  not on the curated list, or IDE-level Copilot autocomplete committed under a
  human's own identity, which leaves no structural trace). Those are
  false-negatives by construction and are out of scope for a precision audit.
- Precision is high partly *because* the design is deliberately conservative:
  anything `Bot`-typed but off the allowlist is reported as `other_bot`, never
  guessed into an AI bucket. That trades recall for precision on purpose.

## One-paragraph version (drop-in)

> A manual precision spot-check was performed on the AI-detection flags. A
> reproducible random sample of 40 flagged PRs (seed 42), plus a full census of
> the 19 PRs flagged by the weakest secondary signal, were reviewed by hand —
> 59 distinct PRs, 68 individual flag assessments — against author/reviewer
> identity, GitHub's structural `Bot`/`User` typing, and the raw commit trail.
> **Zero false positives were found (point-estimate precision 100%; ≥ ~92.5% at
> 95% confidence by the rule of three, computed on the 40-PR random draw —
> the 19-PR census is not a random sample, so it cannot widen that bound).**
> The high precision follows from the
> detection being anchored to GitHub's own actor typing plus a curated allowlist
> rather than username guessing, with non-AI automation (dependabot, renovate,
> etc.) deliberately excluded. The only imperfection found was a single *kind*
> mis-attribution (a Copilot-authored PR with a deleted author account labelled
> "human-opened" rather than "unknown"), which does not affect the correctness
> of its AI-authored verdict.
