# Experiment 2 — Feature Engineering Specification

*Merge-prediction feature set for human-written code (SVM / Random Forest).*
*Design deliverable for Step 8; feeds Steps 9–12. All population numbers below are
computed from `data/processed/*.parquet` (the real 1,494-PR mined dataset), not estimated.*

---

## 1. Scope and prediction target

- **Task.** Binary merge prediction: `y = is_merged` for a closed pull request.
- **Population (lab-guide §2.7.1 "Data Filtering").** Keep PRs that are **human-written**
  (`is_ai_authored == False`) **and closed** (`state ∈ {MERGED, CLOSED}`). In this dataset
  *every* mined PR is already closed, so the closed filter is a no-op here but is stated for
  correctness — open/pending PRs carry no final label and must never enter training.
- **Resulting training population: 1,114 PRs.**

| Split | Count | Share |
|---|---|---|
| Merged (`y=1`) | 827 | **74.2 %** |
| Not-merged (`y=0`) | 287 | **25.8 %** |
| **Total** | **1,114** | imbalance ≈ **2.88 : 1** |

Per-repo (human-written, closed), which the report should show because merge rates differ a lot
by community — this is itself a confound worth naming:

| Repo | Language | N | Merge rate |
|---|---|---|---|
| microsoft/vscode | TypeScript | 195 | 0.851 |
| dotnet/runtime | C# | 189 | 0.836 |
| home-assistant/core | Python | 269 | 0.773 |
| dotnet/aspnetcore | C# | 187 | 0.663 |
| microsoft/semantic-kernel | Python + C# | 274 | 0.624 |

The 26 %-vs-15 % spread in *not-merged* rate across repos means "which repo" is a strong naive
predictor. We deliberately **exclude repo identity from the core feature sets** (V1/V2 below) so
the models are judged on *code* features, not on memorising community merge culture; repo is
reported only as a stratification variable and offered as an optional one-hot control.

---

## 2. The hard constraint (from the real lab guide, not the plan's paraphrase)

Lab guide **§2.4.3 Feature Engineering** names exactly three feature categories and one prohibition:

> Code structure features (number of AST nodes, number of CFG nodes, etc.); code modification
> features (added lines, deleted lines, modified files, etc.); textual features (lengths of
> titles, bodies, commit messages, etc.). **Note that you CANNOT use any review
> comments/discussions as features.** [features are then] normalized to serve as input.

Lab guide **§2.7.3 Step 3** lists the concrete features to extract: number of modified files,
number of added lines, number of deleted lines, commit-message length, PR-description length,
AST-related features, CFG-related features.

**Correction to project-plan §4.4 (worth stating in the report as due diligence).** The plan
claims "the guide's own Step 4 lists 'number of reviewers' and 'number of comments' as features,"
and treats those as spec features that leak. Reading the actual PDF: `num_reviewers` and
`num_comments` appear in **Experiment 1's** descriptive-statistics list (§1.7.4), *not* in
Experiment 2's model-feature list (§2.7.3), and §2.4.3 explicitly forbids review
comments/discussion. So the compliant Experiment 2 feature set **never contained those features
in the first place** — the leakage risk is real but comes from *accidentally* importing
Experiment 1's statistics, exactly the mistake the plan itself made. The spec below is built to
make that boundary explicit and to *measure* the temptation rather than deny it.

---

## 3. Temporal-availability tiers (the backbone of the leakage analysis)

Merge prediction is only meaningful as a *pre-decision* task. Every candidate feature is tagged
with **when its value becomes known** relative to the review process:

- **T0 — submission-time.** Known when the PR is opened. No leakage. (title/body text.)
- **T1 — final-code-state.** Known once the code is final, but the "final" diff *includes commits
  the author pushed in response to review*. So T1 features carry **mild, unavoidable temporal
  leakage** in a strict pre-merge framing — the mined dataset only stores the final aggregate
  diff, not the opening snapshot. (additions/deletions/files, AST/CFG, commit counts.)
- **T2 — review-process outcome.** Only exists *because* review happened: review counts, reviewer
  counts, review/issue comments, approval/changes-requested states, AI-reviewer flags. **Strong
  leakage**, and comment/discussion counts are **outright forbidden** by §2.4.3. An `APPROVED`
  review essentially *is* the merge decision, so T2 features leak the label almost directly.

The whole design turns on this: the spec-compliant set is T0+T1; a stricter variant trims T1's
contamination; and the forbidden T2 features are quarantined into an ablation-only set so we can
*quantify* how much they inflate accuracy instead of hand-waving "they would leak."

---

## 4. Feature catalogue

Notation: **S** = code-structure, **M** = code-modification, **T** = textual, **P** =
process/forbidden. `pr` = `pull_requests`, `fc` = `files_changed`, `cm` = `commits`,
`rv` = `reviews`, `rc` = `review_comments`, `ic` = `issue_comments`.

### 4.1 Code-modification (M) — from `pr`, `fc`, `cm`

| Feature | Formula / source | Tier |
|---|---|---|
| `m_additions` | `pr.additions` (authoritative aggregate) | T1 |
| `m_deletions` | `pr.deletions` | T1 |
| `m_churn` | `additions + deletions` | T1 |
| `m_net_change` | `additions − deletions` | T1 |
| `m_changed_files` | `pr.changed_files` | T1 |
| `m_avg_file_churn` | `churn / max(changed_files,1)` | T1 |
| `m_frac_added` | `additions / (churn+1)` | T1 |
| `m_num_hunks` | count of `@@` markers across `fc.patch` | T1 |
| `m_files_added`, `m_files_modified`, `m_files_removed`, `m_files_renamed` | counts of `fc.status` values per PR. The guide's shorthand is `added/deleted/modified`; the implementation keeps `removed` and `renamed` as their own columns rather than folding `renamed` into `modified` — see the note at the top of `src/features/feature_extraction.py` | T1 |
| `m_num_commits` | `count(cm rows per pr_id)` | **T1, grows during review → dropped in V2** |

> Use `pr.additions/deletions` for line counts, **not** a sum over `fc.patch` — 8.8 % of changed
> files have a null `patch` (GitHub omits large/binary diffs), so summing patches undercounts.

### 4.2 Code-structure (S) — produced by Step 9 (`src/features/ast_cfg.py`)

| Feature | Definition | Tier |
|---|---|---|
| `s_ast_node_count` | total AST nodes over changed code files | T1 |
| `s_ast_max_depth` | max AST nesting depth | T1 |
| `s_cfg_node_count` | CFG basic-block count | T1 |
| `s_cfg_edge_count` | CFG edge count | T1 |
| `s_cyclomatic_proxy` | `E − N + 2·components` (McCabe proxy) | T1 |
| `s_num_functions_touched` | function/method defs intersecting the diff | T1 |
| `s_has_parseable_code` | 1 if PR has ≥1 AST-parseable file, else 0 | T1 |

**Coverage / fidelity caveat — must be reported honestly (rubric rewards it):**
- Only **~72 % of the 1,114 PRs** contain ≥1 file in a language we parse (Python/TS/C#). The
  other ~28 % (pure `.md/.json/.yml/.cpp/.c/.rs/...` changes) get **AST/CFG = 0 and
  `s_has_parseable_code = 0`**, so the model can tell "measured zero structure" apart from "not
  measured." Never silently impute a fake non-zero.
- Fidelity differs by language: **Python** gets a real CFG (`ast` + `staticfg`); **TypeScript/C#**
  get `tree-sitter` ASTs with an *approximate* CFG (basic-block detection over
  `if/for/while/switch/try`). Of ~12.8 k AST-parseable files (~6.7 k TS, ~4.6 k C#, ~1.4 k Py),
  full-fidelity Python CFG covers only the ~1.4 k Python slice, so structure features are
  strongest there — state this, don't hide it.

### 4.3 Textual (T) — from `pr`, `cm`

| Feature | Source | Tier |
|---|---|---|
| `t_title_len`, `t_title_wordcount` | `pr.title` | T0 |
| `t_body_len`, `t_body_wordcount` | `pr.body` | T0 |
| `t_body_is_empty` | `len(pr.body)==0` (only 2.1 % of PRs) | T0 |
| `t_commit_msg_len_first` | length of first commit message (by stored/API order) | T0 |
| `t_commit_msg_len_mean` | mean over `cm.message` lengths | T1 |
| `t_commit_msg_len_total` | sum over `cm.message` lengths | **T1, grows → dropped in V2** |

### 4.4 Metadata flagged as borderline — **excluded from V1/V2**

- `num_labels` (`len(pr.labels)`). Labels *look* like cheap metadata, but many are applied at or
  after merge (this dataset contains e.g. `release-cherry-pick`), so they leak. Kept out of the
  compliant sets; allowed only in the leaky ablation.

### 4.5 Process / forbidden (P) — quarantined, ablation-only

Explicitly **forbidden by §2.4.3** and/or T2-leaky. **Never in a graded/compliant result** — used
only in variant V0 to measure inflation:

`p_num_reviews` (`rv`), `p_num_reviewers` (distinct `rv.reviewer_login`),
`p_num_review_comments` (`rc` — *is* a review comment → forbidden),
`p_num_issue_comments` (`ic` — discussion → forbidden),
`p_has_changes_requested` / `p_has_approved` (`rv.state` — near-label leakage),
`p_is_ai_reviewed`, `p_n_ai_reviewers` (post-review AI signals).

---

## 5. The three feature sets

| Set | Contents | Count | Tiers | Compliant? | Purpose |
|---|---|---|---|---|---|
| **V1 — Per-spec** | S ∪ M ∪ T (§4.1–4.3) | 28 | T0+T1 | ✅ exactly §2.7.3 / §2.4.3 | **Primary graded results** |
| **V2 — Pre-review-strict** | V1 minus `m_num_commits`, `t_commit_msg_len_total` (the features that keep growing during review); **plus** the author-history pair `author_prior_merge_rate` and `author_prior_pr_count`, both computed *only* from PRs the same author closed **strictly before** this PR's `created_at` | 28 | mostly T0, hardened T1 | ✅ stricter than spec | Robustness / honesty check |
| **V0 — Leaky superset** | V1 ∪ P (§4.5) ∪ `num_labels` | 37 | +T2 | ❌ violates §2.4.3 | **Ablation only** — quantify leakage inflation |

*(V2 drops two features and adds two, so it is the same width as V1 — 28 each — which
is why the V1-vs-V2 comparison is not confounded by feature count. Counts exclude the
four carrier columns `pr_id` / `y` / `created_at` / `repo`.)*

**Why three, when the plan asked for two.** The plan wanted a "per-spec (with leaky features)"
vs "pre-review-only" comparison. But the *actual* per-spec set (V1) already excludes the leaky
process features, so that pairing collapses. The honest reconstruction splits the comparison in
two, and both belong in the report:

- **V1 vs V0** answers *"how much do the forbidden review-process signals inflate accuracy?"* —
  the concrete demonstration of **why §2.4.3 bans them**. Expect V0 to look impressive and be
  worthless: `p_has_approved` alone nearly reproduces the label.
- **V1 vs V2** answers *"even within the allowed features, how much does final-code-state
  contamination help vs a strict submission-time view?"* — the subtler, more defensible point.

Report V1 as the headline model; present V0 and V2 as the two flanking comparisons.

**Residual limitation to disclose:** V2 still cannot fully undo T1 leakage — truly reconstructing
the *opening* diff would need per-commit file snapshots we did not mine (the `commits` table stores
aggregate additions/deletions per commit but neither per-file patches nor commit timestamps, so
the opening code snapshot cannot be re-derived without re-mining). V2 is therefore "as strict as
the mined data allows," and the report should say so rather than overclaim a leak-free set.

---

## 6. Preprocessing (required by §2.4.3: "normalized … as input")

1. **Missing AST/CFG** → impute `0`, set `s_has_parseable_code = 0` (see §4.2).
2. **Skew transform.** Count/size features (`m_additions`, `m_churn`, `s_ast_node_count`, …) are
   heavy-tailed → apply `log1p` before scaling.
3. **Scaling.** `StandardScaler` (z-score) **fit on the training split only**, then applied to
   test — fitting on the full set would leak test distribution. Essential for SVM-RBF (distance-
   based); harmless for Random Forest. Booleans pass through as 0/1.
4. **Split.** **Time-based** on `pr.created_at` (range 2021-04 → 2026-07): train = older ~80 %,
   test = newest ~20 %; report the exact cutoff date. This avoids look-ahead leakage and mirrors
   real deployment (predict tomorrow's PRs from yesterday's) — a stronger choice than a random
   split and worth defending on-site.
5. **Imbalance.** With 74/26, run each model **plain vs `class_weight='balanced'`** and report
   **minority (not-merged) recall** alongside accuracy — accuracy alone is inflated by always
   predicting "merge."

---

## 7. Traceability to the lab guide's required outputs

**§2.8 required results → where they come from:** (1) human-written dataset stats → §1 table;
(3) feature-extraction examples → §4 catalogue on a sample PR; (6) feature importance → Random
Forest importances over **V1** (Step 12); (7) model comparison → SVM vs RF × {V0, V1, V2}.

**§2.9 reflection questions this spec is built to answer with numbers:**
- **Q4 "Which feature type contributes most to merge prediction? Why?"** → the S/M/T grouping in
  §4 lets Step 12 aggregate Random-Forest importance *by category*, giving a grounded answer
  instead of a single feature name.
- **Q5 "What additional preprocessing vs Experiment 1?"** → §6 is the checklist: human-written +
  closed filtering, AST/CFG generation, feature vectorisation, missing-coverage handling,
  log/normalisation, and the time-based split — none of which existed in Experiment 1's
  descriptive pipeline.

---

## 8. Handoff to Steps 9–12

- **Step 9** (`src/features/ast_cfg.py`) produces the **S** columns per §4.2, keyed by `pr_id`,
  including `s_has_parseable_code`.
- **Step 10** (`src/features/feature_extraction.py`) joins S with the **M**/**T** columns and emits
  three matrices — `features_v0.parquet`, `features_v1.parquet`, `features_v2.parquet` — plus the
  `y`/`created_at`/`repo` columns, so training can select a variant without recomputation.
- **Steps 11–12** train SVM + RF on each variant with the §6 pipeline and report the §7 outputs.
