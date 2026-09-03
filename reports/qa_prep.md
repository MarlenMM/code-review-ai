# On-Site Q&A Prep — Project II (Step 30)

*Talking points for all 20 official reflection questions (5 per experiment, from the lab
guide §1.9/2.9/3.9/4.9) plus Lecture 1's 6 discussion questions. Every number here was
re-verified in the Step 29 pass — see [`verification_signoff.md`](verification_signoff.md).*

**How to use this.** Each answer has three parts: **Say this** (~20–30 seconds out loud —
the answer on its own is enough), **Numbers** (have these ready, don't recite all of
them), **If pushed** (the follow-up they'll actually ask). Rehearse the *Say this* lines
aloud; do not read the Numbers rows verbatim — pick the one that fits.

**Three delivery rules.**
1. **Lead with the answer, then the evidence.** "Yes — 72.0%, and here's why it mattered"
   beats a two-minute build-up.
2. **Name one real number per answer, not five.** One remembered figure sounds like
   command; five recited figures sound like a script.
3. **When a limitation is the honest answer, say it first.** Section 8 exists because the
   rubric explicitly rewards "identify false positives and false negatives" and "discuss
   limitations" — volunteering a weakness is worth more marks than being caught on it.

---

## 1. Core numbers — the one page to memorize

| | |
|---|---|
| **Dataset** | 1,494 PRs, 5 repos, closed only. 1,075 merged / 419 not = **72.0%** |
| **AI-authored** | **380 (25.4%)**; 1,114 human. Merge rate **65.3% AI vs 74.2% human** |
| **AI-reviewed** | 1,064 PRs (**71.2%**) — Copilot reviewer on all 1,064, coderabbitai on 4 |
| **Detection precision** | 59 PRs / 68 assessments checked by hand, **0 false positives** |
| **Exp2 split** | 1,114 human PRs → **890 train / 224 test**, per-repo time-based |
| **Exp2 shipped model** | `rf_v1_balanced` — **76.8% acc, 42.5% not-merged recall** |
| **Leakage proof** | V0 (forbidden features) **0.97–0.99 ROC-AUC** vs V1's **0.56–0.68**; `p_has_approved` alone = **41.6%** of importance |
| **Exp3 sample** | **16 PRs** (a real subset of the 224), 14:2, **512/512 cells, 0 failures** |
| **Exp3 peak** | **93.75% acc / 0.816 macro-F1** — `few_shot`@`diff_commit_message`, tied by `role_based`@`diff_metadata` |
| **Exp3 pooled** | 84.8% acc / **0.525 macro-F1** / 9.4% not-merged recall — **11 of 16 configs never predict CLOSE** |
| **Exp4 sample** | **3 complete PRs** (2:1), all AI-authored, **148/384 cells, 0 failures** |
| **Exp4 headline** | `role_based`/`few_shot`/`cot` = **0.0 not-merged recall on every tier**; only `self_reflection` (0.60) and `multi_turn` (0.50) ever catch one |
| **Exp4 context** | repo context lifts recall **0.0 → 0.4**, macro-F1 peaks 0.498; `complete` tier regresses to **0.0** |
| **System** | fast = `rf_v1_balanced`; deep = `multi_turn`@`diff_repo_context`. **Measured** on Groq `llama-3.1-8b-instant`; **live today** on Qwen `qwen-plus` (Groq retired the model — see below) |
| **Tests** | 451 Python + 66 extension unit + 7 real-VS-Code integration |

**If you are asked why the provider changed** (the one thing on this page that is
not in the lab reports). Groq retired `llama-3.1-8b-instant` after Experiments 3
and 4 were run, so live calls started returning `404 ... does not exist`. The
answer to give is the *distinction*, not the swap: that model id was doing two
jobs — an **experimental constant** (every Exp3/4 number was measured on it, and
`data/llm_cache/`'s ~700 responses are keyed on it) and a **live dependency**.
Only the second broke. So the live path moved to Qwen, and the grid runners kept
`--provider groq` as their default, which means a re-run still replays Labs 3/4
from disk *exactly* and for free. The cache key includes the model, so the two
cannot mix even by accident. What is **not** claimed: that `multi_turn` @
`diff_repo_context` is still the best config for Qwen — that was measured on
Llama, and re-testing it means re-running the Exp4 grid. Full write-up:
`reports/api_design.md` §8.

**Two bases to have straight, in case you're asked to reproduce a figure.**
1. **"Pooled" means one confusion matrix over all the predictions, not the mean of
   the per-config scores.** For Exp3 accuracy (84.8%) and not-merged recall (9.4%)
   the two are identical, because every config sees the same 16 PRs. Macro-F1 is
   not linear in the confusion counts, so they differ slightly: **pooled 0.525,
   mean-of-16-configs 0.509.** If someone recomputes and gets 0.509, that's the
   mean — same finding, and the reports quote the pooled figure throughout.
2. **The ≥92.5% detection-precision bound comes from the 40-PR random draw**
   (3/40 = 7.5%), not from all 68 assessments. The 19-PR census is a full census
   of the weakest tier, not a random sample, so it can't be used to widen the
   bound — quoting 3/68 would overclaim.

**If you blank on a number:** *"I'd have to check the exact figure — it's in
`results/tables/exp2_metrics.json` — but the direction was X."* Never invent a digit.

---

## 2. Lecture 1 — general discussion questions

### L1. Which clone type is the most difficult to detect, and why?

> **Say this.** Type-4, semantic clones — two functions that do the same thing with no
> syntactic resemblance. Types 1–3 are progressively noisier versions of the *same* text,
> so token or AST similarity still finds them. Type-4 breaks that assumption entirely:
> there's no shared surface form to match, so you need a representation of *behaviour*,
> not structure. That's why Project I reaches for CodeBERT-style embeddings rather than
> tree diffing.

**If pushed — "you didn't do Project I, why should I believe you understand this?"**
Straight answer: I did Project II only. But the same lesson showed up in my own work —
my AST/CFG features capture structure, and structure was my *weakest* feature category
(15.8% of importance). Surface form and meaning aren't the same thing; that's the Type-4
problem in miniature.

### L2. What information is lost when code is represented only as text?

> **Say this.** Everything relational. Text gives you tokens in sequence; it doesn't give
> you scope, control flow, or the fact that a name on line 200 refers to a definition on
> line 12. In my Experiment 2, a text-only view could tell me a PR added 268 lines but not
> whether those lines added a branch. That's exactly what `s_cyclomatic_proxy` measures,
> and you cannot compute it from raw text — you need the parse.

**Numbers.** PR `vscode#326961`: 268 added lines (textual), but AST depth 50, 19 CFG nodes,
23 functions touched — structural facts invisible to a line count.

### L3. What is the difference between a commit and a Pull Request?

*(Also Experiment 1 Q1 — see §3 for the fuller version.)*

> **Say this.** A commit is a Git object: an immutable snapshot with a hash, an author and
> a message. A Pull Request is a GitHub workflow object that wraps one or more commits for
> review, and unlike a commit it's mutable and social — it has reviewers, discussion, labels
> and a final merge/close decision. In my dataset that's literally the table relationship:
> 1,494 PRs carrying 6,861 commits, averaging 4.59 commits per PR.

### L4. Can code review be performed using only the changed lines?

> **Say this.** Sometimes — and my Experiment 4 measured exactly when it isn't enough. On
> AI-generated code, a bare diff gave **0.0** not-merged recall: the model kept saying
> "this looks locally fine, merge it." Adding repository context took that to **0.4**. So
> the changed lines are sufficient to spot local defects, but not to spot a change that is
> internally correct and still wrong for the codebase it's landing in.

**If pushed — "so more context is always better?"** No, and I have the counter-example. My
`complete_se_context` tier stacks every source at once and falls **back to 0.0** recall.
More context isn't monotonically better; the well-chosen mid-tier beat the everything-tier.

### L5. Why might AI-generated code require a different review strategy?

> **Say this.** Because the prompting that works on human code demonstrably stops working.
> The three strategies I carried over from Experiment 3 — role-based, few-shot,
> chain-of-thought — scored **exactly 0.0** not-merged recall on AI-generated code, on
> every context tier, with no exception. Only the two new strategies caught anything. My
> read is that those strategies were exploiting human-specific cues — an author's stated
> intent, a maintainer persona calibrated to human contribution norms — and AI-authored
> PRs don't carry them. What transfers is changing *how* the model reasons, not what it's shown.

**This is the single best answer in the deck — it's the project's central finding.** If you
get one chance to sound like you understand your own results, use this one.

### L6. What makes an AI software engineering tool genuinely useful?

> **Say this.** That it fits the developer's existing loop and is honest about its own
> confidence. Mine is a VS Code command over the working diff, not a separate web app —
> and the default mode is the free, instant model, with the LLM pass opt-in because it
> spends real quota. When the LLM fails it degrades to a warning field, never a 500. And
> the click-to-jump anchoring shows its confidence level rather than pretending every
> comment is certainly located.

**If pushed — "usable, not merely runnable" (the rubric's phrase).** I can install the
packaged `.vsix` and click the button; 7 of my tests run inside a real, unmodified VS Code
instance doing exactly that — real `git diff`, real backend call, real panel, real
click-to-jump navigation.

---

## 3. Experiment 1 — dataset

### E1.1 Differences between a Pull Request and a commit?

> **Say this.** A commit is an atomic, immutable Git snapshot identified by a content
> hash — it has no notion of review or approval. A Pull Request is a GitHub-level, mutable
> object that groups commits for review, and it carries everything the review process
> needs: title, description, labels, reviewers, discussion threads, and a final merge/close
> decision. My data model mirrors that exactly: `pull_requests` is the review unit,
> `commits` is a child table keyed by `pr_id` — 1,494 PRs, 6,861 commits, 4.59 per PR.

**If pushed — "why does the distinction matter for your project?"** Because the label I
predict — merged or not — only exists at PR level. A commit has no merge outcome to learn from.

### E1.2 Why does code review improve software quality?

> **Say this.** Three mechanisms, and I can point at all three in my own data. Defect
> catching — my AI reviewer example flags a concrete `StopIteration` crash from a missing
> dictionary key. Consistency enforcement — a human reviewer in my sample writes "the same
> issue at line 79, for consistency," which is literally that act. And knowledge transfer —
> review threads are a running record of design rationale. What's new in this dataset is
> scale: 71.2% of my PRs now get an automated first-pass review layer *in addition to*
> human review, not instead of it.

### E1.3 In what practical scenarios can merge prediction be applied?

> **Say this.** Four I'd defend. Maintainer triage — surface PRs unlikely to merge for
> early feedback instead of a long silent wait. CI resource allocation — spend expensive
> build minutes on PRs likely to land. Contributor-facing warning before a PR is even
> opened. And backlog health monitoring. My own per-repo merge rates range 63.0% to 81.9%,
> so a flat repository-wide prior already can't distinguish "typical successful PR" from
> "resembles our historically-rejected ones" — that's the gap a model fills.

**If pushed — "would you actually deploy a 76.8% model?"** Not as a gate. As a triage
*prior* that orders a maintainer's queue, where being wrong costs a wasted glance. That's
exactly why the tool is an advisory panel, not a merge blocker.

### E1.4 Differences in review approaches between AI and human reviewers?

> **Say this.** My retrieved examples show it concretely. The AI reviewer's comment is a
> single self-contained technical observation naming a specific failure mode — consistent
> with pattern-matching the diff in isolation. The human comments are conversational turns
> in a thread — "Thanks for the feedback! I'll remove it" — or they reference project-wide
> context beyond the diff. And the AI style is uniform: `copilot-pull-request-reviewer` is
> present on all 1,064 AI-reviewed PRs, one high-volume consistent voice, where human style
> varies per reviewer. That difference is exactly what motivated giving the LLM richer
> context in Experiments 3 and 4.

### E1.5 Does the dataset suffer from class imbalance? How can this be addressed?

> **Say this.** Yes — 72.0% merged versus 28.0% not, and it isn't uniform: 63.0% in
> semantic-kernel up to 81.9% in dotnet/runtime. It's moderate but real, and the important
> consequence is that a trivial always-predict-merged classifier scores 72% accuracy, so
> raw accuracy is misleading on its own. I addressed it three ways in Experiment 2: a
> time-based split rather than random, `class_weight='balanced'` compared against the
> unweighted model, and reporting per-class precision/recall/F1 — specifically minority
> recall — throughout instead of accuracy alone.

**If pushed — "is 72/28 really imbalanced?"** It's mild by ML standards — I wouldn't call
it severe. But it's enough that it changed my model *choice*: I shipped the balanced
variant at 76.8% accuracy over the plain one at 82.1%, because the plain one only catches
10 of the 40 genuinely-not-merged PRs and the balanced one catches 17.

---

## 4. Experiment 2 — features and traditional ML

### E2.1 Why construct intermediate representations like ASTs and CFGs?

> **Say this.** Because raw diff text tells you *how much* changed, not *what kind* of
> change it was. An AST gives you syntactic structure — function boundaries, nesting depth.
> A CFG gives you control flow directly, which is the only way to compute a McCabe-style
> complexity number; there's no way to get that from text. And it earned its place: even
> with real coverage gaps, structure features still carry 14.6–15.8% of Random Forest's
> importance, so they're contributing signal that line counts and text lengths don't already have.

**If pushed — "only 15%? was it worth it?"** That 15.8% is a *floor*, not a verdict. I can
only measure structure for 71.8% of PRs, and among the parseable Python hunks `staticfg`'s
CFG stage fails on about 27% more. So the honest statement is that structure's true ceiling
is unknown — I'd need to fix the coverage before concluding it's weak.

### E2.2 Why does traditional ML require manual feature engineering?

> **Say this.** Because SVM and Random Forest consume fixed-length numeric vectors. They
> have no mechanism for learning a representation from raw code or a graph — a human has
> to decide what to count first. In my case that meant freezing an explicit
> structure/modification/text taxonomy *before* writing any extraction code, then making
> per-column transform decisions: `log1p` for heavy-tailed counts, a sign-preserving log
> for the one feature that goes negative, scaling fit strictly on the training split. All
> of that is human judgment spent before a single model is trained — and it's exactly the
> axis Experiments 3 and 4 differ on, where the LLM reads the diff directly.

### E2.3 For which scenarios are SVM and Random Forest respectively suitable?

> **Say this.** Textbook: SVM with an RBF kernel suits smaller, cleanly-scaled numeric
> spaces where you expect a smooth non-linear boundary; Random Forest suits larger,
> heterogeneous, less carefully-scaled feature sets and gives you interpretability for
> free. My own data adds a sharper point: Random Forest's `class_weight='balanced'`
> behaved exactly as advertised — minority recall up, accuracy down, 0.250 to 0.425 — while
> SVM's did the *opposite*, recall actually fell from 0.100 to 0.025. My explanation is that
> SVM's probabilities route through a separate `CalibratedClassifierCV` pass that isn't
> jointly optimized with the class-weight shift to the decision boundary. So on this
> dataset, when imbalance handling is the point, Random Forest is the predictable choice.

**This is a strong answer — it's a real, unexpected, explained finding rather than a
textbook recital. Lead with the SVM anomaly if time is short.**

### E2.4 Which type of feature contributes most to merge prediction? Why?

> **Say this.** Modification and textual features, roughly equally — 44.2% and 40.0% in V1.
> The reason is coverage: every PR always has line counts and title/body lengths, fully
> populated. Structure trails at 15.8%, but as I said, that's a measurement limitation, not
> a verdict on structure. The genuinely interesting result is in V2, where a single
> non-code feature — the author's prior merge rate — outranks every individual code-derived
> feature at 10.0%. Who wrote it predicts more than any one thing about what they wrote.

### E2.5 What preprocessing did Experiment 2 add compared to Experiment 1?

> **Say this.** Experiment 1 was purely descriptive — flatten to tables, compute
> distributions. Experiment 2 added five things: filtering to human-written closed PRs;
> generating an AST/CFG representation that didn't exist at all in Experiment 1;
> vectorizing every PR into a fixed-length row under an explicit category taxonomy, with
> missing structural coverage marked honestly rather than imputed; column-role-aware
> transforms with the scaler fit only on train; and a per-repository time-based split.
> None of the last four has any equivalent in Experiment 1.

**If pushed — "why per-repository and not one global time cutoff?"** I tried the global
cutoff and it broke: semantic-kernel's test slice collapsed to **1 PR out of 274**, and
test merge rate jumped to 87.9% against 70.8% in train. That's an artifact of my
recency-biased sampling window, not a real temporal boundary. Splitting per repo gives
890/224 with every repo proportionally represented.

---

## 5. Experiment 3 — LLM review of human-written code

### E3.1 Why does code context affect LLM inference?

> **Say this.** Because the model has nothing except what's in the prompt. A human reviewer
> can click the linked issue or browse the file; the LLM's entire understanding is the text
> I hand it. Different tiers hand it different evidence — the description gives the author's
> stated intent, the commit message a compressed narrative, metadata gives process signals
> a maintainer would glance at. The cleanest proof that context carries real signal: the
> same strategy moves from 0.00 to 0.50 not-merged recall purely by changing which context
> tier it reads. Nothing about the model changed.

### E3.2 Advantages of few-shot over zero-shot prompting?

> **Say this.** Demonstrations let the model infer task-specific patterns without any
> parameter updates, and can recalibrate it away from a generic prior. My result is a
> sharper version of that: few-shot had by far the highest minority-class recall — 0.250,
> against 0.125 for role-based and **exactly 0.000** for both zero-shot and
> chain-of-thought — and simultaneously the *lowest* accuracy, 0.750. Those are the same
> fact. Its balanced demonstrations pulled the model away from "always predict the majority
> class," which costs accuracy under an 87.5%-merged reality and buys the only recall in
> the grid. It's Experiment 2's class-weight trade-off reappearing as a prompt-design lever.

### E3.3 Which type of context contributes most to Merge Prediction? Why?

> **Say this.** On raw averages, diff-plus-commit-message leads on both accuracy (0.891)
> and macro-F1 (0.554). The commit message adds a compact statement of intent for very few
> tokens, which is close to what a maintainer actually weighs. But the honest answer is
> that the rankings *disagree*: metadata is last on accuracy at 0.813 yet second on
> macro-F1, because paired with the role-based persona it produced one of the two best
> cells in the whole grid. So it isn't "one context dominates" — it's that which context
> helps depends on the strategy consuming it. That's a genuinely different conclusion from
> Experiment 2, where modification and text features dominated regardless of model.

### E3.4 Why do different prompt designs produce comments of varying quality?

> **Say this.** Because each strategy optimizes a structurally different objective and it
> shows. Zero-shot has no anchor beyond "write a review," so it defaults to generic
> coverage. Few-shot is pulled toward the *style* of real human comments by its
> demonstrations — which is exactly what BLEU and ROUGE measure, and few-shot duly tops
> every text metric. Chain-of-thought forces analyze-then-prioritize. Role-based imposes an
> explicit priority order. My qualitative example shows the difference is in kind, not
> degree: the generated review is accurate and on-topic but three sentences per point,
> where the real reviewer needed two words — "Why drop this test?"

**If pushed — "your BLEU scores are near zero, isn't the generation just bad?"** BLEU
around 1 and ROUGE-L around 0.1 are very low in absolute terms, and I say so rather than
dress it up. But n-gram overlap is largely a length-and-style proxy: a generated review and
a human review can both be correct while sharing almost no n-grams. That's why I paired
every score with a side-by-side example. The *ordering* is still meaningful — and the
ordering produced a real finding, which is the next question.

### E3.5 Advantages and disadvantages of LLMs vs. the Experiment 2 model?

*(The guide writes "the deep learning model used in Experiment 3" — I read that as the
Experiment 2 trained models, since that's the comparison that exists in this project.
Worth saying out loud; it shows you read the guide rather than pattern-matched it.)*

> **Say this.** Advantages: zero training cost — the LLM was never fitted to this dataset,
> yet its best configuration reached 93.75% accuracy and 0.816 macro-F1 with the same
> 14/1/1/0 confusion matrix as my best trained models on the identical 16 PRs. It also does
> something the ML model fundamentally cannot — generate free-text review comments, not
> just a label. And chain-of-thought makes part of its reasoning inspectable.
> Disadvantages: it's volatile. That peak is one cell out of sixteen; pooled across all of
> them macro-F1 is **0.525**, and eleven of the sixteen never predict CLOSE at all. It costs
> real API quota and 60 to 1,090 milliseconds per call where the Random Forest is free and
> instant. So: higher ceiling, much lower floor, and the floor is what you'd ship.

**Know this trap.** The peak matched `rf_v1_plain` and `svm_v1_plain`. It did **not** match
`rf_v1_balanced` — the model I actually deployed — which scores 0.75 accuracy on that same
16-PR subset. If asked "which trained model did it match," say so plainly. See §8.2.

---

## 6. Experiment 4 — reviewing AI-generated code

### E4.1 Why does repository-level context improve review performance?

> **Say this.** Because an AI agent generates a change by pattern-matching the immediate
> task, with no guarantee it modelled how that change fits the rest of the codebase. That's
> the failure repository context is aimed at, and it moved the number: not-merged recall
> from **0.00 to 0.40**, and macro-F1 to its peak of 0.498 across all five tiers. You can
> see the mechanism in the transcripts — given a slightly wider frame, the model starts
> reasoning about whether maintainers would find a rename unnecessary, a judgment the bare
> diff never prompts.

**Always add the caveat unprompted:** it isn't monotonic. Stacking *every* context source
regresses to 0.00. That's a finding, and volunteering it is worth more than being asked.

### E4.2 Which prompt design is best for reviewing AI-generated code? Why?

> **Say this.** Multi-turn — highest macro-F1 at 0.486, recall 0.50, without the accuracy
> collapse self-reflection suffers. And the reason is mechanistic, not incidental: multi-turn
> commits to a diff-only judgment first, then reveals the wider context and asks the model
> to reconsider *specifically in light of it*. So it revises only when the context actually
> contradicts the local view. Self-reflection has the single highest recall, 0.60, but
> critiques its own draft regardless of whether the draft was right — I have a transcript
> where the critique talks a correct MERGE into an incorrect CLOSE. For AI-generated code
> the premise is that the local diff is *sometimes* misleading, not always, and multi-turn
> is the strategy built to tell those apart.

### E4.3 What issues do prompt optimization and context augmentation each address?

> **Say this.** Different deficits, and I kept the axes separable so it's attributable.
> Context augmentation fixes an **information** deficit — the model can't judge what it was
> never shown. Prompt optimization fixes a **disposition** deficit — even given rich
> context, role-based, few-shot and CoT never once caught a not-merged AI PR. They weren't
> short of information; they were defaulting to "looks locally fine, merge." The clean
> confirmation is that the *same* repository-context tier drives self-reflection and
> multi-turn to their best recall while leaving the other three at 0.00 on that identical
> tier. Information without a disposition to use it skeptically accomplished nothing here.

**This is the second-best answer in the deck. It's the one that shows experimental design
thinking rather than result-reporting.**

### E4.4 Are different prompt combinations applicable to all review tasks?

> **Say this.** No — and unevenly. For merge prediction the ranking is unambiguous:
> self-reflection and multi-turn are strictly necessary to catch the minority class at all;
> the other three are structurally incapable of it on this sample. For comment generation
> it's a different picture — role-based is far the most verbose at 8.5 comments per review,
> driven by its priority-order persona, while multi-turn sits second at 5.7, above both
> single-shot alternatives it's compared against but nowhere near role-based; its advantage
> is *specificity* after the second turn, not volume. So the best strategy is task-dependent.
> A system built on this evidence picks multi-turn for the merge path to protect recall,
> and treats comment generation as a separate choice — which is exactly what my backend does.

### E4.5 What else could be explored for reviewing AI-generated code?

> **Say this.** Five, in the order I'd actually do them. First, enable `--fetch-issues` —
> the code exists and is tested, but wasn't switched on for this run, so the issue tier's
> real effect is still unmeasured; that's the cheapest missing result. Second, a real
> repository fetch — full file bodies and an actual call graph instead of my lightweight
> git-heading approximation — to test whether the tier that already helps helps more.
> Third, a larger sample, because three PRs fix a direction and not a magnitude. Fourth, a
> model bigger than 8B. Fifth, retrieval-augmented context and multi-agent review, which is
> what the guide's own extension section suggests.

**If pushed — "why didn't you just enable `--fetch-issues`?"** Operational oversight,
caught after the daily quota was already exhausted. I chose to report the tier honestly as
"no issue data" rather than quietly drop it or re-run and overwrite. It's flagged at the
table itself in the report, not buried.

---

## 7. Live demo script (≈2 minutes)

The rubric asks for a tool that is *usable*, so demo rather than describe.

1. **Open a repo with uncommitted changes.** Say: "this is a real working diff, nothing staged."
2. **Command Palette → "Code Review AI: Review Current Changes."**
3. **Point at the gauge.** "That probability is the Random Forest from Experiment 2 —
   free, instant, no network. This is the default because it always works."
4. **Expand the feature list.** "These are the 28 V1 features it used — I return them
   deliberately so you can see *why*, not just the score."
5. **Switch to deep mode.** "Now it also calls the LLM — Experiment 4's best config,
   multi-turn on repository context. This spends real quota, which is why it's opt-in."
6. **Click a comment → jumps to the line.** "The model returns prose with no location, so
   this anchoring is a client-side heuristic — and it shows its own confidence rather than
   pretending it's certain."

**Have ready if the network/quota fails mid-demo:** *"That's the degradation path working —
it falls back to the fast-mode result plus a warning field rather than a 500. I'd rather
show an honest 'unavailable' than a canned fake review."* A failed deep call is a
demonstration, not a disaster — say so confidently.

---

## 8. Where I'm vulnerable — rehearse these hardest

These are the questions a sharp instructor actually asks. Each has an honest answer;
none requires defending something indefensible.

### 8.1 "Your Experiment 4 conclusions rest on 3 PRs. Is that meaningful?"

> Not as an effect size, and I don't claim one. What it *is* is a fully **paired** design:
> the same 3 PRs went through all 24 tier-by-strategy combinations — 148 cells, zero
> failures. That controls for PR difficulty, so when three strategies score 0.0 recall and
> two score 0.5 on the *identical* PRs with the *identical* context, the difference is
> attributable to the strategy rather than to which PRs I happened to draw. Three PRs fix a
> direction; they cannot fix a magnitude, and every claim in Lab 4 is worded as direction.
> The constraint was Groq's 500,000-tokens-a-day free tier shared across the account — not a
> design choice.

### 8.2 "You say the LLM matched your trained model. Which trained model?"

> `rf_v1_plain` and `svm_v1_plain` — both hit 93.75% accuracy, 0.816 macro-F1 and the same
> 14/1/1/0 matrix on those 16 PRs. It did **not** match `rf_v1_balanced`, which is the model
> I actually deployed; that one scores 0.75 on the same subset. So the accurate claim is
> that the LLM's best configuration reached the operating point of my best
> *accuracy-oriented* models, on a 16-PR sample where accuracy is close to uninformative —
> an always-MERGE baseline scores 87.5% there for free. I'd be overclaiming if I said it
> beat what I shipped.

**Volunteering this before being asked is worth real marks.**

### 8.3 "Your V1 ROC-AUC is 0.56–0.68. That's barely better than chance."

> Correct, and I report it rather than hide it. Three things. That's the honest cost of the
> guide's own constraint — no review-process features. The moment I add them back, V0 hits
> 0.97–0.99, which proves the ceiling isn't the model, it's the information available
> before review happens. Second, the compliant space isn't exhausted: one legitimate
> non-leaky feature — author history — takes V2 to 0.74. Third, it's precisely why the tool
> ships this as a fast triage prior with an opt-in LLM pass, not as an oracle or a merge gate.

### 8.4 "76.8% accuracy is *worse* than always predicting merge (82.1%)."

> Yes — and that's the clearest illustration in the whole project of why accuracy is the
> wrong metric here. Always-MERGE scores 82.1% accuracy, 0.0 minority recall and 0.451
> macro-F1. My model scores 76.8%, 42.5% recall and 0.626 macro-F1. Concretely: of the 40
> PRs in the test set that genuinely didn't merge, the constant baseline flags **zero** and
> my model flags **17**. For a tool whose entire job is surfacing PRs worth a second look,
> a model that never flags anything is worthless no matter what its accuracy says.

### 8.5 "Why deploy V1 when V2 is measurably better?"

> Because V2's advantage comes from author-history features, and those need a
> repo-plus-author lookup against my mined PR history. An arbitrary diff arriving at the
> API — from any contributor, in any repository, not necessarily one of my five — doesn't
> carry that. V1 is the variant computable for *any* diff. If I scoped the extension to the
> five mined repos and authenticated the author, V2 would be the better choice; that's
> written up in `api_design.md` as the concrete next step.

### 8.6 "Why Groq? The plan said Gemini."

> Gemini's free tier turned out to be unusable from this account and region without
> enabling billing, and I exhausted the documented workarounds before switching. Groq's
> free tier worked, so Experiments 3 and 4 both ran on `llama-3.1-8b-instant`. That's
> documented in Lab 3's Problems section rather than quietly retconned. The honest cost:
> 8B is a small model, which bounds review-comment quality and is part of why I read
> direction rather than magnitude throughout.

### 8.7 "How do you know your AI-authorship detection isn't just wrong?"

> Three tiers, and I checked it by hand. The primary signal is GitHub's own `__typename` —
> structural, not a guess. Then an allowlist separates genuine AI agents from non-AI
> automation like dependabot; anything Bot-typed but on neither list is reported as
> `other_bot` rather than guessed into a bucket. Then I manually reviewed 59 distinct
> flagged PRs — 40 random plus a full census of the 19 flagged only by the weaker
> commit-trail signal — 68 assessments, **zero false positives**. On the 40 random
> draws that gives ≥92.5% precision at 95% confidence by the rule of three; I quote the
> bound on the random 40 rather than on all 68, because the census of 19 isn't a random
> sample and using it would overstate the bound. One caveat I'll volunteer: that measures
> precision, not recall. A human who pasted Copilot output with no commit trail is a false
> negative by construction, and I say so in the report.

### 8.8 "Your click-to-jump is a heuristic. What if it points at the wrong line?"

> Then the user sees a weaker confidence marker, because the UI surfaces the anchor's
> precision rather than hiding it behind a green check. The alternative — making the model
> emit line numbers — would have meant changing Experiment 4's prompt contract, which would
> invalidate the completed grid and its cache. I chose not to break a finished experiment
> for a UI convenience. Real deep-mode output drove the design, incidentally: the first pass
> resolved 3 of 4 comments, and indexing unchanged *context* lines rather than only added
> lines took it to 4 of 4.

### 8.9 "What would you do differently if you started again?"

> Enable `--fetch-issues` from the first run — it cost me a whole tier of results for no
> good reason. Budget the LLM quota across experiments up front instead of letting
> Experiment 3 consume most of the day's tokens before Experiment 4 started, which is the
> direct cause of the 3-PR sample. And fix the `staticfg` CFG bug or swap the library
> earlier, because that's what capped my structure features at a coverage I can't defend as
> representative.

---

## 9. If you genuinely don't know

Say one of these and move on. Do not improvise numbers.

- *"I didn't measure that — what I did measure was X, and the closest thing I can say is…"*
- *"That's in `results/tables/…json`; I'd rather check than guess the digit."*
- *"That's a real gap. It's in my limitations section for exactly that reason."*

The rubric rewards discussing limitations. An honest "I didn't test that" followed by what
you *would* test costs almost nothing; a fabricated number that unravels costs a lot.
