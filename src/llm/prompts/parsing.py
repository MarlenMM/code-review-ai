"""Parsers for the two output contracts the Experiment 3 prompts impose.

A prompt that ends with ``DECISION: MERGE`` is only useful if something turns
that back into a label the scorer (Step 17) can compare against the ground
truth. The parser and the output contract are two halves of one design
decision, so they live next to the templates rather than in the eval harness:
if the required format in `templates.py` changes, this file must change with
it, and the shared tests catch a drift between them.

Robustness posture
------------------
LLM output is not guaranteed to obey a format, and the merge-prediction metrics
(accuracy/precision/recall/F1) are only meaningful if an unparseable answer is
recorded as *unparseable* rather than silently coerced to a class. So:

* `parse_merge_prediction` returns ``label=None`` when it genuinely cannot find
  a decision -- Step 17 decides how to handle abstentions (e.g. count them as
  wrong, or report a coverage figure), a policy choice that does not belong in
  the parser.
* Both parsers are tolerant of the cosmetic noise real models add around the
  contract -- markdown bold/`**`, code fences, trailing prose -- because
  penalising the model for wrapping ``DECISION: MERGE`` in ``**...**`` would
  measure formatting compliance, not review ability.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Tokens a model uses for each class, beyond the canonical MERGE/CLOSE the
# contract asks for. Matching is by whole-word regex so "CLOSE" never matches
# inside "disclosed". Two tiers:
#   * STRONG tokens are decision verbs unlikely to appear incidentally in
#     reasoning prose ("merge"/"close"/"accept"/"reject"/...). Safe to scan for.
#   * WEAK tokens ("yes"/"no") are meaningful right after "DECISION:" but far
#     too common in ordinary text to trust in a free scan -- so they are used
#     ONLY when parsing an explicit DECISION line, never in the fallback.
_MERGE_STRONG = ("merged", "merge", "accepted", "accept", "approved", "approve")
_CLOSE_STRONG = ("closed", "close", "rejected", "reject", "declined", "decline")
_MERGE_WEAK = ("yes",)
_CLOSE_WEAK = ("no",)

_DECISION_LINE_RE = re.compile(r"decision\s*[:=]\s*(.+)", re.IGNORECASE)
_CONFIDENCE_LINE_RE = re.compile(r"confidence\s*[:=]\s*([0-9]*\.?[0-9]+)\s*(%?)", re.IGNORECASE)
_MERGE_STRONG_RE = re.compile(r"\b(" + "|".join(_MERGE_STRONG) + r")\b", re.IGNORECASE)
_CLOSE_STRONG_RE = re.compile(r"\b(" + "|".join(_CLOSE_STRONG) + r")\b", re.IGNORECASE)
_MERGE_ALL_RE = re.compile(r"\b(" + "|".join(_MERGE_STRONG + _MERGE_WEAK) + r")\b", re.IGNORECASE)
_CLOSE_ALL_RE = re.compile(r"\b(" + "|".join(_CLOSE_STRONG + _CLOSE_WEAK) + r")\b", re.IGNORECASE)

_REVIEW_BLOCK_RE = re.compile(r"<review>\s*(.*?)\s*</review>", re.IGNORECASE | re.DOTALL)
# Tolerate a missing closing tag (observed in ~19% of real Groq
# llama-3.1-8b-instant responses, likely truncation): match everything after
# the opening tag rather than falling all the way back to the raw response,
# which would leak the literal "<review>" string itself as a bogus first
# bullet (confirmed on ~14% of real responses before this fix).
_REVIEW_OPEN_ONLY_RE = re.compile(r"<review>\s*(.*)", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class MergePrediction:
    """Parsed result of a merge-prediction prompt. `label` is ``"MERGE"``,
    ``"CLOSE"`` or ``None`` (unparseable). `confidence` is the model's stated
    probability that the PR was merged, in ``[0, 1]``, or ``None`` if absent."""

    label: str | None
    confidence: float | None
    raw: str

    @property
    def merged(self) -> bool | None:
        """Boolean view aligned with the dataset's ``y`` (1 = merged): ``True``
        for MERGE, ``False`` for CLOSE, ``None`` if unparseable."""
        if self.label is None:
            return None
        return self.label == "MERGE"


@dataclass(frozen=True)
class GeneratedReview:
    """Parsed result of a review-comment prompt. `comments` is the list of
    individual review comments (bullet lines) with the reasoning/persona
    preamble stripped; `text` is them rejoined for BLEU/ROUGE against the human
    comments (Step 17)."""

    comments: list[str]
    text: str
    raw: str


def _classify_token(text: str, merge_re: re.Pattern, close_re: re.Pattern) -> str | None:
    """Map a snippet to MERGE/CLOSE by which class token appears *last* in it
    (models often write "not closed, so MERGE"); ``None`` if neither appears.
    Caller chooses whether to include the weak yes/no tokens via the patterns."""
    merge_hit = None
    close_hit = None
    for m in merge_re.finditer(text):
        merge_hit = m.start()
    for m in close_re.finditer(text):
        close_hit = m.start()
    if merge_hit is None and close_hit is None:
        return None
    if merge_hit is None:
        return "CLOSE"
    if close_hit is None:
        return "MERGE"
    return "MERGE" if merge_hit > close_hit else "CLOSE"


def parse_merge_prediction(raw: str) -> MergePrediction:
    """Extract the ``DECISION``/``CONFIDENCE`` the merge-prediction contract
    asks for, tolerant of markdown and surrounding prose.

    Strategy: prefer the *last* explicit ``DECISION:`` line (last, because a
    Chain-of-Thought answer may mention "merge" many times while reasoning and
    only the final labelled line is the commitment); the weak yes/no tokens are
    honoured there. Only if there is NO ``DECISION:`` line at all do we fall
    back to the response's *last non-empty line* using strong tokens only -- so
    a model that concluded "...therefore it will be merged." is still
    classified, while ordinary prose containing "no"/"yes" is not mistaken for a
    decision. Genuinely format-less answers return ``label=None`` (abstention)
    rather than a coin-flip.
    """
    if not isinstance(raw, str) or not raw.strip():
        return MergePrediction(label=None, confidence=None, raw=raw if isinstance(raw, str) else "")

    label: str | None = None
    decision_matches = _DECISION_LINE_RE.findall(raw)
    if decision_matches:
        # a DECISION line was given -- classify it (weak tokens allowed) and
        # respect its ambiguity as abstention rather than scanning prose
        label = _classify_token(decision_matches[-1], _MERGE_ALL_RE, _CLOSE_ALL_RE)
    else:
        # no DECISION line -- scan only the final non-empty line, strong tokens
        tail = next((ln for ln in reversed(raw.splitlines()) if ln.strip()), "")
        label = _classify_token(tail, _MERGE_STRONG_RE, _CLOSE_STRONG_RE)

    confidence: float | None = None
    conf_match = _CONFIDENCE_LINE_RE.search(raw)
    if conf_match:
        value = float(conf_match.group(1))
        if conf_match.group(2) == "%":
            value /= 100.0
        confidence = max(0.0, min(1.0, value))

    return MergePrediction(label=label, confidence=confidence, raw=raw)


def _split_comment_lines(block: str) -> list[str]:
    """Split a review block into individual comments. Accepts '-'/'*'/'•'
    bullets or numbered lines; falls back to non-empty lines."""
    comments: list[str] = []
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # strip a leading bullet or "1." / "1)" enumerator
        stripped = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", stripped)
        stripped = stripped.strip()
        if stripped:
            comments.append(stripped)
    return comments


def parse_review_comments(raw: str) -> GeneratedReview:
    """Isolate the review comments from a review-comment response.

    Prefer the ``<review>...</review>`` block the contract asks for; this is
    what lets BLEU/ROUGE (Step 17) compare the *comment body* to human comments
    rather than being polluted by a Chain-of-Thought analysis or a role-based
    preamble that precedes the block. If the closing tag is missing (a real,
    non-rare failure mode -- see `_REVIEW_OPEN_ONLY_RE`), everything after the
    opening tag is used instead of falling all the way back to the raw
    response. Only with no opening tag at all does it fall back to the whole
    response, so nothing is silently dropped.
    """
    if not isinstance(raw, str) or not raw.strip():
        return GeneratedReview(comments=[], text="", raw=raw if isinstance(raw, str) else "")

    match = _REVIEW_BLOCK_RE.search(raw)
    if match:
        block = match.group(1)
    else:
        open_match = _REVIEW_OPEN_ONLY_RE.search(raw)
        block = open_match.group(1) if open_match else raw
    comments = _split_comment_lines(block)
    text = "\n".join(comments)
    return GeneratedReview(comments=comments, text=text, raw=raw)
