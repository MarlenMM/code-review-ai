"""AI-authorship / AI-reviewer detection for mined PRs.

This is the research core of Experiment 1, so the design is deliberately
*layered* rather than a single string match, and every verdict is auditable
(carries a `method`, a `confidence`, and human-readable `evidence`) so a
random sample can be spot-checked by hand in Step 5.

Signal hierarchy, strongest first:

1. Structural primary signal — GitHub's own GraphQL `__typename`. Every actor
   is typed `Bot` or `User` by GitHub itself; this is far more reliable than
   guessing an account's nature from its username.

2. Semantic sub-classification of `Bot` actors — a `Bot` is not necessarily an
   *AI* bot. `dependabot`, `renovate`, `github-actions`, `codecov`, `stale`
   are ordinary automation. We split Bot actors against two curated allowlists
   (AI *coding* agents vs. AI *review* bots); anything Bot-typed but on neither
   list is reported honestly as `other_bot`, never guessed into an AI bucket —
   an unclassified bucket is more defensible than a false positive.

3. Secondary, lower-confidence signal — a PR opened by a *human* whose commit
   trail nonetheless shows AI involvement (an agent's commit identity/email, or
   Copilot's literal "Initial plan" placeholder first commit). These are kept
   as a genuinely distinct `human_opened_ai_assisted` category, reported
   separately from fully agent-authored PRs, and at explicitly lower confidence.

The module consumes the raw record produced by
`github_client.GitHubClient.fetch_pr` and depends on nothing else, so it stays
unit-testable against hand-built dicts with no network or token.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# --------------------------------------------------------------------------- #
# Curated allowlists. Kept as module-level constants (normalized: lowercase,
# no trailing "[bot]") so they're easy to audit, cite in the report, and extend.
# --------------------------------------------------------------------------- #

# Accounts that *author* code (open PRs / write commits) using an LLM agent.
AI_CODING_AGENT_LOGINS: frozenset[str] = frozenset({
    "copilot-swe-agent",   # GitHub's own Copilot coding agent (app/copilot-swe-agent)
    "copilot",             # display login the coding agent's Bot author sometimes uses
    "cursor-agent",        # Cursor background agent
    "devin-ai-integration",  # Cognition's Devin
    "sweep-ai",            # Sweep
    "google-labs-jules",   # Google Jules
    "codegen-sh",          # Codegen
})

# Accounts that *review* code (leave reviews / review comments) using an LLM.
AI_REVIEW_BOT_LOGINS: frozenset[str] = frozenset({
    "copilot-pull-request-reviewer",  # GitHub's own Copilot code review feature
    "coderabbitai",        # CodeRabbit
    "sourcery-ai",         # Sourcery
    "greptile",            # Greptile
    "qodo-merge-pro",      # Qodo Merge (formerly PR-Agent)
    "gemini-code-assist",  # Gemini Code Assist
})

# Bots that are automation but NOT AI — enumerated only so the report can state
# plainly that they were deliberately excluded. Anything Bot-typed and off the
# two AI lists lands in `other_bot` regardless of whether it's listed here.
KNOWN_AUTOMATION_LOGINS: frozenset[str] = frozenset({
    "dependabot", "renovate", "renovate-bot", "github-actions",
    "codecov", "codecov-commenter", "stale", "mergify", "netlify",
    "vercel", "allcontributors", "pre-commit-ci", "semantic-release-bot",
})

# Copilot's coding agent opens every PR with this exact placeholder first commit.
COPILOT_INITIAL_COMMIT = "initial plan"

# GitHub no-reply commit email, e.g. "198982749+Copilot@users.noreply.github.com".
_NOREPLY_EMAIL_RE = re.compile(r"^(?:\d+\+)?(?P<login>[^@]+)@users\.noreply\.github\.com$", re.I)


# --------------------------------------------------------------------------- #
# Confidence levels — named so they read meaningfully in the report and tables.
# --------------------------------------------------------------------------- #

CONF_ALLOWLIST_BOT = 0.95   # allowlisted login AND GitHub-typed Bot: structural + curated
CONF_ALLOWLIST_USER = 0.90  # allowlisted login but typed User (agent on a user account)
CONF_HUMAN = 0.90           # typed User, no AI signal (can't be 1.0 — a human may drive an AI)
CONF_OTHER_BOT = 0.95       # typed Bot, off both AI lists
CONF_COMMIT_IDENTITY = 0.80  # a commit is authored by an allowlisted agent identity
CONF_COMMIT_INITIAL_PLAN = 0.70  # Copilot's "Initial plan" placeholder first commit
CONF_UNKNOWN = 0.0


class ActorCategory(str, Enum):
    HUMAN = "human"
    AI_CODING_AGENT = "ai_coding_agent"
    AI_REVIEW_BOT = "ai_review_bot"
    OTHER_BOT = "other_bot"
    UNKNOWN = "unknown"  # deleted/ghost account, or actor absent from the payload


@dataclass
class ActorClassification:
    login: Optional[str]
    typename: Optional[str]
    category: ActorCategory
    confidence: float
    method: str

    @property
    def is_ai(self) -> bool:
        return self.category in (ActorCategory.AI_CODING_AGENT, ActorCategory.AI_REVIEW_BOT)


# --------------------------------------------------------------------------- #
# Login normalization + atomic actor classification
# --------------------------------------------------------------------------- #

def normalize_login(login: Optional[str]) -> str:
    """Lowercase and strip a trailing '[bot]' suffix.

    GraphQL Bot actors expose a bare login (`dependabot`), while the commit
    trail exposes the suffixed form (`dependabot[bot]`); normalizing both to
    the same key lets one allowlist serve every surface.
    """
    if not login:
        return ""
    login = login.strip().lower()
    if login.endswith("[bot]"):
        login = login[: -len("[bot]")]
    return login


def classify_actor(login: Optional[str], typename: Optional[str]) -> ActorClassification:
    """Classify a single actor from its `{login, __typename}` pair.

    An allowlisted login wins even when GitHub types it `User` (some agents run
    on ordinary user accounts) — but the confidence is lowered and the method
    records the mismatch, so nothing is hidden from a spot-check.
    """
    norm = normalize_login(login)

    if not norm and not typename:
        return ActorClassification(login, typename, ActorCategory.UNKNOWN, CONF_UNKNOWN, "no_actor")

    if norm in AI_CODING_AGENT_LOGINS:
        if typename == "Bot":
            return ActorClassification(login, typename, ActorCategory.AI_CODING_AGENT,
                                       CONF_ALLOWLIST_BOT, "allowlist_coding_agent+bot")
        return ActorClassification(login, typename, ActorCategory.AI_CODING_AGENT,
                                   CONF_ALLOWLIST_USER, "allowlist_coding_agent+user")

    if norm in AI_REVIEW_BOT_LOGINS:
        if typename == "Bot":
            return ActorClassification(login, typename, ActorCategory.AI_REVIEW_BOT,
                                       CONF_ALLOWLIST_BOT, "allowlist_review_bot+bot")
        return ActorClassification(login, typename, ActorCategory.AI_REVIEW_BOT,
                                   CONF_ALLOWLIST_USER, "allowlist_review_bot+user")

    if typename == "Bot":
        method = "typename_bot_known_automation" if norm in KNOWN_AUTOMATION_LOGINS else "typename_bot_other"
        return ActorClassification(login, typename, ActorCategory.OTHER_BOT, CONF_OTHER_BOT, method)

    if typename == "User":
        return ActorClassification(login, typename, ActorCategory.HUMAN, CONF_HUMAN, "typename_user")

    # Login present but no typename (e.g. reconstructed from the commit trail):
    # fall back to the allowlist only, never guess a bare login into `human`.
    return ActorClassification(login, typename, ActorCategory.UNKNOWN, CONF_UNKNOWN, "no_typename")


def _actor_of(node: Optional[dict]) -> tuple[Optional[str], Optional[str]]:
    """Pull (login, __typename) out of a GraphQL actor object, tolerating None."""
    if not node:
        return None, None
    author = node.get("author")
    if not author:
        return None, None
    return author.get("login"), author.get("__typename")


# --------------------------------------------------------------------------- #
# Secondary signal: AI involvement in the commit trail of a human-opened PR
# --------------------------------------------------------------------------- #

def _email_login(email: Optional[str]) -> str:
    """Extract the agent login from a GitHub no-reply commit email, normalized."""
    if not email:
        return ""
    m = _NOREPLY_EMAIL_RE.match(email.strip())
    if not m:
        return ""
    return normalize_login(m.group("login"))


def scan_commits_for_ai(commit_nodes: list[dict]) -> Optional[dict[str, Any]]:
    """Look for AI authorship evidence in the commits themselves.

    Returns None if nothing is found, else a dict with `method`, `confidence`,
    `evidence`, and the matched `login` (when applicable). Commit identity is a
    stronger signal than the placeholder-message heuristic, so it's checked
    first and wins on confidence.
    """
    initial_plan_hit: Optional[dict[str, Any]] = None

    for index, node in enumerate(commit_nodes):
        commit = (node or {}).get("commit") or {}
        commit_author = commit.get("author") or {}
        user = commit_author.get("user") or {}

        # (a) Strongest: the commit's GitHub user, its display name, or its
        #     no-reply email resolves to an allowlisted coding-agent login.
        for candidate in (
            normalize_login(user.get("login")),
            normalize_login(commit_author.get("name")),
            _email_login(commit_author.get("email")),
        ):
            if candidate and candidate in AI_CODING_AGENT_LOGINS:
                oid = (commit.get("oid") or "")[:10]
                return {
                    "method": "commit_agent_identity",
                    "confidence": CONF_COMMIT_IDENTITY,
                    "login": candidate,
                    "evidence": f"commit {oid} authored by agent identity '{candidate}'",
                }

        # (b) Weaker: Copilot's literal "Initial plan" placeholder first commit.
        #     Recorded but not returned yet, so an identity hit on a later
        #     commit can still take precedence.
        if index == 0 and (commit.get("message") or "").strip().lower().startswith(COPILOT_INITIAL_COMMIT):
            initial_plan_hit = {
                "method": "commit_initial_plan",
                "confidence": CONF_COMMIT_INITIAL_PLAN,
                "login": "copilot-swe-agent",
                "evidence": "first commit message is Copilot's 'Initial plan' placeholder",
            }

    return initial_plan_hit


# --------------------------------------------------------------------------- #
# PR-level aggregation — produces one `ai_signals` row per PR
# --------------------------------------------------------------------------- #

@dataclass
class PRDetection:
    pr_id: Optional[str]
    owner: str
    repo: str
    number: int
    author_login: Optional[str]
    author_type: Optional[str]
    author_category: str
    # authorship
    is_ai_authored: bool
    ai_authored_kind: Optional[str]      # "agent_authored" | "human_opened_ai_assisted" | None
    ai_authored_method: str
    ai_authored_confidence: float
    ai_authored_evidence: Optional[str]
    # review
    is_ai_reviewed: bool
    ai_reviewer_logins: list[str] = field(default_factory=list)
    ai_reviewer_sources: list[str] = field(default_factory=list)
    n_ai_reviewers: int = 0

    def as_row(self) -> dict[str, Any]:
        """Flat, parquet-friendly dict (list fields are kept, plus scalar
        conveniences) for Step 4's `ai_signals` table."""
        return {
            "pr_id": self.pr_id,
            "owner": self.owner,
            "repo": self.repo,
            "number": self.number,
            "author_login": self.author_login,
            "author_type": self.author_type,
            "author_category": self.author_category,
            "is_ai_authored": self.is_ai_authored,
            "ai_authored_kind": self.ai_authored_kind,
            "ai_authored_method": self.ai_authored_method,
            "ai_authored_confidence": self.ai_authored_confidence,
            "ai_authored_evidence": self.ai_authored_evidence,
            "is_ai_reviewed": self.is_ai_reviewed,
            "ai_reviewer_logins": self.ai_reviewer_logins,
            "ai_reviewer_logins_str": ",".join(self.ai_reviewer_logins),
            "ai_reviewer_sources": sorted(set(self.ai_reviewer_sources)),
            "n_ai_reviewers": self.n_ai_reviewers,
        }


def _detect_authorship(pr: dict) -> tuple[ActorClassification, dict]:
    """Resolve is_ai_authored from the primary author signal, then fall back to
    the commit-trail secondary signal for human-opened PRs."""
    author_login, author_type = _actor_of(pr)
    author = classify_actor(author_login, author_type)

    # Primary: the PR author is itself an AI coding agent.
    if author.category == ActorCategory.AI_CODING_AGENT:
        return author, {
            "is_ai_authored": True,
            "kind": "agent_authored",
            "method": author.method,
            "confidence": author.confidence,
            "evidence": f"PR author '{author_login}' is an allowlisted coding agent",
        }

    # A review bot or non-AI automation opening a PR is not "AI-authored code".
    if author.category in (ActorCategory.AI_REVIEW_BOT, ActorCategory.OTHER_BOT):
        return author, {"is_ai_authored": False, "kind": None, "method": author.method,
                        "confidence": author.confidence, "evidence": None}

    # Secondary: a human (or unknown) author whose commits show AI involvement.
    commit_nodes = (pr.get("commits") or {}).get("nodes") or []
    hit = scan_commits_for_ai(commit_nodes)
    if hit is not None:
        return author, {
            "is_ai_authored": True,
            "kind": "human_opened_ai_assisted",
            "method": hit["method"],
            "confidence": hit["confidence"],
            "evidence": hit["evidence"],
        }

    return author, {"is_ai_authored": False, "kind": None, "method": author.method,
                    "confidence": author.confidence, "evidence": None}


def _detect_reviewers(pr: dict) -> tuple[list[str], list[str]]:
    """Find AI review-bot involvement across formal reviews, inline review
    comments, and top-level issue comments (bots like CodeRabbit often post as
    issue comments rather than formal reviews). Returns (logins, sources)."""
    logins: list[str] = []
    sources: list[str] = []

    def consider(node: Optional[dict], source: str) -> None:
        login, typename = _actor_of(node)
        cls = classify_actor(login, typename)
        if cls.category == ActorCategory.AI_REVIEW_BOT and login:
            logins.append(login)
            sources.append(source)

    for review in (pr.get("reviews") or {}).get("nodes") or []:
        consider(review, "review")
        for comment in (review.get("comments") or {}).get("nodes") or []:
            consider(comment, "review_comment")

    for comment in (pr.get("comments") or {}).get("nodes") or []:
        consider(comment, "issue_comment")

    # De-duplicate logins while preserving a stable, sorted order.
    unique_logins = sorted(set(logins))
    return unique_logins, sources


def detect_pr(record: dict) -> PRDetection:
    """Run full AI detection on one raw PR record from `GitHubClient.fetch_pr`."""
    pr = record.get("pull_request") or {}
    author_login, author_type = _actor_of(pr)
    author, authorship = _detect_authorship(pr)

    ai_reviewer_logins, ai_reviewer_sources = _detect_reviewers(pr)

    return PRDetection(
        pr_id=pr.get("id"),
        owner=record.get("owner"),
        repo=record.get("repo"),
        number=record.get("number") or pr.get("number"),
        author_login=author_login,
        author_type=author_type,
        author_category=author.category.value,
        is_ai_authored=authorship["is_ai_authored"],
        ai_authored_kind=authorship["kind"],
        ai_authored_method=authorship["method"],
        ai_authored_confidence=authorship["confidence"],
        ai_authored_evidence=authorship["evidence"],
        is_ai_reviewed=bool(ai_reviewer_logins),
        ai_reviewer_logins=ai_reviewer_logins,
        ai_reviewer_sources=ai_reviewer_sources,
        n_ai_reviewers=len(ai_reviewer_logins),
    )
