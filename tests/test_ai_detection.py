from src.mining.ai_detection import (
    ActorCategory,
    classify_actor,
    detect_pr,
    normalize_login,
    scan_commits_for_ai,
    _email_login,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def actor(login, typename):
    return {"login": login, "__typename": typename}


def make_record(author=None, commits=None, reviews=None, comments=None,
                owner="microsoft", repo="vscode", number=1, pr_id="PR_1"):
    """Build a minimal fetch_pr-shaped record for detection tests."""
    return {
        "owner": owner,
        "repo": repo,
        "number": number,
        "pull_request": {
            "id": pr_id,
            "number": number,
            "author": author,
            "commits": {"totalCount": len(commits or []), "nodes": commits or []},
            "reviews": {"totalCount": len(reviews or []), "nodes": reviews or []},
            "comments": {"totalCount": len(comments or []), "nodes": comments or []},
        },
    }


def commit(message="fix", login=None, name=None, email=None, oid="abc123"):
    return {
        "commit": {
            "oid": oid,
            "message": message,
            "author": {
                "name": name,
                "email": email,
                "user": {"login": login, "__typename": "Bot"} if login else None,
            },
        }
    }


# --------------------------------------------------------------------------- #
# login normalization + email parsing
# --------------------------------------------------------------------------- #

def test_normalize_login_strips_bot_suffix_and_lowercases():
    assert normalize_login("Dependabot[bot]") == "dependabot"
    assert normalize_login("copilot-swe-agent") == "copilot-swe-agent"
    assert normalize_login(None) == ""


def test_email_login_parses_noreply_with_numeric_prefix():
    assert _email_login("198982749+Copilot@users.noreply.github.com") == "copilot"
    assert _email_login("dependabot[bot]@users.noreply.github.com") == "dependabot"
    assert _email_login("someone@example.com") == ""
    assert _email_login(None) == ""


# --------------------------------------------------------------------------- #
# atomic actor classification
# --------------------------------------------------------------------------- #

def test_classify_human_user():
    c = classify_actor("octocat", "User")
    assert c.category == ActorCategory.HUMAN
    assert not c.is_ai


def test_classify_coding_agent_bot_high_confidence():
    c = classify_actor("copilot-swe-agent", "Bot")
    assert c.category == ActorCategory.AI_CODING_AGENT
    assert c.confidence == 0.95
    assert c.is_ai


def test_classify_coding_agent_on_user_account_lower_confidence():
    c = classify_actor("devin-ai-integration", "User")
    assert c.category == ActorCategory.AI_CODING_AGENT
    assert c.confidence == 0.90
    assert "user" in c.method


def test_classify_review_bot():
    c = classify_actor("coderabbitai", "Bot")
    assert c.category == ActorCategory.AI_REVIEW_BOT
    assert c.is_ai


def test_classify_non_ai_automation_is_other_bot():
    c = classify_actor("dependabot", "Bot")
    assert c.category == ActorCategory.OTHER_BOT
    assert not c.is_ai
    assert "known_automation" in c.method


def test_classify_unknown_bot_is_other_bot_not_ai():
    c = classify_actor("some-random-ci-bot", "Bot")
    assert c.category == ActorCategory.OTHER_BOT
    assert not c.is_ai


def test_classify_ghost_actor_is_unknown():
    c = classify_actor(None, None)
    assert c.category == ActorCategory.UNKNOWN


# --------------------------------------------------------------------------- #
# commit-trail secondary signal
# --------------------------------------------------------------------------- #

def test_scan_commits_detects_initial_plan_first_commit():
    hit = scan_commits_for_ai([commit(message="Initial plan"), commit(message="real work")])
    assert hit["method"] == "commit_initial_plan"


def test_scan_commits_ignores_initial_plan_if_not_first():
    hit = scan_commits_for_ai([commit(message="real work"), commit(message="Initial plan")])
    assert hit is None


def test_scan_commits_detects_agent_email_identity():
    hit = scan_commits_for_ai([commit(email="1+copilot-swe-agent@users.noreply.github.com")])
    assert hit["method"] == "commit_agent_identity"
    assert hit["confidence"] == 0.80


def test_scan_commits_ignores_ordinary_commits():
    assert scan_commits_for_ai([commit(name="Jane Dev", email="jane@example.com")]) is None


# --------------------------------------------------------------------------- #
# PR-level aggregation
# --------------------------------------------------------------------------- #

def test_detect_agent_authored_pr():
    rec = make_record(author=actor("copilot-swe-agent", "Bot"))
    d = detect_pr(rec)
    assert d.is_ai_authored
    assert d.ai_authored_kind == "agent_authored"
    assert d.ai_authored_confidence == 0.95
    assert d.author_category == "ai_coding_agent"


def test_detect_human_opened_ai_assisted_pr():
    rec = make_record(
        author=actor("human-dev", "User"),
        commits=[commit(message="Initial plan"), commit(message="Implement feature")],
    )
    d = detect_pr(rec)
    assert d.is_ai_authored
    assert d.ai_authored_kind == "human_opened_ai_assisted"
    assert d.ai_authored_confidence < 0.95  # explicitly weaker than agent-authored
    assert d.author_category == "human"


def test_detect_plain_human_pr_is_not_ai_authored():
    rec = make_record(
        author=actor("human-dev", "User"),
        commits=[commit(message="fix bug", name="Human Dev", email="h@example.com")],
    )
    d = detect_pr(rec)
    assert not d.is_ai_authored
    assert d.ai_authored_kind is None


def test_review_bot_via_issue_comment_flags_ai_reviewed():
    rec = make_record(
        author=actor("human-dev", "User"),
        comments=[{"author": actor("coderabbitai", "Bot")}],
    )
    d = detect_pr(rec)
    assert d.is_ai_reviewed
    assert d.ai_reviewer_logins == ["coderabbitai"]
    assert "issue_comment" in d.ai_reviewer_sources


def test_review_bot_via_formal_review_and_review_comment():
    rec = make_record(
        author=actor("human-dev", "User"),
        reviews=[{
            "author": actor("copilot-pull-request-reviewer", "Bot"),
            "comments": {"nodes": [{"author": actor("copilot-pull-request-reviewer", "Bot")}]},
        }],
    )
    d = detect_pr(rec)
    assert d.is_ai_reviewed
    assert d.n_ai_reviewers == 1  # de-duplicated across surfaces
    assert set(d.ai_reviewer_sources) == {"review", "review_comment"}


def test_human_reviewer_does_not_flag_ai_reviewed():
    rec = make_record(
        author=actor("human-dev", "User"),
        reviews=[{"author": actor("senior-maintainer", "User"), "comments": {"nodes": []}}],
    )
    d = detect_pr(rec)
    assert not d.is_ai_reviewed
    assert d.n_ai_reviewers == 0


def test_dependabot_pr_is_not_ai_authored_or_reviewed():
    rec = make_record(author=actor("dependabot", "Bot"))
    d = detect_pr(rec)
    assert not d.is_ai_authored
    assert not d.is_ai_reviewed
    assert d.author_category == "other_bot"


def test_as_row_is_flat_and_parquet_friendly():
    rec = make_record(
        author=actor("copilot-swe-agent", "Bot"),
        comments=[{"author": actor("coderabbitai", "Bot")}],
    )
    row = detect_pr(rec).as_row()
    assert row["is_ai_authored"] is True
    assert row["ai_reviewer_logins_str"] == "coderabbitai"
    assert row["n_ai_reviewers"] == 1
    assert isinstance(row["ai_reviewer_logins"], list)
