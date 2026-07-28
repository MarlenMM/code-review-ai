import pandas as pd

from src.mining.mine_prs import (
    build_sample_numbers,
    build_tables,
    flatten_commits,
    flatten_files,
    flatten_issue_comments,
    flatten_pull_request,
    flatten_review_comments,
    flatten_reviews,
    rebuild_from_existing,
    write_tables,
    TABLE_NAMES,
)
from src.mining.ai_detection import detect_pr
from src.mining.github_client import GitHubClient


class FakeClient:
    """Stands in for GitHubClient in sampling tests: no network, no token."""

    def __init__(self, ai_numbers, closed_numbers):
        self._ai_numbers = ai_numbers
        self._closed_numbers = closed_numbers

    def search_pr_numbers(self, owner, repo, extra_qualifiers, limit=None):
        result = self._ai_numbers[:limit] if limit else list(self._ai_numbers)
        return result

    def list_closed_pr_numbers(self, owner, repo, limit):
        return self._closed_numbers[:limit]


def test_build_sample_numbers_fills_remainder_and_dedupes():
    ai_numbers = [10, 11, 12]
    closed_numbers = [12, 11, 20, 21, 22, 23, 24]  # 12, 11 overlap the AI slice
    client = FakeClient(ai_numbers, closed_numbers)

    numbers, ai_set = build_sample_numbers(client, "o", "r", prs_per_repo=6, ai_authored_cap=100)

    assert ai_set == {10, 11, 12}
    assert numbers[:3] == [10, 11, 12]
    # Fill should skip the two overlapping numbers and take 3 fresh ones.
    assert numbers[3:] == [20, 21, 22]
    assert len(numbers) == 6


def test_build_sample_numbers_respects_ai_authored_cap():
    ai_numbers = list(range(1, 200))  # far more than the cap
    closed_numbers = list(range(1000, 1100))
    client = FakeClient(ai_numbers, closed_numbers)

    numbers, ai_set = build_sample_numbers(client, "o", "r", prs_per_repo=300, ai_authored_cap=100)

    assert len(ai_set) == 100
    assert numbers[:100] == list(range(1, 101))


def test_build_sample_numbers_handles_ai_search_meeting_full_target():
    # If the AI-authored slice (post-cap) already meets the per-repo target,
    # no fill PRs should be requested/added, regardless of what's available.
    ai_numbers = list(range(1, 320))
    closed_numbers = [9999]
    client = FakeClient(ai_numbers, closed_numbers)

    numbers, ai_set = build_sample_numbers(client, "o", "r", prs_per_repo=100, ai_authored_cap=100)
    assert len(numbers) == 100  # target met by the capped AI slice alone, no fill


def test_build_sample_numbers_dedupes_pagination_drift_in_ai_search():
    # Regression: GitHub's search endpoint paginates over live data, so an
    # item can shift between pages and appear twice in one raw response
    # (observed in practice on microsoft/vscode). ai_numbers itself must
    # come back deduplicated.
    client = FakeClient(ai_numbers=[10, 11, 11, 12], closed_numbers=[20, 21])
    numbers, ai_set = build_sample_numbers(client, "o", "r", prs_per_repo=5, ai_authored_cap=100)
    assert numbers.count(11) == 1
    assert ai_set == {10, 11, 12}


def test_build_sample_numbers_dedupes_pagination_drift_in_fill_list():
    # Same drift risk on the recent-closed-PRs listing used to fill the
    # remainder: a duplicate within `candidates` must not produce a
    # duplicate in the final fill.
    client = FakeClient(ai_numbers=[1, 2], closed_numbers=[30, 31, 31, 32])
    numbers, ai_set = build_sample_numbers(client, "o", "r", prs_per_repo=5, ai_authored_cap=100)
    assert numbers.count(31) == 1
    assert len(numbers) == 5
    assert numbers == [1, 2, 30, 31, 32]


def test_build_sample_numbers_ai_cap_never_exceeds_per_repo_target():
    # Regression: a repo whose AI-authored search alone exceeds the per-repo
    # target (e.g. a small smoke-test target with the default 100 AI cap)
    # must not end up fetching more PRs than the target.
    ai_numbers = list(range(1, 200))
    closed_numbers = list(range(1000, 1100))
    client = FakeClient(ai_numbers, closed_numbers)

    numbers, ai_set = build_sample_numbers(client, "o", "r", prs_per_repo=5, ai_authored_cap=100)
    assert len(numbers) == 5
    assert len(ai_set) == 5


# --------------------------------------------------------------------------- #
# flattening
# --------------------------------------------------------------------------- #

def _sample_record(number=42, pr_id="PR_kwABC", author_login="octocat", author_type="User", labels=None):
    return {
        "owner": "microsoft",
        "repo": "vscode",
        "number": number,
        "pull_request": {
            "id": pr_id,
            "number": number,
            "title": "Fix the thing",
            "body": "This fixes the thing.",
            "state": "CLOSED",
            "merged": True,
            "createdAt": "2026-01-01T00:00:00Z",
            "closedAt": "2026-01-02T00:00:00Z",
            "mergedAt": "2026-01-02T00:00:00Z",
            "additions": 10,
            "deletions": 2,
            "changedFiles": 1,
            "author": {"login": author_login, "__typename": author_type},
            "labels": {"totalCount": len(labels or []), "nodes": [{"name": l} for l in (labels or [])]},
            "commits": {
                "totalCount": 1,
                "nodes": [{
                    "commit": {
                        "oid": "abc123",
                        "message": "fix",
                        "additions": 10,
                        "deletions": 2,
                        "author": {"name": "Octo Cat", "email": "octo@example.com", "user": None},
                    }
                }],
            },
            "reviews": {
                "totalCount": 1,
                "nodes": [{
                    "id": "REV_1",
                    "state": "APPROVED",
                    "submittedAt": "2026-01-01T12:00:00Z",
                    "body": "LGTM",
                    "author": {"login": "reviewer1", "__typename": "User"},
                    "comments": {
                        "totalCount": 1,
                        "nodes": [{
                            "id": "RC_1",
                            "body": "nit: rename this",
                            "path": "src/foo.py",
                            "position": 3,
                            "createdAt": "2026-01-01T11:00:00Z",
                            "author": {"login": "reviewer1", "__typename": "User"},
                        }],
                    },
                }],
            },
            "comments": {
                "totalCount": 1,
                "nodes": [{
                    "id": "IC_1",
                    "body": "thanks!",
                    "createdAt": "2026-01-02T00:30:00Z",
                    "author": {"login": "octocat", "__typename": "User"},
                }],
            },
        },
        "files": [
            {"filename": "src/foo.py", "status": "modified", "additions": 10, "deletions": 2, "patch": "@@ -1 +1 @@"},
        ],
    }


def test_flatten_pull_request_shape():
    record = _sample_record(labels=["bug", "good first issue"])
    detection = detect_pr(record).as_row()
    row = flatten_pull_request(record, detection)

    assert row["id"] == "PR_kwABC"
    assert row["repo"] == "microsoft/vscode"
    assert row["number"] == 42
    assert row["is_merged"] is True
    assert row["labels"] == ["bug", "good first issue"]
    assert row["is_ai_authored"] is False
    assert row["is_ai_reviewed"] is False


def test_flatten_commits_review_comments_and_issue_comments():
    record = _sample_record()
    commits = flatten_commits(record)
    reviews = flatten_reviews(record)
    review_comments = flatten_review_comments(record)
    issue_comments = flatten_issue_comments(record)
    files = flatten_files(record)

    assert commits == [{
        "pr_id": "PR_kwABC", "sha": "abc123", "message": "fix",
        "author_name": "Octo Cat", "author_email": "octo@example.com",
        "author_login": None, "additions": 10, "deletions": 2,
    }]
    assert reviews[0]["reviewer_login"] == "reviewer1"
    assert reviews[0]["state"] == "APPROVED"
    assert review_comments[0]["path"] == "src/foo.py"
    assert review_comments[0]["review_id"] == "REV_1"
    assert review_comments[0]["comment_id"] == "RC_1"
    assert issue_comments[0]["body"] == "thanks!"
    assert issue_comments[0]["comment_id"] == "IC_1"
    assert files[0]["filename"] == "src/foo.py"


def test_build_tables_covers_all_seven_tables_and_ai_flags():
    ai_record = _sample_record(number=1, pr_id="PR_AI", author_login="copilot-swe-agent", author_type="Bot")
    human_record = _sample_record(number=2, pr_id="PR_HUMAN", author_login="octocat", author_type="User")

    tables = build_tables([ai_record, human_record])

    assert set(tables.keys()) == set(TABLE_NAMES)
    assert len(tables["pull_requests"]) == 2
    assert len(tables["ai_signals"]) == 2
    assert len(tables["commits"]) == 2  # one commit per PR in the fixture
    assert len(tables["reviews"]) == 2
    assert len(tables["review_comments"]) == 2
    assert len(tables["issue_comments"]) == 2
    assert len(tables["files_changed"]) == 2

    ai_row = next(r for r in tables["pull_requests"] if r["id"] == "PR_AI")
    human_row = next(r for r in tables["pull_requests"] if r["id"] == "PR_HUMAN")
    assert ai_row["is_ai_authored"] is True
    assert ai_row["ai_authored_kind"] == "agent_authored"
    assert human_row["is_ai_authored"] is False


def test_write_tables_produces_readable_parquet_files(tmp_path):
    ai_record = _sample_record(number=1, pr_id="PR_AI", author_login="copilot-swe-agent", author_type="Bot",
                                labels=["ai-authored"])
    human_record = _sample_record(number=2, pr_id="PR_HUMAN", author_login="octocat", author_type="User")
    tables = build_tables([ai_record, human_record])

    written = write_tables(tables, output_dir=tmp_path)

    assert set(written.keys()) == set(TABLE_NAMES)
    for name, path in written.items():
        assert path.exists()

    pr_df = pd.read_parquet(written["pull_requests"])
    assert len(pr_df) == 2
    assert set(pr_df["is_ai_authored"]) == {True, False}
    assert list(pr_df.loc[pr_df["id"] == "PR_AI", "labels"])[0] == ["ai-authored"]


def test_rebuild_from_existing_reads_cache_only_no_network(tmp_path):
    # Write a pull_requests.parquet like a prior run would have produced,
    # then confirm rebuild_from_existing pulls each (owner, repo, number)
    # straight from GitHubClient's disk cache -- no GraphQL/REST call needed,
    # since fetch_pr checks cache before ever touching the network.
    client = GitHubClient(token="fake-token-for-test", cache_dir=tmp_path / "cache")
    client._write_cache("microsoft", "vscode", 1, {"owner": "microsoft", "repo": "vscode", "number": 1,
                                                    "pull_request": {"id": "PR_1"}, "files": []})
    client._write_cache("microsoft", "vscode", 2, {"owner": "microsoft", "repo": "vscode", "number": 2,
                                                    "pull_request": {"id": "PR_2"}, "files": []})

    existing = pd.DataFrame([
        {"owner": "microsoft", "repo_name": "vscode", "number": 1},
        {"owner": "microsoft", "repo_name": "vscode", "number": 2},
    ])
    pr_path = tmp_path / "pull_requests.parquet"
    existing.to_parquet(pr_path, index=False)

    def _no_network(*a, **k):
        raise AssertionError("rebuild_from_existing must not hit the network")

    client._graphql = _no_network
    client._rest_get_paginated = _no_network

    records = rebuild_from_existing(pr_path, client)
    assert [r["pull_request"]["id"] for r in records] == ["PR_1", "PR_2"]
