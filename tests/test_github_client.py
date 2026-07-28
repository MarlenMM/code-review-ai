from src.mining.github_client import GitHubClient, backoff_delay, should_retry


def test_should_retry_on_retryable_codes_only():
    assert should_retry(403, attempt=0)
    assert should_retry(429, attempt=0)
    assert should_retry(502, attempt=0)
    assert not should_retry(404, attempt=0)
    assert not should_retry(401, attempt=0)


def test_should_retry_stops_after_max_retries():
    assert not should_retry(429, attempt=6, max_retries=6)
    assert should_retry(429, attempt=5, max_retries=6)


def test_backoff_delay_respects_retry_after_header():
    assert backoff_delay(attempt=3, retry_after=12.0) == 12.0


def test_backoff_delay_grows_with_attempt_and_is_capped():
    early = backoff_delay(attempt=0)
    late = backoff_delay(attempt=20)
    assert 0 < early < late
    assert late <= 90.0


def test_client_requires_a_token(tmp_path, monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    try:
        GitHubClient(cache_dir=tmp_path / "should-not-be-created")
        assert False, "expected ValueError without a token"
    except ValueError:
        pass


def test_cache_path_layout(tmp_path):
    client = GitHubClient(token="fake-token-for-test", cache_dir=tmp_path)
    path = client._cache_path("octocat", "hello-world", 42)
    assert path == tmp_path / "octocat" / "hello-world" / "42.json"
    assert path.parent.exists()


def test_fetch_pr_writes_and_reuses_cache(tmp_path, monkeypatch):
    client = GitHubClient(token="fake-token-for-test", cache_dir=tmp_path)

    graphql_calls = []
    files_calls = []

    def fake_graphql(query, variables):
        graphql_calls.append(variables)
        return {
            "repository": {
                "pullRequest": {
                    "number": variables["number"],
                    "title": "Test PR",
                    "commits": {"totalCount": 1, "nodes": []},
                    "reviews": {"totalCount": 0, "nodes": []},
                    "comments": {"totalCount": 0, "nodes": []},
                }
            }
        }

    def fake_files(path, params=None):
        files_calls.append(path)
        return [{"filename": "a.py", "status": "modified", "additions": 1, "deletions": 0, "patch": "@@ -1 +1 @@"}]

    monkeypatch.setattr(client, "_graphql", fake_graphql)
    monkeypatch.setattr(client, "_rest_get_paginated", fake_files)

    record = client.fetch_pr("octocat", "hello-world", 1)
    assert record["pull_request"]["title"] == "Test PR"
    assert record["files"][0]["filename"] == "a.py"
    assert len(graphql_calls) == 1
    assert (tmp_path / "octocat" / "hello-world" / "1.json").exists()

    # Second call must come from disk cache, not hit GraphQL/REST again.
    record2 = client.fetch_pr("octocat", "hello-world", 1)
    assert record2 == record
    assert len(graphql_calls) == 1
    assert len(files_calls) == 1


def test_fetch_pr_force_refresh_bypasses_cache(tmp_path, monkeypatch):
    client = GitHubClient(token="fake-token-for-test", cache_dir=tmp_path)
    call_count = {"n": 0}

    def fake_graphql(query, variables):
        call_count["n"] += 1
        return {
            "repository": {
                "pullRequest": {
                    "number": variables["number"],
                    "title": f"Test PR v{call_count['n']}",
                    "commits": {"totalCount": 0, "nodes": []},
                    "reviews": {"totalCount": 0, "nodes": []},
                    "comments": {"totalCount": 0, "nodes": []},
                }
            }
        }

    monkeypatch.setattr(client, "_graphql", fake_graphql)
    monkeypatch.setattr(client, "_rest_get_paginated", lambda path, params=None: [])

    first = client.fetch_pr("octocat", "hello-world", 7)
    second = client.fetch_pr("octocat", "hello-world", 7, force_refresh=True)
    assert first["pull_request"]["title"] == "Test PR v1"
    assert second["pull_request"]["title"] == "Test PR v2"
    assert call_count["n"] == 2


def test_fetch_prs_skips_failures_without_raising(tmp_path, monkeypatch):
    client = GitHubClient(token="fake-token-for-test", cache_dir=tmp_path)

    def fake_fetch_pr(owner, repo, number, force_refresh=False):
        if number == 2:
            raise ValueError("PR not found")
        return {"number": number}

    monkeypatch.setattr(client, "fetch_pr", fake_fetch_pr)
    results = client.fetch_prs("octocat", "hello-world", [1, 2, 3])
    assert [r["number"] for r in results] == [1, 3]


def test_fetch_prs_dedupes_repeated_numbers(tmp_path, monkeypatch):
    # Regression: a caller's sample can contain the same PR number twice
    # (observed in practice from pagination drift on a live GitHub listing).
    # fetch_prs must not fetch or return it twice.
    client = GitHubClient(token="fake-token-for-test", cache_dir=tmp_path)
    calls = []

    def fake_fetch_pr(owner, repo, number, force_refresh=False):
        calls.append(number)
        return {"number": number}

    monkeypatch.setattr(client, "fetch_pr", fake_fetch_pr)
    results = client.fetch_prs("octocat", "hello-world", [1, 2, 2, 3])
    assert [r["number"] for r in results] == [1, 2, 3]
    assert sorted(calls) == [1, 2, 3]


def test_rest_get_paginated_stops_on_short_page(tmp_path, monkeypatch):
    client = GitHubClient(token="fake-token-for-test", cache_dir=tmp_path)

    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    pages = [[{"id": i} for i in range(100)], [{"id": i} for i in range(100, 130)]]

    def fake_request(method, url, **kwargs):
        page = kwargs["params"]["page"]
        return FakeResponse(pages[page - 1])

    monkeypatch.setattr(client, "_request_with_retry", fake_request)
    results = client._rest_get_paginated("/repos/octocat/hello-world/pulls/1/files")
    assert len(results) == 130


# --------------------------------------------------------------------------- #
# fetch_issue (Experiment 4's DIFF_ISSUE / COMPLETE context tiers)
# --------------------------------------------------------------------------- #

class _FakeIssueResponse:
    def __init__(self, status_code, payload=None, headers=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.headers = headers or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        raise RuntimeError(f"HTTP {self.status_code}")


def test_fetch_issue_writes_and_reuses_cache(tmp_path, monkeypatch):
    client = GitHubClient(token="fake-token-for-test", cache_dir=tmp_path)
    calls = []

    def fake_request(method, url, **kwargs):
        calls.append(url)
        return _FakeIssueResponse(200, {
            "number": 42, "title": "Spacing is wrong", "body": "Please fix the spacing.",
            "state": "open",
        })

    monkeypatch.setattr(client.session, "request", fake_request)
    record = client.fetch_issue("octocat", "hello-world", 42)
    assert record == {
        "not_found": False, "number": 42, "title": "Spacing is wrong",
        "body": "Please fix the spacing.", "state": "open", "is_pull_request": False,
    }
    assert len(calls) == 1
    assert (tmp_path / "issues" / "octocat" / "hello-world" / "42.json").exists()

    # second call must come from disk cache, not hit the network again
    record2 = client.fetch_issue("octocat", "hello-world", 42)
    assert record2 == record
    assert len(calls) == 1


def test_fetch_issue_404_cached_as_not_found(tmp_path, monkeypatch):
    client = GitHubClient(token="fake-token-for-test", cache_dir=tmp_path)
    calls = []

    def fake_request(method, url, **kwargs):
        calls.append(url)
        return _FakeIssueResponse(404)

    monkeypatch.setattr(client.session, "request", fake_request)
    record = client.fetch_issue("octocat", "hello-world", 999)
    assert record == {"not_found": True}

    # cached: a second call does not hit the network again
    record2 = client.fetch_issue("octocat", "hello-world", 999)
    assert record2 == {"not_found": True}
    assert len(calls) == 1


def test_fetch_issue_flags_pull_requests(tmp_path, monkeypatch):
    client = GitHubClient(token="fake-token-for-test", cache_dir=tmp_path)

    def fake_request(method, url, **kwargs):
        return _FakeIssueResponse(200, {
            "number": 7, "title": "A PR, not an issue", "body": "...",
            "state": "open", "pull_request": {"url": "..."},
        })

    monkeypatch.setattr(client.session, "request", fake_request)
    record = client.fetch_issue("octocat", "hello-world", 7)
    assert record["is_pull_request"] is True


def test_fetch_issue_force_refresh_bypasses_cache(tmp_path, monkeypatch):
    client = GitHubClient(token="fake-token-for-test", cache_dir=tmp_path)
    call_count = {"n": 0}

    def fake_request(method, url, **kwargs):
        call_count["n"] += 1
        return _FakeIssueResponse(200, {
            "number": 1, "title": f"Title v{call_count['n']}", "body": "b", "state": "open",
        })

    monkeypatch.setattr(client.session, "request", fake_request)
    client.fetch_issue("o", "r", 1)
    client.fetch_issue("o", "r", 1, force_refresh=True)
    assert call_count["n"] == 2


def test_fetch_issue_retries_transient_5xx_then_succeeds(tmp_path, monkeypatch):
    import src.mining.github_client as gh_mod
    monkeypatch.setattr(gh_mod.time, "sleep", lambda s: None)
    client = GitHubClient(token="fake-token-for-test", cache_dir=tmp_path)

    responses = [
        _FakeIssueResponse(502),
        _FakeIssueResponse(200, {"number": 5, "title": "Ok now", "body": "b", "state": "open"}),
    ]

    def fake_request(method, url, **kwargs):
        return responses.pop(0)

    monkeypatch.setattr(client.session, "request", fake_request)
    record = client.fetch_issue("o", "r", 5)
    assert record["title"] == "Ok now"
    assert responses == []  # both scripted responses consumed


def test_fetch_issue_non_retryable_error_raises(tmp_path, monkeypatch):
    client = GitHubClient(token="fake-token-for-test", cache_dir=tmp_path)

    def fake_request(method, url, **kwargs):
        return _FakeIssueResponse(401)

    monkeypatch.setattr(client.session, "request", fake_request)
    try:
        client.fetch_issue("o", "r", 1)
        assert False, "expected an exception for a non-retryable 401"
    except RuntimeError:
        pass


def test_issue_cache_path_is_separate_from_pr_cache(tmp_path):
    client = GitHubClient(token="fake-token-for-test", cache_dir=tmp_path)
    issue_path = client._issue_cache_path("octocat", "hello-world", 42)
    pr_path = client._cache_path("octocat", "hello-world", 42)
    assert issue_path != pr_path
    assert issue_path == tmp_path / "issues" / "octocat" / "hello-world" / "42.json"
