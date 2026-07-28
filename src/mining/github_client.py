"""Authenticated GitHub client: one nested GraphQL query per PR (core fields,
commits, reviews, review comments, issue comments, labels) plus REST for file
patches (GraphQL has no unified-diff field). Every raw PR response is cached
to disk keyed by (owner, repo, number), so mining is idempotent and resumable.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

GITHUB_GRAPHQL_URL = "https://api.github.com/graphql"
GITHUB_REST_URL = "https://api.github.com"

RETRYABLE_STATUS_CODES = {403, 429, 500, 502, 503, 504}
MAX_RETRIES = 6
BASE_BACKOFF_SECONDS = 2.0
MAX_BACKOFF_SECONDS = 90.0

# One nested query per PR: core fields + commits + reviews (with their
# comments) + top-level issue comments + labels. Each list is capped at the
# first 100 nodes (GitHub's per-page max) - fetch_pr logs a warning if a PR
# has more than that, rather than silently truncating without a trace.
PR_QUERY = """
query($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      id
      number
      title
      body
      state
      merged
      createdAt
      closedAt
      mergedAt
      additions
      deletions
      changedFiles
      author { login __typename }
      labels(first: 30) {
        totalCount
        nodes { name }
      }
      commits(first: 100) {
        totalCount
        nodes {
          commit {
            oid
            message
            additions
            deletions
            author {
              name
              email
              user { login __typename }
            }
          }
        }
      }
      reviews(first: 100) {
        totalCount
        nodes {
          id
          state
          submittedAt
          body
          author { login __typename }
          comments(first: 100) {
            totalCount
            nodes {
              id
              body
              path
              position
              createdAt
              author { login __typename }
            }
          }
        }
      }
      comments(first: 100) {
        totalCount
        nodes {
          id
          body
          createdAt
          author { login __typename }
        }
      }
    }
  }
}
"""


def should_retry(status_code: int, attempt: int, max_retries: int = MAX_RETRIES) -> bool:
    return status_code in RETRYABLE_STATUS_CODES and attempt < max_retries


def backoff_delay(attempt: int, retry_after: Optional[float] = None) -> float:
    """Exponential backoff with jitter, or the server-provided Retry-After."""
    if retry_after is not None:
        return retry_after
    delay = min(MAX_BACKOFF_SECONDS, BASE_BACKOFF_SECONDS * (2 ** attempt))
    return delay * (0.5 + random.random() / 2)


class GitHubClient:
    def __init__(
        self,
        token: Optional[str] = None,
        cache_dir: str | Path = "data/raw_cache",
        max_workers: int = 10,
        session: Optional[requests.Session] = None,
    ):
        self.token = token or os.environ.get("GITHUB_TOKEN")
        if not self.token:
            raise ValueError(
                "No GitHub token found. Set GITHUB_TOKEN in .env or pass token= explicitly."
            )
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "code-review-ai-mining-client",
            }
        )

    # ---------- low-level request helpers with retry ----------

    def _request_with_retry(self, method: str, url: str, **kwargs) -> requests.Response:
        attempt = 0
        while True:
            response = self.session.request(method, url, timeout=30, **kwargs)
            if response.status_code < 400:
                return response
            if not should_retry(response.status_code, attempt):
                response.raise_for_status()
            retry_after_header = response.headers.get("Retry-After")
            retry_after = float(retry_after_header) if retry_after_header else None
            delay = backoff_delay(attempt, retry_after)
            logger.warning(
                "%s %s -> %d (attempt %d/%d); retrying in %.1fs",
                method, url, response.status_code, attempt + 1, MAX_RETRIES, delay,
            )
            time.sleep(delay)
            attempt += 1

    def _graphql(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        attempt = 0
        while True:
            response = self._request_with_retry(
                "POST", GITHUB_GRAPHQL_URL, json={"query": query, "variables": variables}
            )
            payload = response.json()
            errors = payload.get("errors")
            if errors:
                rate_limited = any(e.get("type") == "RATE_LIMITED" for e in errors)
                if rate_limited and attempt < MAX_RETRIES:
                    delay = backoff_delay(attempt)
                    logger.warning(
                        "GraphQL rate limited for %s (attempt %d/%d); retrying in %.1fs",
                        variables, attempt + 1, MAX_RETRIES, delay,
                    )
                    time.sleep(delay)
                    attempt += 1
                    continue
                raise RuntimeError(f"GraphQL errors for {variables}: {errors}")
            return payload["data"]

    def _rest_get_paginated(self, path: str, params: Optional[dict[str, Any]] = None) -> list[dict]:
        results: list[dict] = []
        page = 1
        params = dict(params or {})
        params["per_page"] = 100
        while True:
            params["page"] = page
            response = self._request_with_retry("GET", f"{GITHUB_REST_URL}{path}", params=params)
            batch = response.json()
            if not batch:
                break
            results.extend(batch)
            if len(batch) < 100:
                break
            page += 1
        return results

    # ---------- disk cache, keyed by (owner, repo, number) ----------

    def _cache_path(self, owner: str, repo: str, number: int) -> Path:
        repo_dir = self.cache_dir / owner / repo
        repo_dir.mkdir(parents=True, exist_ok=True)
        return repo_dir / f"{number}.json"

    def _read_cache(self, owner: str, repo: str, number: int) -> Optional[dict]:
        path = self._cache_path(owner, repo, number)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            logger.warning("Corrupt cache file %s, will refetch", path)
            return None

    def _write_cache(self, owner: str, repo: str, number: int, data: dict) -> None:
        self._cache_path(owner, repo, number).write_text(json.dumps(data, indent=2))

    # ---------- single-PR fetch (GraphQL core + REST files), cached ----------

    def fetch_pr(
        self, owner: str, repo: str, number: int, force_refresh: bool = False
    ) -> dict[str, Any]:
        if not force_refresh:
            cached = self._read_cache(owner, repo, number)
            if cached is not None:
                return cached

        graphql_data = self._graphql(PR_QUERY, {"owner": owner, "name": repo, "number": number})
        pr = graphql_data.get("repository", {}).get("pullRequest")
        if pr is None:
            raise ValueError(f"PR {owner}/{repo}#{number} not found")

        for node_field in ("commits", "reviews", "comments"):
            block = pr.get(node_field) or {}
            total = block.get("totalCount", 0)
            fetched = len(block.get("nodes", []))
            if total > fetched:
                logger.warning(
                    "%s/%s#%d: only captured %d/%d %s (single-page cap of 100)",
                    owner, repo, number, fetched, total, node_field,
                )

        files = self._rest_get_paginated(f"/repos/{owner}/{repo}/pulls/{number}/files")

        record = {
            "owner": owner,
            "repo": repo,
            "number": number,
            "pull_request": pr,
            "files": files,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
        self._write_cache(owner, repo, number, record)
        return record

    def fetch_prs(
        self,
        owner: str,
        repo: str,
        numbers: Iterable[int],
        force_refresh: bool = False,
        max_workers: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Concurrent fetch over a thread pool. Returns only the PRs that
        succeeded, in the order of `numbers`; failures are logged, not raised,
        so one bad PR (e.g. a server-side 502) doesn't abort the whole batch.

        `numbers` is deduplicated defensively: a caller building a sample from
        paginated GitHub search/list endpoints can occasionally see the same
        PR twice if it shifts between pages mid-fetch (observed in practice on
        a high-traffic repo), and this must not turn into a duplicate row in
        the fetched output.
        """
        numbers = list(dict.fromkeys(numbers))
        results: dict[int, dict] = {}
        workers = max_workers or self.max_workers
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_number = {
                executor.submit(self.fetch_pr, owner, repo, n, force_refresh): n
                for n in numbers
            }
            done = 0
            for future in as_completed(future_to_number):
                n = future_to_number[future]
                try:
                    results[n] = future.result()
                except Exception:
                    logger.exception("Failed to fetch %s/%s#%d", owner, repo, n)
                done += 1
                if done % 25 == 0 or done == len(numbers):
                    logger.info("Fetched %d/%d PRs for %s/%s", done, len(numbers), owner, repo)

        missing = [n for n in numbers if n not in results]
        if missing:
            logger.warning("%s/%s: %d PR(s) could not be fetched: %s", owner, repo, len(missing), missing)
        return [results[n] for n in numbers if n in results]

    # ---------- discovery helpers, used by the mining step to build samples ----------

    def search_pr_numbers(
        self, owner: str, repo: str, extra_qualifiers: str, limit: Optional[int] = None
    ) -> list[int]:
        """Search PRs via the REST search API. `extra_qualifiers` is appended
        to `repo:{owner}/{repo} is:pr` (e.g. 'is:closed author:app/copilot-swe-agent').
        """
        query = f"repo:{owner}/{repo} is:pr {extra_qualifiers}".strip()
        numbers: list[int] = []
        page = 1
        while True:
            response = self._request_with_retry(
                "GET",
                f"{GITHUB_REST_URL}/search/issues",
                params={"q": query, "per_page": 100, "page": page, "sort": "created", "order": "desc"},
            )
            items = response.json().get("items", [])
            if not items:
                break
            numbers.extend(item["number"] for item in items)
            if limit and len(numbers) >= limit:
                return numbers[:limit]
            if len(items) < 100:
                break
            page += 1
            if page > 10:  # GitHub search API caps results at 1,000 (10 pages of 100)
                break
        return numbers

    # ---------- single-issue fetch (Experiment 4's DIFF_ISSUE / COMPLETE
    # context tiers), cached separately from the PR cache ----------

    def _issue_cache_path(self, owner: str, repo: str, number: int) -> Path:
        # A dedicated `issues/` subtree, NOT `_cache_path`'s PR cache: issue
        # and PR numbers share GitHub's per-repo numbering namespace but are
        # different resources, so the two must never collide on disk.
        issue_dir = self.cache_dir / "issues" / owner / repo
        issue_dir.mkdir(parents=True, exist_ok=True)
        return issue_dir / f"{number}.json"

    def fetch_issue(
        self, owner: str, repo: str, number: int, force_refresh: bool = False
    ) -> dict[str, Any]:
        """Fetch one issue's title + body via REST
        (`GET /repos/{owner}/{repo}/issues/{number}`), cached to disk. Used by
        Experiment 4's context population to resolve an issue *reference*
        (`src/llm/prompts/context_aug.py:extract_issue_references`) into the
        issue's actual text for the `DIFF_ISSUE` / `COMPLETE_SE_CONTEXT` tiers.

        GitHub's issues endpoint also serves pull requests (a PR is a superset
        of an issue), so the returned record carries `is_pull_request` and
        callers should typically skip it as "linked issue" context in that
        case. A 404 -- including the common case where a referenced `#N` is
        actually a PR number in a different numbering context, or the issue
        was deleted/transferred -- is cached and returned as
        `{"not_found": True}` rather than raised, since this is an expected,
        routine outcome for many of the bare `#N` references
        `extract_issue_references` finds, not an error condition.
        """
        cache_path = self._issue_cache_path(owner, repo, number)
        if not force_refresh and cache_path.exists():
            try:
                return json.loads(cache_path.read_text())
            except json.JSONDecodeError:
                logger.warning("Corrupt issue cache file %s, will refetch", cache_path)

        url = f"{GITHUB_REST_URL}/repos/{owner}/{repo}/issues/{number}"
        attempt = 0
        while True:
            response = self.session.request("GET", url, timeout=30)
            if response.status_code == 404:
                record = {"not_found": True}
                cache_path.write_text(json.dumps(record))
                return record
            if response.status_code < 400:
                data = response.json()
                record = {
                    "not_found": False,
                    "number": data.get("number"),
                    "title": data.get("title"),
                    "body": data.get("body"),
                    "state": data.get("state"),
                    "is_pull_request": "pull_request" in data,
                }
                cache_path.write_text(json.dumps(record))
                return record
            if not should_retry(response.status_code, attempt):
                response.raise_for_status()
            retry_after_header = response.headers.get("Retry-After")
            retry_after = float(retry_after_header) if retry_after_header else None
            delay = backoff_delay(attempt, retry_after)
            logger.warning(
                "GET %s -> %d (attempt %d/%d); retrying in %.1fs",
                url, response.status_code, attempt + 1, MAX_RETRIES, delay,
            )
            time.sleep(delay)
            attempt += 1

    def list_closed_pr_numbers(self, owner: str, repo: str, limit: int) -> list[int]:
        """Recent closed PR numbers via REST, for filling a sample beyond the
        AI-authored slice found by search_pr_numbers.
        """
        numbers: list[int] = []
        page = 1
        while len(numbers) < limit:
            response = self._request_with_retry(
                "GET",
                f"{GITHUB_REST_URL}/repos/{owner}/{repo}/pulls",
                params={
                    "state": "closed",
                    "sort": "updated",
                    "direction": "desc",
                    "per_page": 100,
                    "page": page,
                },
            )
            batch = response.json()
            if not batch:
                break
            numbers.extend(pr["number"] for pr in batch)
            page += 1
        return numbers[:limit]
