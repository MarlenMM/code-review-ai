"""Mine ~300 closed PRs each from the 5 target repos (plan Section 3.1),
apply Step 3's AI detection to every PR, and write flat parquet tables to
data/processed/ per the Section 3.3 data model.

Sampling strategy (Section 3.2): for each repo, first search for PRs authored
by GitHub's Copilot coding agent (`is:closed author:app/copilot-swe-agent`),
capped at ~100, so every repo has a real AI-authored slice; then fill the
remainder with recent closed PRs up to the per-repo target. A pure random
sample of 300 would likely contain few or zero AI-authored PRs by chance, so
this is a deliberate, over-representing design choice -- documented here and
in the Lab Report 1 write-up, not a hidden shortcut.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.mining.ai_detection import detect_pr
from src.mining.github_client import GitHubClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

TARGET_REPOS: list[tuple[str, str]] = [
    ("microsoft", "vscode"),
    ("dotnet", "runtime"),
    ("dotnet", "aspnetcore"),
    ("microsoft", "semantic-kernel"),
    ("home-assistant", "core"),
]

PRS_PER_REPO = 300
AI_AUTHORED_CAP = 100
AI_AUTHOR_SEARCH_QUALIFIERS = "is:closed author:app/copilot-swe-agent"

OUTPUT_DIR = Path("data/processed")

TABLE_NAMES = (
    "pull_requests", "commits", "files_changed",
    "reviews", "review_comments", "issue_comments", "ai_signals",
)


# --------------------------------------------------------------------------- #
# sampling
# --------------------------------------------------------------------------- #

def build_sample_numbers(
    client: GitHubClient,
    owner: str,
    repo: str,
    prs_per_repo: int = PRS_PER_REPO,
    ai_authored_cap: int = AI_AUTHORED_CAP,
) -> tuple[list[int], set[int]]:
    """Return (ordered PR numbers to fetch, set of AI-authored-search hits)
    for one repo, per the Section 3.2 sampling strategy."""
    effective_cap = min(ai_authored_cap, prs_per_repo)
    ai_numbers_raw = client.search_pr_numbers(owner, repo, AI_AUTHOR_SEARCH_QUALIFIERS, limit=effective_cap)
    # De-dup defensively: GitHub's search/listing endpoints paginate over
    # live, frequently-updated data (especially on a high-traffic repo like
    # vscode), so an item can shift position between page fetches and land
    # on two pages at once -- confirmed to happen in practice, not just
    # theoretical, so every page-based listing gets deduped here.
    ai_numbers = list(dict.fromkeys(ai_numbers_raw))
    ai_number_set = set(ai_numbers)
    logger.info("%s/%s: found %d copilot-swe-agent PR(s) (cap %d)", owner, repo, len(ai_numbers), effective_cap)

    remaining_slots = prs_per_repo - len(ai_numbers)
    fill_numbers: list[int] = []
    seen = set(ai_number_set)
    if remaining_slots > 0:
        # Over-fetch a bit: recent closed PRs will overlap the AI-authored set.
        candidates = client.list_closed_pr_numbers(owner, repo, remaining_slots + len(ai_number_set) + 20)
        for n in candidates:
            if n in seen:
                continue
            seen.add(n)
            fill_numbers.append(n)
            if len(fill_numbers) >= remaining_slots:
                break
    logger.info("%s/%s: filling %d more PR(s) from recent closed PRs", owner, repo, len(fill_numbers))

    return ai_numbers + fill_numbers, ai_number_set


def mine_repo(client: GitHubClient, owner: str, repo: str, prs_per_repo: int = PRS_PER_REPO) -> list[dict[str, Any]]:
    numbers, _ = build_sample_numbers(client, owner, repo, prs_per_repo)
    logger.info("%s/%s: fetching %d PR(s)", owner, repo, len(numbers))
    records = client.fetch_prs(owner, repo, numbers)
    missing = len(numbers) - len(records)
    if missing:
        logger.warning("%s/%s: %d PR(s) targeted but not fetched (excluded, e.g. server-side 502)", owner, repo, missing)
    return records


# --------------------------------------------------------------------------- #
# flattening raw GraphQL+REST records into per-table rows
# --------------------------------------------------------------------------- #

def _labels(pr: dict) -> list[str]:
    return [n["name"] for n in (pr.get("labels") or {}).get("nodes") or [] if n]


def flatten_pull_request(record: dict, detection_row: dict) -> dict:
    pr = record["pull_request"]
    return {
        "id": pr.get("id"),
        "repo": f"{record['owner']}/{record['repo']}",
        "owner": record["owner"],
        "repo_name": record["repo"],
        "number": record["number"],
        "title": pr.get("title"),
        "body": pr.get("body"),
        "author_login": detection_row["author_login"],
        "author_type": detection_row["author_type"],
        "created_at": pr.get("createdAt"),
        "closed_at": pr.get("closedAt"),
        "merged_at": pr.get("mergedAt"),
        "state": pr.get("state"),
        "is_merged": bool(pr.get("merged")),
        "additions": pr.get("additions"),
        "deletions": pr.get("deletions"),
        "changed_files": pr.get("changedFiles"),
        "labels": _labels(pr),
        # ai_signals flags folded in directly on pull_requests too (not only
        # the separate ai_signals table below), since Step 5 reads them from
        # pull_requests.parquet directly.
        "is_ai_authored": detection_row["is_ai_authored"],
        "ai_authored_kind": detection_row["ai_authored_kind"],
        "ai_authored_method": detection_row["ai_authored_method"],
        "ai_authored_confidence": detection_row["ai_authored_confidence"],
        "ai_authored_evidence": detection_row["ai_authored_evidence"],
        "is_ai_reviewed": detection_row["is_ai_reviewed"],
        "ai_reviewer_logins_str": detection_row["ai_reviewer_logins_str"],
        "n_ai_reviewers": detection_row["n_ai_reviewers"],
        "author_category": detection_row["author_category"],
    }


def flatten_commits(record: dict) -> list[dict]:
    pr = record["pull_request"]
    pr_id = pr.get("id")
    rows = []
    for node in (pr.get("commits") or {}).get("nodes") or []:
        commit = (node or {}).get("commit") or {}
        author = commit.get("author") or {}
        user = author.get("user") or {}
        rows.append({
            "pr_id": pr_id,
            "sha": commit.get("oid"),
            "message": commit.get("message"),
            "author_name": author.get("name"),
            "author_email": author.get("email"),
            "author_login": user.get("login"),
            "additions": commit.get("additions"),
            "deletions": commit.get("deletions"),
        })
    return rows


def flatten_files(record: dict) -> list[dict]:
    pr = record["pull_request"]
    pr_id = pr.get("id")
    rows = []
    for f in record.get("files") or []:
        rows.append({
            "pr_id": pr_id,
            "filename": f.get("filename"),
            "status": f.get("status"),
            "additions": f.get("additions"),
            "deletions": f.get("deletions"),
            "patch": f.get("patch"),
        })
    return rows


def flatten_reviews(record: dict) -> list[dict]:
    pr = record["pull_request"]
    pr_id = pr.get("id")
    rows = []
    for r in (pr.get("reviews") or {}).get("nodes") or []:
        author = r.get("author") or {}
        rows.append({
            "pr_id": pr_id,
            "review_id": r.get("id"),
            "reviewer_login": author.get("login"),
            "reviewer_type": author.get("__typename"),
            "state": r.get("state"),
            "submitted_at": r.get("submittedAt"),
        })
    return rows


def flatten_review_comments(record: dict) -> list[dict]:
    pr = record["pull_request"]
    pr_id = pr.get("id")
    rows = []
    for r in (pr.get("reviews") or {}).get("nodes") or []:
        review_id = r.get("id")
        for c in (r.get("comments") or {}).get("nodes") or []:
            author = c.get("author") or {}
            rows.append({
                "pr_id": pr_id,
                "review_id": review_id,
                "comment_id": c.get("id"),
                "author_login": author.get("login"),
                "body": c.get("body"),
                "path": c.get("path"),
                "position": c.get("position"),
                "created_at": c.get("createdAt"),
            })
    return rows


def flatten_issue_comments(record: dict) -> list[dict]:
    pr = record["pull_request"]
    pr_id = pr.get("id")
    rows = []
    for c in (pr.get("comments") or {}).get("nodes") or []:
        author = c.get("author") or {}
        rows.append({
            "pr_id": pr_id,
            "comment_id": c.get("id"),
            "author_login": author.get("login"),
            "body": c.get("body"),
            "created_at": c.get("createdAt"),
        })
    return rows


def flatten_ai_signals(detection_row: dict) -> dict:
    return {
        "pr_id": detection_row["pr_id"],
        "is_ai_authored": detection_row["is_ai_authored"],
        "ai_authored_kind": detection_row["ai_authored_kind"],
        "ai_authored_method": detection_row["ai_authored_method"],
        "ai_authored_confidence": detection_row["ai_authored_confidence"],
        "ai_authored_evidence": detection_row["ai_authored_evidence"],
        "is_ai_reviewed": detection_row["is_ai_reviewed"],
        "ai_reviewer_logins_str": detection_row["ai_reviewer_logins_str"],
        "n_ai_reviewers": detection_row["n_ai_reviewers"],
    }


def build_tables(records: list[dict]) -> dict[str, list[dict]]:
    tables: dict[str, list[dict]] = {name: [] for name in TABLE_NAMES}
    for record in records:
        detection = detect_pr(record).as_row()
        tables["pull_requests"].append(flatten_pull_request(record, detection))
        tables["commits"].extend(flatten_commits(record))
        tables["files_changed"].extend(flatten_files(record))
        tables["reviews"].extend(flatten_reviews(record))
        tables["review_comments"].extend(flatten_review_comments(record))
        tables["issue_comments"].extend(flatten_issue_comments(record))
        tables["ai_signals"].append(flatten_ai_signals(detection))
    return tables


def write_tables(tables: dict[str, list[dict]], output_dir: Path = OUTPUT_DIR) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name in TABLE_NAMES:
        df = pd.DataFrame(tables.get(name, []))
        path = output_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
        written[name] = path
        logger.info("Wrote %s (%d rows)", path, len(df))
    return written


def rebuild_from_existing(pull_requests_path: Path, client: GitHubClient) -> list[dict]:
    """Re-derive raw fetch_pr records (cache-only, no network calls, since
    every PR here was already fetched at least once) for every (owner, repo,
    number) in an existing pull_requests.parquet. Used to regenerate all
    tables after a flatten/detection change -- e.g. a new column, or an
    ai_detection.py fix -- without re-mining from GitHub."""
    existing = pd.read_parquet(pull_requests_path, columns=["owner", "repo_name", "number"])
    records = []
    for owner, repo_name, number in existing.itertuples(index=False):
        records.append(client.fetch_pr(owner, repo_name, int(number)))
    return records


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repos", nargs="*", default=None,
        help="owner/repo pairs to mine (e.g. microsoft/vscode); defaults to all 5 target repos",
    )
    parser.add_argument("--prs-per-repo", type=int, default=PRS_PER_REPO)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--rebuild-from-cache", type=Path, default=None,
        help="Path to an existing pull_requests.parquet: regenerate all tables "
             "from the disk cache for exactly that PR set, no new mining.",
    )
    args = parser.parse_args()

    client = GitHubClient()

    if args.rebuild_from_cache:
        all_records = rebuild_from_existing(args.rebuild_from_cache, client)
        logger.info("Rebuilt %d PR record(s) from cache", len(all_records))
    else:
        repos = TARGET_REPOS
        if args.repos:
            repos = [tuple(r.split("/", 1)) for r in args.repos]
        all_records = []
        for owner, repo in repos:
            all_records.extend(mine_repo(client, owner, repo, args.prs_per_repo))
        logger.info("Fetched %d PR(s) total across %d repo(s)", len(all_records), len(repos))

    tables = build_tables(all_records)
    write_tables(tables, args.output_dir)


if __name__ == "__main__":
    main()
