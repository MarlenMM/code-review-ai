"""End-to-end tests for `src/api/main.py` via FastAPI's `TestClient`. Fast
mode makes no network calls (real ML model, real diff parsing). Deep mode's
LLM call is monkeypatched here to keep the automated suite network-free and
quota-free -- a real live Groq call was separately exercised manually
against a running `uvicorn` server (see `reports/api_design.md`)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.api import main
from src.api.llm_review import DeepReviewResult

client = TestClient(main.app)

SIMPLE_DIFF = """\
diff --git a/src/foo.py b/src/foo.py
index 1234567..89abcde 100644
--- a/src/foo.py
+++ b/src/foo.py
@@ -1,2 +1,3 @@
 def foo():
-    return 1
+    return 2
+    # extra
"""


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_root_redirects_to_the_docs_instead_of_404ing():
    # `/` is the first thing anyone who just ran `uvicorn` opens; a bare
    # `{"detail":"Not Found"}` there is accurate and useless.
    resp = client.get("/", follow_redirects=False)
    assert resp.status_code in (307, 308)
    assert resp.headers["location"] == "/docs"


def test_review_fast_mode_default():
    resp = client.post("/review", json={"diff": SIMPLE_DIFF})
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "fast"
    assert body["n_files_changed"] == 1
    assert body["ml_model"] == "rf_v1_balanced"
    assert 0.0 <= body["merge_probability"] <= 1.0
    assert body["merge_prediction"] in ("MERGE", "CLOSE")
    assert body["review_comments"] is None
    assert body["llm_config"] is None
    assert body["llm_warning"] is None
    assert isinstance(body["features"], dict)
    assert body["features"]["m_changed_files"] == 1.0


def test_review_fast_mode_with_optional_fields():
    resp = client.post("/review", json={
        "diff": SIMPLE_DIFF,
        "title": "Fix foo",
        "description": "Fixes the return value of foo().",
        "commit_messages": ["fix foo", "address review comment"],
        "repo": "octocat/demo",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["features"]["t_title_len"] == len("Fix foo")
    assert body["features"]["m_num_commits"] == 2.0


def test_review_malformed_diff_returns_422():
    resp = client.post("/review", json={"diff": "this is not a diff at all, no headers"})
    assert resp.status_code == 422
    assert "diff" in resp.json()["detail"].lower()


def test_review_empty_diff_rejected_by_schema():
    resp = client.post("/review", json={"diff": ""})
    assert resp.status_code == 422


def test_review_invalid_mode_rejected_by_schema():
    resp = client.post("/review", json={"diff": SIMPLE_DIFF, "mode": "ultra"})
    assert resp.status_code == 422


def test_review_missing_diff_field_rejected():
    resp = client.post("/review", json={})
    assert resp.status_code == 422


def test_review_deep_mode_success(monkeypatch):
    monkeypatch.setattr(
        main, "generate_review_comments",
        lambda files, repo=None, title=None: DeepReviewResult(
            comments=["Consider adding a test.", "Looks reasonable otherwise."],
            config_label="groq/llama-3.1-8b-instant multi_turn @ diff_repo_context",
            warning=None,
        ),
    )
    resp = client.post("/review", json={"diff": SIMPLE_DIFF, "mode": "deep"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "deep"
    assert body["review_comments"] == ["Consider adding a test.", "Looks reasonable otherwise."]
    assert "multi_turn" in body["llm_config"]
    assert body["llm_warning"] is None
    # the ML merge probability is still computed in deep mode
    assert 0.0 <= body["merge_probability"] <= 1.0


def test_review_deep_mode_degrades_gracefully_on_llm_failure(monkeypatch):
    monkeypatch.setattr(
        main, "generate_review_comments",
        lambda files, repo=None, title=None: DeepReviewResult(
            comments=None, config_label=None, warning="Groq quota exhausted: daily cap hit",
        ),
    )
    resp = client.post("/review", json={"diff": SIMPLE_DIFF, "mode": "deep"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["review_comments"] is None
    assert "quota" in body["llm_warning"].lower()
    # fast-mode result is still present and valid despite the LLM failure
    assert 0.0 <= body["merge_probability"] <= 1.0
    assert body["merge_prediction"] in ("MERGE", "CLOSE")
