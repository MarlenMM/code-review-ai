"""FastAPI backend (plan Section 7.1 / Section 9 row 23): a `/review`
endpoint wrapping the best Experiment 2 model (fast, ML-only mode) and
Experiment 4's best config (deep, LLM mode).

    fast (default) -- `src/api/ml_features.py` + `src/api/ml_model.py`:
        parse the diff, compute the V1 feature vector, run the trained
        `rf_v1_balanced` pipeline. No network call, sub-second, free.
    deep -- everything `fast` does, PLUS `src/api/llm_review.py`: one live
        Groq call (Experiment 4's `multi_turn` @ `diff_repo_context`
        config) that generates review comments. Costs real, shared,
        rate-limited quota, so it is opt-in (`mode="deep"`), not the
        default; any LLM failure degrades to the fast-mode result plus a
        `llm_warning` rather than a 500 (see `llm_review.py`,
        `reports/api_design.md`).

Merge probability always comes from the ML model (both modes) -- it is the
reliable, free, always-available signal; review comments only come from the
LLM pass, and are `None` in fast mode. This is the literal reading of plan
§7.1: "fast mode (ML-only, instant, free) and deep mode (adds the LLM
pass)" -- the LLM augments the ML result, it does not replace it.
"""

from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from src.api.diff_parsing import parse_unified_diff
from src.api.llm_review import generate_review_comments
from src.api.ml_features import compute_v1_features
from src.api.ml_model import MODEL_NAME, predict_merge_probability

app = FastAPI(
    title="Code Review AI",
    description=(
        "Wraps Experiment 2's trained merge-prediction model (fast mode) and "
        "Experiment 4's best LLM review-comment config (deep mode) behind one "
        "/review endpoint."
    ),
    version="0.1.0",
)


class ReviewRequest(BaseModel):
    diff: str = Field(..., min_length=1, description="A unified diff, e.g. the output of `git diff`.")
    title: str | None = Field(None, description="PR/change title (optional).")
    description: str | None = Field(None, description="PR/change description or body (optional).")
    commit_messages: list[str] | None = Field(
        None, description="Commit messages for this change, in order (optional)."
    )
    repo: str | None = Field(
        None, description="Repository, e.g. 'owner/name' (optional; only used to phrase the deep-mode prompt)."
    )
    mode: Literal["fast", "deep"] = Field(
        "fast",
        description=(
            "'fast': ML-only merge probability, no network calls. "
            "'deep': also generates review comments via a live Groq LLM call."
        ),
    )


class ReviewResponse(BaseModel):
    mode: str
    n_files_changed: int
    merge_probability: float
    merge_prediction: Literal["MERGE", "CLOSE"]
    ml_model: str
    features: dict[str, float]
    review_comments: list[str] | None = None
    llm_config: str | None = None
    llm_warning: str | None = None


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Send a browser to the docs instead of a bare 404.

    `/` is where anyone who just started `uvicorn` looks first, and
    `{"detail":"Not Found"}` is the API's version of a blank screen: it is
    accurate and it helps nobody. The generated OpenAPI page lists both
    real endpoints and lets you call them, so that is where to land. Scripts
    are unaffected -- they want `/health`, which is unchanged.
    """
    return RedirectResponse(url="/docs")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/review", response_model=ReviewResponse)
def review(request: ReviewRequest) -> ReviewResponse:
    files = parse_unified_diff(request.diff)
    if not files:
        raise HTTPException(
            422,
            "Could not parse any changed files from `diff`; expected unified-diff "
            "format (e.g. `git diff` output).",
        )

    features = compute_v1_features(
        files,
        title=request.title or "",
        description=request.description or "",
        commit_messages=request.commit_messages,
    )
    probability = predict_merge_probability(features)
    prediction: Literal["MERGE", "CLOSE"] = "MERGE" if probability >= 0.5 else "CLOSE"

    review_comments = None
    llm_config = None
    llm_warning = None
    if request.mode == "deep":
        result = generate_review_comments(files, repo=request.repo, title=request.title)
        review_comments = result.comments
        llm_config = result.config_label
        llm_warning = result.warning

    return ReviewResponse(
        mode=request.mode,
        n_files_changed=len(files),
        merge_probability=round(probability, 4),
        merge_prediction=prediction,
        ml_model=MODEL_NAME,
        features=features,
        review_comments=review_comments,
        llm_config=llm_config,
        llm_warning=llm_warning,
    )
