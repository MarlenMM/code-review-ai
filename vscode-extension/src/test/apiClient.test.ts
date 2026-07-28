import assert from "node:assert/strict";
import { test } from "node:test";
import { reviewDiff, type ReviewResponse } from "../apiClient";

function fakeFetchReturning(status: number, body: unknown, statusText = ""): typeof fetch {
  return (async () =>
    ({
      ok: status >= 200 && status < 300,
      status,
      statusText,
      json: async () => body,
      text: async () => JSON.stringify(body),
    }) as Response) as unknown as typeof fetch;
}

const SAMPLE_RESPONSE: ReviewResponse = {
  mode: "fast",
  n_files_changed: 1,
  merge_probability: 0.7,
  merge_prediction: "MERGE",
  ml_model: "rf_v1_balanced",
  features: { m_additions: 2 },
  review_comments: null,
  llm_config: null,
  llm_warning: null,
};

test("returns the parsed response on a 200", async () => {
  const result = await reviewDiff(
    "http://127.0.0.1:8000",
    { diff: "diff --git ..." },
    fakeFetchReturning(200, SAMPLE_RESPONSE),
  );
  assert.deepEqual(result, SAMPLE_RESPONSE);
});

test("posts JSON to /review, joined correctly regardless of trailing slash", async () => {
  const seen: { url?: string; body?: string } = {};
  const fetchFn = (async (url: string, init: RequestInit) => {
    seen.url = url;
    seen.body = init.body as string;
    return { ok: true, status: 200, json: async () => SAMPLE_RESPONSE, text: async () => "" } as Response;
  }) as unknown as typeof fetch;

  await reviewDiff("http://127.0.0.1:8000/", { diff: "x", mode: "deep" }, fetchFn);

  assert.equal(seen.url, "http://127.0.0.1:8000/review");
  assert.deepEqual(JSON.parse(seen.body!), { diff: "x", mode: "deep" });
});

test("throws a readable error on a non-2xx response", async () => {
  await assert.rejects(
    () => reviewDiff(
      "http://127.0.0.1:8000",
      { diff: "x" },
      fakeFetchReturning(422, { detail: "bad diff" }, "Unprocessable Entity"),
    ),
    /422/,
  );
});

test("wraps a network failure with the backend URL", async () => {
  const throwingFetch = (async () => {
    throw new Error("ECONNREFUSED");
  }) as unknown as typeof fetch;

  await assert.rejects(
    () => reviewDiff("http://127.0.0.1:8000", { diff: "x" }, throwingFetch),
    /Could not reach the Code Review AI backend at http:\/\/127\.0\.0\.1:8000/,
  );
});
