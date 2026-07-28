/**
 * Thin client for the Step 23 backend's `POST /review` (`src/api/main.py`).
 * Mirrors that endpoint's Pydantic request/response schema exactly -- see
 * `reports/api_design.md` Section 5 for the contract this is a client of.
 *
 * Uses the extension host's global `fetch` (available since VS Code's
 * bundled Node 18+; no extra runtime dependency) rather than a bundled
 * HTTP client -- `contributes` declares no runtime `dependencies` for
 * exactly this reason. `fetchFn` is injected (defaulting to global
 * `fetch`) so `reviewDiff` is testable without a real backend -- see
 * `src/test/apiClient.test.ts`.
 */

export interface ReviewRequest {
  diff: string;
  title?: string;
  description?: string;
  commit_messages?: string[];
  repo?: string;
  mode?: "fast" | "deep";
}

export interface ReviewResponse {
  mode: string;
  n_files_changed: number;
  merge_probability: number;
  merge_prediction: "MERGE" | "CLOSE";
  ml_model: string;
  features: Record<string, number>;
  review_comments: string[] | null;
  llm_config: string | null;
  llm_warning: string | null;
}

export type FetchFn = typeof fetch;

export async function reviewDiff(
  backendUrl: string,
  request: ReviewRequest,
  fetchFn: FetchFn = fetch,
): Promise<ReviewResponse> {
  const url = new URL("/review", backendUrl).toString();

  let response: Response;
  try {
    response = await fetchFn(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    throw new Error(`Could not reach the Code Review AI backend at ${backendUrl}: ${message}`);
  }

  if (!response.ok) {
    const body = await safeReadText(response);
    throw new Error(`Backend returned ${response.status} ${response.statusText}: ${body}`);
  }

  return (await response.json()) as ReviewResponse;
}

async function safeReadText(response: Response): Promise<string> {
  try {
    return await response.text();
  } catch {
    return "<no response body>";
  }
}
