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
 *
 * Failures are raised as `ReviewError`, which carries a `kind` alongside
 * the message. The panel's error state needs to say something actionable,
 * and "start the backend" is the right advice for a refused connection and
 * the wrong advice for a 422 -- a distinction that is thrown away the
 * moment both become a bare `Error` the caller has to pattern-match on
 * strings to tell apart.
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

/** `unreachable`: the request never got an answer (backend down, wrong
 * URL, DNS). `http`: it answered, with a non-2xx status. */
export type ReviewErrorKind = "unreachable" | "http";

export class ReviewError extends Error {
  constructor(
    message: string,
    readonly kind: ReviewErrorKind,
  ) {
    super(message);
    this.name = "ReviewError";
  }
}

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
    throw new ReviewError(
      `Could not reach the Code Review AI backend at ${backendUrl}: ${message}`,
      "unreachable",
    );
  }

  if (!response.ok) {
    const body = await safeReadText(response);
    throw new ReviewError(
      `Backend returned ${response.status} ${response.statusText}: ${body}`,
      "http",
    );
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
