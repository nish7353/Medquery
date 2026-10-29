// Thin client for the MedQuery REST API. All responses use the envelope
// { success: true, data } or { success: false, error: { code, message } }.

const API_BASE = window.MEDQUERY_API_BASE ?? "/api";

export class ApiError extends Error {
  constructor(message, status = 0, code = "network_error") {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
  }
}

async function request(path, { method = "GET", body, timeoutMs = 120000, raw = false } = {}) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  const isForm = typeof FormData !== "undefined" && body instanceof FormData;
  let res;
  try {
    res = await fetch(API_BASE + path, {
      method,
      // FormData sets its own multipart Content-Type with the boundary.
      headers: body && !isForm ? { "Content-Type": "application/json" } : undefined,
      body: body ? (isForm ? body : JSON.stringify(body)) : undefined,
      signal: ctrl.signal,
    });
  } catch (err) {
    if (err.name === "AbortError") throw err;
    throw new ApiError("Cannot reach the MedQuery server.", 0);
  } finally {
    clearTimeout(timer);
  }
  if (raw && res.ok) return res.blob();
  let payload = null;
  try {
    payload = await res.json();
  } catch {
    /* non-JSON response */
  }
  if (!res.ok || !payload?.success) {
    const e = payload?.error ?? {};
    throw new ApiError(e.message ?? `Request failed (${res.status}).`, res.status, e.code ?? "http_error");
  }
  return payload.data;
}

export const api = {
  health: () => request("/health", { timeoutMs: 10000 }),
  languages: () => request("/languages", { timeoutMs: 10000 }),
  ask: (question, language = "en-IN") => request("/query", { method: "POST", body: { question, language } }),
  transcribe: (audio, language) => {
    const form = new FormData();
    const ext = (audio.type.split("/")[1] || "webm").split(";")[0];
    form.append("file", audio, `question.${ext}`);
    form.append("language", language);
    return request("/transcribe", { method: "POST", body: form, timeoutMs: 45000 });
  },
  /** Returns a WAV Blob of `text` read aloud in `language`. */
  speech: (text, language) => request("/speech", { method: "POST", body: { text, language }, timeoutMs: 45000, raw: true }),
  getQuery: (id) => request(`/query/${encodeURIComponent(id)}`),
  history: () => request("/history"),
  clearHistory: () => request("/history", { method: "DELETE" }),
};
