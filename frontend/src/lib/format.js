// Pure helpers (no DOM access) so they can be unit-tested without rendering.

// `reply` is what the assistant says when the status has no extracted answer to show.
export const STATUS_META = {
  ANSWERED: { tone: "success", label: "Answer found in evidence", icon: "check" },
  LOW_CONFIDENCE: {
    tone: "warning", label: "Partial match", icon: "alert",
    reply: "This answer comes from a passage that only partly matches your question, so treat it with caution.",
  },
  EVIDENCE_ONLY: {
    tone: "info", label: "Evidence retrieved", icon: "book",
    reply: "I found related passages but couldn't pick out a direct answer. Here is what the sources say.",
  },
  NO_ANSWER_IN_EVIDENCE: {
    tone: "info", label: "No direct answer in evidence", icon: "book",
    reply: "The sources I found don't directly answer this. Try rephrasing, or ask your doctor.",
  },
  NO_EVIDENCE: {
    tone: "neutral", label: "No relevant evidence found", icon: "search",
    reply: "I couldn't find reliable information on this. Try naming the condition or medication.",
  },
  KB_NOT_CONFIGURED: {
    tone: "neutral", label: "Sources unavailable", icon: "plug",
    reply: "Medical sources are unavailable right now, so I can't answer this.",
  },
  RETRIEVAL_FAILED: {
    tone: "danger", label: "Retrieval failed", icon: "alert",
    reply: "Something went wrong while searching the sources. Please try again.",
  },
};

export function statusMeta(status) {
  return STATUS_META[status] ?? { tone: "neutral", label: String(status ?? "Unknown"), icon: "info" };
}

export function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

/** Escape `text` and wrap the [start, end) range in <mark>. Invalid ranges are ignored. */
export function highlightRange(text, start, end) {
  const s = String(text ?? "");
  if (!Number.isInteger(start) || !Number.isInteger(end) || start < 0 || end > s.length || start >= end) {
    return escapeHtml(s);
  }
  return escapeHtml(s.slice(0, start)) + "<mark>" + escapeHtml(s.slice(start, end)) + "</mark>" + escapeHtml(s.slice(end));
}

export function confidenceLabel(score) {
  if (typeof score !== "number" || Number.isNaN(score)) return { pct: 0, label: "Unknown" };
  const pct = Math.round(Math.min(Math.max(score, 0), 1) * 100);
  const label = score >= 0.6 ? "High" : score >= 0.3 ? "Moderate" : "Low";
  return { pct, label };
}

/** Accept only http(s) URLs for source links; anything else is rendered as plain text. */
export function safeUrl(url) {
  try {
    const u = new URL(String(url));
    return u.protocol === "http:" || u.protocol === "https:" ? u.href : null;
  } catch {
    return null;
  }
}

export function validateQuestion(text, { min = 3, max = 500 } = {}) {
  const clean = String(text ?? "").replace(/\s+/g, " ").trim();
  if (clean.length < min) return { ok: false, message: `Please enter at least ${min} characters.` };
  if (clean.length > max) return { ok: false, message: `Please keep your question under ${max} characters.` };
  if (!/\p{L}/u.test(clean)) return { ok: false, message: "Please enter a question in words." };
  return { ok: true, value: clean };
}

export function relativeTime(iso, now = Date.now()) {
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return "";
  const sec = Math.max(0, Math.round((now - t) / 1000));
  if (sec < 60) return "just now";
  if (sec < 3600) return `${Math.floor(sec / 60)} min ago`;
  if (sec < 86400) return `${Math.floor(sec / 3600)} h ago`;
  return new Date(t).toLocaleDateString();
}

/** Split `text` around a valid [start, end) range for rendering a highlight; null if invalid. */
export function splitHighlight(text, start, end) {
  const s = String(text ?? "");
  if (!Number.isInteger(start) || !Number.isInteger(end) || start < 0 || end > s.length || start >= end) {
    return null;
  }
  return [s.slice(0, start), s.slice(start, end), s.slice(end)];
}

export function clockTime(iso) {
  const t = Date.parse(iso);
  return Number.isNaN(t) ? "" : new Date(t).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
}

/** Bucket a timestamp into the sidebar's history groups. */
export function dayGroup(iso, now = Date.now()) {
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return "Earlier";
  const startOfToday = new Date(now).setHours(0, 0, 0, 0);
  if (t >= startOfToday) return "Today";
  if (t >= startOfToday - 86400000) return "Yesterday";
  if (t >= startOfToday - 6 * 86400000) return "Previous 7 days";
  return "Earlier";
}

/** Find the first [placeholder] in a template so the UI can pre-select it. */
export function placeholderRange(template) {
  const m = /\[[^\]]+\]/.exec(template);
  return m ? [m.index, m.index + m[0].length] : null;
}

/** Normalise an API error (network, HTTP or envelope) into a user-facing message. */
export function describeError(err) {
  if (!err) return "Unknown error.";
  if (err.name === "AbortError") return "The request took too long. Please try again.";
  if (err.status === 413) return "Your question is too long.";
  if (err.status === 422 && err.message) return err.message;
  // Voice/translation failures carry a message written for users (timeouts, missing key, quota...).
  if ([415, 429, 502, 503, 504].includes(err.status) && err.message && err.code !== "http_error") return err.message;
  if (err.status >= 500) return "The server ran into a problem. Please retry in a moment.";
  if (err.status === 0 || err instanceof TypeError) return "Cannot reach the MedQuery server. Is the backend running?";
  return err.message || "Request failed.";
}
