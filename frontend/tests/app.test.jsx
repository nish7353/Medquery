// Interaction tests: renders the real App in jsdom with a stubbed fetch. Response
// bodies are neutral TEST-ONLY placeholders, not medical content.
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, expect, test } from "vitest";

import App from "../src/App.jsx";

let calls;
let queryHandler;
let history;

function json(status, body) {
  return Promise.resolve({ ok: status < 400, status, json: async () => body });
}

function noEvidenceResult(question) {
  return {
    id: "a1", question, status: "NO_EVIDENCE", answer: null, evidence: [], sources: [],
    retrieval: { status: "EMPTY", adapter: "test", passages_returned: 0 },
    analysis: { question_type: "general", keywords: ["placeholder"], bert: { available: false } },
    notices: [{ level: "warning", code: "x", message: "Placeholder notice." }],
    timings_ms: { total: 12 }, created_at: new Date().toISOString(), disclaimer: "Educational prototype.",
  };
}

function answeredResult(question) {
  return {
    ...noEvidenceResult(question), id: "b2", status: "ANSWERED", notices: [],
    answer: { text: "ALPHA0", confidence: 0.82, method: "extractive", evidence_index: 0, verified: false },
    evidence: [{ text: "Placeholder <b>passage</b> with ALPHA0 inside.", source_id: "test-doc-0",
                 source_title: "Test Document", source_url: "javascript:alert(1)", relevance_score: 0.9,
                 used_for_answer: true, highlight_start: 32, highlight_end: 38, metadata: {} }],
    sources: [{ source_id: "test-doc-0", title: "Test Document", url: "javascript:alert(1)" }],
    retrieval: { status: "OK", adapter: "test", passages_returned: 1 },
  };
}

beforeEach(() => {
  calls = [];
  history = [];
  queryHandler = noEvidenceResult;
  globalThis.fetch = (url, opts = {}) => {
    const path = new URL(url, "http://localhost:8000").pathname;
    calls.push({ path, method: opts.method ?? "GET", body: opts.body });
    if (path === "/api/health") {
      return json(200, { success: true, data: {
        components: { knowledge_base: { configured: true, adapter: "test" } },
        limits: { min_question_chars: 3, max_question_chars: 500 },
      } });
    }
    if (path === "/api/history" && opts.method === "DELETE") { history.length = 0; return json(200, { success: true, data: { deleted: 1 } }); }
    if (path === "/api/history") return json(200, { success: true, data: [...history] });
    if (path === "/api/query") {
      const q = JSON.parse(opts.body).question;
      const res = queryHandler(q);
      if (res.__error) return json(res.__error, { success: false, error: { code: "x", message: "boom" } });
      history.unshift({ id: res.id, question: q, status: res.status, created_at: res.created_at });
      return json(200, { success: true, data: res });
    }
    if (path.startsWith("/api/query/")) return json(200, { success: true, data: answeredResult("from history") });
    return json(404, { success: false, error: { code: "not_found", message: "nf" } });
  };
});
afterEach(cleanup);

const box = () => screen.getByLabelText("Your medical question");
async function ask(text) {
  fireEvent.change(box(), { target: { value: text } });
  fireEvent.submit(box().closest("form"));
}

test("health check drives the availability pill without knowledge-base wording", async () => {
  render(<App />);
  await waitFor(() => expect(screen.getByTestId("availability").dataset.state).toBe("online"));
  expect(screen.getByTestId("availability").textContent).toBe("Assistant available");
  expect(document.body.textContent).not.toMatch(/knowledge base/i);
});

test("quick replies fill the composer and select their placeholder", () => {
  render(<App />);
  fireEvent.click(screen.getAllByTestId("quick-reply")[0]);
  const t = box();
  expect(t.value.slice(t.selectionStart, t.selectionEnd)).toBe("[condition]");
});

test("client-side validation blocks a too-short question", async () => {
  render(<App />);
  await ask("a");
  expect(screen.getByRole("alert").textContent).toMatch(/at least 3/);
  expect(calls.some((c) => c.path === "/api/query")).toBe(false);
});

test("a result without an answer shows the status reply and notices", async () => {
  render(<App />);
  await ask("What is the placeholder?");
  await waitFor(() => screen.getByTestId("status-chip"));
  expect(screen.getByTestId("status-chip").textContent).toBe("No relevant evidence found");
  expect(screen.getByTestId("notice").textContent).toMatch(/Placeholder notice/);
  expect(screen.queryByTestId("answer-text")).toBeNull();
  expect(JSON.parse(calls.findLast((c) => c.path === "/api/query").body).question).toBe("What is the placeholder?");
  await waitFor(() => screen.getByTestId("history-item"));
});

test("an answer renders with confidence, escaped highlighted evidence and safe sources", async () => {
  queryHandler = answeredResult;
  render(<App />);
  await ask("What is the answer token?");
  await waitFor(() => screen.getByTestId("answer-text"));
  expect(screen.getByTestId("answer-text").textContent).toMatch(/^ALPHA0/);
  expect(screen.queryByRole("meter")).toBeNull();
  expect(document.body.textContent).not.toMatch(/confidence/i);
  expect(within(screen.getByTestId("sources")).queryByRole("link")).toBeNull();
  fireEvent.click(screen.getByRole("button", { name: /supporting passages/ }));
  const passage = screen.getByTestId("passage");
  expect(passage.querySelector("mark").textContent).toBe("ALPHA0");
  expect(passage.querySelector("b")).toBeNull();
});

test("a server error offers retry, and retry recovers", async () => {
  queryHandler = () => ({ __error: 500 });
  render(<App />);
  await ask("What is the retry case?");
  await waitFor(() => screen.getByTestId("error-text"));
  expect(screen.getByTestId("error-text").textContent).toMatch(/server ran into a problem/);
  queryHandler = noEvidenceResult;
  fireEvent.click(screen.getByRole("button", { name: /try again/i }));
  await waitFor(() => screen.getByTestId("status-chip"));
});

test("history items reload a stored result; clear empties the list", async () => {
  history.push({ id: "b2", question: "from history", status: "ANSWERED", created_at: new Date().toISOString() });
  render(<App />);
  fireEvent.click(await screen.findByTestId("history-item"));
  await waitFor(() => expect(screen.getByTestId("user-message").textContent).toMatch(/from history/));
  fireEvent.click(screen.getByRole("button", { name: /clear history/i }));
  await screen.findByTestId("history-empty");
});

test("new chat clears the conversation", async () => {
  render(<App />);
  await ask("What is the placeholder?");
  await waitFor(() => screen.getByTestId("status-chip"));
  fireEvent.click(screen.getByRole("button", { name: /new chat/i }));
  expect(screen.queryByTestId("user-message")).toBeNull();
  expect(screen.getByText("How can I help today?")).toBeTruthy();
});

test("the emergency banner can be dismissed", () => {
  render(<App />);
  fireEvent.click(screen.getByRole("button", { name: /dismiss emergency notice/i }));
  expect(screen.queryByRole("region", { name: /emergency notice/i })).toBeNull();
});
