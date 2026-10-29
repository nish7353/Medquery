// Language selection, voice questions and spoken answers, with fetch, the microphone,
// MediaRecorder and Audio stubbed. Strings are neutral TEST-ONLY placeholders.
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

import App from "../src/App.jsx";

let calls;
let configured;
let transcribeResponse;

const LANGS = [
  { code: "en-IN", name: "English", native_name: "English" },
  { code: "kn-IN", name: "Kannada", native_name: "ಕನ್ನಡ" },
  { code: "hi-IN", name: "Hindi", native_name: "हिन्दी" },
];

function reply(status, body, blob) {
  return Promise.resolve({ ok: status < 400, status, json: async () => body, blob: async () => blob });
}

function kannadaResult(question, warnings = []) {
  return {
    id: "k1", question: "What is the placeholder thing?", status: "ANSWERED",
    answer: { text: "ALPHA 5 mg", confidence: 0.9, method: "extractive", evidence_index: 0, verified: false },
    evidence: [{ text: "ALPHA 5 mg passage.", source_id: "doc-0", source_title: "Test Document", relevance_score: 0.9,
                 used_for_answer: true, highlight_start: 0, highlight_end: 10, metadata: {} }],
    sources: [{ source_id: "doc-0", title: "Test Document", url: "https://example.org/doc" }],
    retrieval: { status: "OK", adapter: "test", passages_returned: 1 }, analysis: {}, notices: [],
    timings_ms: { total: 5 }, created_at: new Date().toISOString(), disclaimer: "x",
    localization: {
      language: "kn-IN", original_question: question, english_question: "What is the placeholder thing?",
      question_translated: true, answer_text: "ಆಲ್ಫಾ 5 mg", status_message: "ಸ್ಥಿತಿ", notices: [], complete: true, warnings,
    },
  };
}

let nextWarnings = [];

beforeEach(() => {
  calls = [];
  configured = true;
  nextWarnings = [];
  transcribeResponse = () => reply(200, { success: true, data: { transcript: "ಪ್ಲೇಸ್‌ಹೋಲ್ಡರ್ ಪ್ರಶ್ನೆ", language: "kn-IN" } });
  localStorage.clear();
  globalThis.fetch = (url, opts = {}) => {
    const path = new URL(url, "http://localhost:8000").pathname;
    calls.push({ path, method: opts.method ?? "GET", body: opts.body });
    if (path === "/api/health") return reply(200, { success: true, data: { limits: { min_question_chars: 3, max_question_chars: 500 } } });
    if (path === "/api/languages") {
      return reply(200, { success: true, data: {
        default: "en-IN", multilingual_configured: configured,
        languages: LANGS.map((l) => ({ ...l, available: configured || l.code === "en-IN" })),
      } });
    }
    if (path === "/api/history") return reply(200, { success: true, data: [] });
    if (path === "/api/query") {
      const body = JSON.parse(opts.body);
      return reply(200, { success: true, data: kannadaResult(body.question, nextWarnings) });
    }
    if (path === "/api/transcribe") return transcribeResponse();
    if (path === "/api/speech") return reply(200, null, new Blob(["RIFF"], { type: "audio/wav" }));
    return reply(404, { success: false, error: { code: "nf", message: "nf" } });
  };
  URL.createObjectURL = vi.fn(() => "blob:audio");
  URL.revokeObjectURL = vi.fn();
});
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  delete navigator.mediaDevices;
  delete window.MediaRecorder;
});

const box = () => screen.getByLabelText("Your medical question");
const select = () => screen.getByTestId("language");

async function chooseKannada() {
  await waitFor(() => expect(select().querySelectorAll("option").length).toBe(3));
  fireEvent.change(select(), { target: { value: "kn-IN" } });
}

test("the selected language is sent with the question and the answer is shown translated", async () => {
  render(<App />);
  await chooseKannada();
  fireEvent.change(box(), { target: { value: "ಪ್ಲೇಸ್‌ಹೋಲ್ಡರ್ ಪ್ರಶ್ನೆ" } });
  fireEvent.submit(box().closest("form"));
  await screen.findByTestId("answer-text");
  expect(JSON.parse(calls.find((c) => c.path === "/api/query").body)).toEqual({ question: "ಪ್ಲೇಸ್‌ಹೋಲ್ಡರ್ ಪ್ರಶ್ನೆ", language: "kn-IN" });
  expect(screen.getByTestId("answer-text").textContent).toMatch(/^ಆಲ್ಫಾ 5 mg/);
  expect(screen.getByTestId("answer-text").getAttribute("lang")).toBe("kn-IN");
  expect(screen.getByTestId("english-answer").textContent).toMatch(/ALPHA 5 mg/);
  expect(screen.getByTestId("searched-as").textContent).toMatch(/What is the placeholder thing\?/);
  expect(screen.getByText("Sources (in English)")).toBeTruthy();
  expect(localStorage.getItem("medquery-language")).toBe("kn-IN");
});

test("a changed number in the translation is flagged", async () => {
  nextWarnings = ["numbers_changed"];
  render(<App />);
  await chooseKannada();
  fireEvent.change(box(), { target: { value: "ಪ್ರಶ್ನೆ ಇಲ್ಲಿ" } });
  fireEvent.submit(box().closest("form"));
  expect((await screen.findByTestId("translation-warning")).textContent).toMatch(/Check the numbers/);
});

test("without a Sarvam key only English can be chosen and there is no listen button", async () => {
  configured = false;
  render(<App />);
  await waitFor(() => expect(select().querySelectorAll("option").length).toBe(3));
  const options = [...select().querySelectorAll("option")];
  expect(options.filter((o) => !o.disabled).map((o) => o.value)).toEqual(["en-IN"]);
  expect(options[1].textContent).toMatch(/needs Sarvam key/);
  fireEvent.change(box(), { target: { value: "What is the placeholder thing?" } });
  fireEvent.submit(box().closest("form"));
  await screen.findByTestId("answer-text");
  expect(screen.queryByTestId("listen")).toBeNull();
});

function fakeMicrophone({ deny = false } = {}) {
  const track = { stop: vi.fn() };
  navigator.mediaDevices = {
    getUserMedia: vi.fn(() => (deny
      ? Promise.reject(Object.assign(new Error("denied"), { name: "NotAllowedError" }))
      : Promise.resolve({ getTracks: () => [track] }))),
  };
  class FakeRecorder {
    static isTypeSupported = (t) => t === "audio/webm";
    constructor() { this.state = "inactive"; this.mimeType = "audio/webm"; }
    start() { this.state = "recording"; }
    stop() {
      this.state = "inactive";
      this.ondataavailable({ data: new Blob([new Uint8Array(4096)], { type: "audio/webm" }) });
      this.onstop();
    }
  }
  window.MediaRecorder = FakeRecorder;
  return track;
}

test("a voice question is recorded, transcribed and left in the box for review", async () => {
  const track = fakeMicrophone();
  const now = vi.spyOn(Date, "now");
  now.mockReturnValue(1_000);
  render(<App />);
  await chooseKannada();
  await act(async () => fireEvent.click(screen.getByTestId("mic")));
  expect(screen.getByTestId("voice-status").textContent).toMatch(/Recording 0:00/);
  now.mockReturnValue(4_000);
  await act(async () => fireEvent.click(screen.getByTestId("mic")));
  await waitFor(() => expect(box().value).toBe("ಪ್ಲೇಸ್‌ಹೋಲ್ಡರ್ ಪ್ರಶ್ನೆ"));
  expect(screen.getByTestId("voice-status").textContent).toMatch(/Check the transcript/);
  expect(track.stop).toHaveBeenCalled();
  const upload = calls.find((c) => c.path === "/api/transcribe").body;
  expect(upload.get("language")).toBe("kn-IN");
  expect(upload.get("file").name).toBe("question.webm");
  expect(calls.some((c) => c.path === "/api/query")).toBe(false); // the user submits it
});

test("a blocked microphone explains how to allow it", async () => {
  fakeMicrophone({ deny: true });
  render(<App />);
  await chooseKannada();
  await act(async () => fireEvent.click(screen.getByTestId("mic")));
  expect((await screen.findByRole("alert")).textContent).toMatch(/Microphone access is blocked/);
});

test("transcription errors are shown", async () => {
  fakeMicrophone();
  transcribeResponse = () => reply(504, { success: false, error: { code: "voice_timeout", message: "Sarvam AI speech-to-text took too long." } });
  const now = vi.spyOn(Date, "now");
  now.mockReturnValue(1_000);
  render(<App />);
  await chooseKannada();
  await act(async () => fireEvent.click(screen.getByTestId("mic")));
  now.mockReturnValue(5_000);
  await act(async () => fireEvent.click(screen.getByTestId("mic")));
  expect((await screen.findByRole("alert")).textContent).toMatch(/took too long/);
});

test("listen plays the translated answer and can be stopped", async () => {
  const play = vi.fn(() => Promise.resolve());
  const pause = vi.fn();
  globalThis.Audio = class { play = play; pause = pause; };
  render(<App />);
  await chooseKannada();
  fireEvent.change(box(), { target: { value: "ಪ್ರಶ್ನೆ ಇಲ್ಲಿ" } });
  fireEvent.submit(box().closest("form"));
  const listen = await screen.findByTestId("listen");
  await act(async () => fireEvent.click(listen));
  expect(JSON.parse(calls.find((c) => c.path === "/api/speech").body)).toEqual({ text: "ಆಲ್ಫಾ 5 mg", language: "kn-IN" });
  expect(play).toHaveBeenCalled();
  expect(listen.textContent).toBe("Stop");
  await act(async () => fireEvent.click(listen));
  expect(pause).toHaveBeenCalled();
  expect(listen.textContent).toBe("Listen to answer");
  await act(async () => fireEvent.click(listen)); // replay reuses the audio
  expect(calls.filter((c) => c.path === "/api/speech")).toHaveLength(1);
});
