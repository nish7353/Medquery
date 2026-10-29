import { useEffect, useRef, useState } from "react";

import { api } from "./api.js";
import { describeError } from "./format.js";

// Sarvam's real-time speech-to-text accepts up to 30 s; stop a little before that.
export const MAX_RECORDING_SECONDS = 29;
const MIN_RECORDING_MS = 600;
const MIME_TYPES = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "audio/mp4"];

const SpeechRecognition = typeof window !== "undefined" ? window.SpeechRecognition ?? window.webkitSpeechRecognition : undefined;

function micError(err) {
  if (err?.name === "NotAllowedError" || err?.name === "SecurityError") {
    return "Microphone access is blocked. Allow it in your browser's site settings to ask by voice.";
  }
  if (err?.name === "NotFoundError") return "No microphone was found on this device.";
  return "The microphone could not be started.";
}

/**
 * Voice questions. With `serverVoice` the audio is recorded and transcribed by Sarvam AI on the
 * backend; otherwise the browser's own speech recognition is used, in the selected language.
 * `onText` receives the transcript to put in the composer for the user to review.
 */
export function useVoiceInput({ language, serverVoice, onText, onError }) {
  const [state, setState] = useState("idle"); // idle | recording | transcribing
  const [seconds, setSeconds] = useState(0);
  const recorder = useRef(null);
  const recognition = useRef(null);
  const timer = useRef(null);
  const discard = useRef(false);
  const cb = useRef({ onText, onError });
  cb.current = { onText, onError };

  const canRecord = typeof window !== "undefined" && Boolean(navigator.mediaDevices?.getUserMedia && window.MediaRecorder);
  const supported = serverVoice ? canRecord : Boolean(SpeechRecognition);

  useEffect(() => () => {
    discard.current = true;
    clearInterval(timer.current);
    recorder.current?.state === "recording" && recorder.current.stop();
    recognition.current?.abort();
  }, []);

  async function transcribe(blob) {
    setState("transcribing");
    try {
      const out = await api.transcribe(blob, language);
      cb.current.onText(out.transcript);
    } catch (err) {
      cb.current.onError(describeError(err));
    } finally {
      setState("idle");
    }
  }

  async function startRecording() {
    if (!canRecord) {
      cb.current.onError("Voice recording isn't supported in this browser.");
      return;
    }
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (err) {
      cb.current.onError(micError(err));
      return;
    }
    const mimeType = MIME_TYPES.find((t) => window.MediaRecorder.isTypeSupported?.(t));
    const rec = new window.MediaRecorder(stream, mimeType ? { mimeType } : undefined);
    const chunks = [];
    const started = Date.now();
    rec.ondataavailable = (e) => e.data?.size && chunks.push(e.data);
    rec.onstop = () => {
      clearInterval(timer.current);
      stream.getTracks().forEach((t) => t.stop());
      if (discard.current) return;
      const blob = new Blob(chunks, { type: rec.mimeType || mimeType || "audio/webm" });
      if (Date.now() - started < MIN_RECORDING_MS || blob.size < 1024) {
        setState("idle");
        cb.current.onError("That recording was too short. Press the microphone, speak, then press it again.");
        return;
      }
      transcribe(blob);
    };
    recorder.current = rec;
    discard.current = false;
    rec.start();
    setSeconds(0);
    setState("recording");
    timer.current = setInterval(() => {
      const s = Math.floor((Date.now() - started) / 1000);
      setSeconds(s);
      if (s >= MAX_RECORDING_SECONDS && rec.state === "recording") rec.stop();
    }, 250);
  }

  function startDictation() {
    if (!SpeechRecognition) {
      cb.current.onError("Voice input isn't supported in this browser.");
      return;
    }
    const rec = new SpeechRecognition();
    rec.lang = language;
    rec.interimResults = false;
    rec.onresult = (e) => {
      const text = Array.from(e.results).map((r) => r[0].transcript).join(" ").trim();
      if (text) cb.current.onText(text);
    };
    rec.onerror = (e) => {
      if (e.error === "not-allowed") cb.current.onError(micError({ name: "NotAllowedError" }));
      else if (e.error === "no-speech") cb.current.onError("No speech was heard. Please try again.");
    };
    rec.onend = () => setState("idle");
    recognition.current = rec;
    rec.start();
    setState("recording");
  }

  function toggle() {
    if (state === "transcribing") return;
    if (state === "recording") {
      if (serverVoice) recorder.current?.stop();
      else recognition.current?.stop();
      return;
    }
    if (serverVoice) startRecording();
    else startDictation();
  }

  return { state, seconds, supported, toggle, serverVoice };
}
