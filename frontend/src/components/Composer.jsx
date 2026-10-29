import { ArrowUp, ChevronDown, FileText, Languages, Loader2, Mic, Paperclip, Square, X } from "lucide-react";
import { useLayoutEffect, useRef, useState } from "react";

import { MAX_ATTACHMENT_BYTES } from "../lib/constants.js";
import { MAX_RECORDING_SECONDS, useVoiceInput } from "../lib/useVoiceInput.js";

const clock = (s) => `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;

export default function Composer({
  value, onChange, onSubmit, busy, maxLength, error, onError, attachment, onAttach, textareaRef,
  language, languages, onLanguageChange, serverVoice,
}) {
  const fileRef = useRef(null);
  const [reviewHint, setReviewHint] = useState(false);
  const voice = useVoiceInput({
    language,
    serverVoice,
    onText: (text) => {
      onChange(value ? `${value.trimEnd()} ${text}` : text);
      setReviewHint(true);
      textareaRef.current?.focus();
    },
    onError,
  });
  const recording = voice.state === "recording";
  const transcribing = voice.state === "transcribing";
  const current = languages.find((l) => l.code === language);

  // Grow with the content up to a cap, then scroll.
  useLayoutEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, [value, textareaRef]);

  function pickFile(e) {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;
    if (file.size > MAX_ATTACHMENT_BYTES) {
      onError("That file is larger than 10 MB.");
      return;
    }
    onAttach({
      name: file.name,
      preview: file.type.startsWith("image/") ? URL.createObjectURL(file) : null,
    });
  }

  const nearLimit = value.length > maxLength * 0.8;
  const canSend = !busy && value.trim().length > 0;

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        setReviewHint(false);
        onSubmit();
      }}
      noValidate
      aria-label="Ask MedQuery"
    >
      <div className={`rounded-xl bg-surface p-2 transition-shadow focus-within:ring-2 ${error ? "ring-2 ring-danger" : "focus-within:ring-ink"}`}>
        {attachment && (
          <div className="flex px-2 pt-2">
            <div className="flex items-center gap-2.5 rounded-lg bg-raised p-1.5 pr-2">
              {attachment.preview ? (
                <img src={attachment.preview} alt="" className="h-10 w-10 rounded-md object-cover" />
              ) : (
                <span className="flex h-10 w-10 items-center justify-center rounded-md bg-paper">
                  <FileText className="h-5 w-5" />
                </span>
              )}
              <span className="max-w-[10rem] truncate text-sm">{attachment.name}</span>
              <button
                type="button"
                onClick={() => onAttach(null)}
                className="flex h-8 w-8 items-center justify-center rounded-full text-ink-muted transition-colors hover:bg-paper hover:text-ink"
                aria-label="Remove attachment"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
          </div>
        )}

        <label htmlFor="question" className="sr-only">
          Your medical question
        </label>
        <textarea
          id="question"
          ref={textareaRef}
          rows={1}
          value={value}
          maxLength={maxLength}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault();
              setReviewHint(false);
              onSubmit();
            }
          }}
          lang={language}
          placeholder={language === "en-IN" ? "Describe your symptoms or ask a health question"
            : `Ask in ${current?.name ?? "your language"} or English`}
          aria-invalid={error ? "true" : "false"}
          aria-describedby={error ? "questionError" : undefined}
          className="block w-full resize-none bg-transparent px-3 pb-2 pt-3 text-base text-ink placeholder:text-ink-muted focus:outline-none"
        />

        <div className="flex items-center gap-1.5">
          <input ref={fileRef} type="file" accept="image/*,.pdf" className="hidden" onChange={pickFile} data-testid="file-input" />
          <button
            type="button"
            onClick={() => fileRef.current?.click()}
            className="icon-btn bg-paper"
            aria-label="Attach an image or report"
            title="Attach an image or report"
          >
            <Paperclip className="h-5 w-5" />
          </button>
          <button
            type="button"
            onClick={voice.toggle}
            disabled={!voice.supported || transcribing}
            aria-pressed={recording}
            aria-label={recording ? "Stop recording" : `Ask by voice in ${current?.name ?? "English"}`}
            title={voice.supported ? (recording ? "Stop recording" : "Ask by voice")
              : "Voice input isn't supported in this browser"}
            data-testid="mic"
            className={`icon-btn ${recording ? "bg-danger text-raised hover:bg-danger-hover" : "bg-paper"}`}
          >
            {transcribing ? <Loader2 className="h-5 w-5 animate-spin" />
              : recording ? <Square className="h-4 w-4 fill-current" /> : <Mic className="h-5 w-5" />}
          </button>

          <label className="relative inline-flex h-11 items-center rounded-full bg-paper pl-3 pr-8 text-sm text-ink transition-colors hover:bg-line focus-within:ring-2 focus-within:ring-ink">
            <Languages className="mr-1.5 h-4 w-4 shrink-0" aria-hidden="true" />
            <span className="sr-only">Language</span>
            <select
              value={language}
              onChange={(e) => onLanguageChange(e.target.value)}
              disabled={recording || transcribing}
              data-testid="language"
              className="max-w-[6.5rem] appearance-none truncate bg-transparent font-medium focus:outline-none sm:max-w-none"
            >
              {languages.map((l) => (
                <option key={l.code} value={l.code} disabled={!l.available}>
                  {l.native_name}{l.native_name !== l.name ? ` · ${l.name}` : ""}{l.available ? "" : " (needs Sarvam key)"}
                </option>
              ))}
            </select>
            <ChevronDown className="pointer-events-none absolute right-3 h-4 w-4" aria-hidden="true" />
          </label>

          <span className="ml-auto pr-2 text-xs text-ink-muted" aria-live="polite">
            {nearLimit ? `${value.length} / ${maxLength}` : ""}
          </span>
          <button
            type="submit"
            disabled={!canSend}
            aria-label={busy ? "Sending" : "Send question"}
            className="flex h-12 w-12 items-center justify-center rounded-full bg-lime text-ink transition-colors hover:bg-lime-hover disabled:cursor-not-allowed disabled:opacity-40"
          >
            {busy ? <Loader2 className="h-5 w-5 animate-spin" /> : <ArrowUp className="h-5 w-5" />}
          </button>
        </div>
      </div>
      <p className="sr-only" aria-live="polite">
        {recording ? "Recording" : transcribing ? "Transcribing your question" : ""}
      </p>
      {(recording || transcribing || reviewHint) && !error && (
        <p className="mt-2 flex items-center gap-2 px-3 text-sm text-ink-muted" data-testid="voice-status">
          {recording && (
            <>
              <span className="h-2 w-2 animate-pulse rounded-full bg-danger" aria-hidden="true" />
              Recording {clock(voice.seconds)} / {clock(MAX_RECORDING_SECONDS)}. Press stop when you&apos;re done.
            </>
          )}
          {transcribing && "Transcribing your question…"}
          {!recording && !transcribing && reviewHint && "Check the transcript and edit it if needed, then send."}
        </p>
      )}
      {error && (
        <p id="questionError" role="alert" className="mt-2 px-3 text-sm text-danger">
          {error}
        </p>
      )}
    </form>
  );
}
