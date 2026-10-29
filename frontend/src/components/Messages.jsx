import { AlertTriangle, ArrowUpRight, ChevronDown, FileText, ImageOff, Info, Languages, Loader2, RotateCcw, Square, Volume2 } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { api } from "../lib/api.js";
import { QUICK_REPLIES } from "../lib/constants.js";
import { clockTime, describeError, safeUrl, splitHighlight, statusMeta } from "../lib/format.js";
import { LogoMark } from "./Logo.jsx";

const TONE_CHIP = {
  success: "bg-success-bg text-success",
  warning: "bg-warning-bg text-warning",
  info: "bg-info-bg text-info",
  neutral: "bg-line text-ink-muted",
  danger: "bg-danger-bg text-danger",
};

function hostname(url) {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return "";
  }
}

export function UserMessage({ message }) {
  return (
    <div className="flex animate-fade-up justify-end" data-testid="user-message">
      <div className="flex max-w-[85%] flex-col items-end gap-1.5 sm:max-w-[70%]">
        {message.attachment && (
          <div className="flex items-center gap-2.5 rounded-lg bg-surface p-1.5 pr-4">
            {message.attachment.preview ? (
              <img src={message.attachment.preview} alt="" className="h-11 w-11 rounded-md object-cover" />
            ) : (
              <span className="flex h-11 w-11 items-center justify-center rounded-md bg-paper">
                <FileText className="h-5 w-5" />
              </span>
            )}
            <span className="max-w-[12rem] truncate text-sm">{message.attachment.name}</span>
          </div>
        )}
        <p className="whitespace-pre-wrap rounded-lg rounded-br-md bg-sage px-5 py-3 text-base text-ink">{message.text}</p>
        <time className="px-2 text-xs text-ink-muted">{clockTime(message.time)}</time>
      </div>
    </div>
  );
}

function BotShell({ children, time, label }) {
  return (
    <div className="flex animate-fade-up gap-3" data-testid="bot-message">
      <LogoMark className="mt-1 hidden h-9 w-9 shrink-0 sm:block" />
      <div className="min-w-0 flex-1">
        <div className="mb-2 flex flex-wrap items-center gap-2 pl-1">
          <span className="text-sm font-semibold">MedQuery</span>
          {label}
          {time && <time className="ml-auto text-xs text-ink-muted">{clockTime(time)}</time>}
        </div>
        {children}
      </div>
    </div>
  );
}

/** Dark ink panel: reserved for explaining what the assistant did and why. */
function AssistantNote({ icon: Icon, title, children, testId }) {
  return (
    <div className="flex gap-3 rounded-lg bg-ink px-4 py-3.5 text-surface" data-testid={testId}>
      <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-lime text-ink">
        <Icon className="h-4 w-4" />
      </span>
      <p className="text-sm">
        <span className="block font-semibold">{title}</span>
        <span className="text-surface/85">{children}</span>
      </p>
    </div>
  );
}

export function BotPending() {
  return (
    <BotShell>
      <div className="inline-flex items-center gap-3 rounded-xl rounded-tl-md bg-surface px-5 py-4">
        <span className="flex gap-1.5" aria-hidden="true">
          {[0, 150, 300].map((d) => (
            <span key={d} className="h-2 w-2 animate-blink rounded-full bg-ink" style={{ animationDelay: `${d}ms` }} />
          ))}
        </span>
        <span className="text-sm text-ink-muted">Checking medical sources</span>
      </div>
    </BotShell>
  );
}

export function BotError({ message, onRetry }) {
  return (
    <BotShell>
      <div role="alert" className="rounded-xl rounded-tl-md bg-danger-bg px-5 py-4">
        <p className="flex items-start gap-2 text-sm text-danger">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
          <span data-testid="error-text">{message}</span>
        </p>
        <button
          type="button"
          onClick={onRetry}
          className="mt-3 inline-flex h-11 items-center gap-2 rounded-full bg-ink px-5 text-sm font-medium text-surface transition-colors hover:bg-ink-hover"
        >
          <RotateCcw className="h-4 w-4" />
          Try again
        </button>
      </div>
    </BotShell>
  );
}

/** Reads `text` aloud via the backend's text-to-speech. Audio is fetched once, then replayed. */
function ListenButton({ text, language }) {
  const [state, setState] = useState("idle"); // idle | loading | playing
  const [error, setError] = useState(null);
  const audio = useRef(null);
  const url = useRef(null);

  useEffect(() => () => {
    audio.current?.pause();
    if (url.current) URL.revokeObjectURL(url.current);
  }, []);

  async function toggle() {
    if (state === "loading") return;
    if (state === "playing") {
      audio.current.pause();
      audio.current.currentTime = 0;
      setState("idle");
      return;
    }
    setError(null);
    try {
      if (!url.current) {
        setState("loading");
        url.current = URL.createObjectURL(await api.speech(text, language));
      }
      audio.current = audio.current ?? new Audio();
      audio.current.src = url.current;
      audio.current.onended = () => setState("idle");
      await audio.current.play();
      setState("playing");
    } catch (err) {
      setState("idle");
      setError(err instanceof Error && err.name === "NotAllowedError" ? "Your browser blocked audio playback." : describeError(err));
    }
  }

  const Icon = state === "loading" ? Loader2 : state === "playing" ? Square : Volume2;
  return (
    <div className="mt-3 flex flex-wrap items-center gap-2">
      <button
        type="button"
        onClick={toggle}
        aria-pressed={state === "playing"}
        data-testid="listen"
        className="inline-flex h-10 items-center gap-2 rounded-full bg-paper px-4 text-sm font-medium text-ink transition-colors hover:bg-line"
      >
        <Icon className={`h-4 w-4 ${state === "loading" ? "animate-spin" : ""} ${state === "playing" ? "fill-current" : ""}`} />
        {state === "loading" ? "Preparing audio…" : state === "playing" ? "Stop" : "Listen to answer"}
      </button>
      {error && <span role="alert" className="text-sm text-danger">{error}</span>}
    </div>
  );
}

function Passage({ ev, n }) {
  const parts = splitHighlight(ev.text, ev.highlight_start, ev.highlight_end);
  return (
    <li className="rounded-lg bg-raised px-4 py-3.5">
      <div className="mb-1.5 flex items-center gap-2 text-sm">
        <span className="flex h-6 min-w-6 items-center justify-center rounded-full bg-lime-soft px-1.5 text-xs font-semibold">{n}</span>
        <span className="truncate font-medium">{ev.source_title || ev.source_id}</span>
        {ev.used_for_answer && <span className="shrink-0 rounded-full bg-butter px-2 py-0.5 text-xs">Used for answer</span>}
      </div>
      <p className="text-sm text-ink-muted" data-testid="passage">
        {parts ? (
          <>
            {parts[0]}
            <mark>{parts[1]}</mark>
            {parts[2]}
          </>
        ) : (
          ev.text
        )}
      </p>
    </li>
  );
}

export function BotAnswer({ result, attachmentIgnored, canSpeak = false }) {
  const [showEvidence, setShowEvidence] = useState(false);
  const meta = statusMeta(result.status);
  // Non-English questions carry a translated copy; the English answer and evidence stay authoritative.
  const loc = result.localization;
  const lang = loc?.language ?? "en-IN";
  const reply = loc?.status_message ?? meta.reply;
  const sources = result.sources ?? [];
  const evidence = result.evidence ?? [];
  const answer = result.answer;

  // Citation number for the passage the answer was extracted from.
  const answerSource = answer ? evidence[answer.evidence_index]?.source_id : null;
  const citeIndex = answerSource ? sources.findIndex((s) => s.source_id === answerSource) : -1;

  return (
    <BotShell
      time={result.created_at}
      label={
        <span className={`rounded-full px-2.5 py-0.5 text-xs font-medium ${TONE_CHIP[meta.tone]}`} data-testid="status-chip">
          {meta.label}
        </span>
      }
    >
      <article className="rounded-xl rounded-tl-md bg-surface p-5 sm:p-6">
        {loc?.question_translated && (
          <p className="mb-3 flex items-start gap-2 text-sm text-ink-muted" data-testid="searched-as">
            <Languages className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
            <span>
              Searched in English for <span lang="en" className="text-ink">“{loc.english_question}”</span>
            </span>
          </p>
        )}
        {answer ? (
          <p className="text-base text-ink" data-testid="answer-text" lang={lang}>
            {loc?.answer_text ?? answer.text}
            {citeIndex >= 0 && (
              <sup className="ml-1 rounded-full bg-lime-soft px-1.5 py-px text-[11px] font-semibold">{citeIndex + 1}</sup>
            )}
          </p>
        ) : (
          <p className="text-base text-ink" lang={lang} data-testid="status-reply">{reply}</p>
        )}
        {loc && answer && (
          <p className="mt-2 text-sm text-ink-muted" data-testid="english-answer">
            English original: <span lang="en" className="text-ink">{answer.text}</span>
          </p>
        )}
        {canSpeak && <ListenButton text={answer ? loc?.answer_text ?? answer.text : reply} language={lang} />}

        {(answer && meta.reply) || attachmentIgnored || result.notices?.length || loc?.warnings?.length ? (
          <div className="mt-4 space-y-2">
            {loc?.warnings?.includes("numbers_changed") && (
              <AssistantNote icon={AlertTriangle} title="Check the numbers" testId="translation-warning">
                Some numbers differ between the translation and the English answer. Rely on the English text.
              </AssistantNote>
            )}
            {loc?.warnings?.includes("partial_translation") && (
              <AssistantNote icon={Languages} title="Partly in English" testId="translation-warning">
                Some of this reply couldn&apos;t be translated, so it is shown in English.
              </AssistantNote>
            )}
            {answer && meta.reply && (
              <AssistantNote icon={Info} title={meta.label}>
                <span lang={lang}>{reply}</span>
              </AssistantNote>
            )}
            {attachmentIgnored && (
              <AssistantNote icon={ImageOff} title="Attachment not analysed">
                Image and report analysis isn&apos;t available yet, so I answered from your text only.
              </AssistantNote>
            )}
            {(result.notices ?? []).map((n, i) => (
              <AssistantNote key={i} icon={AlertTriangle} title="Note" testId="notice">
                <span lang={loc?.notices?.[i] ? lang : "en"}>{loc?.notices?.[i] ?? n.message}</span>
              </AssistantNote>
            ))}
          </div>
        ) : null}

        {sources.length > 0 && (
          <div className="mt-5 border-t border-line pt-4">
            <h3 className="eyebrow mb-2.5">{loc ? "Sources (in English)" : "Sources"}</h3>
            <ol className="flex flex-wrap gap-2" data-testid="sources">
              {sources.map((s, i) => {
                const url = safeUrl(s.url);
                const body = (
                  <>
                    <span className="flex h-6 min-w-6 items-center justify-center rounded-full bg-lime-soft px-1.5 text-xs font-semibold">
                      {i + 1}
                    </span>
                    <span className="max-w-[14rem] truncate">{s.title || s.source_id}</span>
                    {url && <span className="hidden text-ink-muted sm:inline">{hostname(url)}</span>}
                    {url && <ArrowUpRight className="h-4 w-4 shrink-0 text-ink-muted" />}
                  </>
                );
                const cls = "inline-flex min-h-[40px] items-center gap-2 rounded-full bg-raised py-1.5 pl-1.5 pr-3.5 text-sm";
                return (
                  <li key={s.source_id + i}>
                    {url ? (
                      <a href={url} target="_blank" rel="noopener noreferrer" className={`${cls} transition-colors hover:bg-lime-soft`}>
                        {body}
                      </a>
                    ) : (
                      <span className={cls}>{body}</span>
                    )}
                  </li>
                );
              })}
            </ol>
          </div>
        )}

        {evidence.length > 0 && (
          <div className="mt-4">
            <button
              type="button"
              onClick={() => setShowEvidence((v) => !v)}
              aria-expanded={showEvidence}
              className="inline-flex h-10 items-center gap-1.5 rounded-full px-3 text-sm font-medium text-ink-muted transition-colors hover:bg-paper hover:text-ink"
            >
              {showEvidence ? "Hide" : "Show"} supporting passages ({evidence.length})
              <ChevronDown className={`h-4 w-4 transition-transform ${showEvidence ? "rotate-180" : ""}`} />
            </button>
            {showEvidence && (
              <ol className="mt-2 space-y-2">
                {evidence.map((ev, i) => {
                  const n = sources.findIndex((s) => s.source_id === ev.source_id);
                  return <Passage key={i} ev={ev} n={n >= 0 ? n + 1 : i + 1} />;
                })}
              </ol>
            )}
          </div>
        )}
      </article>
    </BotShell>
  );
}

export function QuickReplies({ onPick, className = "" }) {
  return (
    <div className={`flex flex-wrap gap-2 ${className}`} role="group" aria-label="Suggested questions">
      {QUICK_REPLIES.map((q) => (
        <button
          key={q.id}
          type="button"
          onClick={() => onPick(q.template)}
          data-testid="quick-reply"
          className="inline-flex h-11 items-center rounded-full bg-surface px-5 text-sm font-medium text-ink transition-colors hover:bg-raised"
        >
          {q.label}
        </button>
      ))}
    </div>
  );
}

export function Welcome({ onPick }) {
  return (
    <div className="flex animate-fade-up flex-col items-center pt-[10vh] text-center">
      <LogoMark className="h-14 w-14" />
      <h2 className="mt-6 text-[32px] font-semibold leading-[38px] sm:text-display">How can I help today?</h2>
      <p className="mt-3 max-w-sm text-base text-ink-muted">Every answer links to the sources it came from.</p>
      <QuickReplies onPick={onPick} className="mt-8 max-w-xl justify-center" />
    </div>
  );
}
