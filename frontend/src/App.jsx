import { ShieldAlert } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import Composer from "./components/Composer.jsx";
import { EmergencyBanner, Header } from "./components/Header.jsx";
import { BotAnswer, BotError, BotPending, QuickReplies, UserMessage, Welcome } from "./components/Messages.jsx";
import Sidebar from "./components/Sidebar.jsx";
import { api } from "./lib/api.js";
import { describeError, placeholderRange, validateQuestion } from "./lib/format.js";

const BANNER_KEY = "medquery-emergency-dismissed";
const LANGUAGE_KEY = "medquery-language";
// Shown until /api/languages answers; English always works.
const FALLBACK_LANGUAGES = [{ code: "en-IN", name: "English", native_name: "English", available: true }];

function sessionGet(key) {
  try { return sessionStorage.getItem(key); } catch { return null; }
}
function sessionSet(key, value) {
  try { sessionStorage.setItem(key, value); } catch { /* storage unavailable */ }
}
function localGet(key) {
  try { return localStorage.getItem(key); } catch { return null; }
}
function localSet(key, value) {
  try { localStorage.setItem(key, value); } catch { /* storage unavailable */ }
}

let nextId = 0;
const uid = () => `m${++nextId}`;

export default function App() {
  const [availability, setAvailability] = useState("checking");
  const [limits, setLimits] = useState({ min: 3, max: 500 });
  const [history, setHistory] = useState([]);
  const [messages, setMessages] = useState([]);
  const [activeId, setActiveId] = useState(null);
  const [draft, setDraft] = useState("");
  const [attachment, setAttachment] = useState(null);
  const [fieldError, setFieldError] = useState(null);
  const [busy, setBusy] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [bannerVisible, setBannerVisible] = useState(() => sessionGet(BANNER_KEY) !== "1");
  const [languages, setLanguages] = useState(FALLBACK_LANGUAGES);
  const [language, setLanguage] = useState(() => localGet(LANGUAGE_KEY) || "en-IN");
  const [multilingual, setMultilingual] = useState(false);

  const textareaRef = useRef(null);
  const endRef = useRef(null);
  const selectAfterRender = useRef(null);

  const refreshHistory = useCallback(async () => {
    try { setHistory(await api.history()); } catch { /* keep the current list */ }
  }, []);

  useEffect(() => {
    api.health()
      .then((h) => {
        setAvailability("online");
        setLimits({ min: h.limits?.min_question_chars ?? 3, max: h.limits?.max_question_chars ?? 500 });
      })
      .catch(() => setAvailability("offline"));
    api.languages()
      .then((l) => {
        setLanguages(l.languages);
        setMultilingual(l.multilingual_configured);
        // A remembered language that is no longer available falls back to English.
        setLanguage((cur) => (l.languages.some((x) => x.code === cur && x.available) ? cur : l.default));
      })
      .catch(() => setLanguage("en-IN"));
    refreshHistory();
  }, [refreshHistory]);

  useEffect(() => {
    endRef.current?.scrollIntoView?.({ behavior: "smooth", block: "end" });
  }, [messages]);

  // Select the [placeholder] once a quick reply has been written into the textarea.
  useEffect(() => {
    const range = selectAfterRender.current;
    if (!range || !textareaRef.current) return;
    selectAfterRender.current = null;
    textareaRef.current.focus();
    textareaRef.current.setSelectionRange(range[0], range[1]);
  }, [draft]);

  useEffect(() => {
    if (!sidebarOpen) return;
    const onKey = (e) => e.key === "Escape" && setSidebarOpen(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [sidebarOpen]);

  function changeLanguage(code) {
    setLanguage(code);
    localSet(LANGUAGE_KEY, code);
    setFieldError(null);
  }

  async function runQuery(botId, question, hadAttachment, lang) {
    setBusy(true);
    try {
      const result = await api.ask(question, lang);
      setMessages((ms) => ms.map((m) => (m.id === botId ? { ...m, kind: "answer", result, attachmentIgnored: hadAttachment } : m)));
      setActiveId(result.id);
      refreshHistory();
    } catch (err) {
      setMessages((ms) => ms.map((m) => (m.id === botId ? { ...m, kind: "error", error: describeError(err) } : m)));
    } finally {
      setBusy(false);
    }
  }

  function submit() {
    if (busy) return;
    const v = validateQuestion(draft, limits);
    if (!v.ok) {
      setFieldError(attachment && !draft.trim() ? "Add your question in words. Image analysis isn't available yet." : v.message);
      textareaRef.current?.focus();
      return;
    }
    const botId = uid();
    const now = new Date().toISOString();
    setMessages((ms) => [
      ...ms,
      { id: uid(), role: "user", text: v.value, time: now, attachment },
      { id: botId, role: "bot", kind: "pending", question: v.value, hadAttachment: Boolean(attachment), language },
    ]);
    setDraft("");
    setAttachment(null);
    setFieldError(null);
    runQuery(botId, v.value, Boolean(attachment), language);
  }

  function retry(msg) {
    if (busy) return;
    setMessages((ms) => ms.map((m) => (m.id === msg.id ? { ...m, kind: "pending" } : m)));
    runQuery(msg.id, msg.question, msg.hadAttachment, msg.language);
  }

  function pickTemplate(template) {
    selectAfterRender.current = placeholderRange(template);
    setDraft(template);
    setFieldError(null);
  }

  function newChat() {
    setMessages([]);
    setActiveId(null);
    setDraft("");
    setAttachment(null);
    setFieldError(null);
    setSidebarOpen(false);
    textareaRef.current?.focus();
  }

  async function openHistory(id) {
    if (busy) return;
    setSidebarOpen(false);
    try {
      const r = await api.getQuery(id);
      setActiveId(r.id);
      setMessages([
        { id: uid(), role: "user", text: r.localization?.original_question ?? r.question, time: r.created_at },
        { id: uid(), role: "bot", kind: "answer", result: r },
      ]);
    } catch (err) {
      setMessages([{ id: uid(), role: "bot", kind: "error", error: describeError(err), question: "" }]);
    }
  }

  async function clearHistory() {
    try { await api.clearHistory(); } catch { /* ignore */ }
    setActiveId(null);
    refreshHistory();
  }

  const title = messages.find((m) => m.role === "user")?.text ?? "New consultation";
  const last = messages.at(-1);

  return (
    <div className="flex h-dvh overflow-hidden">
      <a
        href="#question"
        className="sr-only z-50 rounded-full bg-raised px-4 py-2 text-sm font-medium text-ink shadow-pop focus:not-sr-only focus:fixed focus:left-3 focus:top-3"
      >
        Skip to question
      </a>

      <Sidebar
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        history={history}
        activeId={activeId}
        onSelect={openHistory}
        onNewChat={newChat}
        onClearHistory={clearHistory}
      />

      <main className="flex min-w-0 flex-1 flex-col">
        <Header title={title} availability={availability} onOpenSidebar={() => setSidebarOpen(true)} />
        {bannerVisible && (
          <EmergencyBanner
            onDismiss={() => {
              setBannerVisible(false);
              sessionSet(BANNER_KEY, "1");
            }}
          />
        )}

        <div className="scrollbar-thin flex-1 overflow-y-auto">
          <div className="mx-auto max-w-3xl px-4 py-4 sm:px-6" role="log" aria-live="polite" aria-busy={busy} aria-label="Conversation">
            {messages.length === 0 ? (
              <Welcome onPick={pickTemplate} />
            ) : (
              <div className="space-y-7">
                {messages.map((m) => {
                  if (m.role === "user") return <UserMessage key={m.id} message={m} />;
                  if (m.kind === "pending") return <BotPending key={m.id} />;
                  if (m.kind === "error") return <BotError key={m.id} message={m.error} onRetry={() => retry(m)} />;
                  return <BotAnswer key={m.id} result={m.result} attachmentIgnored={m.attachmentIgnored} canSpeak={multilingual} />;
                })}
                {last?.kind === "answer" && !busy && <QuickReplies onPick={pickTemplate} className="sm:pl-12" />}
              </div>
            )}
            <div ref={endRef} />
          </div>
        </div>

        <div className="shrink-0 bg-paper pb-[max(env(safe-area-inset-bottom),0.75rem)] pt-2">
          <div className="mx-auto max-w-3xl px-4 sm:px-6">
            <Composer
              value={draft}
              onChange={(v) => {
                setDraft(v);
                if (fieldError) setFieldError(null);
              }}
              onSubmit={submit}
              busy={busy}
              maxLength={limits.max}
              error={fieldError}
              onError={setFieldError}
              attachment={attachment}
              onAttach={setAttachment}
              textareaRef={textareaRef}
              language={language}
              languages={languages}
              onLanguageChange={changeLanguage}
              serverVoice={multilingual}
            />
            <footer className="mt-3 flex items-start justify-center gap-2 px-2 text-center text-xs text-ink-muted">
              <ShieldAlert className="h-4 w-4 shrink-0" aria-hidden="true" />
              <p>
                MedQuery shares general health information and does not replace a licensed physician. Always consult a
                qualified doctor about your health.
              </p>
            </footer>
          </div>
        </div>
      </main>
    </div>
  );
}
