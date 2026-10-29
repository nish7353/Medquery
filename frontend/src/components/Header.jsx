import { Lock, Menu, Phone, ShieldCheck, Siren, X } from "lucide-react";

import { EMERGENCY_NUMBER } from "../lib/constants.js";

const AVAILABILITY = {
  checking: { dot: "bg-ink-muted", text: "Connecting…" },
  online: { dot: "bg-success", text: "Assistant available" },
  offline: { dot: "bg-danger", text: "Assistant offline" },
};

export function Header({ title, availability, onOpenSidebar }) {
  const a = AVAILABILITY[availability];
  // Only claim encryption when the page actually arrived over HTTPS.
  const encrypted = typeof window !== "undefined" && window.location.protocol === "https:";

  return (
    <header className="flex h-[72px] shrink-0 items-center gap-3 px-4 sm:px-6">
      <button type="button" onClick={onOpenSidebar} className="icon-btn lg:hidden" aria-label="Open history" aria-controls="sidebar">
        <Menu className="h-5 w-5" />
      </button>

      <h1 className="min-w-0 flex-1 truncate text-h3 font-semibold">{title}</h1>

      <div className="flex items-center gap-2">
        <span
          className="chip hidden sm:inline-flex"
          title={encrypted ? "This page is served over an encrypted HTTPS connection." : "Your questions are processed on this server only."}
        >
          {encrypted ? <Lock className="h-4 w-4" /> : <ShieldCheck className="h-4 w-4" />}
          {encrypted ? "Encrypted" : "Private session"}
        </span>
        <span role="status" aria-live="polite" data-testid="availability" data-state={availability} className="chip">
          <span className={`h-2 w-2 rounded-full ${a.dot}`} aria-hidden="true" />
          {a.text}
        </span>
      </div>
    </header>
  );
}

export function EmergencyBanner({ onDismiss }) {
  return (
    <div role="region" aria-label="Emergency notice" className="mx-4 mb-2 sm:mx-6">
      <div className="mx-auto flex max-w-3xl items-center gap-3 rounded-lg bg-danger-bg py-2 pl-2 pr-2 sm:pl-3">
        <span className="hidden h-9 w-9 shrink-0 items-center justify-center rounded-full bg-raised text-danger sm:flex">
          <Siren className="h-[18px] w-[18px]" />
        </span>
        <p className="min-w-0 flex-1 pl-2 text-sm text-danger sm:pl-0">
          <span className="font-semibold">Medical emergency?</span> Chest pain, trouble breathing or heavy bleeding need help now.
        </p>
        <a
          href={`tel:${EMERGENCY_NUMBER}`}
          className="inline-flex h-10 shrink-0 items-center gap-1.5 rounded-full bg-danger px-4 text-sm font-medium text-raised transition-colors hover:bg-danger-hover"
        >
          <Phone className="h-4 w-4" />
          Call {EMERGENCY_NUMBER}
        </a>
        <button
          type="button"
          onClick={onDismiss}
          className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full text-danger transition-colors hover:bg-raised"
          aria-label="Dismiss emergency notice"
        >
          <X className="h-4 w-4" />
        </button>
      </div>
    </div>
  );
}
