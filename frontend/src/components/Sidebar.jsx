import { History, Plus, Search, Trash2, X } from "lucide-react";
import { useMemo, useState } from "react";

import { dayGroup, relativeTime, statusMeta } from "../lib/format.js";
import { LogoMark } from "./Logo.jsx";

const GROUP_ORDER = ["Today", "Yesterday", "Previous 7 days", "Earlier"];

export default function Sidebar({ open, onClose, history, activeId, onSelect, onNewChat, onClearHistory }) {
  const [filter, setFilter] = useState("");

  const groups = useMemo(() => {
    const needle = filter.trim().toLowerCase();
    const byGroup = new Map();
    for (const item of history) {
      if (needle && !item.question.toLowerCase().includes(needle)) continue;
      const g = dayGroup(item.created_at);
      if (!byGroup.has(g)) byGroup.set(g, []);
      byGroup.get(g).push(item);
    }
    return GROUP_ORDER.filter((g) => byGroup.has(g)).map((g) => [g, byGroup.get(g)]);
  }, [history, filter]);

  return (
    <>
      <div
        className={`fixed inset-0 z-30 bg-ink/25 transition-opacity lg:hidden ${open ? "opacity-100" : "pointer-events-none opacity-0"}`}
        onClick={onClose}
        aria-hidden="true"
      />
      <aside
        id="sidebar"
        aria-label="Consultation history"
        className={`fixed inset-y-0 left-0 z-40 flex w-[260px] flex-col bg-surface transition-transform duration-200 lg:static lg:m-3 lg:mr-0 lg:w-[240px] lg:translate-x-0 lg:rounded-xl ${
          open ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <div className="flex h-[72px] shrink-0 items-center justify-between px-5">
          <a href="/" className="flex items-center gap-2.5 rounded-full">
            <LogoMark />
            <span className="text-h3 font-semibold">MedQuery</span>
          </a>
          <button type="button" onClick={onClose} className="icon-btn bg-paper lg:hidden" aria-label="Close history">
            <X className="h-5 w-5" />
          </button>
        </div>

        <div className="space-y-3 px-4 pb-4">
          <button
            type="button"
            onClick={onNewChat}
            className="flex h-12 w-full items-center justify-center gap-2 rounded-full bg-ink text-[15px] font-medium text-surface transition-colors hover:bg-ink-hover"
          >
            <Plus className="h-5 w-5" />
            New chat
          </button>
          <label className="relative block">
            <span className="sr-only">Search history</span>
            <Search className="pointer-events-none absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-ink-muted" />
            <input
              type="search"
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              placeholder="Search history"
              className="h-11 w-full rounded-full bg-paper pl-10 pr-4 text-sm text-ink placeholder:text-ink-muted focus:outline-none focus:ring-2 focus:ring-ink"
            />
          </label>
        </div>

        <nav className="scrollbar-thin flex-1 overflow-y-auto px-3 pb-4" aria-label="Recent consultations">
          {history.length === 0 ? (
            <div className="mx-1 mt-2 flex flex-col items-center rounded-lg bg-paper px-4 py-8 text-center">
              <History className="mb-2 h-5 w-5 text-ink-muted" strokeWidth={2} />
              <p className="text-sm text-ink-muted" data-testid="history-empty">No consultations yet</p>
            </div>
          ) : groups.length === 0 ? (
            <p className="px-3 py-6 text-sm text-ink-muted">No matches for “{filter}”.</p>
          ) : (
            groups.map(([label, items]) => (
              <div key={label} className="mt-4 first:mt-0">
                <h2 className="eyebrow px-3 pb-2">{label}</h2>
                <ul className="space-y-1">
                  {items.map((h) => {
                    const active = h.id === activeId;
                    return (
                      <li key={h.id}>
                        <button
                          type="button"
                          data-testid="history-item"
                          onClick={() => onSelect(h.id)}
                          aria-current={active ? "true" : undefined}
                          className={`flex min-h-[44px] w-full flex-col justify-center rounded-lg px-3.5 py-2 text-left transition-colors ${
                            active ? "bg-lime" : "hover:bg-paper"
                          }`}
                        >
                          <span className="block w-full truncate text-sm font-medium text-ink">{h.question}</span>
                          <span className={`block text-xs ${active ? "text-ink" : "text-ink-muted"}`}>
                            {relativeTime(h.created_at)} · {statusMeta(h.status).label}
                          </span>
                        </button>
                      </li>
                    );
                  })}
                </ul>
              </div>
            ))
          )}
        </nav>

        {history.length > 0 && (
          <div className="p-3 pt-0">
            <button
              type="button"
              onClick={onClearHistory}
              className="flex h-11 w-full items-center gap-2 rounded-full px-4 text-sm text-ink-muted transition-colors hover:bg-danger-bg hover:text-danger"
            >
              <Trash2 className="h-4 w-4" />
              Clear history
            </button>
          </div>
        )}
      </aside>
    </>
  );
}
