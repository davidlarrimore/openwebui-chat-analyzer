"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import { cn } from "@/lib/utils";
import { apiGet } from "@/lib/api";
import type { SummaryEvent, SummaryEventsResponse, SummaryStatus } from "@/lib/types";

type SummarizerSnapshot = {
  state: string | null;
  total: number;
  completed: number;
  message?: string | null;
};

type SummarizerListener = (payload: { status: SummarizerSnapshot | null; events: SummaryEvent[] }) => void;

interface SummarizerProgressContextValue {
  status: SummarizerSnapshot | null;
  updateStatus: (status: SummarizerSnapshot | null) => void;
  clearStatus: () => void;
  subscribe: (listener: SummarizerListener) => () => void;
}

const SummarizerProgressContext = createContext<SummarizerProgressContextValue | undefined>(undefined);

const PANEL_HIDE_DELAY_MS = 4000;
const POLL_INTERVAL_ACTIVE_MS = 2000;
const POLL_INTERVAL_IDLE_MS = 10000;

function toSnapshot(status: SummaryStatus | null, fallback: SummarizerSnapshot | null, message?: string | null) {
  if (!status) {
    return fallback ?? null;
  }
  const total = Math.max(0, Math.trunc(status.total ?? 0));
  const completed = Math.max(0, Math.trunc(status.completed ?? 0));
  const rawMessage =
    typeof message === "string" && message.trim().length
      ? message.trim()
      : typeof status.message === "string"
        ? status.message.trim()
        : fallback?.message ?? "";
  return {
    state: status.state ?? null,
    total,
    completed,
    message: rawMessage || null
  };
}

export function SummarizerProgressProvider({ children }: { children: React.ReactNode }) {
  const [status, setStatus] = useState<SummarizerSnapshot | null>(null);
  const [visible, setVisible] = useState(false);
  const hideTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pollTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lastEventIdRef = useRef<string | null>(null);
  const listenersRef = useRef<Set<SummarizerListener>>(new Set());
  const abortRef = useRef(false);
  const statusRef = useRef<SummarizerSnapshot | null>(null);

  const applyStatus = useCallback((next: SummarizerSnapshot | null) => {
    setStatus((previous) => {
      let resolved = next;
      if (previous === null && next === null) {
        resolved = previous;
      } else if (previous && next) {
        if (
          previous.state === next.state &&
          previous.total === next.total &&
          previous.completed === next.completed &&
          (previous.message || "") === (next.message || "")
        ) {
          resolved = previous;
        }
      }
      statusRef.current = resolved ?? null;
      return resolved;
    });
  }, []);

  const updateStatus = useCallback(
    (next: SummarizerSnapshot | null) => {
      applyStatus(next);
    },
    [applyStatus]
  );

  const clearStatus = useCallback(() => {
    if (hideTimerRef.current) {
      clearTimeout(hideTimerRef.current);
      hideTimerRef.current = null;
    }
    statusRef.current = null;
    applyStatus(null);
    setVisible(false);
  }, [applyStatus]);

  const subscribe = useCallback((listener: SummarizerListener) => {
    listenersRef.current.add(listener);
    return () => {
      listenersRef.current.delete(listener);
    };
  }, []);

  useEffect(() => {
    return () => {
      if (hideTimerRef.current) {
        clearTimeout(hideTimerRef.current);
        hideTimerRef.current = null;
      }
      if (pollTimeoutRef.current) {
        clearTimeout(pollTimeoutRef.current);
        pollTimeoutRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    if (hideTimerRef.current) {
      clearTimeout(hideTimerRef.current);
      hideTimerRef.current = null;
    }

    if (status?.state === "running") {
      setVisible(true);
      return;
    }

    if (status) {
      hideTimerRef.current = setTimeout(() => {
        setVisible(false);
        hideTimerRef.current = null;
      }, PANEL_HIDE_DELAY_MS);
    } else {
      setVisible(false);
    }
  }, [status]);

  useEffect(() => {
    abortRef.current = false;

    const schedulePoll = (delay: number) => {
      if (pollTimeoutRef.current) {
        clearTimeout(pollTimeoutRef.current);
      }
      pollTimeoutRef.current = setTimeout(runPoll, delay);
    };

    const runPoll = async () => {
      if (abortRef.current) {
        return;
      }

      let nextDelay = POLL_INTERVAL_IDLE_MS;
      let events: SummaryEvent[] = [];
      let snapshot: SummarizerSnapshot | null = statusRef.current;

      try {
        const [statusPayload, eventsPayload] = await Promise.all([
          apiGet<SummaryStatus>("api/v1/summaries/status"),
          (async () => {
            const path = lastEventIdRef.current
              ? `api/v1/summaries/events?after=${encodeURIComponent(lastEventIdRef.current)}`
              : "api/v1/summaries/events";
            return apiGet<SummaryEventsResponse>(path);
          })()
        ]);

        const queue = eventsPayload ?? { events: [] };
        const fetchedEvents = Array.isArray(queue.events) ? queue.events : [];

        if (queue.reset) {
          lastEventIdRef.current = null;
        }

        if (fetchedEvents.length) {
          lastEventIdRef.current = fetchedEvents[fetchedEvents.length - 1]?.event_id ?? lastEventIdRef.current;
          events = fetchedEvents;
        }

        const latestMessageFromEvents =
          events.length && typeof events[events.length - 1]?.message === "string"
            ? (events[events.length - 1].message as string)
            : undefined;

        snapshot = toSnapshot(statusPayload, snapshot, latestMessageFromEvents ?? snapshot?.message ?? null);
        applyStatus(snapshot);
        nextDelay = snapshot?.state === "running" ? POLL_INTERVAL_ACTIVE_MS : POLL_INTERVAL_IDLE_MS;
      } catch (error) {
        console.error("Failed to poll summarizer status", error);
        nextDelay = POLL_INTERVAL_IDLE_MS;
      }

      if (events.length) {
        listenersRef.current.forEach((listener) => {
          try {
            listener({ status: snapshot, events });
          } catch {
            // ignore listener errors
          }
        });
      } else {
        listenersRef.current.forEach((listener) => {
          try {
            listener({ status: snapshot, events: [] });
          } catch {
            // ignore listener errors
          }
        });
      }

      if (!abortRef.current) {
        schedulePoll(nextDelay);
      }
    };

    schedulePoll(0);

    return () => {
      abortRef.current = true;
      if (pollTimeoutRef.current) {
        clearTimeout(pollTimeoutRef.current);
        pollTimeoutRef.current = null;
      }
    };
  }, [applyStatus]);

  const contextValue = useMemo(
    () => ({
      status,
      updateStatus,
      clearStatus,
      subscribe
    }),
    [status, updateStatus, clearStatus, subscribe]
  );

  return (
    <SummarizerProgressContext.Provider value={contextValue}>
      {children}
      <SummarizerProgressPanel status={visible ? status : null} />
    </SummarizerProgressContext.Provider>
  );
}

export function useSummarizerProgress(): SummarizerProgressContextValue {
  const context = useContext(SummarizerProgressContext);
  if (!context) {
    throw new Error("useSummarizerProgress must be used within a SummarizerProgressProvider");
  }
  return context;
}

function SummarizerProgressPanel({ status }: { status: SummarizerSnapshot | null }) {
  if (!status) {
    return null;
  }

  const total = Math.max(0, Math.trunc(status.total ?? 0));
  const completed = Math.max(0, Math.trunc(status.completed ?? 0));
  const clampedCompleted = total > 0 ? Math.min(completed, total) : completed;
  const percent = total > 0 ? Math.min(100, Math.floor((clampedCompleted / total) * 100)) : 0;
  const hasTotal = total > 0;
  const progressSummary = hasTotal
    ? `${clampedCompleted.toLocaleString()} / ${total.toLocaleString()} (${percent}%)`
    : null;
  const message = (status.message || "").trim();

  return (
    <div
      className="fixed bottom-4 right-4 z-50 w-80 max-w-[calc(100vw-2rem)] overflow-hidden rounded-lg border border-border/70 bg-card/95 shadow-lg backdrop-blur supports-[backdrop-filter]:bg-card/80"
      role="status"
      aria-live="polite"
    >
      <div className="flex items-center justify-between gap-2 border-b border-border/60 px-4 py-2">
        <span className="text-sm font-semibold text-foreground">Summarizer</span>
        {progressSummary ? <span className="text-xs text-muted-foreground">{progressSummary}</span> : null}
      </div>
      <div className="space-y-3 px-4 py-3">
        {message ? <p className="text-sm text-foreground">{message}</p> : null}
        <div className="h-2 w-full rounded-full bg-muted">
          <div
            className={cn("h-full rounded-full bg-primary transition-all", hasTotal ? "" : "animate-pulse")}
            style={{ width: hasTotal ? `${percent}%` : "35%" }}
          />
        </div>
      </div>
    </div>
  );
}
