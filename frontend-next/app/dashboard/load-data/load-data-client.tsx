"use client";

import { ChangeEvent, FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { apiGet, apiPost } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { useToast } from "@/components/ui/use-toast";
import { useSummarizerProgress } from "@/components/summarizer-progress-provider";
import type { DirectConnectDefaultResult } from "@/lib/direct-connect-defaults";
import type {
  AppMetadata,
  DatasetMeta,
  DirectConnectSettings,
  SummaryEvent,
  SummaryEventsResponse,
  SummaryStatus,
  UploadResult,
  UploadStats
} from "@/lib/types";

interface LoadDataClientProps {
  initialMeta: DatasetMeta | null;
  initialError: string | null;
  defaults: DirectConnectDefaultResult;
}

interface LogEntry {
  id: string;
  icon: string;
  message: string;
  timestamp: Date;
}

type ProgressState = { completed: number; total: number } | null;

interface DatasetSourceInfo {
  label: string;
  detail: string;
}

interface DatasetSummary {
  ready: boolean;
  sourceLabel: string;
  sourceDetail: string;
  lastLoadedDisplay: string;
  lastLoadedRelative: string;
  chatUploaded: string;
  userUploaded: string;
  modelUploaded: string;
  chatCountDisplay: string;
  userCountDisplay: string;
  modelCountDisplay: string;
  messageCountDisplay: string;
  dateRange: string;
}

const SUMMARY_POLL_INTERVAL_MS = 2000;
const TERMINAL_SUMMARY_STATES = new Set(["idle", "completed", "failed", "cancelled"]);
const SUMMARY_WARNING_EVENT_TYPES = new Set(["invalid_chat", "empty_context"]);
type LogSource = "direct" | "admin";
const PROCESSING_LOG_META: Record<
  LogSource,
  { label: string; description: string; empty: string }
> = {
  direct: {
    label: "Direct Connect",
    description: "Direct Connect sync activity and summarizer updates appear here.",
    empty: "Sync results and summarizer updates will appear here."
  },
  admin: {
    label: "Admin Tools",
    description: "Delete and summary rebuild operations appear here.",
    empty: "Run an admin action to view progress logs here."
  }
};
const DEFAULT_PROCESSING_LOG_META = {
  label: "Processing",
  description: "Processing log collects updates from Direct Connect and Admin actions.",
  empty: "Trigger a sync or admin action to see messages here."
};

interface SummaryProgressTracker {
  lastCompleted: number;
  lastTotal: number;
  lastPercent: number;
  lastSummary: string | null;
}

function createProgressTracker(): SummaryProgressTracker {
  return { lastCompleted: -1, lastTotal: -1, lastPercent: -1, lastSummary: null };
}

function safeUuid(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return Math.random().toString(36).slice(2);
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

function toNumber(value: unknown, fallback = 0): number {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string") {
    const trimmed = value.trim();
    if (!trimmed) {
      return fallback;
    }
    const numeric = Number(trimmed);
    return Number.isNaN(numeric) ? fallback : numeric;
  }
  return fallback;
}

function parseDate(value: unknown): Date | null {
  if (value instanceof Date) {
    return Number.isNaN(value.getTime()) ? null : value;
  }
  if (typeof value === "number") {
    if (!Number.isFinite(value)) {
      return null;
    }
    if (value < 1e12) {
      return new Date(value * 1000);
    }
    return new Date(value);
  }
  if (typeof value === "string") {
    const trimmed = value.trim();
    if (!trimmed) {
      return null;
    }
    const numeric = Number(trimmed);
    if (!Number.isNaN(numeric)) {
      return parseDate(numeric);
    }
    const parsed = new Date(trimmed);
    return Number.isNaN(parsed.getTime()) ? null : parsed;
  }
  return null;
}

function formatDateTime(value: unknown, useLocalTime: boolean): string {
  const date = parseDate(value);
  if (!date) {
    return "N/A";
  }
  const options: Intl.DateTimeFormatOptions = {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit"
  };
  if (!useLocalTime) {
    options.timeZone = "UTC";
    options.timeZoneName = "short";
  }
  return new Intl.DateTimeFormat(undefined, options).format(date);
}

function formatDay(value: unknown, useLocalTime: boolean): string {
  const date = parseDate(value);
  if (!date) {
    return "N/A";
  }
  const options: Intl.DateTimeFormatOptions = {
    year: "numeric",
    month: "2-digit",
    day: "2-digit"
  };
  if (!useLocalTime) {
    options.timeZone = "UTC";
  }
  return new Intl.DateTimeFormat(undefined, options).format(date);
}

function formatRelativeTime(value: unknown): string {
  const date = parseDate(value);
  if (!date) {
    return "";
  }

  const now = new Date();
  const seconds = Math.max((now.getTime() - date.getTime()) / 1000, 0);

  if (seconds < 60) {
    return "1 minute ago";
  }

  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) {
    const safeMinutes = Math.max(1, minutes);
    const unit = safeMinutes === 1 ? "minute" : "minutes";
    return `${safeMinutes} ${unit} ago`;
  }

  const hours = Math.floor(seconds / 3600);
  if (hours < 24) {
    const safeHours = Math.max(1, hours);
    const unit = safeHours === 1 ? "hour" : "hours";
    return `${safeHours} ${unit} ago`;
  }

  const days = Math.max(1, Math.floor(seconds / 86400));
  const unit = days === 1 ? "day" : "days";
  return `${days} ${unit} ago`;
}

function getMetadata(meta: DatasetMeta | null): AppMetadata {
  return ((meta?.app_metadata ?? {}) as AppMetadata) || {};
}

function isDatasetReady(meta: DatasetMeta | null): boolean {
  if (!meta) {
    return false;
  }
  if ((meta.chat_count ?? 0) > 0) {
    return true;
  }
  const metadata = getMetadata(meta);
  const counts = [
    metadata.chat_count,
    metadata.message_count,
    metadata.user_count,
    metadata.model_count
  ].map((value) => toNumber(value, 0));
  return counts.some((count) => count > 0);
}

function determineDatasetSource(meta: DatasetMeta | null): DatasetSourceInfo {
  if (!meta) {
    return { label: "Not Loaded", detail: "" };
  }

  const metadata = getMetadata(meta);
  const rawSource = typeof meta.source === "string" ? meta.source : "";
  const datasetSourceLabelRaw =
    (typeof metadata.dataset_source === "string" && metadata.dataset_source) || rawSource || "Unknown source";

  const normalizedLabel = datasetSourceLabelRaw.trim();
  const normalizedLower = normalizedLabel.toLowerCase();
  const normalizedRawSource = rawSource.trim();
  const isUrlSource = normalizedLabel.startsWith("http://") || normalizedLabel.startsWith("https://");
  const isOpenwebuiSource = normalizedRawSource.startsWith("openwebui:");
  const isFileUploadSource =
    normalizedLower === "local upload" ||
    normalizedRawSource.startsWith("upload:") ||
    normalizedRawSource.startsWith("default:") ||
    normalizedRawSource.startsWith("json:") ||
    normalizedRawSource.startsWith("chat export");

  if (isUrlSource || isOpenwebuiSource) {
    const detail = isUrlSource
      ? normalizedLabel
      : normalizedRawSource.includes(":")
      ? normalizedRawSource.split(":", 2)[1]?.trim() || normalizedLabel
      : normalizedLabel;
    return { label: "Direct Connect", detail };
  }

  if (isFileUploadSource && meta.chat_count > 0) {
    const detail = normalizedRawSource.includes(":")
      ? normalizedRawSource.split(":", 2)[1]?.trim() || ""
      : normalizedLower === "local upload"
      ? ""
      : normalizedLabel;
    return { label: "File Upload", detail };
  }

  if (meta.chat_count > 0 && normalizedLabel && normalizedLower !== "unknown source") {
    return { label: normalizedLabel, detail: "" };
  }

  return { label: "Not Loaded", detail: "" };
}

function formatNumberDisplay(value: unknown): string {
  const numeric = toNumber(value, Number.NaN);
  if (Number.isNaN(numeric)) {
    if (value === null || value === undefined || value === "") {
      return "0";
    }
    return String(value);
  }
  return Math.trunc(numeric).toLocaleString();
}

function buildDateRange(metadata: AppMetadata, useLocalTime: boolean): string {
  const firstDay = formatDay(metadata.first_chat_day, useLocalTime);
  const lastDay = formatDay(metadata.last_chat_day, useLocalTime);

  if (firstDay === "N/A" && lastDay === "N/A") {
    return "N/A";
  }
  if (firstDay !== "N/A" && lastDay !== "N/A") {
    return `${firstDay} – ${lastDay}`;
  }
  return firstDay !== "N/A" ? firstDay : lastDay;
}

function computeDatasetSummary(meta: DatasetMeta | null, useLocalTime: boolean): DatasetSummary {
  const metadata = getMetadata(meta);
  const ready = isDatasetReady(meta);
  const sourceInfo = determineDatasetSource(meta);
  const lastLoadedRaw =
    metadata.dataset_pulled_at ??
    metadata.chat_uploaded_at ??
    metadata.users_uploaded_at ??
    metadata.models_uploaded_at;

  return {
    ready,
    sourceLabel: sourceInfo.label,
    sourceDetail: sourceInfo.detail,
    lastLoadedDisplay: formatDateTime(lastLoadedRaw, useLocalTime),
    lastLoadedRelative: formatRelativeTime(lastLoadedRaw),
    chatUploaded: formatDateTime(metadata.chat_uploaded_at, useLocalTime),
    userUploaded: formatDateTime(metadata.users_uploaded_at, useLocalTime),
    modelUploaded: formatDateTime(metadata.models_uploaded_at, useLocalTime),
    chatCountDisplay: formatNumberDisplay(metadata.chat_count ?? meta?.chat_count ?? 0),
    userCountDisplay: formatNumberDisplay(metadata.user_count ?? meta?.user_count ?? 0),
    modelCountDisplay: formatNumberDisplay(metadata.model_count ?? meta?.model_count ?? 0),
    messageCountDisplay: formatNumberDisplay(metadata.message_count ?? meta?.message_count ?? 0),
    dateRange: buildDateRange(metadata, useLocalTime)
  };
}

function formatCount(value: number, noun: string): string {
  const safeValue = Math.max(0, Math.trunc(value));
  const label = safeValue === 1 ? noun : `${noun}s`;
  return `${safeValue.toLocaleString()} ${label}`;
}

function formatBucket(value: number, noun: string): string {
  const safeValue = Math.max(0, Math.trunc(value));
  const plural = safeValue === 1 ? noun : `${noun}s`;
  return `${safeValue.toLocaleString()} ${plural}`;
}

function formatList(values: string[]): string {
  if (!values.length) {
    return "";
  }
  if (values.length === 1) {
    return values[0];
  }
  if (values.length === 2) {
    return `${values[0]} and ${values[1]}`;
  }
  return `${values.slice(0, -1).join(", ")}, and ${values[values.length - 1]}`;
}

function isTerminalSummaryState(state: string | null | undefined): boolean {
  if (state === null || state === undefined) {
    return true;
  }
  return TERMINAL_SUMMARY_STATES.has(state);
}

function formatSummaryEvent(event: Record<string, unknown>): string {
  const type = typeof event.type === "string" ? event.type : "";
  if (type === "chat") {
    const chatId = typeof event.chat_id === "string" ? event.chat_id : "unknown";
    const outcome = typeof event.outcome === "string" ? event.outcome : "";
    if (outcome === "failed") {
      return `Failed to summarize chat ${chatId}`;
    }
    return `Unexpected outcome while summarizing chat ${chatId}`;
  }
  if (type === "chunk") {
    const chatId = typeof event.chat_id === "string" ? event.chat_id : "unknown";
    const chunkIndex = Number(event.chunk_index) || 0;
    const chunkCount = Number(event.chunk_count) || 0;
    if (chunkIndex > 0 && chunkCount > 0) {
      return `Summarizing chunk ${chunkIndex}/${chunkCount} for chat ${chatId}`;
    }
    return `Summarizing chat ${chatId}`;
  }
  const message = typeof event.message === "string" ? event.message.trim() : "";
  return message || "Processing error encountered.";
}

function determineSummaryEventIcon(event: SummaryEvent): string {
  const type = typeof event.type === "string" ? event.type : "";
  const outcome = typeof event.outcome === "string" ? event.outcome : "";
  if (type === "chunk") {
    return "🧩";
  }
  if (type === "chat") {
    if (outcome === "failed") {
      return "⚠️";
    }
    if (outcome === "generated") {
      return "✅";
    }
    if (outcome === "skipped") {
      return "ℹ️";
    }
  }
  if (type === "error") {
    return "❌";
  }
  if (type === "start") {
    return "🚀";
  }
  if (type === "complete") {
    return "✅";
  }
  if (type === "cancelled") {
    return "⚠️";
  }
  if (SUMMARY_WARNING_EVENT_TYPES.has(type)) {
    return "⚠️";
  }
  return "ℹ️";
}

function inferLogSourceFromReason(reason: string | null | undefined): LogSource {
  if (!reason) {
    return "direct";
  }
  const normalized = reason.trim().toLowerCase();
  if (!normalized) {
    return "direct";
  }
  if (normalized.includes("manual") || normalized.includes("rebuild") || normalized.includes("admin")) {
    return "admin";
  }
  return "direct";
}

function processSummaryEvents(
  events: SummaryEvent[],
  appendLog: (icon: string, message: string) => void,
  seenEvents: Set<string>
): string | null {
  let latestMessage: string | null = null;
  for (const event of events) {
    const eventId = typeof event.event_id === "string" ? event.event_id : "";
    if (!eventId || seenEvents.has(eventId)) {
      continue;
    }
    seenEvents.add(eventId);
    const messageRaw = typeof event.message === "string" ? event.message.trim() : "";
    const message = messageRaw || formatSummaryEvent(event as Record<string, unknown>);
    if (!message) {
      continue;
    }
    appendLog(determineSummaryEventIcon(event), message);
    latestMessage = message;
  }
  return latestMessage;
}

function updateSummaryStatus(
  status: SummaryStatus,
  appendLog: (icon: string, message: string) => void,
  setProgress: (progress: ProgressState) => void,
  tracker?: SummaryProgressTracker,
  latestMessage?: string | null,
  syncGlobalProgress?: (snapshot: SummaryStatus, message?: string | null) => void
): void {
  const total = Math.max(0, Math.trunc(status.total ?? 0));
  const completed = Math.max(0, Math.trunc(status.completed ?? 0));

  if (status.state === "running") {
    setProgress({
      completed,
      total
    });

    if (tracker && total > 0) {
      const percent = Math.floor((completed / total) * 100);
      const isInitial = tracker.lastCompleted < 0 || tracker.lastTotal !== total;
      const advancedFivePercent = percent >= tracker.lastPercent + 5;
      const reachedMilestone = total > 0 && completed === total;
      const completedDelta = completed - tracker.lastCompleted;
      const minStep = Math.max(1, Math.floor(total / 20));
      const advancedByCount = completedDelta >= minStep;
      const earlyProgress = completed > tracker.lastCompleted && completed <= Math.min(total, 10);
      const progressSummary = `${completed.toLocaleString()} / ${total.toLocaleString()} (${percent}%)`;
      const summaryChanged = tracker.lastSummary !== progressSummary;

      if ((isInitial || advancedFivePercent || advancedByCount || earlyProgress || reachedMilestone) && summaryChanged) {
        tracker.lastCompleted = completed;
        tracker.lastTotal = total;
        tracker.lastPercent = percent;
        tracker.lastSummary = progressSummary;
        appendLog("📊", `Summary progress: ${progressSummary}`);
      }
    }
  } else if (isTerminalSummaryState(status.state)) {
    if (total > 0) {
      setProgress({
        completed: completed || total,
        total
      });
    } else {
      setProgress(null);
    }
  }

  if (syncGlobalProgress) {
    syncGlobalProgress(status, latestMessage ?? null);
  }
}

function formatLogTimestamp(date: Date): string {
  if (!(date instanceof Date) || Number.isNaN(date.getTime())) {
    return "--:--:--";
  }
  return date.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false
  });
}

function LogPanel({ entries, emptyMessage }: { entries: LogEntry[]; emptyMessage: string }) {
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (container) {
      container.scrollTop = container.scrollHeight;
    }
  }, [entries.length]);

  return (
    <div className="rounded-md border border-border/60 bg-zinc-950 text-zinc-100 shadow-inner dark:border-border/40">
      <div ref={containerRef} className="h-72 overflow-y-auto px-3 py-2">
        {entries.length ? (
          <div className="space-y-1 font-mono text-xs leading-relaxed">
            {entries.map((entry) => (
              <div className="whitespace-pre-wrap text-left text-zinc-100" key={entry.id}>
                {formatLogTimestamp(entry.timestamp)} {entry.icon} {entry.message}
              </div>
            ))}
          </div>
        ) : (
          <p className="text-sm text-zinc-300">{emptyMessage}</p>
        )}
      </div>
    </div>
  );
}

function ProgressBar({ progress }: { progress: ProgressState }) {
  if (!progress || progress.total <= 0) {
    return null;
  }
  const percentage = Math.min(100, Math.round((progress.completed / progress.total) * 100));
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-xs text-muted-foreground">
        <span>Progress</span>
        <span>
          {progress.completed} / {progress.total}
        </span>
      </div>
      <div className="h-2 rounded bg-muted">
        <div className="h-2 rounded bg-primary transition-all" style={{ width: `${percentage}%` }} />
      </div>
    </div>
  );
}

export default function LoadDataClient({ defaults, initialError, initialMeta }: LoadDataClientProps) {
  const { toast } = useToast();
  const abortRef = useRef(false);
  const {
    updateStatus: setGlobalSummarizerStatus,
    subscribe: subscribeToSummarizer
  } = useSummarizerProgress();
  const directEventIdsRef = useRef<Set<string>>(new Set<string>());
  const adminEventIdsRef = useRef<Set<string>>(new Set<string>());
  const directTrackerRef = useRef<SummaryProgressTracker>(createProgressTracker());
  const adminTrackerRef = useRef<SummaryProgressTracker>(createProgressTracker());
  const activeJobRef = useRef<LogSource | null>(null);
  const lastReasonRef = useRef<string | null>(null);
  const trimmedDefaultsHost = defaults.host.trim();
  const trimmedDatabaseHost = defaults.databaseHost.trim();
  const resolvedInitialHost =
    defaults.hostSource === "database" ? trimmedDatabaseHost : trimmedDatabaseHost || trimmedDefaultsHost;
  const databaseApiKey = defaults.databaseApiKey;
  const trimmedDefaultsApiKey = defaults.apiKey.trim();
  const resolvedInitialApiKey =
    defaults.apiKeySource === "database" ? databaseApiKey : databaseApiKey || trimmedDefaultsApiKey;
  const databaseDefaultsRef = useRef<{ host: string; apiKey: string }>({
    host: trimmedDatabaseHost,
    apiKey: databaseApiKey
  });
  const hasEditedHostRef = useRef(false);
  const hasEditedApiKeyRef = useRef(false);

  const [datasetMeta, setDatasetMeta] = useState<DatasetMeta | null>(initialMeta);
  const [metaError, setMetaError] = useState<string | null>(initialError);
  const [hostname, setHostname] = useState(() => resolvedInitialHost);
  const [apiKey, setApiKey] = useState(() => resolvedInitialApiKey);

  const [isSyncing, setIsSyncing] = useState(false);
  const [directLogs, setDirectLogs] = useState<LogEntry[]>([]);
  const [directProgress, setDirectProgress] = useState<ProgressState>(null);
  const [logSource, setLogSource] = useState<LogSource | null>(null);

  const [isResetting, setIsResetting] = useState(false);
  const [isRebuilding, setIsRebuilding] = useState(false);
  const [adminLogs, setAdminLogs] = useState<LogEntry[]>([]);
  const [adminProgress, setAdminProgress] = useState<ProgressState>(null);

  const [isRefreshing, setIsRefreshing] = useState(false);
  const [isHydrated, setIsHydrated] = useState(false);

  const activeSource = useMemo<LogSource | null>(() => {
    if (logSource) {
      return logSource;
    }
    if (directLogs.length) {
      return "direct";
    }
    if (adminLogs.length) {
      return "admin";
    }
    return null;
  }, [adminLogs, directLogs, logSource]);

  const activeMeta = useMemo(() => {
    if (activeSource) {
      return PROCESSING_LOG_META[activeSource];
    }
    return DEFAULT_PROCESSING_LOG_META;
  }, [activeSource]);

  const activeLogs = useMemo((): LogEntry[] => {
    if (activeSource === "admin") {
      return adminLogs;
    }
    if (activeSource === "direct") {
      return directLogs;
    }
    return [];
  }, [activeSource, adminLogs, directLogs]);

  const activeProgress = useMemo((): ProgressState => {
    if (activeSource === "admin") {
      return adminProgress;
    }
    if (activeSource === "direct") {
      return directProgress;
    }
    return null;
  }, [activeSource, adminProgress, directProgress]);

  useEffect(() => {
    setIsHydrated(true);
  }, []);

  useEffect(() => {
    // Hydrate Direct Connect inputs with database defaults after client mount.
    if (!isHydrated) {
      return;
    }

    let cancelled = false;

    const hydrateFromDatabase = async () => {
      try {
        const payload = await apiGet<DirectConnectSettings>("api/v1/admin/settings/direct-connect");
        if (cancelled || abortRef.current || !payload) {
          return;
        }

        const hostValue = typeof payload.host === "string" ? payload.host.trim() : "";
        const apiKeyValue = typeof payload.api_key === "string" ? payload.api_key : "";
        const databaseHostValue =
          typeof payload.database_host === "string" ? payload.database_host.trim() : "";
        const databaseApiKeyValue =
          typeof payload.database_api_key === "string" ? payload.database_api_key : "";

        databaseDefaultsRef.current = {
          host: databaseHostValue || hostValue,
          apiKey: databaseApiKeyValue || apiKeyValue
        };

        if (!hasEditedHostRef.current) {
          const resolvedHost = hostValue || trimmedDefaultsHost;
          setHostname((current) => (current === resolvedHost ? current : resolvedHost));
        }

        if (!hasEditedApiKeyRef.current) {
          const resolvedApiKey = apiKeyValue || trimmedDefaultsApiKey;
          setApiKey((current) => (current === resolvedApiKey ? current : resolvedApiKey));
        }
      } catch {
        // ignore failures; fallback defaults remain applied.
      }
    };

    hydrateFromDatabase();

    return () => {
      cancelled = true;
    };
  }, [defaults.apiKey, defaults.host, isHydrated, trimmedDefaultsApiKey, trimmedDefaultsHost]);

  useEffect(() => {
    abortRef.current = false;
    return () => {
      abortRef.current = true;
    };
  }, []);

  const appendDirectLog = useCallback((icon: string, message: string) => {
    if (abortRef.current) {
      return;
    }
    setDirectLogs((previous) => [...previous, { id: safeUuid(), icon, message, timestamp: new Date() }]);
  }, []);

  const appendAdminLog = useCallback((icon: string, message: string) => {
    if (abortRef.current) {
      return;
    }
    setAdminLogs((previous) => [...previous, { id: safeUuid(), icon, message, timestamp: new Date() }]);
  }, []);

  const handleHostnameChange = useCallback((event: ChangeEvent<HTMLInputElement>) => {
    hasEditedHostRef.current = true;
    setHostname(event.target.value);
  }, []);

  const handleApiKeyChange = useCallback((event: ChangeEvent<HTMLInputElement>) => {
    hasEditedApiKeyRef.current = true;
    setApiKey(event.target.value);
  }, []);

  const handleResetToDefaults = useCallback(() => {
    hasEditedHostRef.current = true;
    hasEditedApiKeyRef.current = true;
    const dbDefaults = databaseDefaultsRef.current;
    setHostname(dbDefaults.host);
    setApiKey(dbDefaults.apiKey);
  }, []);

  const updateDirectProgress = useCallback(
    (progress: ProgressState) => {
      if (abortRef.current) {
        return;
      }
      setDirectProgress(progress);
    },
    []
  );

  const updateAdminProgress = useCallback(
    (progress: ProgressState) => {
      if (abortRef.current) {
        return;
      }
      setAdminProgress(progress);
    },
    []
  );

  useEffect(() => {
    if (!isHydrated) {
      return;
    }
    let cancelled = false;

    const replayEvents = async () => {
      try {
        const response = await apiGet<SummaryEventsResponse>("api/v1/summaries/events?limit=200");
        if (cancelled || abortRef.current) {
          return;
        }
        const events = Array.isArray(response.events) ? response.events : [];
        if (!events.length) {
          return;
        }

        let activeLog: LogSource | null = activeJobRef.current;

        for (const event of events) {
          const eventId = typeof event.event_id === "string" ? event.event_id : "";
          if (!eventId) {
            continue;
          }

          if (typeof event.reason === "string" && event.type === "start") {
            const inferred = inferLogSourceFromReason(event.reason);
            if (activeJobRef.current !== inferred) {
              if (inferred === "direct") {
                directEventIdsRef.current = new Set<string>();
                directTrackerRef.current = createProgressTracker();
              } else {
                adminEventIdsRef.current = new Set<string>();
                adminTrackerRef.current = createProgressTracker();
              }
            }
            activeLog = inferred;
            lastReasonRef.current = event.reason;
            activeJobRef.current = inferred;
          } else if (!activeLog) {
            activeLog = inferLogSourceFromReason(typeof event.reason === "string" ? event.reason : lastReasonRef.current);
          }

          const target = activeLog ?? "direct";
          const seen = target === "admin" ? adminEventIdsRef.current : directEventIdsRef.current;
          const append = target === "admin" ? appendAdminLog : appendDirectLog;
          processSummaryEvents([event], append, seen);
        }
      } catch {
        // ignore hydration failures
      }
    };

    replayEvents();

    return () => {
      cancelled = true;
    };
  }, [appendAdminLog, appendDirectLog, isHydrated]);

  useEffect(() => {
    return subscribeToSummarizer(({ status: nextStatus, events }) => {
      if (abortRef.current) {
        return;
      }

      let target: LogSource | null = activeJobRef.current;

      if (events.length) {
        for (const event of events) {
          if (typeof event.reason === "string" && event.type === "start") {
            const inferred = inferLogSourceFromReason(event.reason);
            if (activeJobRef.current !== inferred) {
              if (inferred === "direct") {
                directEventIdsRef.current = new Set<string>();
                directTrackerRef.current = createProgressTracker();
              } else {
                adminEventIdsRef.current = new Set<string>();
                adminTrackerRef.current = createProgressTracker();
              }
            }
            target = inferred;
            activeJobRef.current = inferred;
            lastReasonRef.current = event.reason;
            break;
          }
        }
      }

      if (!target && events.length) {
        target = inferLogSourceFromReason(
          typeof events[events.length - 1]?.reason === "string" ? events[events.length - 1].reason : lastReasonRef.current
        );
        activeJobRef.current = target;
      }

      if (events.length) {
        const chosenTarget = target ?? "direct";
        const seen = chosenTarget === "admin" ? adminEventIdsRef.current : directEventIdsRef.current;
        const append = chosenTarget === "admin" ? appendAdminLog : appendDirectLog;
        processSummaryEvents(events, append, seen);
      }

      if (nextStatus) {
        const chosenTarget = target ?? "direct";
        const tracker = chosenTarget === "admin" ? adminTrackerRef.current : directTrackerRef.current;
        const updateProgressFn = chosenTarget === "admin" ? updateAdminProgress : updateDirectProgress;
        const append = chosenTarget === "admin" ? appendAdminLog : appendDirectLog;
        updateSummaryStatus(nextStatus, append, updateProgressFn, tracker, null, setGlobalSummarizerStatus);

        if (isTerminalSummaryState(nextStatus.state)) {
          if (chosenTarget === "direct") {
            directTrackerRef.current = createProgressTracker();
            directEventIdsRef.current = new Set<string>();
          } else {
            adminTrackerRef.current = createProgressTracker();
            adminEventIdsRef.current = new Set<string>();
          }
          activeJobRef.current = null;
          lastReasonRef.current = null;
        }
      }
    });
  }, [
    appendAdminLog,
    appendDirectLog,
    setGlobalSummarizerStatus,
    subscribeToSummarizer,
    updateAdminProgress,
    updateDirectProgress
  ]);


  const refreshDatasetMeta = useCallback(async () => {
    try {
      const meta = await apiGet<DatasetMeta>("api/v1/datasets/meta");
      if (!abortRef.current) {
        setDatasetMeta(meta);
        setMetaError(null);
      }
      return meta;
    } catch (error) {
      if (!abortRef.current) {
        setMetaError(error instanceof Error ? error.message : "Unable to reach backend API.");
      }
      throw error;
    }
  }, []);

  const syncGlobalProgress = useCallback(
    (status: SummaryStatus, latestMessage?: string | null) => {
      const total = Math.max(0, Math.trunc(status.total ?? 0));
      const completed = Math.max(0, Math.trunc(status.completed ?? 0));
      const rawMessage =
        typeof latestMessage === "string" && latestMessage.trim().length
          ? latestMessage.trim()
          : typeof status.message === "string"
            ? status.message.trim()
            : "";
      const message = rawMessage || null;

      if (status.state === "running") {
        setGlobalSummarizerStatus({
          state: status.state,
          total,
          completed,
          message
        });
        return;
      }

      if (status.state === "completed" || status.state === "failed" || status.state === "cancelled") {
        setGlobalSummarizerStatus({
          state: status.state,
          total,
          completed,
          message
        });
        return;
      }

      setGlobalSummarizerStatus(null);
    },
    [setGlobalSummarizerStatus]
  );

  const pollSummaries = useCallback(
    async ({
      onStatus,
      initialStatus
    }: {
      onStatus: (status: SummaryStatus) => void;
      initialStatus?: SummaryStatus | null;
    }): Promise<SummaryStatus> => {
      let current = initialStatus ?? null;

      if (current) {
        onStatus(current);
        if (isTerminalSummaryState(current.state)) {
          return current;
        }
      }

      while (true) {
        let nextStatus: SummaryStatus;
        try {
          nextStatus = await apiGet<SummaryStatus>("api/v1/summaries/status");
        } catch (error) {
          const message = error instanceof Error ? error.message : "Unable to poll summarizer status.";
          onStatus({
            state: "failed",
            total: 0,
            completed: 0,
            message,
            last_event: null
          });
          throw error;
        }

        current = nextStatus;
        onStatus(current);
        if (isTerminalSummaryState(current.state)) {
          return current;
        }

        await sleep(SUMMARY_POLL_INTERVAL_MS);
      }
    },
    []
  );

  const datasetSummary = useMemo(
    () => computeDatasetSummary(datasetMeta, isHydrated),
    [datasetMeta, isHydrated]
  );
  const metadata = useMemo(() => getMetadata(datasetMeta), [datasetMeta]);
  const datasetHasChats = useMemo(() => toNumber(metadata.chat_count ?? datasetMeta?.chat_count ?? 0) > 0, [metadata, datasetMeta]);

  const toastShownRef = useRef(false);
  useEffect(() => {
    if (toastShownRef.current) {
      return;
    }
    const hasDefaults = Boolean(defaults.host.trim() || defaults.apiKey.trim());
    if (hasDefaults) {
      toast({
        title: "Defaults applied",
        description: "Open WebUI host and API key were pre-filled from environment defaults."
      });
      toastShownRef.current = true;
    }
  }, [defaults.apiKey, defaults.host, toast]);

  const handleRefreshMeta = useCallback(async () => {
    setIsRefreshing(true);
    try {
      await refreshDatasetMeta();
    } finally {
      if (!abortRef.current) {
        setIsRefreshing(false);
      }
    }
  }, [refreshDatasetMeta]);

  const handleSync = useCallback(
    async (event?: FormEvent<HTMLFormElement>) => {
      event?.preventDefault();
      if (isSyncing) {
        return;
      }

      const trimmedHost = hostname.trim();
      if (!trimmedHost) {
        toast({
          title: "Hostname required",
          description: "Provide the Open WebUI base URL before syncing.",
          variant: "destructive"
        });
        return;
      }

      setIsSyncing(true);
      setLogSource("direct");
      setDirectLogs([]);
      setDirectProgress(null);
      appendDirectLog("🔌", `Connecting to Open WebUI at ${trimmedHost}...`);

      try {
        const payload: Record<string, string> = { hostname: trimmedHost };
        if (apiKey.trim()) {
          payload.api_key = apiKey.trim();
        }

        const result = await apiPost<UploadResult>("api/v1/openwebui/sync", payload);
        const datasetRecord = (result.dataset ?? {}) as Record<string, unknown>;
        const stats = (result.stats ?? {}) as UploadStats;

        const chatCount = toNumber(datasetRecord["chat_count"] ?? stats.total_chats ?? 0);
        const userCount = toNumber(datasetRecord["user_count"] ?? stats.total_users ?? 0);
        const messageCount = toNumber(datasetRecord["message_count"] ?? stats.total_messages ?? 0);
        const modelCount = toNumber(datasetRecord["model_count"] ?? stats.total_models ?? 0);

        const sourceMatched = Boolean(stats.source_matched);
        const mode = typeof stats.mode === "string" ? stats.mode : "";
        const normalizedHostname =
          (typeof stats.normalized_hostname === "string" && stats.normalized_hostname.trim()) || trimmedHost;
        const newChats = toNumber(stats.new_chats ?? 0);
        const newUsers = toNumber(stats.new_users ?? 0);
        const newModels = toNumber(stats.new_models ?? 0);
        const modelsChanged = Boolean(stats.models_changed);
        const queuedChatIds = Array.isArray(stats.queued_chat_ids) ? stats.queued_chat_ids : [];
        const summarizerEnqueued =
          typeof stats.summarizer_enqueued === "boolean" ? stats.summarizer_enqueued : chatCount > 0;
        let summarizerHadIssues = false;

        if (Object.keys(stats).length) {
          if (sourceMatched) {
            if (mode === "noop") {
              appendDirectLog(
                "ℹ️",
                `Dataset already up to date for ${normalizedHostname}; no new chats, users, or models detected.`
              );
            } else {
              const bucketFragments: string[] = [];
              if (newChats) {
                bucketFragments.push(formatBucket(newChats, "chat"));
              }
              if (newUsers) {
                bucketFragments.push(formatBucket(newUsers, "user"));
              }
              if (newModels) {
                bucketFragments.push(formatBucket(newModels, "model"));
              }

              if (bucketFragments.length) {
                appendDirectLog("➕", `Added ${formatList(bucketFragments)} from ${normalizedHostname}.`);
              } else if (modelsChanged) {
                appendDirectLog("🔄", `Updated model metadata from ${normalizedHostname}.`);
              } else {
                appendDirectLog("ℹ️", `No new chats, users, or models detected for ${normalizedHostname}.`);
              }
            }
          } else {
            appendDirectLog("📦", `Loaded dataset from ${normalizedHostname}.`);
            appendDirectLog(
              "📥",
              `Retrieved ${formatCount(chatCount, "chat")} and ${formatCount(userCount, "user")}.`
            );
            appendDirectLog(
              "🗂️",
              `Captured ${formatCount(messageCount, "message")} and ${formatCount(modelCount, "model")}.`
            );
          }
        } else {
          appendDirectLog("📥", `Retrieved ${formatCount(chatCount, "chat")}`);
          appendDirectLog("🙋", `Retrieved ${formatCount(userCount, "user")}`);
          appendDirectLog("🤖", `Retrieved ${formatCount(modelCount, "model")}`);
        }

        // Refresh dataset metadata immediately after sync completes
        await refreshDatasetMeta();

        if (summarizerEnqueued) {
          const targetCount = queuedChatIds.length ? queuedChatIds.length : newChats || chatCount;
          appendDirectLog("🤖", `Creating new summaries for ${formatCount(targetCount, "chat")}.`);
          appendDirectLog("🧠", "Summarizer is now running in the background.");
          directEventIdsRef.current = new Set<string>();
          directTrackerRef.current = createProgressTracker();
          activeJobRef.current = "direct";
          lastReasonRef.current = "openwebui_sync";
          const progressTracker = directTrackerRef.current;

          let finalStatus: SummaryStatus | null = null;
          try {
            finalStatus = await pollSummaries({
              onStatus: (status) =>
                updateSummaryStatus(status, appendDirectLog, updateDirectProgress, progressTracker, null, syncGlobalProgress)
            });
          } catch (error) {
            const message = error instanceof Error ? error.message : "Unknown error";
            appendDirectLog("⚠️", `Summary job failed: ${message}`);
            summarizerHadIssues = true;
          }

          if (finalStatus) {
            if (finalStatus.state === "failed") {
              const message = finalStatus.message ?? "Summary job failed.";
              appendDirectLog("⚠️", `Summary job failed: ${message}`);
              summarizerHadIssues = true;
              toast({
                title: "Summary job failed",
                description: message,
                variant: "destructive"
              });
            } else if (finalStatus.state === "cancelled") {
              appendDirectLog("⚠️", "Summary job cancelled (dataset changed).");
              summarizerHadIssues = true;
            } else if (!isTerminalSummaryState(finalStatus.state)) {
              appendDirectLog("ℹ️", "Summaries still running in the background.");
            }
          }
        } else {
          appendDirectLog("ℹ️", "Summarizer skipped (no new chats to process).");
        }

        const summaryIcon = summarizerHadIssues ? "⚠️" : "✅";
        const summaryMessage = summarizerEnqueued
          ? summarizerHadIssues
            ? "Import job complete with summarizer warnings; summarizer is still running in the background."
            : "Import job complete; summarizer is running in the background."
          : summarizerHadIssues
            ? "Import job complete with warnings."
            : "Import job complete.";
        appendDirectLog(summaryIcon, summaryMessage);
        toast({
          title: "Open WebUI data synced",
          description: summarizerEnqueued
            ? summarizerHadIssues
              ? "Chats imported, but the summarizer reported warnings. Review the Summaries sidebar for details."
              : "Chats imported; summarizer is running in the background."
            : "Chats and metadata imported successfully."
        });
      } catch (error) {
        const message = error instanceof Error ? error.message : "Unknown error";
        appendDirectLog("❌", `Failed to connect: ${message}`);
      toast({
        title: "Sync failed",
        description: `Failed to sync data: ${message}`,
        variant: "destructive"
      });
    } finally {
      activeJobRef.current = null;
      lastReasonRef.current = null;
      directTrackerRef.current = createProgressTracker();
      directEventIdsRef.current = new Set<string>();
      if (!abortRef.current) {
        setIsSyncing(false);
        setDirectProgress(null);
      }
    }
    },
    [
      apiKey,
      appendDirectLog,
      hostname,
      isSyncing,
      pollSummaries,
      refreshDatasetMeta,
      syncGlobalProgress,
      toast,
      updateDirectProgress
    ]
  );

  const handleReset = useCallback(async () => {
    if (isResetting) {
      return;
    }
    if (typeof window !== "undefined") {
      const confirmed = window.confirm(
        "This will permanently delete all chats, messages, users, models, and reset app metadata. Are you sure?"
      );
      if (!confirmed) {
        return;
      }
    }

    setIsResetting(true);
    setLogSource("admin");
    setAdminLogs([]);
    setAdminProgress(null);
    appendAdminLog("🗑️", "Deleting chats, messages, users, and models...");

    try {
      await apiPost<UploadResult>("api/v1/datasets/reset");
      await refreshDatasetMeta();
      appendAdminLog("✅", "Dataset reset complete.");
      toast({
        title: "Dataset reset",
        description: "All analyzer data deleted successfully."
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : "Failed to delete all data.";
      appendAdminLog("❌", `Failed to delete data: ${message}`);
      toast({
        title: "Delete failed",
        description: message,
        variant: "destructive"
      });
    } finally {
      if (!abortRef.current) {
        setIsResetting(false);
        setAdminProgress(null);
      }
    }
  }, [appendAdminLog, isResetting, refreshDatasetMeta, toast]);

  const handleRerunSummaries = useCallback(async () => {
    if (isRebuilding) {
      return;
    }

    setIsRebuilding(true);
    setLogSource("admin");
    setAdminLogs([]);
    setAdminProgress(null);
    appendAdminLog("🔁", "Queuing manual summary rebuild...");

    adminEventIdsRef.current = new Set<string>();
    adminTrackerRef.current = createProgressTracker();
    activeJobRef.current = "admin";
    lastReasonRef.current = "manual_rebuild";
    const progressTracker = adminTrackerRef.current;
    try {
      let initialStatus: SummaryStatus;
      try {
        initialStatus = await apiPost<SummaryStatus>("api/v1/summaries/rebuild");
      } catch (error) {
        const message = error instanceof Error ? error.message : "Unable to rebuild summaries.";
        appendAdminLog("❌", `Failed to start summarizer: ${message}`);
        toast({
          title: "Summary job failed",
          description: message,
          variant: "destructive"
        });
        return;
      }

      updateSummaryStatus(initialStatus, appendAdminLog, updateAdminProgress, progressTracker, null, syncGlobalProgress);

      const state = initialStatus.state;
      if (state === "failed") {
        const message = initialStatus.message ?? "Summary job failed to start.";
        appendAdminLog("⚠️", `Summary job failed: ${message}`);
        toast({
          title: "Summary job failed",
          description: message,
          variant: "destructive"
        });
        return;
      }

      if ((state === null || state === "idle") && !datasetHasChats) {
        const message = initialStatus.message ?? "No chats available to summarize.";
        appendAdminLog("ℹ️", message);
        return;
      }

      appendAdminLog("🧠", "Rebuilding chat summaries...");

      let finalStatus: SummaryStatus;
      try {
        finalStatus = await pollSummaries({
          onStatus: (status) =>
            updateSummaryStatus(status, appendAdminLog, updateAdminProgress, progressTracker, null, syncGlobalProgress),
          initialStatus
        });
      } catch (error) {
        const message = error instanceof Error ? error.message : "Unknown error";
        appendAdminLog("⚠️", `Summary job failed: ${message}`);
        toast({
          title: "Summary job failed",
          description: message,
          variant: "destructive"
        });
        return;
      }

      if (finalStatus.state === "completed") {
        appendAdminLog("✅", "Chat summaries rebuilt");
        toast({
          title: "Chat summaries rebuilt",
          description: "Dashboard analytics will refresh with the latest summaries."
        });
        try {
          await refreshDatasetMeta();
        } catch (error) {
          const message = error instanceof Error ? error.message : "Unknown error";
          appendAdminLog("⚠️", `Summary job failed: ${message}`);
          toast({
            title: "Summary job failed",
            description: message,
            variant: "destructive"
          });
          return;
        }
      } else if (finalStatus.state === "failed") {
        const message = finalStatus.message ?? "Summary job failed.";
        appendAdminLog("⚠️", `Summary job failed: ${message}`);
        toast({
          title: "Summary job failed",
          description: message,
          variant: "destructive"
        });
      } else if (finalStatus.state === "cancelled") {
        appendAdminLog("⚠️", "Summary job cancelled (dataset changed).");
      } else {
        appendAdminLog("ℹ️", "Summaries still running in the background.");
      }
    } finally {
      activeJobRef.current = null;
      lastReasonRef.current = null;
      adminTrackerRef.current = createProgressTracker();
      adminEventIdsRef.current = new Set<string>();
      if (!abortRef.current) {
        setIsRebuilding(false);
        setAdminProgress(null);
      }
    }
  }, [
    appendAdminLog,
    datasetHasChats,
    isRebuilding,
    pollSummaries,
    refreshDatasetMeta,
    syncGlobalProgress,
    toast,
    updateAdminProgress
  ]);

  const handleBackendRetry = useCallback(async () => {
    setMetaError(null);
    try {
      await refreshDatasetMeta();
    } catch {
      // error state already handled inside refresh
    }
  }, [refreshDatasetMeta]);

  const directConnectReady = datasetSummary.ready && datasetSummary.sourceLabel === "Direct Connect";

  const deleteDisabled = !datasetSummary.ready;

  return (
    <div className="space-y-8">
      <section className="space-y-2">
        <h1 className="text-2xl font-semibold tracking-tight text-foreground">Load Data</h1>
        <p className="text-sm text-muted-foreground">
          Connect directly to Open WebUI or manage the local dataset powering the analytics experience.
        </p>
      </section>

      {metaError && !datasetMeta && (
        <Card>
          <CardHeader>
            <CardTitle>Backend unavailable</CardTitle>
            <CardDescription>
              We could not reach the FastAPI service. Start the backend and retry once it reports healthy.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap items-center gap-3">
              <Button onClick={handleBackendRetry} type="button">
                Retry now
              </Button>
              <p className="text-sm text-muted-foreground">{metaError}</p>
            </div>
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <CardTitle>Loaded Data</CardTitle>
            <CardDescription>Monitor the current dataset status and upload history.</CardDescription>
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <span className="inline-flex items-center gap-2 rounded-full border border-border/70 px-3 py-1 text-sm font-medium">
              <span>{datasetSummary.ready ? "🟢" : "🔴"}</span>
              <span>{datasetSummary.ready ? "Dataset loaded" : "Dataset not loaded"}</span>
            </span>
            <Button disabled={isRefreshing || isSyncing || isResetting || isRebuilding} onClick={handleRefreshMeta} size="sm" type="button" variant="outline">
              {isRefreshing ? "Refreshing…" : "Refresh status"}
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 lg:grid-cols-4">
            <div className="rounded-lg border border-border bg-card p-4 shadow-sm lg:col-span-2">
              <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Data Source</p>
              <div className="mt-2 space-y-2 text-sm">
                <p className="font-medium text-foreground">{datasetSummary.sourceLabel}</p>
                {datasetSummary.sourceLabel === "Direct Connect" && datasetSummary.sourceDetail ? (
                  datasetSummary.sourceDetail.startsWith("http") ? (
                    <a
                      className="text-sm text-primary underline"
                      href={datasetSummary.sourceDetail}
                      rel="noreferrer"
                      target="_blank"
                    >
                      {datasetSummary.sourceDetail}
                    </a>
                  ) : (
                    <p className="text-sm text-muted-foreground">{datasetSummary.sourceDetail}</p>
                  )
                ) : (
                  datasetSummary.sourceDetail && <p className="text-sm text-muted-foreground">{datasetSummary.sourceDetail}</p>
                )}
                <p className="text-sm text-muted-foreground" suppressHydrationWarning>
                  Last pulled: {datasetSummary.lastLoadedDisplay}
                  {datasetSummary.lastLoadedRelative && datasetSummary.lastLoadedRelative !== "0 days" ? (
                    <span className="ml-1 text-xs text-muted-foreground/80" suppressHydrationWarning>
                      ({datasetSummary.lastLoadedRelative})
                    </span>
                  ) : null}
                </p>
                <p className="text-sm text-muted-foreground">
                  Message count: {datasetSummary.messageCountDisplay}
                </p>
              </div>
            </div>
            <div className="rounded-lg border border-border bg-card p-4 shadow-sm">
              <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Chats</p>
              <div className="mt-2 space-y-1 text-sm text-muted-foreground">
                <p suppressHydrationWarning>Uploaded: {datasetSummary.chatUploaded}</p>
                <p>Count: {datasetSummary.chatCountDisplay}</p>
                <p suppressHydrationWarning>Range: {datasetSummary.dateRange}</p>
              </div>
            </div>
            <div className="rounded-lg border border-border bg-card p-4 shadow-sm">
              <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Users</p>
              <div className="mt-2 space-y-1 text-sm text-muted-foreground">
                <p suppressHydrationWarning>Uploaded: {datasetSummary.userUploaded}</p>
                <p>Count: {datasetSummary.userCountDisplay}</p>
              </div>
            </div>
            <div className="rounded-lg border border-border bg-card p-4 shadow-sm">
              <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Models</p>
              <div className="mt-2 space-y-1 text-sm text-muted-foreground">
                <p suppressHydrationWarning>Uploaded: {datasetSummary.modelUploaded}</p>
                <p>Count: {datasetSummary.modelCountDisplay}</p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>
              🔌 Direct Connect to Open WebUI {directConnectReady ? "✅" : ""}
            </CardTitle>
            <CardDescription>
              Pull the latest chats and user records directly from an Open WebUI instance. Provide the base URL and an API token with sufficient permissions.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form className="space-y-4" onSubmit={handleSync}>
              <div className="space-y-2">
                <label className="text-sm font-medium" htmlFor="hostname">
                  Hostname
                </label>
                <Input
                  autoComplete="url"
                  disabled={isSyncing}
                  id="hostname"
                  placeholder="http://localhost:3000"
                  value={hostname}
                  onChange={handleHostnameChange}
                />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium" htmlFor="api-key">
                  API Key (Bearer token)
                </label>
                <Input
                  autoComplete="off"
                  disabled={isSyncing}
                  id="api-key"
                  placeholder="sk-..."
                  type="password"
                  value={apiKey}
                  onChange={handleApiKeyChange}
                />
              </div>
              <div className="flex items-center gap-3">
                <Button disabled={isSyncing} type="submit">
                  {isSyncing ? "Loading…" : "Load data"}
                </Button>
                <Button
                  disabled={isSyncing}
                  onClick={handleResetToDefaults}
                  type="button"
                  variant="outline"
                >
                  Reset to defaults
                </Button>
              </div>
            </form>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="space-y-2">
            <CardTitle>Processing log</CardTitle>
            <CardDescription>{activeMeta.description}</CardDescription>
            {activeSource && (
              <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                Source: {activeMeta.label}
              </p>
            )}
          </CardHeader>
          <CardContent className="space-y-4">
            <ProgressBar progress={activeProgress} />
            <LogPanel entries={activeLogs} emptyMessage={activeMeta.empty} />
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>🛡️ Admin Tools</CardTitle>
          <CardDescription>These actions affect all loaded data. Proceed only if you understand the consequences.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-wrap gap-3">
            <Button disabled={deleteDisabled || isResetting} onClick={handleReset} type="button" variant="destructive">
              {isResetting ? "Deleting…" : "Delete all data"}
            </Button>
            <Button disabled={!datasetHasChats || isRebuilding} onClick={handleRerunSummaries} type="button" variant="outline">
              {isRebuilding ? "Rebuilding…" : "Rerun summaries"}
            </Button>
          </div>
          <p className="text-xs text-muted-foreground">
            Progress updates and results appear in the Processing log.
          </p>
        </CardContent>
      </Card>

      {!datasetSummary.ready && (
        <Card>
          <CardHeader>
            <CardTitle>How to enable Direct Connect</CardTitle>
            <CardDescription>Populate the analyzer with chats, users, and models from Open WebUI.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4 text-sm text-muted-foreground">
            <ol className="list-decimal space-y-2 pl-5">
              <li>Open Open WebUI and sign in with an account that can access the Admin panel.</li>
              <li>Generate an API token from Settings → Personal Access Tokens.</li>
              <li>Confirm the base URL where Open WebUI is running, for example <code>http://localhost:3000</code>.</li>
              <li>Enter the URL and token in the Direct Connect panel and press <strong>Load data</strong>.</li>
            </ol>
            <div>
              <p className="font-medium text-foreground">What you&apos;ll see after loading data</p>
              <ul className="mt-2 list-disc space-y-1 pl-5">
                <li>Overview metrics covering chats, messages, users, and token usage.</li>
                <li>Time analysis charts for daily activity and conversation cadence.</li>
                <li>Model usage breakdowns highlighting popular assistant models.</li>
                <li>Sentiment and content analysis once summaries are available.</li>
                <li>Search and filter tools across conversations, messages, and exports.</li>
              </ul>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
