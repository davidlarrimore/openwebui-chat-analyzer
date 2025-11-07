"use client";

import { useSession } from "next-auth/react";
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { useToast } from "@/components/ui/use-toast";
import {
  ApiError,
  apiGet,
  cancelSummaryJob,
  getProcessLogs,
  type ProcessLogEvent,
} from "@/lib/api";
import type {
  SummaryEvent,
  SummaryEventsResponse,
  SummaryStatus,
} from "@/lib/types";
import {
  deriveJobTypeFromLog,
  inferJobSubtitle,
  normaliseJobKey,
  updateDataSyncStages,
  type JobLifecycleStatus,
  type JobState,
} from "@/lib/job-monitor";
import { JobMonitorPanel } from "@/components/job-monitor-panel";

type SummarizerSnapshot = {
  state: string | null;
  total: number;
  completed: number;
  message?: string | null;
};

type SummarizerListener = (payload: {
  status: SummarizerSnapshot | null;
  events: SummaryEvent[];
}) => void;

interface JobMonitorContextValue {
  jobs: JobState[];
  isPanelCollapsed: boolean;
  setPanelCollapsed: (collapsed: boolean) => void;
  status: SummarizerSnapshot | null;
  updateStatus: (status: SummarizerSnapshot | null) => void;
  clearStatus: () => void;
  subscribe: (listener: SummarizerListener) => () => void;
}

const JobMonitorContext = createContext<JobMonitorContextValue | undefined>(
  undefined,
);

const SUMMARY_POLL_ACTIVE_MS = 2000;
const SUMMARY_POLL_IDLE_MS = 10000;
const LOG_POLL_INTERVAL_MS = 2500;
const JOB_RETENTION_AFTER_COMPLETE_MS = 6000;
const MAX_LOGS_PER_JOB = 200;
const MAX_LOG_SIGNATURE_CACHE = 800;

const STATUS_PRIORITY: Record<JobLifecycleStatus, number> = {
  running: 0,
  error: 1,
  pending: 2,
  cancelled: 3,
  success: 4,
};

function loadToastState(): Map<string, { started?: boolean; finished?: boolean }> {
  if (typeof window === 'undefined') {
    return new Map();
  }
  try {
    const stored = sessionStorage.getItem('job-toast-state');
    if (stored) {
      const parsed = JSON.parse(stored);
      return new Map<string, { started?: boolean; finished?: boolean }>(Object.entries(parsed));
    }
  } catch (error) {
    console.error('Failed to load toast state', error);
  }
  return new Map<string, { started?: boolean; finished?: boolean }>();
}

function saveToastState(state: Map<string, { started?: boolean; finished?: boolean }>): void {
  if (typeof window === 'undefined') {
    return;
  }
  try {
    const obj = Object.fromEntries(state);
    sessionStorage.setItem('job-toast-state', JSON.stringify(obj));
  } catch (error) {
    console.error('Failed to save toast state', error);
  }
}

function toSnapshot(
  status: SummaryStatus | null,
  fallback: SummarizerSnapshot | null,
  message?: string | null,
): SummarizerSnapshot | null {
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
    message: rawMessage || null,
  };
}

export function SummarizerProgressProvider({
  children,
}: {
  children: ReactNode;
}) {
  const { data: session, status: authStatus } = useSession();
  const hasValidSession =
    authStatus === "authenticated" && Boolean(session?.accessToken);
  const { toast } = useToast();
  const [status, setStatus] = useState<SummarizerSnapshot | null>(null);
  const statusRef = useRef<SummarizerSnapshot | null>(null);
  const listenersRef = useRef<Set<SummarizerListener>>(new Set());
  const [jobs, setJobs] = useState<JobState[]>([]);
  const jobsRef = useRef<Map<string, JobState>>(new Map());
  const removalTimersRef = useRef<Map<string, ReturnType<typeof setTimeout>>>(
    new Map(),
  );
  const [isPanelCollapsed, setIsPanelCollapsed] = useState(true);

  const pollTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lastEventIdRef = useRef<string | null>(null);
  const abortRef = useRef(false);
  const unauthorizedRef = useRef(false);
  const summarizerJobIdRef = useRef<string | null>(null);

  const processedLogSignaturesRef = useRef<Set<string>>(new Set());
  const processedLogOrderRef = useRef<string[]>([]);
  const logPollTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const toastStateRef = useRef<Map<string, { started?: boolean; finished?: boolean }>>(loadToastState());

  const publishJobs = useCallback(() => {
    const ordered = Array.from(jobsRef.current.values()).sort((a, b) => {
      const statusDiff =
        STATUS_PRIORITY[a.status] - STATUS_PRIORITY[b.status];
      if (statusDiff !== 0) {
        return statusDiff;
      }
      const timeA = new Date(a.updatedAt).getTime();
      const timeB = new Date(b.updatedAt).getTime();
      return timeB - timeA;
    });
    setJobs(ordered);
  }, []);

  const cancelRemoval = useCallback((jobKey: string) => {
    const timer = removalTimersRef.current.get(jobKey);
    if (timer) {
      clearTimeout(timer);
      removalTimersRef.current.delete(jobKey);
    }
  }, []);

  const scheduleRemoval = useCallback(
    (jobKey: string, delay: number = JOB_RETENTION_AFTER_COMPLETE_MS) => {
      cancelRemoval(jobKey);
      const timer = setTimeout(() => {
        jobsRef.current.delete(jobKey);
        removalTimersRef.current.delete(jobKey);
        toastStateRef.current.delete(jobKey);
        saveToastState(toastStateRef.current);
        publishJobs();
      }, delay);
      removalTimersRef.current.set(jobKey, timer);
    },
    [cancelRemoval, publishJobs],
  );

  const clearPreviousDataRun = useCallback(() => {
    jobsRef.current.forEach((job, key) => {
      if (job.type === "dataSync" || job.type === "summarizer") {
        jobsRef.current.delete(key);
        toastStateRef.current.delete(key);
        cancelRemoval(key);
      }
    });
    saveToastState(toastStateRef.current);
    summarizerJobIdRef.current = null;
  }, [cancelRemoval]);

  const onJobStart = useCallback(
    (job: JobState, hadJobs: boolean) => {
      if (!hadJobs) {
        setIsPanelCollapsed(true);
      }
      const entry = toastStateRef.current.get(job.jobKey) ?? {};
      if (entry.started) {
        return;
      }
      entry.started = true;
      toastStateRef.current.set(job.jobKey, entry);
      saveToastState(toastStateRef.current);
      const descriptionParts: string[] = [];
      if (job.subtitle) {
        descriptionParts.push(job.subtitle);
      }
      if (job.progress && job.progress.total > 0) {
        descriptionParts.push(
          `${job.progress.completed}/${job.progress.total}`,
        );
      }
      toast({
        title: `${job.label} started`,
        description: descriptionParts.length
          ? descriptionParts.join(" • ")
          : undefined,
        duration: 3500,
      });
    },
    [toast],
  );

  const onJobFinish = useCallback(
    (job: JobState) => {
      const entry = toastStateRef.current.get(job.jobKey) ?? {};
      if (entry.finished) {
        return;
      }
      entry.finished = true;
      toastStateRef.current.set(job.jobKey, entry);
      saveToastState(toastStateRef.current);
      toast({
        title: job.status === "error"
          ? `${job.label} failed`
          : `${job.label} complete`,
        description: job.lastMessage ?? job.subtitle ?? undefined,
        variant: job.status === "error" ? "destructive" : "default",
        duration: 4500,
      });
    },
    [toast],
  );

  const upsertJob = useCallback(
    (
      jobKey: string,
      builder: (current: JobState | undefined) => JobState | null,
    ) => {
      const hadJobs = jobsRef.current.size > 0;
      const existing = jobsRef.current.get(jobKey);
      const next = builder(existing);

      if (next === null) {
        jobsRef.current.delete(jobKey);
        toastStateRef.current.delete(jobKey);
        saveToastState(toastStateRef.current);
        cancelRemoval(jobKey);
        publishJobs();
        return;
      }

      if (!existing && next.type === "dataSync") {
        clearPreviousDataRun();
      }

      jobsRef.current.set(jobKey, next);

      if (!existing) {
        onJobStart(next, hadJobs);
      } else if (existing.status !== next.status) {
        if (next.status === "success" || next.status === "error" || next.status === "cancelled") {
          onJobFinish(next);
        }
      }

      if (next.status === "success" || next.status === "error" || next.status === "cancelled") {
        if (next.type === "dataSync" || next.type === "summarizer") {
          cancelRemoval(jobKey);
        } else {
          scheduleRemoval(jobKey);
        }
      } else {
        cancelRemoval(jobKey);
      }

      publishJobs();
    },
    [
      cancelRemoval,
      onJobFinish,
      onJobStart,
      publishJobs,
      scheduleRemoval,
      clearPreviousDataRun,
    ],
  );

  const updateSummarizerJobFromSnapshot = useCallback(
    (snapshot: SummarizerSnapshot | null) => {
      const jobId = summarizerJobIdRef.current;
      if (!snapshot || !jobId) {
        return;
      }
      const jobKey = normaliseJobKey("summarizer", jobId);
      upsertJob(jobKey, (current) => {
        if (!current && snapshot.state && snapshot.state !== "running") {
          return null;
        }
        const percent =
          snapshot.total > 0
            ? Math.min(
              100,
              Math.floor((snapshot.completed / snapshot.total) * 100),
            )
            : snapshot.state === "running"
              ? 0
              : 100;
        let status: JobLifecycleStatus = current?.status ?? "running";
        if (snapshot.state === "running") {
          status = "running";
        } else if (snapshot.state === "failed") {
          status = "error";
        } else if (snapshot.state === "cancelled") {
          status = "cancelled";
        } else if (snapshot.state) {
          status = "success";
        }
        const completedAt =
          status === "success" || status === "error" || status === "cancelled"
            ? current?.completedAt ?? new Date().toISOString()
            : current?.completedAt;
        const progress = {
          completed: snapshot.completed,
          total: snapshot.total,
          percent,
          message: snapshot.message ?? current?.lastMessage ?? null,
        };
        const next: JobState = {
          ...(current ?? {
            jobKey,
            jobId,
            type: "summarizer",
            label: "Summarizer",
            subtitle: "Chat summaries",
            logs: [],
          }),
          jobKey,
          jobId,
          type: "summarizer",
          label: "Summarizer",
          subtitle: current?.subtitle ?? "Chat summaries",
          status,
          startedAt: current?.startedAt ?? new Date().toISOString(),
          completedAt,
          updatedAt: new Date().toISOString(),
          progress,
          lastMessage: progress.message ?? current?.lastMessage ?? null,
        };
        return next;
      });
    },
    [upsertJob],
  );

  const applyStatus = useCallback(
    (next: SummarizerSnapshot | null) => {
      const current = statusRef.current;
      let resolved = next;
      if (
        current &&
        next &&
        current.state === next.state &&
        current.total === next.total &&
        current.completed === next.completed &&
        (current.message ?? "") === (next.message ?? "")
      ) {
        resolved = current;
      }
      statusRef.current = resolved;
      setStatus(resolved);
      if (resolved) {
        updateSummarizerJobFromSnapshot(resolved);
      } else {
        const jobId = summarizerJobIdRef.current;
        if (jobId) {
          const jobKey = normaliseJobKey("summarizer", jobId);
          scheduleRemoval(jobKey, JOB_RETENTION_AFTER_COMPLETE_MS);
        }
      }
    },
    [scheduleRemoval, updateSummarizerJobFromSnapshot],
  );

  const updateStatus = useCallback(
    (next: SummarizerSnapshot | null) => {
      applyStatus(next);
    },
    [applyStatus],
  );

  const clearStatus = useCallback(() => {
    statusRef.current = null;
    setStatus(null);
  }, []);

  const subscribe = useCallback((listener: SummarizerListener) => {
    listenersRef.current.add(listener);
    return () => {
      listenersRef.current.delete(listener);
    };
  }, []);

  const handleSummarizerEvents = useCallback(
    (events: SummaryEvent[], snapshot: SummarizerSnapshot | null) => {
      if (!events.length) {
        return;
      }
      events.forEach((event) => {
        const jobId =
          event.job_id !== null && event.job_id !== undefined
            ? String(event.job_id)
            : summarizerJobIdRef.current;
        if (!jobId) {
          return;
        }
        summarizerJobIdRef.current = jobId;
        const jobKey = normaliseJobKey("summarizer", jobId);
        const percent =
          snapshot && snapshot.total > 0
            ? Math.min(
              100,
              Math.floor((snapshot.completed / snapshot.total) * 100),
            )
            : snapshot?.state === "running"
              ? 0
              : 100;
        upsertJob(jobKey, (current) => {
          let status: JobLifecycleStatus = current?.status ?? "running";
          if (event.outcome === "failed" || /fail/i.test(event.message ?? "")) {
            status = "error";
          } else if (
            /complete/i.test(event.message ?? "") ||
            event.outcome === "completed"
          ) {
            status = "success";
          } else if (snapshot?.state === "running") {
            status = "running";
          }
          const progress = snapshot
            ? {
              completed: snapshot.completed,
              total: snapshot.total,
              percent,
              message:
                snapshot.message ??
                event.message ??
                current?.progress?.message ??
                null,
            }
            : current?.progress ?? null;
          const next: JobState = {
            ...(current ?? {
              jobKey,
              jobId,
              type: "summarizer",
              label: "Summarizer",
              subtitle: "Chat summaries",
              logs: [],
            }),
            jobKey,
            jobId,
            type: "summarizer",
            label: "Summarizer",
            subtitle: "Chat summaries",
            status,
            startedAt: current?.startedAt ?? event.timestamp ?? new Date().toISOString(),
            completedAt:
              status === "success" || status === "error" || status === "cancelled"
                ? current?.completedAt ?? event.timestamp ?? new Date().toISOString()
                : current?.completedAt,
            updatedAt: event.timestamp ?? new Date().toISOString(),
            progress,
            lastMessage: progress?.message ?? current?.lastMessage ?? event.message ?? null,
          };
          return next;
        });
      });
    },
    [upsertJob],
  );

  const applyLogEvent = useCallback(
    (log: ProcessLogEvent) => {
      if (!log.job_id) {
        return;
      }
      const jobType = deriveJobTypeFromLog(log);
      const jobId = String(log.job_id);
      if (jobType === "summarizer") {
        summarizerJobIdRef.current = jobId;
      }
      const jobKey = normaliseJobKey(jobType, jobId);

      upsertJob(jobKey, (current) => {
        const defaultLabel =
          jobType === "summarizer"
            ? "Summarizer"
            : jobType === "dataSync"
              ? "Data Load"
              : "Background Job";
        const logs = [...(current?.logs ?? []), log].slice(-MAX_LOGS_PER_JOB);
        let status: JobLifecycleStatus = current?.status ?? "pending";
        if (log.phase === "error" || /failed/i.test(log.message)) {
          status = "error";
        } else if (log.phase === "done") {
          status = "success";
        } else if (status === "pending") {
          status = "running";
        }

        const next: JobState = {
          jobKey,
          jobId,
          type: jobType,
          label: current?.label ?? defaultLabel,
          subtitle:
            current?.subtitle ??
            inferJobSubtitle(jobType, log) ??
            undefined,
          status,
          startedAt: current?.startedAt ?? log.timestamp,
          completedAt:
            status === "success" || status === "error" || status === "cancelled"
              ? current?.completedAt ?? log.timestamp
              : current?.completedAt,
          updatedAt: log.timestamp,
          logs,
          stages:
            jobType === "dataSync"
              ? updateDataSyncStages(current?.stages, log.phase)
              : current?.stages,
          progress: jobType === "summarizer" ? current?.progress : undefined,
          lastMessage: log.message,
        };
        return next;
      });
    },
    [upsertJob],
  );

  const processLogs = useCallback(
    (logs: ProcessLogEvent[]) => {
      if (!logs.length) {
        return;
      }
      const ordered = [...logs].sort(
        (a, b) =>
          new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime(),
      );
      ordered.forEach((log) => {
        const signature = `${log.timestamp}|${log.level}|${log.phase}|${log.message}|${log.job_id ?? ""}`;
        if (processedLogSignaturesRef.current.has(signature)) {
          return;
        }
        processedLogSignaturesRef.current.add(signature);
        processedLogOrderRef.current.push(signature);
        if (processedLogOrderRef.current.length > MAX_LOG_SIGNATURE_CACHE) {
          const removeCount =
            processedLogOrderRef.current.length - MAX_LOG_SIGNATURE_CACHE;
          const trimmed = processedLogOrderRef.current.splice(0, removeCount);
          trimmed.forEach((item) =>
            processedLogSignaturesRef.current.delete(item),
          );
        }
        applyLogEvent(log);
      });
    },
    [applyLogEvent],
  );

  useEffect(() => {
    const removalTimers = removalTimersRef.current;
    return () => {
      removalTimers.forEach((timer) => clearTimeout(timer));
      removalTimers.clear();
      if (pollTimeoutRef.current) {
        clearTimeout(pollTimeoutRef.current);
        pollTimeoutRef.current = null;
      }
      if (logPollTimeoutRef.current) {
        clearTimeout(logPollTimeoutRef.current);
        logPollTimeoutRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    if (!hasValidSession) {
      unauthorizedRef.current = false;
      abortRef.current = true;
      if (pollTimeoutRef.current) {
        clearTimeout(pollTimeoutRef.current);
        pollTimeoutRef.current = null;
      }
      if (logPollTimeoutRef.current) {
        clearTimeout(logPollTimeoutRef.current);
        logPollTimeoutRef.current = null;
      }
      lastEventIdRef.current = null;
      summarizerJobIdRef.current = null;
      statusRef.current = null;
      setStatus(null);
      jobsRef.current.clear();
      publishJobs();
      return;
    }

    if (unauthorizedRef.current) {
      return;
    }

    abortRef.current = false;

    const scheduleSummaryPoll = (delay: number) => {
      if (pollTimeoutRef.current) {
        clearTimeout(pollTimeoutRef.current);
      }
      pollTimeoutRef.current = setTimeout(runSummaryPoll, delay);
    };

    const runSummaryPoll = async () => {
      if (abortRef.current) {
        return;
      }

      let nextDelay = SUMMARY_POLL_IDLE_MS;
      let events: SummaryEvent[] = [];
      let snapshot = statusRef.current;

      try {
        const statusPayload = await apiGet<SummaryStatus>(
          "api/v1/summaries/status",
          undefined,
          { skipAuthRedirect: true },
        );

        const isRunning = statusPayload?.state === "running";
        if (isRunning) {
          const path = lastEventIdRef.current
            ? `api/v1/summaries/events?after=${encodeURIComponent(lastEventIdRef.current)}`
            : "api/v1/summaries/events";
          const queue = await apiGet<SummaryEventsResponse>(
            path,
            undefined,
            { skipAuthRedirect: true },
          );

          const fetchedEvents = Array.isArray(queue?.events)
            ? queue.events
            : [];

          if (queue?.reset) {
            lastEventIdRef.current = null;
          }

          if (fetchedEvents.length) {
            lastEventIdRef.current =
              fetchedEvents[fetchedEvents.length - 1]?.event_id ??
              lastEventIdRef.current;
            events = fetchedEvents;
          }
        } else {
          lastEventIdRef.current = null;
        }

        const latestMessage =
          events.length && typeof events[events.length - 1]?.message === "string"
            ? (events[events.length - 1].message as string)
            : undefined;

        snapshot = toSnapshot(statusPayload, snapshot, latestMessage ?? snapshot?.message ?? null);

        if (events.length) {
          handleSummarizerEvents(events, snapshot);
        }

        applyStatus(snapshot);
        nextDelay =
          snapshot?.state === "running"
            ? SUMMARY_POLL_ACTIVE_MS
            : SUMMARY_POLL_IDLE_MS;
      } catch (error) {
        if (error instanceof ApiError && error.status === 401) {
          abortRef.current = true;
          unauthorizedRef.current = true;
          lastEventIdRef.current = null;
          statusRef.current = null;
          setStatus(null);
          return;
        }
        console.error("Failed to poll summarizer status", error);
        nextDelay = SUMMARY_POLL_IDLE_MS;
      }

      listenersRef.current.forEach((listener) => {
        try {
          listener({ status: snapshot, events });
        } catch {
          // ignore listener errors
        }
      });

      if (!abortRef.current) {
        scheduleSummaryPoll(nextDelay);
      }
    };

    scheduleSummaryPoll(0);

    return () => {
      abortRef.current = true;
      if (pollTimeoutRef.current) {
        clearTimeout(pollTimeoutRef.current);
        pollTimeoutRef.current = null;
      }
    };
  }, [applyStatus, authStatus, handleSummarizerEvents, hasValidSession, publishJobs]);

  useEffect(() => {
    if (!hasValidSession) {
      return;
    }

    let cancelled = false;

    const scheduleLogPoll = (delay: number) => {
      if (logPollTimeoutRef.current) {
        clearTimeout(logPollTimeoutRef.current);
      }
      logPollTimeoutRef.current = setTimeout(runLogPoll, delay);
    };

    const runLogPoll = async () => {
      try {
        const response = await getProcessLogs(undefined, 200);
        if (cancelled) {
          return;
        }
        processLogs(response.logs);
      } catch (error) {
        console.error("Failed to fetch process logs", error);
      } finally {
        if (!cancelled) {
          scheduleLogPoll(LOG_POLL_INTERVAL_MS);
        }
      }
    };

    scheduleLogPoll(0);

    return () => {
      cancelled = true;
      if (logPollTimeoutRef.current) {
        clearTimeout(logPollTimeoutRef.current);
        logPollTimeoutRef.current = null;
      }
    };
  }, [authStatus, hasValidSession, processLogs, session?.accessToken]);

  const handleCancelJob = useCallback(
    async (jobId: string, jobType: JobState["type"]) => {
      if (jobType !== "summarizer") {
        return;
      }
      try {
        await cancelSummaryJob();
        toast({
          title: "Summarizer stopped",
          description: "The summarizer job has been cancelled",
          duration: 3000,
        });
      } catch (error) {
        console.error("Failed to cancel summarizer job", error);
        toast({
          title: "Failed to stop",
          description: "Could not cancel the summarizer job",
          variant: "destructive",
          duration: 4000,
        });
      }
    },
    [toast],
  );

  const contextValue = useMemo<JobMonitorContextValue>(
    () => ({
      jobs,
      isPanelCollapsed,
      setPanelCollapsed: setIsPanelCollapsed,
      status,
      updateStatus,
      clearStatus,
      subscribe,
    }),
    [clearStatus, isPanelCollapsed, jobs, status, subscribe, updateStatus],
  );

  return (
    <JobMonitorContext.Provider value={contextValue}>
      {children}
      {hasValidSession ? (
        <JobMonitorPanel
          jobs={jobs}
          isCollapsed={isPanelCollapsed}
          onCollapseChange={setIsPanelCollapsed}
          onCancelJob={handleCancelJob}
        />
      ) : null}
    </JobMonitorContext.Provider>
  );
}

export function useJobMonitor(): JobMonitorContextValue {
  const context = useContext(JobMonitorContext);
  if (!context) {
    throw new Error(
      "useJobMonitor must be used within a SummarizerProgressProvider",
    );
  }
  return context;
}

export function useSummarizerProgress(): Pick<
  JobMonitorContextValue,
  "status" | "updateStatus" | "clearStatus" | "subscribe"
> {
  const context = useJobMonitor();
  return {
    status: context.status,
    updateStatus: context.updateStatus,
    clearStatus: context.clearStatus,
    subscribe: context.subscribe,
  };
}
