"use client";

import * as React from "react";
import {
  Activity,
  Clock,
  FileText,
  MinusSquare,
  Monitor,
  NotebookPen,
  Radio,
  ChevronDown,
  ChevronRight,
} from "lucide-react";
import { cn } from "@/lib/utils";
import type { ProcessLogEvent } from "@/lib/api";
import type { JobState, JobStageState } from "@/lib/job-monitor";

interface JobMonitorPanelProps {
  jobs: JobState[];
  isCollapsed: boolean;
  onCollapseChange: (collapsed: boolean) => void;
}

export function JobMonitorPanel({
  jobs,
  isCollapsed,
  onCollapseChange,
}: JobMonitorPanelProps) {
  const [logVisibility, setLogVisibility] = React.useState<Record<string, boolean>>({});

  React.useEffect(() => {
    // Reset log expansion when jobs disappear
    const keys = new Set(jobs.map((job) => job.jobKey));
    setLogVisibility((prev) => {
      const next: Record<string, boolean> = {};
      keys.forEach((key) => {
        if (prev[key]) {
          next[key] = true;
        }
      });
      return next;
    });
  }, [jobs]);

  const activeCount = jobs.filter(
    (job) => job.status === "running" || job.status === "pending",
  ).length;
  const totalCount = jobs.length;

  if (isCollapsed) {
    return (
      <div className="fixed bottom-4 right-4 z-50">
        <button
          type="button"
          className="pointer-events-auto inline-flex items-center gap-2 rounded-t-lg rounded-bl-lg border border-border bg-card px-3 py-2 text-sm font-medium shadow-lg transition hover:bg-card/90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
          aria-expanded="false"
          aria-controls="job-monitor-panel"
          onClick={() => onCollapseChange(false)}
        >
          <Monitor className="h-4 w-4" aria-hidden />
          <span className="font-semibold">Jobs</span>
          <span
            className={cn(
              "rounded px-1.5 py-0.5 text-xs",
              totalCount > 0 ? "bg-primary/10 text-primary" : "bg-muted text-muted-foreground"
            )}
          >
            {totalCount > 0 ? activeCount || totalCount : "0"}
          </span>
        </button>
      </div>
    );
  }

  return (
    <div className="fixed bottom-4 right-4 z-50">
      <div
        id="job-monitor-panel"
        role="region"
        aria-label="Job activity monitor"
        className="pointer-events-auto w-[min(24rem,calc(100vw-2rem))]"
      >
        <div className="relative rounded-lg border border-border bg-card shadow-xl">
          <button
            type="button"
            className="absolute -top-5 left-4 inline-flex h-8 items-center justify-center gap-2 rounded-full border border-border bg-card px-3 text-xs font-medium shadow focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
            aria-expanded="true"
            aria-controls="job-monitor-panel"
            onClick={() => onCollapseChange(true)}
          >
            <MinusSquare className="h-4 w-4" aria-hidden />
            Collapse
          </button>
          <div className="overflow-hidden rounded-lg">
            <header className="flex items-center justify-between border-b border-border/80 px-4 pt-6 pb-3">
              <div className="flex flex-col gap-1">
                <div className="flex items-center gap-2">
                  <Activity className="h-4 w-4 text-primary" aria-hidden />
                  <span className="text-sm font-semibold tracking-tight">
                    Job Activity
                  </span>
                </div>
                <span className="text-xs text-muted-foreground">
                  {totalCount
                    ? `${activeCount} running`
                    : "No jobs currently running"}
                </span>
              </div>
              <span className="text-xs text-muted-foreground">
                Total {totalCount}
              </span>
            </header>

            {totalCount === 0 ? (
              <div className="px-4 py-6 text-sm text-muted-foreground">
                No background jobs are running right now. New activity will appear here automatically.
              </div>
            ) : (
              <div className="max-h-[360px] space-y-3 overflow-y-auto px-4 py-3">
                {jobs.map((job) => {
                  const isLogsOpen = logVisibility[job.jobKey] ?? false;
                  return (
                    <JobCard
                      key={job.jobKey}
                      job={job}
                      showLogs={isLogsOpen}
                      onToggleLogs={() =>
                        setLogVisibility((prev) => ({
                          ...prev,
                          [job.jobKey]: !isLogsOpen,
                        }))
                      }
                    />
                  );
                })}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

interface JobCardProps {
  job: JobState;
  showLogs: boolean;
  onToggleLogs: () => void;
}

function JobCard({ job, showLogs, onToggleLogs }: JobCardProps) {
  const duration = React.useMemo(
    () => formatDuration(job.startedAt, job.completedAt),
    [job.startedAt, job.completedAt],
  );
  const updatedRelative = React.useMemo(
    () => formatRelativeTime(job.updatedAt),
    [job.updatedAt],
  );
  const logEntries = React.useMemo(
    () => job.logs.slice(-40),
    [job.logs],
  );

  return (
    <article className="rounded-lg border border-border/80 bg-muted/20 p-3 shadow-sm">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0 flex-1 space-y-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-sm font-semibold leading-tight">
              {job.label}
            </span>
            {job.subtitle ? (
              <span className="truncate text-xs text-muted-foreground">
                {job.subtitle}
              </span>
            ) : null}
            <StatusBadge status={job.status} />
          </div>
          <div className="flex flex-wrap items-center gap-3 text-xs text-muted-foreground">
            {job.startedAt ? (
              <span className="inline-flex items-center gap-1">
                <Clock className="h-3 w-3" aria-hidden />
                Started {formatTime(job.startedAt)}
              </span>
            ) : null}
            {duration ? (
              <span className="inline-flex items-center gap-1">
                <Radio className="h-3 w-3" aria-hidden />
                {duration}
              </span>
            ) : null}
            {updatedRelative ? (
              <span className="inline-flex items-center gap-1">
                <NotebookPen className="h-3 w-3" aria-hidden />
                Updated {updatedRelative}
              </span>
            ) : null}
          </div>
        </div>
        <button
          type="button"
          className="inline-flex items-center gap-1 rounded border border-border/70 px-2 py-1 text-xs font-medium transition hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
          onClick={onToggleLogs}
          aria-expanded={showLogs}
          aria-label={showLogs ? "Hide logs" : "Show logs"}
        >
          {showLogs ? (
            <ChevronDown className="h-3 w-3" aria-hidden />
          ) : (
            <ChevronRight className="h-3 w-3" aria-hidden />
          )}
          Logs
        </button>
      </div>

      <div className="mt-3 space-y-3">
        {job.progress ? <JobProgress progress={job.progress} /> : null}
        {Array.isArray(job.stages) && job.stages.length ? (
          <JobStages stages={job.stages} />
        ) : null}
        {job.lastMessage ? (
          <p className="text-xs text-muted-foreground">{job.lastMessage}</p>
        ) : null}
      </div>

      {showLogs ? (
        <div className="mt-3 rounded-md border border-border/70 bg-background/80 p-2">
          {logEntries.length ? (
            <div className="max-h-48 space-y-1 overflow-y-auto font-mono text-[11px]">
              {logEntries.map((entry, index) => (
                <div
                  key={`${entry.timestamp}-${index}`}
                  className={cn(
                    "flex items-start gap-2 rounded px-2 py-1",
                    logLevelClass(entry.level),
                  )}
                >
                  <span className="shrink-0 text-muted-foreground">
                    {formatLogTimestamp(entry.timestamp)}
                  </span>
                  <span className="shrink-0 uppercase text-[10px] font-semibold text-muted-foreground">
                    {entry.phase}
                  </span>
                  <span className="flex-1 break-words">{entry.message}</span>
                  {entry.details ? (
                    <span className="shrink-0 text-muted-foreground">
                      {formatDetails(entry.details)}
                    </span>
                  ) : null}
                </div>
              ))}
            </div>
          ) : (
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <FileText className="h-3.5 w-3.5" aria-hidden />
              Waiting for log entries…
            </div>
          )}
        </div>
      ) : null}
    </article>
  );
}

function JobProgress({
  progress,
}: {
  progress: NonNullable<JobState["progress"]>;
}) {
  return (
    <section className="space-y-1">
      <div className="flex items-center justify-between text-xs">
        <span className="font-medium text-muted-foreground">Progress</span>
        <span className="text-muted-foreground">{progress.percent}%</span>
      </div>
      <div className="h-2 w-full rounded-full bg-muted">
        <div
          className="h-2 rounded-full bg-primary transition-all"
          style={{ width: `${Math.max(3, progress.percent)}%` }}
        />
      </div>
      {progress.message ? (
        <p className="text-xs text-muted-foreground">
          {progress.message}
        </p>
      ) : null}
    </section>
  );
}

function JobStages({ stages }: { stages: JobStageState[] }) {
  return (
    <section className="flex flex-wrap gap-2 text-xs">
      {stages.map((stage) => (
        <span
          key={stage.id}
          className={cn(
            "inline-flex items-center gap-1 rounded-full px-2 py-1 font-medium",
            stageStateClass(stage.state),
          )}
        >
          <span className="h-2 w-2 rounded-full bg-current" aria-hidden />
          {stage.label}
        </span>
      ))}
    </section>
  );
}

function StatusBadge({ status }: { status: JobState["status"] }) {
  const { label, className } = React.useMemo(() => {
    switch (status) {
      case "running":
        return {
          label: "Running",
          className:
            "bg-blue-100 text-blue-700 dark:bg-blue-900/50 dark:text-blue-200",
        };
      case "pending":
        return {
          label: "Pending",
          className:
            "bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-200",
        };
      case "success":
        return {
          label: "Complete",
          className:
            "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-200",
        };
      case "error":
      default:
        return {
          label: "Failed",
          className:
            "bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-200",
        };
    }
  }, [status]);

  return (
    <span className={cn("rounded-full px-2 py-0.5 text-xs font-semibold", className)}>
      {label}
    </span>
  );
}

function stageStateClass(state: JobStageState["state"]) {
  switch (state) {
    case "current":
      return "bg-primary/20 text-primary";
    case "complete":
      return "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-200";
    case "error":
      return "bg-destructive/20 text-destructive-foreground";
    case "pending":
    default:
      return "bg-muted text-muted-foreground";
  }
}

function logLevelClass(level: ProcessLogEvent["level"]) {
  switch (level) {
    case "error":
      return "bg-red-500/5 text-red-500 dark:text-red-300";
    case "warning":
      return "bg-amber-500/5 text-amber-600 dark:text-amber-300";
    case "info":
      return "bg-blue-500/5 text-blue-600 dark:text-blue-300";
    case "debug":
    default:
      return "bg-muted/40 text-muted-foreground";
  }
}

function formatTime(timestamp?: string) {
  if (!timestamp) {
    return "";
  }
  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) {
    return "";
  }
  return date.toLocaleTimeString(undefined, {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function formatLogTimestamp(timestamp: string) {
  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) {
    return timestamp;
  }
  return date.toLocaleTimeString(undefined, {
    hour12: false,
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function formatRelativeTime(timestamp?: string) {
  if (!timestamp) {
    return "";
  }
  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) {
    return "";
  }
  const diffMs = Date.now() - date.getTime();
  const seconds = Math.max(0, Math.round(diffMs / 1000));
  if (seconds < 60) {
    return `${seconds}s ago`;
  }
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) {
    return `${minutes}m ago`;
  }
  const hours = Math.round(minutes / 60);
  return `${hours}h ago`;
}

function formatDuration(start?: string, end?: string) {
  if (!start) {
    return "";
  }
  const startDate = new Date(start);
  const endDate = end ? new Date(end) : new Date();
  if (
    Number.isNaN(startDate.getTime()) ||
    Number.isNaN(endDate.getTime()) ||
    endDate.getTime() < startDate.getTime()
  ) {
    return "";
  }
  const diffSeconds = Math.round(
    (endDate.getTime() - startDate.getTime()) / 1000,
  );
  const hours = Math.floor(diffSeconds / 3600);
  const minutes = Math.floor((diffSeconds % 3600) / 60);
  const seconds = diffSeconds % 60;
  const parts: string[] = [];
  if (hours > 0) {
    parts.push(`${hours}h`);
  }
  if (minutes > 0 || hours) {
    parts.push(`${minutes}m`);
  }
  parts.push(`${seconds}s`);
  return parts.join(" ");
}

function formatDetails(details: Record<string, unknown>) {
  const entries = Object.entries(details);
  if (!entries.length) {
    return "";
  }
  return entries
    .map(([key, value]) => `${key}: ${String(value)}`)
    .join(" • ");
}
