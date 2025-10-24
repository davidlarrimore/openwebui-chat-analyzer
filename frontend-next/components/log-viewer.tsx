"use client";

import * as React from "react";
import { cn } from "@/lib/utils";
import { getProcessLogs, type ProcessLogEvent } from "@/lib/api";

interface LogViewerProps {
  className?: string;
  jobId?: string;
  pollInterval?: number; // milliseconds between polls (default 2000)
  maxLogs?: number; // maximum number of logs to display (default 200)
  supplementalLogs?: ProcessLogEvent[];
}

export function LogViewer({
  className,
  jobId,
  pollInterval = 2000,
  maxLogs = 200,
  supplementalLogs = [],
}: LogViewerProps) {
  const [logs, setLogs] = React.useState<ProcessLogEvent[]>([]);
  const [isAutoScroll, setIsAutoScroll] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);
  const scrollRef = React.useRef<HTMLDivElement>(null);
  const lastLogCountRef = React.useRef(0);

  // Fetch logs on mount and at regular intervals
  React.useEffect(() => {
    const fetchLogs = async () => {
      try {
        const response = await getProcessLogs(jobId, maxLogs);
        setLogs(response.logs);
        setError(null);
      } catch (err) {
        console.error("Failed to fetch logs:", err);
        setError(err instanceof Error ? err.message : "Failed to fetch logs");
      }
    };

    // Initial fetch
    fetchLogs();

    // Set up polling
    const intervalId = setInterval(fetchLogs, pollInterval);

    return () => clearInterval(intervalId);
  }, [jobId, maxLogs, pollInterval]);

  const combinedLogs = React.useMemo(() => {
    if (!supplementalLogs.length) {
      return logs;
    }
    const merged = [...logs, ...supplementalLogs];
    merged.sort((a, b) => {
      const timeA = new Date(a.timestamp).getTime();
      const timeB = new Date(b.timestamp).getTime();
      return timeA - timeB;
    });
    return merged;
  }, [logs, supplementalLogs]);

  // Auto-scroll to bottom when new logs arrive
  React.useEffect(() => {
    if (isAutoScroll && combinedLogs.length > lastLogCountRef.current) {
      scrollRef.current?.scrollTo({
        top: scrollRef.current.scrollHeight,
        behavior: "smooth",
      });
    }
    lastLogCountRef.current = combinedLogs.length;
  }, [combinedLogs, isAutoScroll]);

  // Detect user scroll to disable auto-scroll
  const handleScroll = React.useCallback(() => {
    if (!scrollRef.current) return;

    const { scrollTop, scrollHeight, clientHeight } = scrollRef.current;
    const isAtBottom = scrollHeight - scrollTop - clientHeight < 50;

    if (isAutoScroll !== isAtBottom) {
      setIsAutoScroll(isAtBottom);
    }
  }, [isAutoScroll]);

  const getLevelColor = (level: ProcessLogEvent["level"]) => {
    switch (level) {
      case "error":
        return "text-red-600 dark:text-red-400";
      case "warning":
        return "text-yellow-600 dark:text-yellow-400";
      case "info":
        return "text-blue-600 dark:text-blue-400";
      case "debug":
        return "text-gray-500 dark:text-gray-400";
      default:
        return "text-gray-700 dark:text-gray-300";
    }
  };

  const getPhaseIcon = (phase: ProcessLogEvent["phase"]) => {
    switch (phase) {
      case "connect":
        return "🔌";
      case "fetch":
        return "📥";
      case "persist":
        return "💾";
      case "summarize":
        return "📝";
      case "done":
        return "✅";
      case "error":
        return "❌";
      default:
        return "•";
    }
  };

  const formatTimestamp = (timestamp: string) => {
    try {
      const date = new Date(timestamp);
      return date.toLocaleTimeString("en-US", {
        hour12: false,
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
        fractionalSecondDigits: 3,
      });
    } catch {
      return timestamp;
    }
  };

  return (
    <div className={cn("flex flex-col h-full", className)}>
      {/* Header with controls */}
      <div className="flex items-center justify-between mb-2 pb-2 border-b">
        <div className="text-xs font-medium text-muted-foreground">
          Process Logs {jobId && <span className="ml-2">Job: {jobId}</span>}
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setIsAutoScroll(!isAutoScroll)}
            className={cn(
              "text-xs px-2 py-1 rounded",
              isAutoScroll
                ? "bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-200"
                : "bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300"
            )}
            title={isAutoScroll ? "Auto-scroll enabled" : "Auto-scroll disabled"}
          >
            {isAutoScroll ? "Auto-scroll: ON" : "Auto-scroll: OFF"}
          </button>
          <div className="text-xs text-muted-foreground">
            {combinedLogs.length} {combinedLogs.length === 1 ? "entry" : "entries"}
          </div>
        </div>
      </div>

      {/* Log display area */}
      <div
        ref={scrollRef}
        onScroll={handleScroll}
        className="flex-1 overflow-y-auto font-mono text-xs bg-muted/30 rounded p-3 space-y-1"
      >
        {error && (
          <div className="text-red-600 dark:text-red-400 mb-2">
            Error: {error}
          </div>
        )}

        {combinedLogs.length === 0 && !error && (
          <div className="text-muted-foreground">
            No log entries yet. Logs will appear here as operations run...
          </div>
        )}

        {combinedLogs.map((log, index) => (
          <div
            key={`${log.timestamp}-${index}`}
            className={cn(
              "flex items-start gap-2 p-1 rounded hover:bg-muted/50",
              getLevelColor(log.level)
            )}
          >
            <span className="flex-shrink-0 w-[90px] text-muted-foreground">
              {formatTimestamp(log.timestamp)}
            </span>
            <span className="flex-shrink-0 w-4" title={log.phase}>
              {getPhaseIcon(log.phase)}
            </span>
            <span className="flex-shrink-0 w-[60px] uppercase text-[10px] font-semibold">
              {log.level}
            </span>
            <span className="flex-1 break-words">{log.message}</span>
            {log.details && (
              <span className="flex-shrink-0 text-muted-foreground text-[10px]">
                {JSON.stringify(log.details)}
              </span>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
