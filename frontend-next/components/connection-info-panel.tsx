"use client";

import * as React from "react";
import { useSummarizerProgress } from "@/components/summarizer-progress-provider";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { cn } from "@/lib/utils";
import {
  getOpenWebUISettings,
  updateOpenWebUISettings,
  testOpenWebUIConnection,
  getSyncStatus,
  getSyncScheduler,
  updateSyncScheduler,
  runSync,
  rebuildSummaries,
  getSummaryStatus,
  getSummaryEvents,
  type OpenWebUISettingsResponse,
  type SyncStatusResponse,
  type SyncSchedulerConfig,
  type ProcessLogEvent,
} from "@/lib/api";
import { toast } from "@/components/ui/use-toast";
import { LogViewer } from "@/components/log-viewer";
import type { SummaryEvent, SummaryStatus } from "@/lib/types";

const SUMMARY_POLL_INTERVAL_MS = 2000;
const TERMINAL_SUMMARY_STATES = new Set(["idle", "completed", "failed", "cancelled"]);
const SUMMARY_WARNING_EVENT_TYPES = new Set(["invalid_chat", "empty_context"]);

function isTerminalSummaryState(state: string | null | undefined): boolean {
  if (state === null || state === undefined) {
    return true;
  }
  return TERMINAL_SUMMARY_STATES.has(state);
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

function formatSummaryEventDetails(event: SummaryEvent): string {
  const type = typeof event.type === "string" ? event.type : "";
  if (type === "chat") {
    const chatId = typeof event.chat_id === "string" ? event.chat_id : "unknown";
    const outcome = typeof event.outcome === "string" ? event.outcome : "";
    if (outcome === "generated" || outcome === "success") {
      return `Generated summary for chat ${chatId}`;
    }
    if (outcome === "failed") {
      return `Failed to summarize chat ${chatId}`;
    }
    if (outcome === "skipped") {
      return `Skipped chat ${chatId}`;
    }
    return `Processed chat ${chatId}`;
  }
  if (type === "chunk") {
    const chatId = typeof event.chat_id === "string" ? event.chat_id : "unknown";
    const chunkIndex = typeof event.chunk_index === "number" ? event.chunk_index : Number(event.chunk_index ?? NaN);
    const chunkCount = typeof event.chunk_count === "number" ? event.chunk_count : Number(event.chunk_count ?? NaN);
    if (Number.isFinite(chunkIndex) && Number.isFinite(chunkCount) && chunkCount > 0) {
      return `Summarizing chunk ${chunkIndex}/${chunkCount} for chat ${chatId}`;
    }
    return `Summarizing chat ${chatId}`;
  }
  if (type === "skip") {
    const chatId = typeof event.chat_id === "string" ? event.chat_id : "unknown";
    return `Skipped existing summary for chat ${chatId}`;
  }
  if (type === "invalid_chat") {
    return "Skipped chat without a valid chat_id";
  }
  if (type === "empty_context") {
    const chatId = typeof event.chat_id === "string" ? event.chat_id : "unknown";
    return `No summarizable content for chat ${chatId}`;
  }
  if (type === "start") {
    return "Summarizer job started";
  }
  if (type === "complete") {
    return "Summarizer job complete";
  }
  if (type === "error") {
    return "Summarizer job error";
  }
  return "";
}

function determineSummaryEventLevel(event: SummaryEvent): ProcessLogEvent["level"] {
  const type = typeof event.type === "string" ? event.type : "";
  const outcome = typeof event.outcome === "string" ? event.outcome : "";
  if (type === "error" || outcome === "failed") {
    return "error";
  }
  if (SUMMARY_WARNING_EVENT_TYPES.has(type) || outcome === "skipped") {
    return "warning";
  }
  return "info";
}

function convertSummaryEventToLog(event: SummaryEvent): ProcessLogEvent | null {
  const messageRaw = typeof event.message === "string" ? event.message.trim() : "";
  const fallback = formatSummaryEventDetails(event);
  const message = messageRaw || fallback;
  if (!message) {
    return null;
  }
  const timestamp =
    typeof event.timestamp === "string" && event.timestamp.trim().length
      ? event.timestamp
      : new Date().toISOString();
  const log: ProcessLogEvent = {
    timestamp,
    level: determineSummaryEventLevel(event),
    job_id: event.job_id !== null && event.job_id !== undefined ? String(event.job_id) : null,
    phase: "summarize",
    message,
  };

  const details: Record<string, unknown> = {};
  if (event.event_id) details.event_id = event.event_id;
  if (event.type) details.type = event.type;
  if (event.outcome) details.outcome = event.outcome;
  if (Object.keys(details).length > 0) {
    log.details = details;
  }

  return log;
}

interface ConnectionInfoState {
  hostname: string;
  apiKey: string;
  mode: "full" | "incremental";
  lastSync: string | null;
  datasetLoaded: boolean;
  isStale: boolean;
  hostSource?: "database" | "environment" | "default";
  apiKeySource?: "database" | "environment" | "empty";
  isLoading: boolean;
  isSaving: boolean;
  isRebuildingSummaries: boolean;
  isEditMode: boolean;
  isTesting: boolean;
  error: string | null;
  schedulerDrawerOpen: boolean;
  schedulerEnabled: boolean;
  schedulerInterval: number;
}

interface ConnectionInfoPanelProps {
  className?: string;
  initialSettings?: OpenWebUISettingsResponse;
}

export function ConnectionInfoPanel({ className, initialSettings }: ConnectionInfoPanelProps) {
  const [state, setState] = React.useState<ConnectionInfoState>({
    hostname: initialSettings?.host || "",
    apiKey: initialSettings?.api_key || "",
    mode: "full",
    lastSync: null,
    datasetLoaded: false,
    isStale: false,
    hostSource: initialSettings?.host_source,
    apiKeySource: initialSettings?.api_key_source,
    isLoading: false,
    isSaving: false,
    isRebuildingSummaries: false,
    isEditMode: false,
    isTesting: false,
    error: null,
    schedulerDrawerOpen: false,
    schedulerEnabled: false,
    schedulerInterval: 60,
  });

  // Track original values to detect changes
  const [originalValues, setOriginalValues] = React.useState({
    hostname: initialSettings?.host || "",
    apiKey: initialSettings?.api_key || "",
  });

  const [summaryLogs, setSummaryLogs] = React.useState<ProcessLogEvent[]>([]);
  const summaryEventIdsRef = React.useRef<Set<string>>(new Set());
  const lastSummaryEventIdRef = React.useRef<string | null>(null);
  const activeSummaryJobIdRef = React.useRef<string | null>(null);

  const { updateStatus: setGlobalSummarizerStatus } = useSummarizerProgress();

  const syncGlobalSummarizerStatus = React.useCallback(
    (status: SummaryStatus | null, latestMessage?: string | null) => {
      if (!status) {
        setGlobalSummarizerStatus(null);
        return;
      }

      const total = Math.max(0, Math.trunc(status.total ?? 0));
      const completed = Math.max(0, Math.trunc(status.completed ?? 0));
      const rawMessage =
        typeof latestMessage === "string" && latestMessage.trim().length
          ? latestMessage.trim()
          : typeof status.message === "string"
            ? status.message.trim()
            : "";
      const message = rawMessage || null;

      if (
        status.state === "running" ||
        status.state === "completed" ||
        status.state === "failed" ||
        status.state === "cancelled"
      ) {
        setGlobalSummarizerStatus({
          state: status.state ?? null,
          total,
          completed,
          message,
        });
        return;
      }

      setGlobalSummarizerStatus(null);
    },
    [setGlobalSummarizerStatus]
  );

  const appendManualSummaryLog = React.useCallback(
    (message: string, level: ProcessLogEvent["level"] = "info") => {
      setSummaryLogs(prev => {
        const nextEntry: ProcessLogEvent = {
          timestamp: new Date().toISOString(),
          level,
          job_id: null,
          phase: "summarize",
          message,
        };
        const next = [...prev, nextEntry];
        return next.length > 200 ? next.slice(-200) : next;
      });
    },
    []
  );

  const resetSummaryLogState = React.useCallback(() => {
    summaryEventIdsRef.current = new Set();
    lastSummaryEventIdRef.current = null;
    activeSummaryJobIdRef.current = null;
    setSummaryLogs([]);
  }, []);

  const appendSummaryEventsToLogs = React.useCallback((events: SummaryEvent[]) => {
    if (!events.length) {
      return;
    }
    setSummaryLogs(prev => {
      const next = [...prev];
      let changed = false;
      for (const event of events) {
        const eventJobId =
          event.job_id !== null && event.job_id !== undefined ? String(event.job_id) : null;
        const activeJobId = activeSummaryJobIdRef.current;
        if (activeJobId) {
          if (!eventJobId || eventJobId !== activeJobId) {
            continue;
          }
        } else if (eventJobId) {
          activeSummaryJobIdRef.current = eventJobId;
        }

        const eventId =
          typeof event.event_id === "string" && event.event_id.trim().length ? event.event_id.trim() : null;
        if (eventId) {
          if (summaryEventIdsRef.current.has(eventId)) {
            continue;
          }
          summaryEventIdsRef.current.add(eventId);
        }
        const logEntry = convertSummaryEventToLog(event);
        if (!logEntry) {
          continue;
        }
        next.push(logEntry);
        changed = true;
      }
      if (!changed) {
        return prev;
      }
      return next.length > 200 ? next.slice(-200) : next;
    });
  }, []);

  const fetchSummaryEventsFeed = React.useCallback(async () => {
    try {
      const after = lastSummaryEventIdRef.current ?? undefined;
      const response = await getSummaryEvents(after);
      if (response.reset) {
        summaryEventIdsRef.current = new Set();
        lastSummaryEventIdRef.current = null;
      }
      const events = Array.isArray(response.events) ? response.events : [];
      if (events.length) {
        appendSummaryEventsToLogs(events);
        const lastEvent = events[events.length - 1];
        const lastId =
          typeof lastEvent?.event_id === "string" && lastEvent.event_id.trim().length
            ? lastEvent.event_id.trim()
            : null;
        if (lastId) {
          lastSummaryEventIdRef.current = lastId;
        }
      }
    } catch (err) {
      console.error("Failed to fetch summary events:", err);
    }
  }, [appendSummaryEventsToLogs]);

  // Fetch settings, sync status, and scheduler config on mount
  React.useEffect(() => {
    if (!initialSettings) {
      fetchSettings();
    }
    fetchSyncStatus();
    fetchSchedulerConfig();
  }, [initialSettings]);

  const fetchSyncStatus = async () => {
    try {
      const syncStatus = await getSyncStatus();
      setState(prev => ({
        ...prev,
        lastSync: syncStatus.last_sync_at,
        mode: syncStatus.recommended_mode,
        isStale: syncStatus.is_stale,
      }));
    } catch (err) {
      console.error("Failed to fetch sync status:", err);
    }
  };

  const fetchSchedulerConfig = async () => {
    try {
      const config = await getSyncScheduler();
      setState(prev => ({
        ...prev,
        schedulerEnabled: config.enabled,
        schedulerInterval: config.interval_minutes,
      }));
    } catch (err) {
      console.error("Failed to fetch scheduler config:", err);
    }
  };

  const fetchSettings = async () => {
    setState(prev => ({ ...prev, isLoading: true, error: null }));
    try {
      const settings = await getOpenWebUISettings();
      setState(prev => ({
        ...prev,
        hostname: settings.host,
        apiKey: settings.api_key,
        hostSource: settings.host_source,
        apiKeySource: settings.api_key_source,
        isLoading: false,
      }));
      setOriginalValues({
        hostname: settings.host,
        apiKey: settings.api_key,
      });
    } catch (err) {
      setState(prev => ({
        ...prev,
        isLoading: false,
        error: err instanceof Error ? err.message : "Failed to fetch settings",
      }));
    }
  };

  const handleSaveSettings = async () => {
    setState(prev => ({ ...prev, isSaving: true, error: null }));
    try {
      const updates: { host?: string; api_key?: string } = {};

      // Only send values that changed
      if (state.hostname !== originalValues.hostname) {
        updates.host = state.hostname;
      }
      if (state.apiKey !== originalValues.apiKey) {
        updates.api_key = state.apiKey;
      }

      if (Object.keys(updates).length > 0) {
        const updatedSettings = await updateOpenWebUISettings(updates);
        setState(prev => ({
          ...prev,
          hostname: updatedSettings.host,
          apiKey: updatedSettings.api_key,
          hostSource: updatedSettings.host_source,
          apiKeySource: updatedSettings.api_key_source,
          isSaving: false,
          isEditMode: false, // Exit edit mode on successful save
        }));
        setOriginalValues({
          hostname: updatedSettings.host,
          apiKey: updatedSettings.api_key,
        });
        // TODO: Show success toast notification
      } else {
        setState(prev => ({ ...prev, isSaving: false, isEditMode: false }));
      }
    } catch (err) {
      setState(prev => ({
        ...prev,
        isSaving: false,
        error: err instanceof Error ? err.message : "Failed to save settings",
      }));
    }
  };

  const handleCancelEdit = () => {
    setState(prev => ({
      ...prev,
      hostname: originalValues.hostname,
      apiKey: originalValues.apiKey,
      error: null,
      isEditMode: false, // Exit edit mode on cancel
    }));
  };

  const hasChanges = state.hostname !== originalValues.hostname || state.apiKey !== originalValues.apiKey;

  const handleTestConnection = async () => {
    setState(prev => ({ ...prev, isTesting: true }));

    try {
      // If in edit mode, test the current form values
      // Otherwise, test the stored settings
      const testRequest = state.isEditMode
        ? { host: state.hostname, api_key: state.apiKey }
        : undefined;

      const result = await testOpenWebUIConnection(testRequest);

      if (result.status === "ok") {
        // Success toast
        const details: string[] = [];
        if (result.meta?.version) {
          details.push(`Version: ${result.meta.version}`);
        }
        if (result.meta?.chat_count !== undefined) {
          details.push(`${result.meta.chat_count} chats found`);
        }
        if (result.attempts > 1) {
          details.push(`Connected after ${result.attempts} attempts`);
        }

        toast({
          title: "Connection Successful",
          description: details.length > 0 ? details.join(" • ") : "Successfully connected to OpenWebUI",
          variant: "default",
          duration: 5000,
        });
      } else {
        // Error toast
        toast({
          title: "Connection Failed",
          description: result.detail || "Could not connect to OpenWebUI instance",
          variant: "destructive",
          duration: 7000,
        });
      }
    } catch (err) {
      toast({
        title: "Connection Test Failed",
        description: err instanceof Error ? err.message : "An unexpected error occurred",
        variant: "destructive",
        duration: 7000,
      });
    } finally {
      setState(prev => ({ ...prev, isTesting: false }));
    }
  };

  const handleLoadData = async () => {
    setState(prev => ({ ...prev, isLoading: true, error: null }));

    try {
      const result = await runSync({
        hostname: state.hostname,
        api_key: state.apiKey,
        mode: state.mode,
      });

      // Refresh sync status after successful sync
      await fetchSyncStatus();

      toast({
        title: "Data Sync Complete",
        description: result.detail || `Synced data successfully in ${state.mode} mode`,
        variant: "default",
        duration: 5000,
      });

      setState(prev => ({ ...prev, isLoading: false, datasetLoaded: true }));
    } catch (err) {
      setState(prev => ({
        ...prev,
        isLoading: false,
        error: err instanceof Error ? err.message : "Failed to sync data",
      }));

      toast({
        title: "Data Sync Failed",
        description: err instanceof Error ? err.message : "An unexpected error occurred",
        variant: "destructive",
        duration: 7000,
      });
    }
  };

  const handleRerunSummaries = React.useCallback(async () => {
    if (state.isRebuildingSummaries) {
      return;
    }

    setState(prev => ({ ...prev, isRebuildingSummaries: true, error: null }));
    resetSummaryLogState();
    appendManualSummaryLog("Queuing manual summary rebuild...");

    try {
      let currentStatus = await rebuildSummaries();
      syncGlobalSummarizerStatus(currentStatus);
      if (currentStatus.message) {
        appendManualSummaryLog(currentStatus.message);
      }

      const lastEvent = currentStatus.last_event;
      const rawJobId =
        lastEvent && typeof lastEvent === "object" ? (lastEvent as Record<string, unknown>).job_id : undefined;
      const initialJobId =
        rawJobId !== undefined && rawJobId !== null && `${rawJobId}`.trim().length ? String(rawJobId) : null;
      if (initialJobId) {
        activeSummaryJobIdRef.current = initialJobId;
      }

      await fetchSummaryEventsFeed();

      while (!isTerminalSummaryState(currentStatus.state)) {
        await sleep(SUMMARY_POLL_INTERVAL_MS);
        await fetchSummaryEventsFeed();
        currentStatus = await getSummaryStatus();
        syncGlobalSummarizerStatus(currentStatus);
      }

      await fetchSummaryEventsFeed();

      if (currentStatus.state === "completed") {
        appendManualSummaryLog("Summary job completed successfully.");
        toast({
          title: "Chat summaries rebuilt",
          description: "Dashboard analytics will refresh with the latest summaries.",
          variant: "default",
          duration: 5000,
        });
        setState(prev => ({ ...prev, datasetLoaded: true }));
        await fetchSyncStatus();
      } else if (currentStatus.state === "failed") {
        const message = currentStatus.message ?? "Summary job failed.";
        appendManualSummaryLog(message, "error");
        toast({
          title: "Summary job failed",
          description: message,
          variant: "destructive",
          duration: 7000,
        });
      } else if (currentStatus.state === "cancelled") {
        const message = currentStatus.message ?? "Summary job cancelled (dataset changed).";
        appendManualSummaryLog(message, "warning");
        toast({
          title: "Summary job cancelled",
          description: message,
          variant: "destructive",
          duration: 7000,
        });
      } else {
        const message = currentStatus.message ?? "Summaries still running in the background.";
        appendManualSummaryLog(message);
        toast({
          title: "Summaries running",
          description: message,
          variant: "default",
          duration: 5000,
        });
      }
    } catch (err) {
      console.error("Failed to rebuild summaries:", err);
      const message = err instanceof Error ? err.message : "Unable to rebuild summaries";
      appendManualSummaryLog(message, "error");
      toast({
        title: "Summary job failed",
        description: message,
        variant: "destructive",
        duration: 7000,
      });
      syncGlobalSummarizerStatus(null);
    } finally {
      setState(prev => ({ ...prev, isRebuildingSummaries: false }));
      activeSummaryJobIdRef.current = null;
    }
  }, [
    state.isRebuildingSummaries,
    appendManualSummaryLog,
    fetchSummaryEventsFeed,
    fetchSyncStatus,
    resetSummaryLogState,
    syncGlobalSummarizerStatus,
  ]);

  const handleEditDataSource = () => {
    setState(prev => ({ ...prev, isEditMode: true, error: null }));
  };

  const handleModeToggle = () => {
    setState((prev) => ({
      ...prev,
      mode: prev.mode === "full" ? "incremental" : "full",
    }));
  };

  return (
    <div className={cn("w-full space-y-6", className)}>
      {/* Header Bar */}
      <div className="flex items-center justify-between border-b bg-card p-4 rounded-lg shadow-sm">
        <h1 className="text-2xl font-bold">Connection Info</h1>

        <div className="flex items-center gap-3">
          {/* Dataset Loaded Pill */}
          {/* TODO: Connect to actual dataset state from backend */}
          <div
            className={cn(
              "rounded-full px-3 py-1 text-xs font-medium",
              state.datasetLoaded
                ? "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200"
                : "bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-200"
            )}
          >
            {state.datasetLoaded ? "Dataset Loaded" : "No Dataset"}
          </div>

          {/* Last Sync */}
          <div className="text-sm text-muted-foreground">
            Last Sync: {state.lastSync ? new Date(state.lastSync).toLocaleString() : "Never"}
          </div>

          {/* Staleness Pill */}
          {state.lastSync && (
            <div
              className={cn(
                "rounded-full px-3 py-1 text-xs font-medium",
                state.isStale
                  ? "bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-200"
                  : "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200"
              )}
            >
              {state.isStale ? "Stale" : "Up to date"}
            </div>
          )}

          {/* Mode Toggle */}
          <Button
            variant="outline"
            size="sm"
            onClick={handleModeToggle}
            className="min-w-[120px]"
            title={state.mode === "full" ? "Full sync replaces all data" : "Incremental sync adds new data"}
          >
            Mode: {state.mode === "full" ? "Full" : "Incremental"}
          </Button>

          {/* Test Connection Button */}
          <Button
            variant="outline"
            size="sm"
            onClick={handleTestConnection}
            disabled={state.isTesting || state.isSaving || state.isLoading}
          >
            {state.isTesting ? "Testing..." : "Test Connection"}
          </Button>

          {/* Load Data Button */}
          <Button
            variant="default"
            size="sm"
            onClick={handleLoadData}
            disabled={state.isLoading || state.isSaving || state.isTesting || state.isRebuildingSummaries}
          >
            {state.isLoading ? "Loading..." : "Load Data"}
          </Button>

          {/* Rerun Summaries Button */}
          <Button
            variant="outline"
            size="sm"
            onClick={handleRerunSummaries}
            disabled={state.isRebuildingSummaries || state.isLoading || state.isSaving || state.isTesting}
          >
            {state.isRebuildingSummaries ? "Rebuilding..." : "Rerun Summaries"}
          </Button>

          {/* Scheduler Settings Button */}
          <Button
            variant="outline"
            size="sm"
            onClick={() => setState(prev => ({ ...prev, schedulerDrawerOpen: true }))}
          >
            Scheduler
          </Button>
        </div>
      </div>

      {/* Scheduler Settings Drawer */}
      {state.schedulerDrawerOpen && (
        <div className="fixed inset-0 z-50 bg-black/50" onClick={() => setState(prev => ({ ...prev, schedulerDrawerOpen: false }))}>
          <div
            className="fixed right-0 top-0 bottom-0 w-96 bg-card shadow-xl p-6 overflow-y-auto"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-xl font-bold">Scheduler Settings</h2>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setState(prev => ({ ...prev, schedulerDrawerOpen: false }))}
              >
                ✕
              </Button>
            </div>

            <div className="space-y-6">
              {/* Enable/Disable Toggle */}
              <div className="flex items-center justify-between">
                <div>
                  <Label htmlFor="scheduler-enabled" className="text-base">Enable Scheduler</Label>
                  <p className="text-xs text-muted-foreground mt-1">
                    Automatically sync data at regular intervals
                  </p>
                </div>
                <Button
                  id="scheduler-enabled"
                  variant={state.schedulerEnabled ? "default" : "outline"}
                  size="sm"
                  onClick={async () => {
                    try {
                      const newEnabled = !state.schedulerEnabled;
                      await updateSyncScheduler({ enabled: newEnabled });
                      setState(prev => ({ ...prev, schedulerEnabled: newEnabled }));
                      toast({
                        title: newEnabled ? "Scheduler Enabled" : "Scheduler Disabled",
                        description: newEnabled
                          ? `Automatic syncs will run every ${state.schedulerInterval} minutes`
                          : "Automatic syncs have been disabled",
                        variant: "default",
                      });
                    } catch (err) {
                      toast({
                        title: "Update Failed",
                        description: err instanceof Error ? err.message : "Failed to update scheduler",
                        variant: "destructive",
                      });
                    }
                  }}
                >
                  {state.schedulerEnabled ? "ON" : "OFF"}
                </Button>
              </div>

              {/* Interval Selector */}
              <div className="space-y-2">
                <Label htmlFor="scheduler-interval">Sync Interval (minutes)</Label>
                <Input
                  id="scheduler-interval"
                  type="number"
                  min="5"
                  max="1440"
                  value={state.schedulerInterval}
                  onChange={(e) => setState(prev => ({ ...prev, schedulerInterval: parseInt(e.target.value) || 60 }))}
                  disabled={!state.schedulerEnabled}
                />
                <p className="text-xs text-muted-foreground">
                  Minimum: 5 minutes, Maximum: 1440 minutes (24 hours)
                </p>
              </div>

              {/* Save Interval Button */}
              <Button
                variant="default"
                className="w-full"
                disabled={!state.schedulerEnabled}
                onClick={async () => {
                  try {
                    await updateSyncScheduler({ interval_minutes: state.schedulerInterval });
                    toast({
                      title: "Interval Updated",
                      description: `Scheduler will now run every ${state.schedulerInterval} minutes`,
                      variant: "default",
                    });
                  } catch (err) {
                    toast({
                      title: "Update Failed",
                      description: err instanceof Error ? err.message : "Failed to update interval",
                      variant: "destructive",
                    });
                  }
                }}
              >
                Save Interval
              </Button>

              {/* Info */}
              <div className="bg-muted p-3 rounded text-sm space-y-1">
                <p><strong>Note:</strong> The scheduler runs automatic incremental syncs in the background.</p>
                <p className="text-muted-foreground">
                  You can still manually trigger syncs using the &quot;Load Data&quot; button.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Two Column Layout */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Left Column: Settings Form */}
        <Card>
          <CardHeader>
            <div className="flex w-full items-center gap-4">
              <CardTitle className="text-lg font-semibold">Data Source Settings</CardTitle>
              <Button
                variant="secondary"
                size="sm"
                className="ml-auto"
                onClick={handleEditDataSource}
                disabled={state.isEditMode}
              >
                Edit Data Source
              </Button>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            {state.isLoading ? (
              <div className="flex items-center justify-center p-8">
                <p className="text-sm text-muted-foreground">Loading settings...</p>
              </div>
            ) : (
              <>
                {state.error && (
                  <div className="rounded-md bg-destructive/10 p-3 text-sm text-destructive">
                    {state.error}
                  </div>
                )}

                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <Label htmlFor="hostname">Hostname</Label>
                    {state.hostSource && (
                      <span className="text-xs text-muted-foreground">
                        Source: {state.hostSource}
                      </span>
                    )}
                  </div>
                  <Input
                    id="hostname"
                    type="text"
                    placeholder="https://example.com"
                    value={state.hostname}
                    onChange={(e) =>
                      setState((prev) => ({ ...prev, hostname: e.target.value }))
                    }
                    disabled={!state.isEditMode || state.isSaving}
                  />
                  <p className="text-xs text-muted-foreground">
                    The base URL of your Open WebUI instance
                  </p>
                </div>

                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <Label htmlFor="apiKey">API Key</Label>
                    {state.apiKeySource && (
                      <span className="text-xs text-muted-foreground">
                        Source: {state.apiKeySource}
                      </span>
                    )}
                  </div>
                  <Input
                    id="apiKey"
                    type="password"
                    placeholder="sk-..."
                    value={state.apiKey}
                    onChange={(e) =>
                      setState((prev) => ({ ...prev, apiKey: e.target.value }))
                    }
                    disabled={!state.isEditMode || state.isSaving}
                  />
                  <p className="text-xs text-muted-foreground">
                    Your Open WebUI API key for authentication
                  </p>
                </div>

                {/* Show Save/Cancel buttons only in edit mode */}
                {state.isEditMode && (
                  <div className="flex gap-2 pt-4">
                    <Button
                      variant="default"
                      className="flex-1"
                      onClick={handleSaveSettings}
                      disabled={!hasChanges || state.isSaving}
                    >
                      {state.isSaving ? "Saving..." : "Save Settings"}
                    </Button>
                    <Button
                      variant="outline"
                      className="flex-1"
                      onClick={handleCancelEdit}
                      disabled={state.isSaving}
                    >
                      Cancel
                    </Button>
                  </div>
                )}
              </>
            )}
          </CardContent>
        </Card>

        {/* Right Column: Processing Log */}
        <Card className="flex flex-col">
          <CardHeader>
            <CardTitle>Processing Log</CardTitle>
          </CardHeader>
          <CardContent className="flex-1 flex flex-col min-h-[400px] max-h-[600px]">
            <LogViewer className="flex-1" pollInterval={2000} maxLogs={200} supplementalLogs={summaryLogs} />
          </CardContent>
        </Card>
      </div>

      {/* Additional sections can be added below */}
      {/* TODO: Add connection status indicators */}
      {/* TODO: Add data sync statistics */}
      {/* TODO: Add recent activity timeline */}
    </div>
  );
}
