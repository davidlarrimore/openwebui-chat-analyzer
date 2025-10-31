"use client";

import * as React from "react";
import { useRouter } from "next/navigation";
import { useSummarizerProgress } from "@/components/summarizer-progress-provider";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { cn } from "@/lib/utils";
import {
  apiGet,
  getOpenWebUISettings,
  updateOpenWebUISettings,
  getAnonymizationSettings,
  updateAnonymizationSettings,
  testOpenWebUIConnection,
  getSyncStatus,
  getSyncScheduler,
  updateSyncScheduler,
  runSync,
  rebuildSummaries,
  getSummaryStatus,
  getSummarizerSettings,
  updateSummarizerSettings,
  getAvailableOllamaModels,
  type OpenWebUISettingsResponse,
  type OpenWebUISettingsUpdate,
  type SyncStatusResponse,
  type SyncSchedulerConfig,
  type ProcessLogEvent,
} from "@/lib/api";
import { toast } from "@/components/ui/use-toast";
import type { SummaryStatus, DatasetMeta, OllamaModelTag } from "@/lib/types";

const SUMMARY_POLL_INTERVAL_MS = 2000;
const TERMINAL_SUMMARY_STATES = new Set(["idle", "completed", "failed", "cancelled"]);
function getTemperatureLabel(temperature: number): string {
  if (temperature === 0.0) return "Strict";
  if (temperature <= 0.2) return "Precise";
  if (temperature <= 0.5) return "Balanced";
  if (temperature <= 0.8) return "Creative";
  if (temperature <= 1.1) return "Inventive";
  if (temperature <= 1.5) return "Experimental";
  return "Max";
}

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

interface ConnectionInfoState {
  hostname: string;
  apiKey: string;
  mode: "full" | "incremental";
  lastSync: string | null;
  datasetLoaded: boolean;
  isStale: boolean;
  chatCount: number;
  userCount: number;
  modelCount: number;
  messageCount: number;
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
  schedulerNextRunAt: string | null;
  schedulerLastRunAt: string | null;
  schedulerCountdown: string;
  summaryModel: string;
  summaryModelSource?: "database" | "environment" | "default";
  summaryTemperature: number;
  summaryTemperatureSource?: "database" | "environment" | "default";
  summarizerEnabled: boolean;
  summarizerEnabledSource?: "database" | "environment" | "default";
  availableSummaryModels: string[];
  summaryModelError: string | null;
  isLoadingSummaryModels: boolean;
  isUpdatingSummaryModel: boolean;
  isUpdatingSummaryTemperature: boolean;
  isUpdatingSummarizerEnabled: boolean;
}

interface ConnectionInfoPanelProps {
  className?: string;
  initialSettings?: OpenWebUISettingsResponse;
}

export function ConnectionInfoPanel({ className, initialSettings }: ConnectionInfoPanelProps) {
  const router = useRouter();
  const [state, setState] = React.useState<ConnectionInfoState>({
    hostname: initialSettings?.host || "",
    apiKey: initialSettings?.api_key || "",
    mode: "incremental",
    lastSync: null,
    datasetLoaded: false,
    isStale: false,
    chatCount: 0,
    userCount: 0,
    modelCount: 0,
    messageCount: 0,
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
    schedulerNextRunAt: null,
    schedulerLastRunAt: null,
    schedulerCountdown: "",
    summaryModel: "",
    summaryModelSource: undefined,
    summaryTemperature: 0.2,
    summaryTemperatureSource: undefined,
    summarizerEnabled: true,
    summarizerEnabledSource: undefined,
    availableSummaryModels: [],
    summaryModelError: null,
    isLoadingSummaryModels: false,
    isUpdatingSummaryModel: false,
    isUpdatingSummaryTemperature: false,
    isUpdatingSummarizerEnabled: false,
  });

  // Track original values to detect changes
  const [originalValues, setOriginalValues] = React.useState({
    hostname: initialSettings?.host || "",
    apiKey: initialSettings?.api_key || "",
  });

  const [anonymization, setAnonymization] = React.useState({
    enabled: true,
    source: "default" as "default" | "database" | "environment",
    isSaving: false,
    error: null as string | null,
    confirmOpen: false,
  });

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

  const fetchSummarizerConfig = React.useCallback(async () => {
    setState(prev => ({
      ...prev,
      isLoadingSummaryModels: true,
      summaryModelError: null,
    }));
    try {
      const [modelsResponse, summarizerSettings] = await Promise.all([
        getAvailableOllamaModels(),
        getSummarizerSettings(),
      ]);

      const modelNames = Array.from(
        new Set(
          modelsResponse.models
            .filter((model) => model.supports_completions === true)
            .map((model) => (typeof model?.name === "string" ? model.name.trim() : ""))
            .filter((name): name is string => Boolean(name)),
        ),
      ).sort((a, b) => a.localeCompare(b));

      let selectedModel = (summarizerSettings.model || "").trim();
      if (!selectedModel && modelNames.length > 0) {
        selectedModel = modelNames[0];
      }

      const options = modelNames.slice();
      if (selectedModel && !options.includes(selectedModel)) {
        options.unshift(selectedModel);
      }

      setState(prev => ({
        ...prev,
        availableSummaryModels: options,
        summaryModel: selectedModel,
        summaryModelSource: summarizerSettings.model_source,
        summaryTemperature: summarizerSettings.temperature,
        summaryTemperatureSource: summarizerSettings.temperature_source,
        summarizerEnabled: summarizerSettings.enabled,
        summarizerEnabledSource: summarizerSettings.enabled_source,
        isLoadingSummaryModels: false,
        summaryModelError: null,
      }));
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to load summarizer settings";
      setState(prev => ({
        ...prev,
        isLoadingSummaryModels: false,
        summaryModelError: message,
        availableSummaryModels: [],
      }));
    }
  }, []);

  const fetchSettings = React.useCallback(async () => {
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
  }, []);

  const fetchAnonymizationSettings = React.useCallback(async () => {
    try {
      const result = await getAnonymizationSettings();
      setAnonymization(prev => ({
        ...prev,
        enabled: Boolean(result.enabled),
        source: result.source,
        error: null,
      }));
    } catch (err) {
      setAnonymization(prev => ({
        ...prev,
        error: err instanceof Error ? err.message : "Failed to fetch anonymization settings",
      }));
    }
  }, []);

  const fetchDatasetMeta = React.useCallback(async () => {
    try {
      const meta = await apiGet<DatasetMeta>("api/v1/datasets/meta");
      setState(prev => ({
        ...prev,
        chatCount: meta.chat_count,
        userCount: meta.user_count,
        modelCount: meta.model_count,
        messageCount: meta.message_count,
        datasetLoaded: meta.chat_count > 0 || meta.user_count > 0 || meta.model_count > 0,
      }));
    } catch (err) {
      console.error("Failed to fetch dataset metadata:", err);
      setState(prev => ({
        ...prev,
        chatCount: 0,
        userCount: 0,
        modelCount: 0,
        messageCount: 0,
        datasetLoaded: false,
      }));
    }
  }, []);

  const fetchSyncStatus = React.useCallback(async () => {
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
  }, []);

  const fetchSchedulerConfig = React.useCallback(async () => {
    try {
      const config = await getSyncScheduler();
      setState(prev => ({
        ...prev,
        schedulerEnabled: config.enabled,
        schedulerInterval: config.interval_minutes,
        schedulerNextRunAt: config.next_run_at,
        schedulerLastRunAt: config.last_run_at,
      }));
    } catch (err) {
      console.error("Failed to fetch scheduler config:", err);
    }
  }, []);

  // Poll scheduler config every 10 seconds to update next_run_at
  React.useEffect(() => {
    const interval = setInterval(() => {
      void fetchSchedulerConfig();
    }, 10000);
    return () => clearInterval(interval);
  }, [fetchSchedulerConfig]);

  // Update countdown timer every second
  React.useEffect(() => {
    const updateCountdown = () => {
      if (!state.schedulerEnabled || !state.schedulerNextRunAt) {
        setState(prev => ({ ...prev, schedulerCountdown: "" }));
        return;
      }

      const now = new Date();
      const nextRun = new Date(state.schedulerNextRunAt);
      const diffMs = nextRun.getTime() - now.getTime();

      if (diffMs <= 0) {
        setState(prev => ({ ...prev, schedulerCountdown: "Running soon..." }));
        return;
      }

      const minutes = Math.floor(diffMs / 60000);
      const seconds = Math.floor((diffMs % 60000) / 1000);

      if (minutes > 0) {
        setState(prev => ({ ...prev, schedulerCountdown: `${minutes}m ${seconds}s` }));
      } else {
        setState(prev => ({ ...prev, schedulerCountdown: `${seconds}s` }));
      }
    };

    updateCountdown();
    const interval = setInterval(updateCountdown, 1000);
    return () => clearInterval(interval);
  }, [state.schedulerEnabled, state.schedulerNextRunAt]);

  // Monitor for auto-sync starts via process logs
  React.useEffect(() => {
    let lastCheckTime = Date.now();

    const checkForAutoSync = async () => {
      try {
        const logs = await apiGet<ProcessLogEvent[]>("api/v1/sync/logs");

        // Look for recent "Automatic sync triggered" messages
        const recentAutoSyncLog = logs.find(log =>
          log.message?.includes("Automatic sync triggered by scheduler") &&
          new Date(log.timestamp).getTime() > lastCheckTime
        );

        if (recentAutoSyncLog) {
          toast({
            title: "Auto-Sync Started",
            description: `Scheduled data sync is now running`,
            variant: "default",
            duration: 4000,
          });
          lastCheckTime = Date.now();
        }
      } catch (err) {
        console.error("Failed to check process logs:", err);
      }
    };

    const interval = setInterval(checkForAutoSync, 5000);
    return () => clearInterval(interval);
  }, []);

  // Fetch settings, sync status, scheduler config, and dataset metadata on mount
  React.useEffect(() => {
    if (!initialSettings) {
      fetchSettings();
    }
    fetchAnonymizationSettings();
    fetchSyncStatus();
    fetchSchedulerConfig();
    fetchDatasetMeta();
    fetchSummarizerConfig();
  }, [
    initialSettings,
    fetchSettings,
    fetchAnonymizationSettings,
    fetchSyncStatus,
    fetchSchedulerConfig,
    fetchDatasetMeta,
    fetchSummarizerConfig,
  ]);

  const applyAnonymizationUpdate = async (enabled: boolean) => {
    setAnonymization(prev => ({ ...prev, isSaving: true, error: null, enabled }));
    try {
      const result = await updateAnonymizationSettings({ enabled });
      setAnonymization(prev => ({
        ...prev,
        enabled: Boolean(result.enabled),
        source: result.source,
        isSaving: false,
      }));

      // Refresh all pages to show updated user names
      toast({
        title: "Anonymization Updated",
        description: "Refreshing dashboard data...",
        variant: "default",
        duration: 3000,
      });
      router.refresh();
    } catch (err) {
      setAnonymization(prev => ({
        ...prev,
        enabled: !enabled,
        isSaving: false,
        error: err instanceof Error ? err.message : "Failed to update anonymization settings",
      }));
    }
  };

  const handleAnonymizationToggle = (nextValue: boolean) => {
    if (!nextValue) {
      setAnonymization(prev => ({ ...prev, confirmOpen: true }));
      return;
    }
    if (!anonymization.enabled) {
      void applyAnonymizationUpdate(true);
    }
  };

  const confirmDisableAnonymization = () => {
    setAnonymization(prev => ({ ...prev, confirmOpen: false }));
    void applyAnonymizationUpdate(false);
  };

  const cancelDisableAnonymization = () => {
    setAnonymization(prev => ({ ...prev, confirmOpen: false }));
  };

  const handleSaveSettings = async () => {
    setState(prev => ({ ...prev, isSaving: true, error: null }));
    try {
      const updates: OpenWebUISettingsUpdate = {};

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

  const summaryModelUnavailable =
    !!state.summaryModel &&
    !state.summaryModelError &&
    !state.isLoadingSummaryModels &&
    !state.availableSummaryModels.includes(state.summaryModel);

  const hasChanges =
    state.hostname !== originalValues.hostname ||
    state.apiKey !== originalValues.apiKey;

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

      // Refresh sync status and dataset metadata after successful sync
      await fetchSyncStatus();
      await fetchDatasetMeta();

      toast({
        title: "Data Sync Complete",
        description: result.detail || `Synced data successfully in ${state.mode} mode`,
        variant: "default",
        duration: 5000,
      });

      setState(prev => ({ ...prev, isLoading: false }));
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

  const handleSummaryModelChange = async (event: React.ChangeEvent<HTMLSelectElement>) => {
    const nextModel = event.target.value.trim();
    if (!nextModel || state.isUpdatingSummaryModel) {
      return;
    }

    const previousModel = state.summaryModel;
    setState(prev => ({
      ...prev,
      summaryModel: nextModel,
      isUpdatingSummaryModel: true,
      summaryModelError: null,
    }));

    try {
      const updated = await updateSummarizerSettings({ model: nextModel });
      setState(prev => {
        const options = prev.availableSummaryModels.includes(updated.model)
          ? prev.availableSummaryModels
          : [updated.model, ...prev.availableSummaryModels];
        return {
          ...prev,
          summaryModel: updated.model,
          summaryModelSource: updated.model_source,
          summaryTemperature: updated.temperature,
          summaryTemperatureSource: updated.temperature_source,
          availableSummaryModels: options,
          isUpdatingSummaryModel: false,
          summaryModelError: null,
        };
      });
      toast({
        title: "Summarizer model updated",
        description: `Summaries will now run with ${nextModel}`,
        variant: "default",
        duration: 5000,
      });
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to update summarizer model";
      setState(prev => ({
        ...prev,
        summaryModel: previousModel,
        isUpdatingSummaryModel: false,
        summaryModelError: message,
      }));
      toast({
        title: "Update failed",
        description: message,
        variant: "destructive",
        duration: 7000,
      });
    }
  };

  const handleTemperatureChange = async (newTemperature: number) => {
    if (state.isUpdatingSummaryTemperature) {
      return;
    }

    const previousTemperature = state.summaryTemperature;
    setState(prev => ({
      ...prev,
      summaryTemperature: newTemperature,
      isUpdatingSummaryTemperature: true,
    }));

    try {
      const updated = await updateSummarizerSettings({ temperature: newTemperature });
      setState(prev => ({
        ...prev,
        summaryTemperature: updated.temperature,
        summaryTemperatureSource: updated.temperature_source,
        isUpdatingSummaryTemperature: false,
      }));

      const tempLabel = getTemperatureLabel(newTemperature);
      toast({
        title: "Temperature updated",
        description: `Summarizer temperature set to ${newTemperature.toFixed(1)} (${tempLabel})`,
        variant: "default",
        duration: 3000,
      });
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to update temperature";
      setState(prev => ({
        ...prev,
        summaryTemperature: previousTemperature,
        isUpdatingSummaryTemperature: false,
      }));
      toast({
        title: "Update failed",
        description: message,
        variant: "destructive",
        duration: 5000,
      });
    }
  };

  const handleSummarizerEnabledChange = async (newEnabled: boolean) => {
    if (state.isUpdatingSummarizerEnabled) {
      return;
    }

    const previousEnabled = state.summarizerEnabled;
    setState(prev => ({
      ...prev,
      summarizerEnabled: newEnabled,
      isUpdatingSummarizerEnabled: true,
    }));

    try {
      const updated = await updateSummarizerSettings({ enabled: newEnabled });
      setState(prev => ({
        ...prev,
        summarizerEnabled: updated.enabled,
        summarizerEnabledSource: updated.enabled_source,
        isUpdatingSummarizerEnabled: false,
      }));

      toast({
        title: newEnabled ? "Summarizer enabled" : "Summarizer disabled",
        description: newEnabled
          ? "Chat summaries will be generated for new uploads"
          : "Chat summaries will not be generated until re-enabled",
        variant: "default",
        duration: 3000,
      });
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to update summarizer enabled state";
      setState(prev => ({
        ...prev,
        summarizerEnabled: previousEnabled,
        isUpdatingSummarizerEnabled: false,
      }));
      toast({
        title: "Update failed",
        description: message,
        variant: "destructive",
        duration: 5000,
      });
    }
  };

  const handleRerunSummaries = React.useCallback(async () => {
    if (state.isRebuildingSummaries) {
      return;
    }

    setState(prev => ({ ...prev, isRebuildingSummaries: true, error: null }));
    toast({
      title: "Queuing summary rebuild",
      description: "Summarizer job has been submitted.",
      variant: "default",
      duration: 3500,
    });

    try {
      let currentStatus = await rebuildSummaries();
      syncGlobalSummarizerStatus(currentStatus);

      while (!isTerminalSummaryState(currentStatus.state)) {
        await sleep(SUMMARY_POLL_INTERVAL_MS);
        currentStatus = await getSummaryStatus();
        syncGlobalSummarizerStatus(currentStatus);
      }

      if (currentStatus.state === "completed") {
        toast({
          title: "Chat summaries rebuilt",
          description: "Dashboard analytics will refresh with the latest summaries.",
          variant: "default",
          duration: 5000,
        });
        await fetchSyncStatus();
        await fetchDatasetMeta();
      } else if (currentStatus.state === "failed") {
        const message = currentStatus.message ?? "Summary job failed.";
        toast({
          title: "Summary job failed",
          description: message,
          variant: "destructive",
          duration: 7000,
        });
        setState(prev => ({ ...prev, error: message }));
      } else if (currentStatus.state === "cancelled") {
        const message = currentStatus.message ?? "Summary job cancelled (dataset changed).";
        toast({
          title: "Summary job cancelled",
          description: message,
          variant: "destructive",
          duration: 7000,
        });
      } else {
        const message = currentStatus.message ?? "Summaries still running in the background.";
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
      toast({
        title: "Summary rebuild failed",
        description: message,
        variant: "destructive",
        duration: 7000,
      });
      syncGlobalSummarizerStatus(null);
      setState(prev => ({ ...prev, error: message }));
    } finally {
      setState(prev => ({ ...prev, isRebuildingSummaries: false }));
    }
  }, [
    state.isRebuildingSummaries,
    syncGlobalSummarizerStatus,
    fetchSyncStatus,
    fetchDatasetMeta,
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
      {/* System Status Section */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <span className="text-lg">📊 System Status</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Connection Status */}
            <div className="space-y-2 p-4 border rounded-lg bg-muted/30">
              <div className="text-sm font-medium text-muted-foreground">Connection Status</div>
              <div className="flex items-center gap-2">
                <div className={cn(
                  "h-3 w-3 rounded-full",
                  state.hostname ? "bg-green-500" : "bg-red-500"
                )} />
                <span className="font-semibold">
                  {state.hostname ? "Configured" : "Not Configured"}
                </span>
              </div>
              {state.hostname && (
                <div className="text-xs text-muted-foreground truncate">
                  {state.hostname}
                </div>
              )}
            </div>

            {/* Dataset Status */}
            <div className="space-y-2 p-4 border rounded-lg bg-muted/30">
              <div className="text-sm font-medium text-muted-foreground">Dataset Status</div>
              <div className="space-y-1">
                <div
                  className={cn(
                    "inline-flex rounded-full px-3 py-1 text-xs font-semibold",
                    state.datasetLoaded
                      ? "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200"
                      : "bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-200"
                  )}
                >
                  {state.datasetLoaded ? "✓ Data Loaded" : "⚠ No Data Loaded"}
                </div>
                {state.datasetLoaded && (
                  <div className="text-xs text-muted-foreground space-y-0.5 mt-2">
                    <div>{state.chatCount.toLocaleString()} chats</div>
                    <div>{state.userCount.toLocaleString()} users</div>
                    <div>{state.modelCount.toLocaleString()} models</div>
                  </div>
                )}
              </div>
            </div>

            {/* Last Sync Status */}
            <div className="space-y-2 p-4 border rounded-lg bg-muted/30">
              <div className="text-sm font-medium text-muted-foreground">Last Sync</div>
              <div className="space-y-1">
                <div className="text-sm font-semibold">
                  {state.lastSync ? new Date(state.lastSync).toLocaleString() : "Never"}
                </div>
                {state.lastSync && (
                  <div
                    className={cn(
                      "inline-flex rounded-full px-2 py-0.5 text-xs font-medium",
                      state.isStale
                        ? "bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-200"
                        : "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200"
                    )}
                  >
                    {state.isStale ? "Stale" : "Current"}
                  </div>
                )}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Quick Actions Section */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <span className="text-lg">⚡ Quick Actions</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
            {/* Sync Data Now */}
            <Button
              variant="default"
              className="h-auto flex-col items-start p-4 gap-2"
              onClick={handleLoadData}
              disabled={state.isLoading || state.isSaving || state.isTesting || state.isRebuildingSummaries}
            >
              <div className="text-lg">🔄</div>
              <div className="text-left">
                <div className="font-semibold">{state.isLoading ? "Syncing..." : "Sync Data Now"}</div>
                <div className="text-xs opacity-90 font-normal">
                  Run {state.mode} sync immediately
                </div>
              </div>
            </Button>

            {/* Test Connection */}
            <Button
              variant="outline"
              className="h-auto flex-col items-start p-4 gap-2"
              onClick={handleTestConnection}
              disabled={state.isTesting || state.isSaving || state.isLoading}
            >
              <div className="text-lg">🔌</div>
              <div className="text-left">
                <div className="font-semibold">{state.isTesting ? "Testing..." : "Test Connection"}</div>
                <div className="text-xs text-muted-foreground font-normal">
                  Verify OpenWebUI access
                </div>
              </div>
            </Button>

            {/* Rebuild Summaries */}
            <Button
              variant="outline"
              className="h-auto flex-col items-start p-4 gap-2"
              onClick={handleRerunSummaries}
              disabled={!state.summarizerEnabled || state.isRebuildingSummaries || state.isLoading || state.isSaving || state.isTesting}
            >
              <div className="text-lg">🧠</div>
              <div className="text-left">
                <div className="font-semibold">
                  {state.isRebuildingSummaries ? "Rebuilding..." : "Rebuild Summaries"}
                </div>
                <div className="text-xs text-muted-foreground font-normal">
                  {!state.summarizerEnabled ? "Summarizer is disabled" : "Regenerate all AI summaries"}
                </div>
              </div>
            </Button>

            {/* Scheduler Settings */}
            <Button
              variant="outline"
              className="h-auto flex-col items-start p-4 gap-2"
              onClick={() => setState(prev => ({ ...prev, schedulerDrawerOpen: true }))}
            >
              <div className="text-lg">⏰</div>
              <div className="text-left">
                <div className="font-semibold">Scheduler</div>
                <div className="text-xs text-muted-foreground font-normal">
                  {state.schedulerEnabled ? (
                    state.schedulerCountdown ? (
                      `Next: ${state.schedulerCountdown}`
                    ) : (
                      `Active (${state.schedulerInterval}m)`
                    )
                  ) : (
                    "Disabled"
                  )}
                </div>
              </div>
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Summarizer Configuration */}
      <Card>
        <CardHeader>
          <div className="flex items-start justify-between gap-3">
            <div>
              <CardTitle className="flex items-center gap-2 text-lg">
                <span>🧠 Summarizer Settings</span>
              </CardTitle>
              <p className="text-sm text-muted-foreground">
                Configure the Ollama model and temperature for automated chat summaries.
              </p>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={fetchSummarizerConfig}
              disabled={state.isLoadingSummaryModels}
            >
              {state.isLoadingSummaryModels ? "Refreshing..." : "Refresh"}
            </Button>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Summarizer Enable/Disable Toggle */}
          <div className="flex items-center justify-between rounded-lg border p-3">
            <div className="flex-1 space-y-1">
              <Label htmlFor="summarizer-enabled" className="font-medium">
                Enable Summarizer
              </Label>
              <p className="text-xs text-muted-foreground">
                Generate summaries for chats during data loading
              </p>
            </div>
            <div className="flex items-center gap-2">
              {state.summarizerEnabledSource && (
                <span className="text-xs px-2 py-0.5 rounded-full bg-muted text-muted-foreground">
                  {state.summarizerEnabledSource}
                </span>
              )}
              <Switch
                id="summarizer-enabled"
                checked={state.summarizerEnabled}
                onCheckedChange={handleSummarizerEnabledChange}
                disabled={state.isUpdatingSummarizerEnabled}
              />
            </div>
          </div>

          {state.summaryModelError ? (
            <div className="flex items-start justify-between gap-3 rounded-md bg-destructive/10 p-3 text-sm text-destructive">
              <div className="flex-1">
                <span>⚠️ {state.summaryModelError}</span>
              </div>
              <Button
                variant="outline"
                size="sm"
                onClick={fetchSummarizerConfig}
                disabled={state.isLoadingSummaryModels}
              >
                Retry
              </Button>
            </div>
          ) : (
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label htmlFor="summarizer-model" className="font-medium">
                  Summarizer Model
                </Label>
                {state.summaryModelSource && (
                  <span className="text-xs px-2 py-0.5 rounded-full bg-muted text-muted-foreground">
                    {state.summaryModelSource}
                  </span>
                )}
              </div>
              <select
                id="summarizer-model"
                className={cn(
                  "w-full rounded-md border border-input bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-ring",
                  state.isUpdatingSummaryModel ? "opacity-75" : "",
                )}
                value={state.summaryModel}
                onChange={handleSummaryModelChange}
                disabled={
                  state.isLoadingSummaryModels ||
                  state.isUpdatingSummaryModel ||
                  state.availableSummaryModels.length === 0
                }
              >
                {state.availableSummaryModels.length === 0 ? (
                  <option value="">
                    {state.isLoadingSummaryModels ? "Loading models..." : "No models detected"}
                  </option>
                ) : (
                  state.availableSummaryModels.map(model => (
                    <option key={model} value={model}>
                      {model}
                    </option>
                  ))
                )}
              </select>
              <p className="text-xs text-muted-foreground">
                This model is used when rebuilding summaries or running manual summarizer jobs.
              </p>
              {summaryModelUnavailable && (
                <p className="text-xs text-destructive">
                  The configured model was not found on the Ollama server. Select another option before running summaries.
                </p>
              )}

              {/* Temperature Slider */}
              <div className="space-y-3 pt-4 border-t">
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label htmlFor="summarizer-temperature" className="font-medium">
                      Temperature
                    </Label>
                    <div className="text-xs text-muted-foreground">
                      {getTemperatureLabel(state.summaryTemperature)} — {state.summaryTemperature.toFixed(1)}
                    </div>
                  </div>
                  {state.summaryTemperatureSource && (
                    <span className="text-xs px-2 py-0.5 rounded-full bg-muted text-muted-foreground">
                      {state.summaryTemperatureSource}
                    </span>
                  )}
                </div>
                <input
                  id="summarizer-temperature"
                  type="range"
                  min="0"
                  max="2"
                  step="0.1"
                  value={state.summaryTemperature}
                  onChange={(e) => {
                    const newValue = parseFloat(e.target.value);
                    setState(prev => ({ ...prev, summaryTemperature: newValue }));
                  }}
                  onMouseUp={(e) => {
                    const target = e.target as HTMLInputElement;
                    handleTemperatureChange(parseFloat(target.value));
                  }}
                  onTouchEnd={(e) => {
                    const target = e.target as HTMLInputElement;
                    handleTemperatureChange(parseFloat(target.value));
                  }}
                  disabled={state.isLoadingSummaryModels || state.isUpdatingSummaryTemperature}
                  className={cn(
                    "w-full h-2 bg-muted rounded-lg appearance-none cursor-pointer",
                    "[&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-primary",
                    "[&::-moz-range-thumb]:w-4 [&::-moz-range-thumb]:h-4 [&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:bg-primary [&::-moz-range-thumb]:border-0",
                    state.isUpdatingSummaryTemperature ? "opacity-50" : ""
                  )}
                />
                <div className="flex justify-between text-xs text-muted-foreground px-1">
                  <span>Strict (0.0)</span>
                  <span>Balanced (0.5)</span>
                  <span>Experimental (1.5)</span>
                  <span>Max (2.0)</span>
                </div>
                <p className="text-xs text-muted-foreground">
                  Controls creativity vs. consistency in generated summaries. Lower values follow instructions more closely, higher values are more creative.
                </p>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Sync Configuration Section */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <span className="text-lg">⚙️ Sync Configuration</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {/* Sync Mode Selector */}
            <div className="flex items-center justify-between p-4 border rounded-lg">
              <div className="space-y-1">
                <div className="font-medium">Sync Mode</div>
                <div className="text-sm text-muted-foreground">
                  {state.mode === "full"
                    ? "Full sync replaces all existing data"
                    : "Incremental sync adds only new data"}
                </div>
              </div>
              <Button
                variant={state.mode === "full" ? "default" : "outline"}
                onClick={handleModeToggle}
                className="min-w-[140px]"
              >
                {state.mode === "full" ? "Full Sync" : "Incremental"}
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Scheduler Settings Drawer */}
      {state.schedulerDrawerOpen && (
        <div className="fixed inset-0 z-50 bg-black/50" onClick={() => setState(prev => ({ ...prev, schedulerDrawerOpen: false }))}>
          <div
            className="fixed right-0 top-0 bottom-0 w-96 bg-card shadow-xl p-6 overflow-y-auto"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-xl font-bold">⏰ Scheduler</h2>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setState(prev => ({ ...prev, schedulerDrawerOpen: false }))}
              >
                ✕
              </Button>
            </div>

            <div className="space-y-6">
              {/* Manual Sync Action */}
              <div className="p-4 border-2 border-dashed rounded-lg space-y-3">
                <div className="font-medium">Manual Sync</div>
                <p className="text-sm text-muted-foreground">
                  Run a data sync immediately without waiting for the scheduled interval
                </p>
                <Button
                  variant="default"
                  className="w-full"
                  onClick={async () => {
                    setState(prev => ({ ...prev, schedulerDrawerOpen: false }));
                    await handleLoadData();
                  }}
                  disabled={state.isLoading || state.isSaving || state.isTesting || state.isRebuildingSummaries}
                >
                  {state.isLoading ? "Syncing..." : "🔄 Run Sync Now"}
                </Button>
              </div>

              <div className="h-px bg-border" />

              {/* Countdown Timer Display */}
              {state.schedulerEnabled && state.schedulerCountdown && (
                <div className="p-4 bg-muted/50 rounded-lg">
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="text-sm font-medium">Next Sync In</div>
                      <p className="text-xs text-muted-foreground mt-1">
                        Automatic sync countdown
                      </p>
                    </div>
                    <div className="text-2xl font-mono font-bold text-primary">
                      {state.schedulerCountdown}
                    </div>
                  </div>
                </div>
              )}

              {/* Enable/Disable Toggle */}
              <div className="flex items-center justify-between">
                <div>
                  <Label htmlFor="scheduler-enabled" className="text-base">Automatic Scheduling</Label>
                  <p className="text-xs text-muted-foreground mt-1">
                    Sync data automatically at regular intervals
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
                <p><strong>📌 Note:</strong> The scheduler runs automatic incremental syncs in the background.</p>
                <p className="text-muted-foreground">
                  Manual syncs use the mode configured in &quot;Sync Configuration&quot; section.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Anonymization Settings */}
      <Card>
        <CardHeader className="flex flex-col gap-2">
          <div className="flex items-center justify-between">
            <CardTitle className="text-lg">🛡️ Anonymization Settings</CardTitle>
            <span className="text-xs px-2 py-0.5 rounded-full bg-muted text-muted-foreground">
              {anonymization.source}
            </span>
          </div>
          <p className="text-sm text-muted-foreground">
            Control whether the dashboard displays privacy-preserving pseudonyms (recommended) or real user names pulled from Open WebUI exports.
          </p>
        </CardHeader>
        <CardContent>
          {anonymization.error && (
            <div className="mb-4 rounded-md bg-destructive/10 p-3 text-sm text-destructive flex items-center gap-2">
              <span>⚠️</span>
              <span>{anonymization.error}</span>
            </div>
          )}
          <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between rounded-md border border-dashed p-4">
            <div className="space-y-2 max-w-xl">
              <p className="text-sm font-medium">Anonymization mode</p>
              <p className="text-xs text-muted-foreground">
                When enabled, the API returns persistent pseudonyms for every user. Disabling this exposes the actual names contained in your dataset.
              </p>
            </div>
            <div className="flex items-center gap-3">
              <span className="text-sm text-muted-foreground">
                {anonymization.enabled ? "On" : "Off"}
              </span>
              <Switch
                id="anonymization-toggle"
                checked={anonymization.enabled}
                disabled={anonymization.isSaving}
                onCheckedChange={handleAnonymizationToggle}
                aria-label="Toggle anonymization mode"
              />
            </div>
          </div>
          {anonymization.isSaving && (
            <p className="mt-2 text-xs text-muted-foreground">Saving anonymization preference…</p>
          )}
        </CardContent>
      </Card>

      {/* Data Source Configuration */}
      <Card>
        <CardHeader>
          <div className="flex w-full items-center justify-between">
            <div className="flex items-center gap-2">
              <CardTitle className="text-lg">🔗 Data Source Configuration</CardTitle>
            </div>
            {!state.isEditMode && (
              <Button
                variant="outline"
                size="sm"
                onClick={handleEditDataSource}
              >
                ✏️ Edit Credentials
              </Button>
            )}
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
                <div className="rounded-md bg-destructive/10 p-3 text-sm text-destructive flex items-center gap-2">
                  <span>⚠️</span>
                  <span>{state.error}</span>
                </div>
              )}

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Hostname */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <Label htmlFor="hostname" className="font-medium">OpenWebUI Hostname</Label>
                    {state.hostSource && (
                      <span className="text-xs px-2 py-0.5 rounded-full bg-muted text-muted-foreground">
                        {state.hostSource}
                      </span>
                    )}
                  </div>
                  <Input
                    id="hostname"
                    type="text"
                    placeholder="https://your-openwebui.example.com"
                    value={state.hostname}
                    onChange={(e) =>
                      setState((prev) => ({ ...prev, hostname: e.target.value }))
                    }
                    disabled={!state.isEditMode || state.isSaving}
                    className={!state.isEditMode ? "bg-muted" : ""}
                  />
                  {!state.isEditMode && (
                    <p className="text-xs text-muted-foreground">
                      Base URL for your OpenWebUI instance
                    </p>
                  )}
                </div>

                {/* API Key */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <Label htmlFor="apiKey" className="font-medium">API Key</Label>
                    {state.apiKeySource && (
                      <span className="text-xs px-2 py-0.5 rounded-full bg-muted text-muted-foreground">
                        {state.apiKeySource}
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
                    className={!state.isEditMode ? "bg-muted" : ""}
                  />
                  {!state.isEditMode && (
                    <p className="text-xs text-muted-foreground">
                      Authentication key for API access
                    </p>
                  )}
                </div>

              </div>

              {/* Show Save/Cancel buttons only in edit mode */}
              {state.isEditMode && (
                <div className="flex gap-3 pt-2">
                  <Button
                    variant="default"
                    className="flex-1"
                    onClick={handleSaveSettings}
                    disabled={!hasChanges || state.isSaving}
                  >
                    {state.isSaving ? "💾 Saving..." : "💾 Save Changes"}
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

    {anonymization.confirmOpen && (
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-background/80 backdrop-blur p-4">
        <div className="w-full max-w-md rounded-lg border border-border bg-background shadow-lg overflow-hidden">
          <div className="p-6 space-y-4">
            <div className="space-y-2">
              <h3 className="text-lg font-semibold">Turn off anonymization?</h3>
              <p className="text-sm text-muted-foreground">
                Disabling anonymization mode will expose real user names across the dashboard. Make sure you are comfortable sharing this information before proceeding.
              </p>
            </div>
            <div className="flex flex-col gap-2 sm:flex-row sm:justify-end">
              <Button variant="outline" onClick={cancelDisableAnonymization} className="sm:w-auto w-full">
                Keep Anonymization On
              </Button>
              <Button variant="destructive" onClick={confirmDisableAnonymization} className="sm:w-auto w-full">
                Yes, Turn Off Anonymization
              </Button>
            </div>
          </div>
        </div>
      </div>
    )}

    </div>
  );
}
