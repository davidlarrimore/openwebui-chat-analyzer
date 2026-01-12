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
  getSummarizerConnections,
  getSummarizerModels,
  validateSummarizerModel,
  getProcessLogs,
  type OpenWebUISettingsResponse,
  type OpenWebUISettingsUpdate,
  type SyncStatusResponse,
  type SyncSchedulerConfig,
  type ProcessLogEvent,
  type ProviderType,
  type ProviderConnection,
  type SummarizerModel,
} from "@/lib/api";
import { fetchHealthStatus, type HealthStatus } from "@/lib/health";
import { toast } from "@/components/ui/use-toast";
import type { SummaryStatus, DatasetMeta, OllamaModelTag } from "@/lib/types";

const SUMMARY_POLL_INTERVAL_MS = 2000;
const TERMINAL_SUMMARY_STATES = new Set(["idle", "completed", "failed", "cancelled"]);
const CONNECTION_LABELS: Record<ProviderType, string> = {
  ollama: "Ollama",
  openai: "OpenAI",
  litellm: "LiteLLM",
  openwebui: "Open WebUI",
};
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
  selectedConnection: ProviderType;
  availableConnections: ProviderConnection[];
  isLoadingConnections: boolean;
  summaryModel: string;
  summaryModelSource?: "database" | "environment" | "default";
  summaryTemperature: number;
  summaryTemperatureSource?: "database" | "environment" | "default";
  summarizerEnabled: boolean;
  summarizerEnabledSource?: "database" | "environment" | "default";
  availableSummaryModels: string[];
  modelsByConnection: Record<ProviderType, string[]>;
  summaryModelError: string | null;
  isLoadingSummaryModels: boolean;
  isUpdatingSummaryModel: boolean;
  isUpdatingSummaryTemperature: boolean;
  isUpdatingSummarizerEnabled: boolean;
  isValidatingSummaryModels: boolean;
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
    selectedConnection: "ollama",
    availableConnections: [],
    isLoadingConnections: false,
    summaryModel: "",
    summaryModelSource: undefined,
    summaryTemperature: 0.2,
    summaryTemperatureSource: undefined,
    summarizerEnabled: true,
    summarizerEnabledSource: undefined,
    availableSummaryModels: [],
    modelsByConnection: {
      ollama: [],
      openai: [],
      litellm: [],
      openwebui: [],
    },
    summaryModelError: null,
    isLoadingSummaryModels: false,
    isUpdatingSummaryModel: false,
    isUpdatingSummaryTemperature: false,
    isUpdatingSummarizerEnabled: false,
    isValidatingSummaryModels: false,
  });

  const selectedConnectionRef = React.useRef<ProviderType>(state.selectedConnection);

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
  const [openwebuiHealth, setOpenwebuiHealth] = React.useState<HealthStatus | null>(null);
  const [isCheckingOpenwebui, setIsCheckingOpenwebui] = React.useState(false);

  React.useEffect(() => {
    selectedConnectionRef.current = state.selectedConnection;
  }, [state.selectedConnection]);

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

  const fetchSummarizerConfig = React.useCallback(async (autoValidateMissing = false, forceAutoValidate = false) => {
    setState(prev => ({
      ...prev,
      isLoadingConnections: true,
      isLoadingSummaryModels: true,
      summaryModelError: null,
    }));

    try {
      // Fetch connections and settings in parallel
      const [connectionsResponse, summarizerSettings] = await Promise.all([
        getSummarizerConnections(),
        getSummarizerSettings(),
      ]);

      const connections = connectionsResponse.connections || [];
      setState(prev => ({
        ...prev,
        availableConnections: connections,
        isLoadingConnections: false,
      }));

      // Determine which connection to use (simple priority: saved → first available → ollama)
      const savedConnection = summarizerSettings.connection?.toLowerCase() as ProviderType | undefined;
      const firstAvailable = connections.find(conn => conn.available)?.type as ProviderType | undefined;
      const connectionToUse = savedConnection ?? firstAvailable ?? "ollama";

      // Fetch models only for the selected connection
      const shouldAutoValidate = autoValidateMissing;
      const shouldForceValidate = autoValidateMissing && forceAutoValidate;

      const modelsResponse = await getSummarizerModels(
        connectionToUse,
        true,  // Include unvalidated models
        shouldAutoValidate,
        shouldForceValidate
      );

      const modelNames = modelsResponse.models
        .map(model => model.name.trim())
        .sort((a, b) => a.localeCompare(b));

      // Determine which model to select
      const savedModel = (summarizerSettings.model || "").trim();
      const modelInList = savedModel && modelNames.includes(savedModel);
      const selectedModel = modelInList ? savedModel : (modelNames[0] || "");

      // Build options list (include saved model even if not in list for display)
      const options = modelInList
        ? modelNames
        : (savedModel ? [savedModel, ...modelNames] : modelNames);

      setState(prev => ({
        ...prev,
        selectedConnection: connectionToUse,
        availableSummaryModels: options,
        modelsByConnection: {
          ...prev.modelsByConnection,
          [connectionToUse]: modelNames,
        },
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
        isLoadingConnections: false,
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

  const refreshOpenwebuiHealth = React.useCallback(
    async (hostOverride?: string) => {
      const hostToCheck = (hostOverride ?? originalValues.hostname ?? state.hostname ?? "").trim();
      if (!hostToCheck) {
        setOpenwebuiHealth(null);
        return;
      }
      setIsCheckingOpenwebui(true);
      try {
        const result = await fetchHealthStatus("openwebui");
        setOpenwebuiHealth(result);
      } catch (err) {
        const message = err instanceof Error ? err.message : "Unable to fetch OpenWebUI status";
        setOpenwebuiHealth({
          service: "openwebui",
          status: "error",
          attempts: 0,
          elapsed_seconds: 0,
          detail: message,
        });
      } finally {
        setIsCheckingOpenwebui(false);
      }
    },
    [originalValues.hostname, state.hostname]
  );

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
        const { logs } = await getProcessLogs();

        // Look for recent "Automatic sync triggered" messages
        const recentAutoSyncLog = logs.find((log: ProcessLogEvent) =>
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

  React.useEffect(() => {
    void refreshOpenwebuiHealth();
  }, [refreshOpenwebuiHealth]);

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

  const handleValidateModels = React.useCallback(async () => {
    if (state.isLoadingSummaryModels || state.isValidatingSummaryModels) {
      return;
    }
    setState(prev => ({ ...prev, isValidatingSummaryModels: true }));
    try {
      await fetchSummarizerConfig(true, true);
      toast({
        title: "Validation complete",
        description: "Models revalidated for the selected connection.",
        variant: "default",
        duration: 4000,
      });
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to validate models";
      toast({
        title: "Validation failed",
        description: message,
        variant: "destructive",
        duration: 5000,
      });
    } finally {
      setState(prev => ({ ...prev, isValidatingSummaryModels: false }));
    }
  }, [fetchSummarizerConfig, state.isLoadingSummaryModels, state.isValidatingSummaryModels]);

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
        await refreshOpenwebuiHealth(updatedSettings.host);
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
      void refreshOpenwebuiHealth();
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

  const handleConnectionChange = async (newConnection: ProviderType) => {
    if (state.isLoadingSummaryModels || newConnection === state.selectedConnection) {
      return;
    }

    setState(prev => ({
      ...prev,
      selectedConnection: newConnection,
      isLoadingSummaryModels: true,
      summaryModelError: null,
    }));

    try {
      // Fetch models for the new connection (including unvalidated)
      const modelsResponse = await getSummarizerModels(newConnection, true);

      const modelNames = modelsResponse.models
        .map(model => model.name.trim())
        .sort((a, b) => a.localeCompare(b));

      const selectedModel = modelNames.length > 0 ? modelNames[0] : "";

      setState(prev => ({
        ...prev,
        availableSummaryModels: modelNames,
        summaryModel: selectedModel,
        isLoadingSummaryModels: false,
        summaryModelError: null,
        modelsByConnection: {
          ...prev.modelsByConnection,
          [newConnection]: modelNames,
        },
      }));

      // Persist the connection change to the backend
      try {
        await updateSummarizerSettings({ connection: newConnection });
        toast({
          title: "Connection changed",
          description: `Switched to ${newConnection}. ${selectedModel ? 'Select a model to update.' : 'No models available.'}`,
          variant: "default",
          duration: 3000,
        });
      } catch (persistErr) {
        const persistMessage = persistErr instanceof Error ? persistErr.message : "Failed to persist connection change";
        // Show warning but don't fail the operation since the UI state is already updated
        toast({
          title: "Connection changed (not persisted)",
          description: `Switched to ${newConnection} locally, but failed to save: ${persistMessage}`,
          variant: "destructive",
          duration: 5000,
        });
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : `Failed to load models from ${newConnection}`;
      setState(prev => ({
        ...prev,
        isLoadingSummaryModels: false,
        summaryModelError: message,
        availableSummaryModels: [],
        modelsByConnection: {
          ...prev.modelsByConnection,
          [newConnection]: [],
        },
      }));
      toast({
        title: "Connection error",
        description: message,
        variant: "destructive",
        duration: 5000,
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

  const openwebuiStatus = !state.hostname
    ? "not_configured"
    : isCheckingOpenwebui || openwebuiHealth === null
      ? "loading"
      : openwebuiHealth?.status ?? "error";

  const openwebuiStatusDetail = React.useMemo(() => {
    if (!state.hostname) {
      return "Add a host to enable sync.";
    }
    if (!openwebuiHealth) {
      return isCheckingOpenwebui ? "Checking connection..." : "Status unavailable.";
    }
    if (openwebuiHealth.status !== "ok") {
      return openwebuiHealth.detail || "Unable to reach OpenWebUI.";
    }
    const meta = openwebuiHealth.meta ?? {};
    const parts: string[] = [];
    if (typeof meta.chat_count === "number") {
      parts.push(`${meta.chat_count} chats`);
    }
    if (typeof meta.version === "string" && meta.version.trim()) {
      parts.push(`v${meta.version.trim()}`);
    }
    if (typeof meta.host === "string" && meta.host.trim()) {
      parts.push(meta.host.trim());
    }
    return parts.join(" • ") || "Connected";
  }, [isCheckingOpenwebui, openwebuiHealth, state.hostname]);

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
                  openwebuiStatus === "ok"
                    ? "bg-emerald-500"
                    : openwebuiStatus === "loading"
                      ? "bg-amber-400"
                      : "bg-destructive"
                )} />
                <span className="font-semibold">
                  {openwebuiStatus === "ok"
                    ? "Online"
                    : openwebuiStatus === "loading"
                      ? "Checking..."
                      : openwebuiStatus === "not_configured"
                        ? "Not Configured"
                        : "Offline"}
                </span>
              </div>
              <div className="text-xs text-muted-foreground truncate">
                {openwebuiStatusDetail}
              </div>
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

      {/* Summarizer Configuration - Moved to Summarizer Tab */}
      <Card className="border-2 border-dashed">
        <CardHeader>
          <div className="flex items-center gap-2">
            <span className="text-2xl">🧠</span>
            <div>
              <CardTitle className="text-lg">Summarizer Configuration</CardTitle>
              <p className="text-sm text-muted-foreground mt-1">
                Summarizer configuration has moved to the dedicated Summarizer tab
              </p>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground mb-4">
            Configure LLM provider, model selection, temperature, and metrics extraction in the new Summarizer admin panel.
          </p>
          <Button
            variant="outline"
            onClick={() => router.push("/dashboard/admin/summarizer")}
          >
            Open Summarizer Settings →
          </Button>
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
