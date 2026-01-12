"use client";

import { useEffect, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Loader2, CheckCircle2, XCircle, AlertTriangle, Info } from "lucide-react";
import {
  apiGet,
  apiPost,
  ProviderType,
  getSummarizerConnections,
  getSummarizerModels,
  updateSummarizerSettings,
  type ProviderConnection,
} from "@/lib/api";
import { MonitoringDashboard } from "@/components/summarizer/monitoring-dashboard";
import { cn } from "@/lib/utils";
import { toast } from "@/components/ui/use-toast";

interface AvailableMetric {
  name: string;
  description: string;
  enabled_by_default: boolean;
  requires_messages: boolean;
  features: string[];
}

interface SummarizerSettings {
  enabled: boolean;
  connection?: string;
  model?: string;
  models?: Record<string, string>;
  temperature: number;
  temperature_source?: "database" | "environment" | "default";
  model_source?: "database" | "environment" | "default";
  enabled_source?: "database" | "environment" | "default";
}

interface SummarizerStatistics {
  total_processed: number;
  total_failed: number;
  success_rate: number;
  avg_latency_ms?: number;
  by_metric?: Record<string, {
    success: number;
    failed: number;
    avg_latency_ms?: number;
  }>;
}

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

export function SummarizerClient() {
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [testing, setTesting] = useState(false);
  const [isLoadingConnections, setIsLoadingConnections] = useState(false);
  const [isLoadingModels, setIsLoadingModels] = useState(false);
  const [isValidating, setIsValidating] = useState(false);
  const [isUpdatingModel, setIsUpdatingModel] = useState(false);
  const [isUpdatingTemperature, setIsUpdatingTemperature] = useState(false);

  const [availableMetrics, setAvailableMetrics] = useState<AvailableMetric[]>([]);
  const [availableConnections, setAvailableConnections] = useState<ProviderConnection[]>([]);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [selectedConnection, setSelectedConnection] = useState<ProviderType>("ollama");
  const [settings, setSettings] = useState<SummarizerSettings>({
    enabled: false,
    temperature: 0.2,
  });
  const [selectedMetrics, setSelectedMetrics] = useState<Set<string>>(new Set());
  const [statistics, setStatistics] = useState<SummarizerStatistics | null>(null);
  const [modelError, setModelError] = useState<string | null>(null);

  const [connectionStatus, setConnectionStatus] = useState<{
    status: "unknown" | "success" | "error";
    message?: string;
  }>({ status: "unknown" });

  // Load initial data
  useEffect(() => {
    Promise.all([
      loadAvailableMetrics(),
      loadConnections(),
      loadSummarizerSettings(),
      loadStatistics(),
    ]).finally(() => setLoading(false));
  }, []);

  const loadAvailableMetrics = async () => {
    try {
      const response = await apiGet<{ metrics: AvailableMetric[] }>("/api/v1/metrics/available");
      const metrics = response.metrics || [];
      setAvailableMetrics(metrics);

      // Initialize selected metrics with defaults
      const defaults = new Set(
        metrics.filter(m => m.enabled_by_default).map(m => m.name)
      );
      setSelectedMetrics(defaults);
    } catch (error) {
      console.error("Failed to load available metrics:", error);
    }
  };

  const loadConnections = async () => {
    setIsLoadingConnections(true);
    try {
      const response = await getSummarizerConnections();
      const connections = response.connections || [];
      setAvailableConnections(connections);
    } catch (error) {
      console.error("Failed to load connections:", error);
    } finally {
      setIsLoadingConnections(false);
    }
  };

  const loadSummarizerSettings = async (autoLoadModels = true) => {
    try {
      const config = await apiGet<SummarizerSettings>("/api/v1/admin/summarizer/settings");
      setSettings(config);

      // Set the selected connection
      if (config.connection) {
        setSelectedConnection(config.connection as ProviderType);
        if (autoLoadModels) {
          await loadModels(config.connection as ProviderType);
        }
      }
    } catch (error) {
      console.error("Failed to load summarizer settings:", error);
    }
  };

  const loadModels = async (connection: ProviderType, autoValidate = false) => {
    setIsLoadingModels(true);
    setModelError(null);
    try {
      const response = await getSummarizerModels(connection, true, autoValidate, false);
      const modelNames = response.models
        .map(model => model.name.trim())
        .filter(name => name !== "")
        .sort((a, b) => a.localeCompare(b));

      setAvailableModels(modelNames);

      // If current model is not in the list, clear it
      if (settings.model && !modelNames.includes(settings.model)) {
        setSettings(prev => ({ ...prev, model: modelNames[0] || "" }));
      }
    } catch (error: any) {
      const message = error instanceof Error ? error.message : "Failed to load models";
      setModelError(message);
      console.error("Failed to load models:", error);
    } finally {
      setIsLoadingModels(false);
    }
  };

  const loadStatistics = async () => {
    try {
      const stats = await apiGet<SummarizerStatistics>("/api/v1/admin/summarizer/statistics");
      setStatistics(stats);
    } catch (error) {
      console.error("Failed to load statistics:", error);
    }
  };

  const testConnection = async () => {
    if (!settings.connection) {
      setConnectionStatus({
        status: "error",
        message: "No connection configured. Please configure a connection in the Connection tab.",
      });
      return;
    }

    setTesting(true);
    try {
      const result = await apiPost<{ status: string; message: string }>(
        "/api/v1/admin/summarizer/test-connection",
        {}
      );
      setConnectionStatus({
        status: result.status === "success" ? "success" : "error",
        message: result.message,
      });
    } catch (error: any) {
      setConnectionStatus({
        status: "error",
        message: error.message || "Connection test failed",
      });
    } finally {
      setTesting(false);
    }
  };

  const saveConfiguration = async () => {
    setSaving(true);
    try {
      await apiPost("/api/v1/admin/summarizer/settings", {
        enabled: settings.enabled,
        selected_metrics: Array.from(selectedMetrics),
      });

      await loadStatistics();
      setConnectionStatus({
        status: "success",
        message: "Configuration saved successfully",
      });
    } catch (error: any) {
      setConnectionStatus({
        status: "error",
        message: error.message || "Failed to save configuration",
      });
    } finally {
      setSaving(false);
    }
  };

  const toggleMetric = (metricName: string) => {
    setSelectedMetrics(prev => {
      const next = new Set(prev);
      if (next.has(metricName)) {
        next.delete(metricName);
      } else {
        next.add(metricName);
      }
      return next;
    });
  };

  const handleConnectionChange = async (newConnection: ProviderType) => {
    if (isLoadingModels || isLoadingConnections) return;

    setSelectedConnection(newConnection);
    setModelError(null);

    try {
      // Load models for the new connection
      await loadModels(newConnection);

      // Persist the connection change
      await updateSummarizerSettings({ connection: newConnection });

      toast({
        title: "Connection changed",
        description: `Switched to ${CONNECTION_LABELS[newConnection]}. Select a model to complete setup.`,
        duration: 3000,
      });
    } catch (error: any) {
      const message = error instanceof Error ? error.message : "Failed to change connection";
      toast({
        title: "Connection change failed",
        description: message,
        variant: "destructive",
        duration: 5000,
      });
    }
  };

  const handleModelChange = async (newModel: string) => {
    if (isUpdatingModel) return;

    const previousModel = settings.model;
    setSettings(prev => ({ ...prev, model: newModel }));
    setIsUpdatingModel(true);

    try {
      const updated = await updateSummarizerSettings({ model: newModel });
      setSettings(prev => ({
        ...prev,
        model: updated.model,
        model_source: updated.model_source,
      }));

      toast({
        title: "Model updated",
        description: `Summaries will now use ${newModel}`,
        duration: 3000,
      });
    } catch (error: any) {
      const message = error instanceof Error ? error.message : "Failed to update model";
      setSettings(prev => ({ ...prev, model: previousModel }));
      toast({
        title: "Update failed",
        description: message,
        variant: "destructive",
        duration: 5000,
      });
    } finally {
      setIsUpdatingModel(false);
    }
  };

  const handleTemperatureChange = async (newTemperature: number) => {
    if (isUpdatingTemperature) return;

    const previousTemperature = settings.temperature;
    setSettings(prev => ({ ...prev, temperature: newTemperature }));
    setIsUpdatingTemperature(true);

    try {
      const updated = await updateSummarizerSettings({ temperature: newTemperature });
      setSettings(prev => ({
        ...prev,
        temperature: updated.temperature,
        temperature_source: updated.temperature_source,
      }));

      const tempLabel = getTemperatureLabel(newTemperature);
      toast({
        title: "Temperature updated",
        description: `Set to ${newTemperature.toFixed(1)} (${tempLabel})`,
        duration: 3000,
      });
    } catch (error: any) {
      const message = error instanceof Error ? error.message : "Failed to update temperature";
      setSettings(prev => ({ ...prev, temperature: previousTemperature }));
      toast({
        title: "Update failed",
        description: message,
        variant: "destructive",
        duration: 5000,
      });
    } finally {
      setIsUpdatingTemperature(false);
    }
  };

  const handleValidateModels = async () => {
    if (isLoadingModels || isValidating) return;

    setIsValidating(true);
    try {
      await loadModels(selectedConnection, true);
      toast({
        title: "Validation complete",
        description: "Models revalidated for the selected connection.",
        duration: 3000,
      });
    } catch (error: any) {
      const message = error instanceof Error ? error.message : "Validation failed";
      toast({
        title: "Validation failed",
        description: message,
        variant: "destructive",
        duration: 5000,
      });
    } finally {
      setIsValidating(false);
    }
  };

  const handleRefresh = async () => {
    await Promise.all([
      loadConnections(),
      loadSummarizerSettings(false),
      loadModels(selectedConnection),
    ]);
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center p-8">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Summarizer Configuration</h1>
        <p className="text-muted-foreground mt-2">
          Configure AI model settings, select metrics to extract, and monitor summarization performance.
        </p>
      </div>

      {/* Enable/Disable Summarizer Toggle */}
      <div className="flex items-center justify-between rounded-lg border p-4 bg-card">
        <div className="flex-1 space-y-1">
          <Label htmlFor="enable-summarizer" className="font-medium cursor-pointer">
            Enable Summarizer
          </Label>
          <p className="text-sm text-muted-foreground">
            Automatically extract metrics from new conversations
          </p>
        </div>
        <Switch
          id="enable-summarizer"
          checked={settings.enabled}
          onCheckedChange={(checked) => setSettings({ ...settings, enabled: checked })}
        />
      </div>

      {/* Model Configuration Card */}
      <Card>
        <CardHeader>
          <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
            <div>
              <CardTitle>Model Configuration</CardTitle>
              <CardDescription>
                Configure the LLM provider, model, and temperature for automated chat summaries
              </CardDescription>
            </div>
            <div className="flex w-full justify-end lg:w-auto">
              <div className="inline-flex w-full max-w-sm justify-end rounded-md shadow-sm lg:w-auto" role="group">
                <Button
                  variant="outline"
                  size="sm"
                  className="rounded-none first:rounded-l-md last:rounded-r-md first:border-r-0"
                  onClick={handleRefresh}
                  disabled={isLoadingModels}
                >
                  {isLoadingModels ? "⏳ Refreshing..." : "🔁 Refresh"}
                </Button>
                <Button
                  size="sm"
                  variant="default"
                  className="rounded-none first:rounded-l-md last:rounded-r-md bg-emerald-600 text-white hover:bg-emerald-700"
                  onClick={handleValidateModels}
                  disabled={isLoadingModels || isValidating}
                >
                  {isValidating ? "🧪 Validating..." : "✅ Validate"}
                </Button>
              </div>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Connection Type Selector */}
          <div className="space-y-2">
            <Label className="font-medium">Connection Type</Label>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
              {availableConnections.map((conn) => {
                const isSelected = conn.type === selectedConnection;
                const isDisabled = !conn.available || isLoadingConnections || isLoadingModels;

                return (
                  <button
                    key={conn.type}
                    type="button"
                    onClick={() => !isDisabled && handleConnectionChange(conn.type)}
                    disabled={isDisabled}
                    className={cn(
                      "flex flex-col items-center justify-center rounded-lg border-2 p-3 text-sm font-medium transition-all",
                      isSelected && conn.available
                        ? "border-primary bg-primary/10 text-primary"
                        : "border-border bg-background hover:border-primary/50",
                      !conn.available && "opacity-50 cursor-not-allowed",
                      isDisabled && "cursor-wait"
                    )}
                    title={!conn.available && conn.reason ? conn.reason : undefined}
                  >
                    <span>{CONNECTION_LABELS[conn.type] ?? conn.type}</span>
                    {!conn.available && (
                      <span className="text-xs text-muted-foreground mt-1">
                        Unavailable
                      </span>
                    )}
                    {isSelected && conn.available && (
                      <span className="text-xs mt-1">✓</span>
                    )}
                  </button>
                );
              })}
            </div>
            {selectedConnection && (() => {
              const selectedConn = availableConnections.find(c => c.type === selectedConnection);
              return selectedConn && !selectedConn.available && selectedConn.reason ? (
                <p className="text-xs text-muted-foreground">
                  {selectedConn.reason}
                </p>
              ) : null;
            })()}
          </div>

          {/* Model Selector */}
          {modelError ? (
            <div className="flex items-start justify-between gap-3 rounded-md bg-destructive/10 p-3 text-sm text-destructive">
              <div className="flex-1">
                <span>⚠️ {modelError}</span>
              </div>
              <Button
                variant="outline"
                size="sm"
                onClick={handleRefresh}
                disabled={isLoadingModels}
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
                {settings.model_source && (
                  <span className="text-xs px-2 py-0.5 rounded-full bg-muted text-muted-foreground">
                    {settings.model_source}
                  </span>
                )}
              </div>
              <select
                id="summarizer-model"
                className={cn(
                  "w-full rounded-md border border-input bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-ring",
                  isUpdatingModel ? "opacity-75" : "",
                )}
                value={settings.model || ""}
                onChange={(e) => handleModelChange(e.target.value)}
                disabled={
                  isLoadingModels ||
                  isUpdatingModel ||
                  availableModels.length === 0
                }
              >
                {availableModels.length === 0 ? (
                  <option value="">
                    {isLoadingModels ? "Loading models..." : "No models detected"}
                  </option>
                ) : (
                  availableModels.map(model => (
                    <option key={model} value={model}>
                      {model}
                    </option>
                  ))
                )}
              </select>
              <p className="text-xs text-muted-foreground">
                This model is used when rebuilding summaries or running manual summarizer jobs.
              </p>
              {settings.model && !availableModels.includes(settings.model) && (
                <p className="text-xs text-destructive">
                  The configured model was not found on the selected provider. Select another option before running summaries.
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
                      {getTemperatureLabel(settings.temperature)} — {settings.temperature.toFixed(1)}
                    </div>
                  </div>
                  {settings.temperature_source && (
                    <span className="text-xs px-2 py-0.5 rounded-full bg-muted text-muted-foreground">
                      {settings.temperature_source}
                    </span>
                  )}
                </div>
                <input
                  id="summarizer-temperature"
                  type="range"
                  min="0"
                  max="2"
                  step="0.1"
                  value={settings.temperature}
                  onChange={(e) => {
                    const newValue = parseFloat(e.target.value);
                    setSettings(prev => ({ ...prev, temperature: newValue }));
                  }}
                  onMouseUp={(e) => {
                    const target = e.target as HTMLInputElement;
                    handleTemperatureChange(parseFloat(target.value));
                  }}
                  onTouchEnd={(e) => {
                    const target = e.target as HTMLInputElement;
                    handleTemperatureChange(parseFloat(target.value));
                  }}
                  disabled={isLoadingModels || isUpdatingTemperature}
                  className={cn(
                    "w-full h-2 bg-muted rounded-lg appearance-none cursor-pointer",
                    "[&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-primary",
                    "[&::-moz-range-thumb]:w-4 [&::-moz-range-thumb]:h-4 [&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:bg-primary [&::-moz-range-thumb]:border-0",
                    isUpdatingTemperature ? "opacity-50" : ""
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

          {/* Test Connection */}
          <div className="pt-4 border-t flex items-center gap-3">
            <Button
              onClick={testConnection}
              disabled={testing || !settings.connection}
              variant="outline"
              size="sm"
            >
              {testing && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              Test Connection
            </Button>

            {connectionStatus.status !== "unknown" && (
              <div className="flex items-center gap-2">
                {connectionStatus.status === "success" ? (
                  <CheckCircle2 className="h-4 w-4 text-green-600" />
                ) : (
                  <XCircle className="h-4 w-4 text-red-600" />
                )}
                <span className="text-sm text-muted-foreground">
                  {connectionStatus.message}
                </span>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Metric Selection Card */}
      <Card>
        <CardHeader>
          <CardTitle>Metric Selection</CardTitle>
          <CardDescription>
            Choose which metrics to extract during summarization. Some metrics require full message history.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {availableMetrics.length === 0 ? (
            <Alert>
              <Info className="h-4 w-4" />
              <AlertDescription>No metrics available</AlertDescription>
            </Alert>
          ) : (
            <div className="space-y-3">
              {availableMetrics.map((metric) => (
                <div
                  key={metric.name}
                  className="flex items-start justify-between p-3 border rounded-lg hover:bg-accent/50 transition-colors"
                >
                  <div className="flex-1 space-y-1">
                    <div className="flex items-center gap-2">
                      <Label htmlFor={`metric-${metric.name}`} className="font-medium cursor-pointer">
                        {metric.name.charAt(0).toUpperCase() + metric.name.slice(1)}
                      </Label>
                      {metric.requires_messages && (
                        <Badge variant="secondary" className="text-xs">
                          Full History
                        </Badge>
                      )}
                    </div>
                    <p className="text-sm text-muted-foreground">{metric.description}</p>
                    {metric.features.length > 0 && (
                      <div className="flex flex-wrap gap-1 mt-1">
                        {metric.features.map((feature) => (
                          <Badge key={feature} variant="outline" className="text-xs">
                            {feature.replace(/_/g, " ")}
                          </Badge>
                        ))}
                      </div>
                    )}
                  </div>
                  <Switch
                    id={`metric-${metric.name}`}
                    checked={selectedMetrics.has(metric.name)}
                    onCheckedChange={() => toggleMetric(metric.name)}
                  />
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Statistics Card */}
      {statistics && (
        <Card>
          <CardHeader>
            <CardTitle>Performance Statistics</CardTitle>
            <CardDescription>
              Summarization success rates and performance metrics
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-3 gap-4">
              <div>
                <Label className="text-sm text-muted-foreground">Total Processed</Label>
                <p className="text-2xl font-bold mt-1">{statistics.total_processed}</p>
              </div>
              <div>
                <Label className="text-sm text-muted-foreground">Success Rate</Label>
                <p className="text-2xl font-bold mt-1">
                  {(statistics.success_rate * 100).toFixed(1)}%
                </p>
              </div>
              {statistics.avg_latency_ms && (
                <div>
                  <Label className="text-sm text-muted-foreground">Avg Latency</Label>
                  <p className="text-2xl font-bold mt-1">{statistics.avg_latency_ms.toFixed(0)}ms</p>
                </div>
              )}
            </div>

            {statistics.by_metric && Object.keys(statistics.by_metric).length > 0 && (
              <div className="mt-6 space-y-3">
                <Label className="text-sm font-medium">By Metric</Label>
                {Object.entries(statistics.by_metric).map(([metric, stats]) => (
                  <div key={metric} className="flex items-center justify-between p-2 border rounded">
                    <span className="text-sm font-medium capitalize">{metric}</span>
                    <div className="flex items-center gap-4 text-sm">
                      <span className="text-green-600">{stats.success} success</span>
                      {stats.failed > 0 && (
                        <span className="text-red-600">{stats.failed} failed</span>
                      )}
                      {stats.avg_latency_ms && (
                        <span className="text-muted-foreground">
                          {stats.avg_latency_ms.toFixed(0)}ms
                        </span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Monitoring Dashboard */}
      <MonitoringDashboard />

      {/* Save Button */}
      <div className="flex justify-end">
        <Button
          onClick={saveConfiguration}
          disabled={saving || !settings.connection || selectedMetrics.size === 0}
          size="lg"
        >
          {saving && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
          Save Configuration
        </Button>
      </div>
    </div>
  );
}
