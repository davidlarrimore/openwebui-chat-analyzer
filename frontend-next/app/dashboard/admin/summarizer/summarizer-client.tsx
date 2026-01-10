"use client";

import { useEffect, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Loader2, CheckCircle2, XCircle, AlertTriangle, Info } from "lucide-react";
import { apiGet, apiPost, ProviderType } from "@/lib/api";
import { MonitoringDashboard } from "@/components/summarizer/monitoring-dashboard";

interface AvailableMetric {
  name: string;
  description: string;
  sprint: number;
  enabled_by_default: boolean;
  requires_messages: boolean;
  features: string[];
}

interface SummarizerSettings {
  enabled: boolean;
  connection?: string;
  model?: string;
  models?: Record<string, string>;
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

export function SummarizerClient() {
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [testing, setTesting] = useState(false);

  const [availableMetrics, setAvailableMetrics] = useState<AvailableMetric[]>([]);
  const [settings, setSettings] = useState<SummarizerSettings>({
    enabled: false,
  });
  const [selectedMetrics, setSelectedMetrics] = useState<Set<string>>(new Set());
  const [statistics, setStatistics] = useState<SummarizerStatistics | null>(null);

  const [connectionStatus, setConnectionStatus] = useState<{
    status: "unknown" | "success" | "error";
    message?: string;
  }>({ status: "unknown" });

  // Load initial data
  useEffect(() => {
    Promise.all([
      loadAvailableMetrics(),
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

  const loadSummarizerSettings = async () => {
    try {
      const config = await apiGet<SummarizerSettings>("/api/v1/admin/summarizer/settings");
      setSettings(config);
    } catch (error) {
      console.error("Failed to load summarizer settings:", error);
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

  if (loading) {
    return (
      <div className="flex items-center justify-center p-8">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  const currentConnection = settings.connection;
  const currentModel = settings.model;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Summarizer Configuration</h1>
        <p className="text-muted-foreground mt-2">
          Configure AI model settings, select metrics to extract, and monitor summarization performance.
        </p>
      </div>

      {/* Connection Status Card */}
      <Card>
        <CardHeader>
          <CardTitle>Connection Status</CardTitle>
          <CardDescription>
            Current LLM provider connection and model configuration
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {!currentConnection ? (
            <Alert>
              <AlertTriangle className="h-4 w-4" />
              <AlertDescription>
                No connection configured. Please configure a connection in the{" "}
                <a href="/dashboard/admin/connection" className="font-medium underline">
                  Connection tab
                </a>
                .
              </AlertDescription>
            </Alert>
          ) : (
            <>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label className="text-sm text-muted-foreground">Provider</Label>
                  <p className="text-sm font-medium mt-1 capitalize">{currentConnection}</p>
                </div>
                <div>
                  <Label className="text-sm text-muted-foreground">Model</Label>
                  <p className="text-sm font-medium mt-1">{currentModel || "Not selected"}</p>
                </div>
              </div>

              <div className="flex items-center gap-3">
                <Button
                  onClick={testConnection}
                  disabled={testing}
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
            </>
          )}
        </CardContent>
      </Card>

      {/* Metric Selection Card */}
      <Card>
        <CardHeader>
          <CardTitle>Metric Selection</CardTitle>
          <CardDescription>
            Choose which metrics to extract during summarization. Metrics marked with Sprint 3 require full message history.
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
                      <Badge variant="outline" className="text-xs">
                        Sprint {metric.sprint}
                      </Badge>
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

          <div className="pt-4 border-t">
            <div className="flex items-center justify-between">
              <div>
                <Label htmlFor="enable-summarizer" className="font-medium">
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
          </div>
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

      {/* Advanced Monitoring Dashboard (Sprint 5) */}
      <MonitoringDashboard />

      {/* Save Button */}
      <div className="flex justify-end">
        <Button
          onClick={saveConfiguration}
          disabled={saving || !currentConnection || selectedMetrics.size === 0}
          size="lg"
        >
          {saving && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
          Save Configuration
        </Button>
      </div>
    </div>
  );
}
