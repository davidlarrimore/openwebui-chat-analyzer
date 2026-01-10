"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Loader2, Download, RefreshCw, AlertTriangle, CheckCircle2 } from "lucide-react";
import { apiGet, apiPost } from "@/lib/api";

interface OverallStats {
  total_attempts: number;
  total_successes: number;
  total_failures: number;
  success_rate: number;
  avg_latency_ms: number;
  total_tokens: number | null;
  total_retries: number;
}

interface MetricStats {
  total_attempts: number;
  total_successes: number;
  total_failures: number;
  success_rate: number;
  avg_latency_ms: number;
  avg_tokens: number | null;
  total_retries: number;
}

interface FailureLog {
  timestamp: string;
  chat_id: string;
  metric_name: string;
  provider: string;
  model: string;
  error: string;
  latency_ms: number;
  retry_count: number;
}

export function MonitoringDashboard() {
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [exporting, setExporting] = useState(false);

  const [overallStats, setOverallStats] = useState<OverallStats | null>(null);
  const [metricStats, setMetricStats] = useState<Record<string, MetricStats>>({});
  const [recentFailures, setRecentFailures] = useState<FailureLog[]>([]);

  const [exportStatus, setExportStatus] = useState<{
    status: "idle" | "success" | "error";
    message?: string;
  }>({ status: "idle" });

  useEffect(() => {
    loadMonitoringData();
  }, []);

  const loadMonitoringData = async () => {
    try {
      const [overall, byMetric, failures] = await Promise.all([
        apiGet<OverallStats>("/api/v1/admin/summarizer/monitoring/overall"),
        apiGet<{ metrics: Record<string, MetricStats> }>("/api/v1/admin/summarizer/monitoring/by-metric"),
        apiGet<{ failures: FailureLog[] }>("/api/v1/admin/summarizer/monitoring/recent-failures?limit=10"),
      ]);

      setOverallStats(overall);
      setMetricStats(byMetric.metrics || {});
      setRecentFailures(failures.failures || []);
    } catch (error) {
      console.error("Failed to load monitoring data:", error);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  const handleRefresh = async () => {
    setRefreshing(true);
    await loadMonitoringData();
  };

  const handleExport = async () => {
    setExporting(true);
    setExportStatus({ status: "idle" });

    try {
      const result = await apiPost<{ status: string; message: string; path?: string }>(
        "/api/v1/admin/summarizer/monitoring/export",
        {}
      );

      if (result.status === "success") {
        setExportStatus({
          status: "success",
          message: result.message || "Logs exported successfully",
        });
      } else {
        setExportStatus({
          status: "error",
          message: result.message || "Export failed",
        });
      }
    } catch (error: any) {
      setExportStatus({
        status: "error",
        message: error.message || "Export failed",
      });
    } finally {
      setExporting(false);
      setTimeout(() => setExportStatus({ status: "idle" }), 5000);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center p-8">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  const hasData = overallStats && overallStats.total_attempts > 0;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Performance Monitoring</h2>
          <p className="text-sm text-muted-foreground">
            Real-time metrics for summarizer operations
          </p>
        </div>
        <div className="flex gap-2">
          <Button
            onClick={handleRefresh}
            disabled={refreshing}
            variant="outline"
            size="sm"
          >
            {refreshing ? (
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            ) : (
              <RefreshCw className="mr-2 h-4 w-4" />
            )}
            Refresh
          </Button>
          <Button
            onClick={handleExport}
            disabled={exporting || !hasData}
            variant="outline"
            size="sm"
          >
            {exporting ? (
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            ) : (
              <Download className="mr-2 h-4 w-4" />
            )}
            Export Logs
          </Button>
        </div>
      </div>

      {exportStatus.status !== "idle" && (
        <Alert variant={exportStatus.status === "error" ? "destructive" : "default"}>
          {exportStatus.status === "success" ? (
            <CheckCircle2 className="h-4 w-4" />
          ) : (
            <AlertTriangle className="h-4 w-4" />
          )}
          <AlertDescription>{exportStatus.message}</AlertDescription>
        </Alert>
      )}

      {!hasData ? (
        <Alert>
          <AlertDescription>
            No monitoring data available yet. Metrics will appear after summarizer operations are performed.
          </AlertDescription>
        </Alert>
      ) : (
        <>
          {/* Overall Statistics */}
          <Card>
            <CardHeader>
              <CardTitle>Overall Statistics</CardTitle>
              <CardDescription>Aggregated metrics across all extraction types</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-4 gap-4">
                <div>
                  <p className="text-sm text-muted-foreground">Success Rate</p>
                  <p className="text-2xl font-bold">
                    {(overallStats.success_rate * 100).toFixed(1)}%
                  </p>
                  <p className="text-xs text-muted-foreground">
                    {overallStats.total_successes}/{overallStats.total_attempts} attempts
                  </p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Avg Latency</p>
                  <p className="text-2xl font-bold">
                    {overallStats.avg_latency_ms.toFixed(0)}ms
                  </p>
                </div>
                {overallStats.total_tokens && (
                  <div>
                    <p className="text-sm text-muted-foreground">Total Tokens</p>
                    <p className="text-2xl font-bold">
                      {overallStats.total_tokens.toLocaleString()}
                    </p>
                  </div>
                )}
                <div>
                  <p className="text-sm text-muted-foreground">Total Retries</p>
                  <p className="text-2xl font-bold">{overallStats.total_retries}</p>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Per-Metric Statistics */}
          <Card>
            <CardHeader>
              <CardTitle>Metrics Breakdown</CardTitle>
              <CardDescription>Performance by metric type</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {Object.entries(metricStats).map(([metricName, stats]) => (
                  <div key={metricName} className="flex items-center justify-between border-b pb-3 last:border-0">
                    <div className="flex items-center gap-3">
                      <span className="font-medium capitalize">{metricName}</span>
                      <Badge variant={stats.success_rate >= 0.9 ? "default" : "destructive"}>
                        {(stats.success_rate * 100).toFixed(1)}%
                      </Badge>
                    </div>
                    <div className="flex items-center gap-6 text-sm text-muted-foreground">
                      <div>
                        <span className="text-green-600">{stats.total_successes}</span>
                        {" / "}
                        <span className="text-red-600">{stats.total_failures}</span>
                      </div>
                      <div>{stats.avg_latency_ms.toFixed(0)}ms</div>
                      {stats.total_retries > 0 && (
                        <div className="text-orange-600">{stats.total_retries} retries</div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Recent Failures */}
          {recentFailures.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle>Recent Failures</CardTitle>
                <CardDescription>Last 10 failed extraction attempts</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {recentFailures.map((failure, idx) => (
                    <div key={idx} className="border-b pb-3 last:border-0">
                      <div className="flex items-start justify-between">
                        <div className="space-y-1">
                          <div className="flex items-center gap-2">
                            <Badge variant="destructive" className="text-xs">
                              {failure.metric_name}
                            </Badge>
                            <span className="text-xs text-muted-foreground">
                              {new Date(failure.timestamp).toLocaleString()}
                            </span>
                          </div>
                          <p className="text-sm text-muted-foreground">
                            Chat: {failure.chat_id} | Provider: {failure.provider}/{failure.model}
                          </p>
                          <p className="text-sm text-red-600">{failure.error}</p>
                        </div>
                        <div className="text-right text-xs text-muted-foreground">
                          <div>{failure.latency_ms.toFixed(0)}ms</div>
                          {failure.retry_count > 0 && (
                            <div className="text-orange-600">{failure.retry_count} retries</div>
                          )}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </>
      )}
    </div>
  );
}
