import { logAuthEvent } from "./logger";
import type { SummaryEventsResponse, SummaryStatus, OllamaModelTag } from "./types";

const ALLOWED_PATH = /^\/?api\/v1\//;
const AUTH_OPTIONAL_PATHS = new Set([
  "/api/v1/auth/status",
  "/api/v1/auth/bootstrap"
]);

export class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

function normalise(path: string) {
  const formatted = path.startsWith("/") ? path : `/${path}`;
  if (!ALLOWED_PATH.test(formatted)) {
    throw new Error(`Disallowed API path: ${formatted}`);
  }
  return formatted;
}

interface ResponseOptions {
  skipAuthRedirect?: boolean;
  requestPath?: string;
  requestMethod?: string;
  isServer?: boolean;
}

interface RequestOptions {
  skipAuthRedirect?: boolean;
}

async function handleJsonResponse<T>(response: Response, options: ResponseOptions = {}): Promise<T> {
  const contentType = response.headers.get("content-type") ?? "";
  if (!response.ok) {
    const context = {
      path: options.requestPath,
      method: options.requestMethod,
      status: response.status,
      redirected: response.redirected,
      source: options.isServer ? "server" : "client"
    };
    // Handle 401 Unauthorized - redirect to login on client side
    if (response.status === 401 && !options.skipAuthRedirect && typeof window !== "undefined") {
      logAuthEvent("warn", "API request returned 401; redirecting to login.", context);
      // Clear any stale session and redirect to login
      window.location.href = "/login?error=SessionExpired";
      throw new Error("Session expired, redirecting to login...");
    }
    logAuthEvent("error", "API request failed.", context);
    const message = response.statusText || `Request failed with status ${response.status}`;
    throw new ApiError(response.status, message);
  }
  if (contentType.includes("application/json")) {
    return (await response.json()) as T;
  }
  return {} as T;
}

export async function apiGet<T>(path: string, init?: RequestInit, options?: RequestOptions): Promise<T> {
  return request<T>("GET", path, undefined, init, options);
}

export async function apiPost<T>(path: string, body?: unknown, init?: RequestInit, options?: RequestOptions): Promise<T> {
  return request<T>("POST", path, body, init, options);
}

export async function apiPut<T>(path: string, body?: unknown, init?: RequestInit, options?: RequestOptions): Promise<T> {
  return request<T>("PUT", path, body, init, options);
}

async function request<T>(
  method: string,
  path: string,
  body?: unknown,
  init?: RequestInit,
  options: RequestOptions = {}
): Promise<T> {
  const normalised = normalise(path);
  const skipAuth = AUTH_OPTIONAL_PATHS.has(normalised);
  const headers = new Headers(init?.headers);

  if (body !== undefined && !headers.has("content-type")) {
    headers.set("content-type", "application/json");
  }

  if (typeof window === "undefined") {
    const [{ BACKEND_BASE_URL }, { getServerAuthSession }] = await Promise.all([import("./config"), import("./auth")]);
    const session = await getServerAuthSession();
    if (!skipAuth && session?.accessToken && !headers.has("authorization")) {
      headers.set("authorization", `Bearer ${session.accessToken}`);
    }

    const response = await fetch(`${BACKEND_BASE_URL.replace(/\/$/, "")}${normalised}`, {
      ...init,
      method,
      headers,
      body: body !== undefined ? JSON.stringify(body) : undefined,
      cache: "no-store"
    });

    return handleJsonResponse<T>(response, {
      skipAuthRedirect: skipAuth || options.skipAuthRedirect,
      requestPath: normalised,
      requestMethod: method,
      isServer: true
    });
  }

  if (skipAuth) {
    headers.set("x-next-auth-skip", "true");
  }

  const response = await fetch(`/api/proxy${normalised}`, {
    ...init,
    method,
    headers,
    body: body !== undefined ? JSON.stringify(body) : undefined
  });

  return handleJsonResponse<T>(response, {
    skipAuthRedirect: skipAuth || options.skipAuthRedirect,
    requestPath: normalised,
    requestMethod: method,
    isServer: false
  });
}

// ============================================================================
// OpenWebUI Settings API
// ============================================================================

export interface OpenWebUISettingsResponse {
  host: string;
  api_key: string;
  database_host: string;
  database_api_key: string;
  host_source: "database" | "environment" | "default";
  api_key_source: "database" | "environment" | "empty";
}

export interface OpenWebUISettingsUpdate {
  host?: string;
  api_key?: string;
}

export interface AnonymizationSettingsResponse {
  enabled: boolean;
  source: "database" | "environment" | "default";
}

export interface AnonymizationSettingsUpdate {
  enabled: boolean;
}

/**
 * Fetch current OpenWebUI Direct Connect settings
 */
export async function getOpenWebUISettings(): Promise<OpenWebUISettingsResponse> {
  return apiGet<OpenWebUISettingsResponse>("api/v1/admin/settings/direct-connect");
}

/**
 * Update OpenWebUI Direct Connect settings
 * @param settings - Object with optional host and api_key fields
 */
export async function updateOpenWebUISettings(
  settings: OpenWebUISettingsUpdate
): Promise<OpenWebUISettingsResponse> {
  return apiPut<OpenWebUISettingsResponse>("api/v1/admin/settings/direct-connect", settings);
}


// ============================================================================
// Anonymization Settings API
// ============================================================================

export async function getAnonymizationSettings(): Promise<AnonymizationSettingsResponse> {
  return apiGet<AnonymizationSettingsResponse>("api/v1/admin/settings/anonymization");
}

export async function updateAnonymizationSettings(
  payload: AnonymizationSettingsUpdate
): Promise<AnonymizationSettingsResponse> {
  return apiPut<AnonymizationSettingsResponse>("api/v1/admin/settings/anonymization", payload);
}

// ============================================================================
// Summarizer Settings API
// ============================================================================

export interface SummarizerSettingsResponse {
  model: string;
  temperature: number;
  enabled: boolean;
  model_source: "database" | "environment" | "default";
  temperature_source: "database" | "environment" | "default";
  enabled_source: "database" | "environment" | "default";
}

export interface SummarizerSettingsUpdate {
  model?: string;
  temperature?: number;
  enabled?: boolean;
}

export async function getSummarizerSettings(): Promise<SummarizerSettingsResponse> {
  return apiGet<SummarizerSettingsResponse>("api/v1/admin/settings/summarizer");
}

export async function updateSummarizerSettings(
  payload: SummarizerSettingsUpdate
): Promise<SummarizerSettingsResponse> {
  return apiPut<SummarizerSettingsResponse>("api/v1/admin/settings/summarizer", payload);
}

export interface OllamaModel {
  name: string;
  supports_completions: boolean;
  created_at?: string;
  updated_at?: string;
  model?: string;
  modified_at?: string;
  size?: number;
  digest?: string;
  details?: {
    parameter_size?: string;
    quantization_level?: string;
  };
}

export interface OllamaModelsResponse {
  models: OllamaModel[];
  sync_stats?: {
    added: number;
    removed: number;
    tested: number;
    errors: string[];
  };
}

export async function getAvailableOllamaModels(): Promise<OllamaModelsResponse> {
  return apiGet<OllamaModelsResponse>("api/v1/admin/ollama/models");
}

// ============================================================================
// OpenWebUI Health Test API
// ============================================================================

export interface OpenWebUIHealthTestRequest {
  host?: string;
  api_key?: string;
}

export interface OpenWebUIHealthTestResponse {
  service: string;
  status: "ok" | "error";
  attempts: number;
  elapsed_seconds: number;
  detail?: string;
  meta?: {
    version?: string;
    chat_count?: number;
  };
}

/**
 * Test connection to OpenWebUI instance
 * @param request - Optional host and api_key to test; if omitted, uses stored settings
 */
export async function testOpenWebUIConnection(
  request?: OpenWebUIHealthTestRequest
): Promise<OpenWebUIHealthTestResponse> {
  return apiPost<OpenWebUIHealthTestResponse>(
    "api/v1/health/openwebui",
    request || {}
  );
}

// ============================================================================
// Sync Status and Control API
// ============================================================================

export interface SyncStatusResponse {
  last_sync_at: string | null;
  last_watermark: string | null;
  has_data: boolean;
  recommended_mode: "full" | "incremental";
  is_stale: boolean;
  staleness_threshold_hours: number;
  local_counts: {
    chats: number;
    messages: number;
    users: number;
    models: number;
  } | null;
}

export interface SyncRunRequest {
  hostname: string;
  api_key?: string;
  mode?: "full" | "incremental";
}

export interface SyncRunResponse {
  detail: string;
  dataset: Record<string, unknown>;
  stats: Record<string, unknown>;
}

export interface SyncSchedulerConfig {
  enabled: boolean;
  interval_minutes: number;
  last_run_at: string | null;
  next_run_at: string | null;
}

export interface SyncSchedulerUpdate {
  enabled?: boolean;
  interval_minutes?: number;
}

/**
 * Get current sync status and watermark
 */
export async function getSyncStatus(): Promise<SyncStatusResponse> {
  return apiGet<SyncStatusResponse>("api/v1/sync/status");
}

/**
 * Run a sync operation with OpenWebUI
 * @param request - Sync request with hostname, optional API key, and sync mode
 */
export async function runSync(request: SyncRunRequest): Promise<SyncRunResponse> {
  return apiPost<SyncRunResponse>("api/v1/openwebui/sync", request);
}

/**
 * Get sync scheduler configuration
 */
export async function getSyncScheduler(): Promise<SyncSchedulerConfig> {
  return apiGet<SyncSchedulerConfig>("api/v1/sync/scheduler");
}

/**
 * Update sync scheduler configuration
 * @param config - Scheduler updates (enabled, interval_minutes)
 */
export async function updateSyncScheduler(config: SyncSchedulerUpdate): Promise<SyncSchedulerConfig> {
  return apiPost<SyncSchedulerConfig>("api/v1/sync/scheduler", config);
}

// ============================================================================
// Process Logs API
// ============================================================================

export interface ProcessLogEvent {
  timestamp: string;
  level: "debug" | "info" | "warning" | "error";
  job_id: string | null;
  phase: "connect" | "fetch" | "persist" | "summarize" | "done" | "error";
  message: string;
  details?: Record<string, unknown>;
}

export interface ProcessLogsResponse {
  logs: ProcessLogEvent[];
  total: number;
}

/**
 * Get recent process log events
 * @param jobId - Optional job ID to filter logs
 * @param limit - Maximum number of logs to return (default 100)
 */
export async function getProcessLogs(
  jobId?: string,
  limit?: number
): Promise<ProcessLogsResponse> {
  const params = new URLSearchParams();
  if (jobId) params.set("job_id", jobId);
  if (limit !== undefined) params.set("limit", limit.toString());

  const query = params.toString();
  const path = query ? `api/v1/logs?${query}` : "api/v1/logs";

  return apiGet<ProcessLogsResponse>(path);
}

// ============================================================================
// Summaries API
// ============================================================================

/**
 * Trigger a full rebuild of chat summaries.
 */
export async function rebuildSummaries(): Promise<SummaryStatus> {
  const payload = await apiPost<SummaryStatus | { status: SummaryStatus }>("api/v1/summaries/rebuild");
  if (payload && typeof (payload as { status?: SummaryStatus }).status === "object") {
    return (payload as { status: SummaryStatus }).status;
  }
  return payload as SummaryStatus;
}

/**
 * Fetch the current summarizer job status.
 */
export async function getSummaryStatus(): Promise<SummaryStatus> {
  return apiGet<SummaryStatus>("api/v1/summaries/status");
}

/**
 * Fetch recent summarizer events.
 */
export async function getSummaryEvents(after?: string, limit?: number): Promise<SummaryEventsResponse> {
  const params = new URLSearchParams();
  if (after) {
    params.set("after", after);
  }
  if (limit !== undefined) {
    params.set("limit", limit.toString());
  }
  const query = params.toString();
  const path = query ? `api/v1/summaries/events?${query}` : "api/v1/summaries/events";
  return apiGet<SummaryEventsResponse>(path, undefined, { skipAuthRedirect: true });
}

// ============================================================================
// Ollama Models API
// ============================================================================

export async function getOllamaModels(): Promise<OllamaModelTag[]> {
  return apiGet<OllamaModelTag[]>("api/v1/genai/models");
}
