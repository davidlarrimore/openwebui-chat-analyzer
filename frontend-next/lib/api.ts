import { logAuthEvent } from "./logger";
import type { SummaryEventsResponse, SummaryStatus, OllamaModelTag } from "./types";

const DATA_PATH = /^\/?api\/v1\//;
const BACKEND_ROOT = /^\/?api\/backend\//i;
const AUTH_OPTIONAL_PATHS = new Set([
  "/api/backend/auth/status",
  "/api/backend/auth/bootstrap",
  "/api/backend/auth/login",
  "/api/backend/auth/oidc/login"
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
  if (BACKEND_ROOT.test(formatted)) {
    return formatted;
  }
  if (!DATA_PATH.test(formatted)) {
    throw new Error(`Disallowed API path: ${formatted}`);
  }
  return `/api/backend${formatted}`;
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
  const payload = body !== undefined ? JSON.stringify(body) : undefined;

  const execute = async (): Promise<{ response: Response; isServer: boolean }> => {
    if (typeof window === "undefined") {
      const [{ getServerConfig }, { cookies, headers: nextHeaders }] = await Promise.all([
        import("./config"),
        import("next/headers")
      ]);
      const { BACKEND_BASE_URL } = getServerConfig();
      const cookieStore = cookies();
      if (!headers.has("cookie")) {
        const cookieHeader = cookieStore
          .getAll()
          .map((entry) => `${entry.name}=${entry.value}`)
          .join("; ");
        if (cookieHeader) {
          headers.set("cookie", cookieHeader);
        }
      }
      headers.set("x-analyzer-internal", "true");
      const forwarded = nextHeaders().get("x-forwarded-for");
      if (forwarded && !headers.has("x-forwarded-for")) {
        headers.set("x-forwarded-for", forwarded);
      }
      const response = await fetch(`${BACKEND_BASE_URL.replace(/\/$/, "")}${normalised}`, {
        ...init,
        method,
        headers,
        body: payload,
        cache: "no-store"
      });
      return { response, isServer: true };
    }

    const response = await fetch(normalised, {
      ...init,
      method,
      headers,
      body: payload,
      credentials: "include"
    });
    return { response, isServer: false };
  };

  const attemptRefresh = async (isServer: boolean): Promise<boolean> => {
    if (normalised === "/api/backend/auth/refresh" || isServer) {
      return false;
    }
    try {
      const refreshResponse = await fetch("/api/backend/auth/refresh", {
        method: "POST",
        credentials: "include"
      });
      return refreshResponse.ok;
    } catch (error) {
      console.error("Session refresh failed", error);
      return false;
    }
  };

  let { response, isServer } = await execute();

  if (
    response.status === 401 &&
    !skipAuth &&
    !options.skipAuthRedirect &&
    (await attemptRefresh(isServer))
  ) {
    ({ response } = await execute());
  }

  if (
    response.status === 401 &&
    !skipAuth &&
    !options.skipAuthRedirect &&
    typeof window !== "undefined"
  ) {
    const { pathname, search } = window.location;
    const onAuthPage = pathname.startsWith("/login") || pathname.startsWith("/register");
    const hittingAuthEndpoint = normalised.startsWith("/api/backend/auth/");

    if (!onAuthPage && !hittingAuthEndpoint) {
      const loginUrl = new URL("/login", window.location.origin);
      loginUrl.searchParams.set("error", "SessionExpired");
      loginUrl.searchParams.set("callbackUrl", pathname + search);
      logAuthEvent("warn", "Session expired; redirecting to login.", {
        path: normalised,
        method,
        status: response.status,
        redirected: true,
        source: "client"
      });
      window.location.href = loginUrl.toString();
    }
  }

  return handleJsonResponse<T>(response, {
    skipAuthRedirect: skipAuth || options.skipAuthRedirect,
    requestPath: normalised,
    requestMethod: method,
    isServer
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
  connection: string;
  model_source: "database" | "environment" | "default";
  temperature_source: "database" | "environment" | "default";
  enabled_source: "database" | "environment" | "default";
  connection_source: "database" | "environment" | "default";
}

export interface SummarizerSettingsUpdate {
  model?: string;
  temperature?: number;
  enabled?: boolean;
  connection?: string;
}

export async function getSummarizerSettings(): Promise<SummarizerSettingsResponse> {
  return apiGet<SummarizerSettingsResponse>("api/v1/admin/settings/summarizer");
}

export async function updateSummarizerSettings(
  payload: SummarizerSettingsUpdate
): Promise<SummarizerSettingsResponse> {
  return apiPut<SummarizerSettingsResponse>("api/v1/admin/settings/summarizer", payload);
}

// ============================================================================
// Summarizer Provider Connections API
// ============================================================================

export type ProviderType = "ollama" | "openai" | "openwebui";

export interface ProviderConnection {
  type: ProviderType;
  available: boolean;
  reason?: string | null;
}

export interface SummarizerConnectionsResponse {
  connections: ProviderConnection[];
}

export interface SummarizerModel {
  name: string;
  display_name: string;
  provider: string;
  validated: boolean;
  metadata?: Record<string, unknown> | null;
}

export interface SummarizerModelsResponse {
  models: SummarizerModel[];
}

export interface ValidateModelRequest {
  connection: ProviderType;
  model: string;
}

export interface ValidateModelResponse {
  valid: boolean;
}

/**
 * Get status of all available LLM provider connections
 */
export async function getSummarizerConnections(): Promise<SummarizerConnectionsResponse> {
  return apiGet<SummarizerConnectionsResponse>("api/v1/admin/summarizer/connections");
}

/**
 * Get models available from a specific provider
 * @param connection - Provider type (ollama | openai | openwebui)
 * @param includeUnvalidated - Whether to include unvalidated models (default: true)
 */
export async function getSummarizerModels(
  connection: ProviderType,
  includeUnvalidated: boolean = true,
  autoValidateMissing = false,
  forceValidate = false
): Promise<SummarizerModelsResponse> {
  const params = new URLSearchParams();
  params.set("connection", connection);
  params.set("include_unvalidated", includeUnvalidated.toString());
  if (autoValidateMissing) {
    params.set("auto_validate_missing", "true");
  }
  if (forceValidate) {
    params.set("force_validate", "true");
  }

  return apiGet<SummarizerModelsResponse>(
    `api/v1/admin/summarizer/models?${params.toString()}`
  );
}

/**
 * Validate if a model supports text completion
 * @param connection - Provider type
 * @param model - Model name to validate
 */
export async function validateSummarizerModel(
  connection: ProviderType,
  model: string
): Promise<ValidateModelResponse> {
  return apiPost<ValidateModelResponse>("api/v1/admin/summarizer/validate-model", {
    connection,
    model,
  });
}

// ============================================================================
// Ollama Models API (Legacy)
// ============================================================================

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
 * Cancel the currently running summarizer job.
 */
export async function cancelSummaryJob(): Promise<SummaryStatus> {
  const payload = await apiPost<SummaryStatus | { status: SummaryStatus }>("api/v1/summaries/cancel");
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
