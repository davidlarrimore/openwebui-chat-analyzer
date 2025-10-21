import { apiGet } from "./api";

export const HEALTH_SERVICES = ["ollama", "database"] as const;

export type HealthService = (typeof HEALTH_SERVICES)[number];

export type HealthStatus = {
  service: HealthService;
  status: "ok" | "error";
  attempts: number;
  elapsed_seconds: number;
  detail?: string;
  meta?: Record<string, unknown>;
};

export async function fetchHealthStatus(service: HealthService): Promise<HealthStatus> {
  return apiGet<HealthStatus>(`/api/v1/health/${service}`);
}
