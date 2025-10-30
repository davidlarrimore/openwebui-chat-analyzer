"use client";

import { useEffect, useState } from "react";
import { fetchHealthStatus, HealthService, HealthStatus, HEALTH_SERVICES } from "@/lib/health";
import { cn } from "@/lib/utils";

const POLL_INTERVAL_MS = 10000;

const SERVICE_PRESENTATION: Record<HealthService, { label: string; icon: string }> = {
  backend: { label: "Backend API", icon: "🛰️" },
  ollama: { label: "Ollama", icon: "🧠" },
  database: { label: "Database", icon: "🗄️" }
};

type ServiceLookup = {
  id: HealthService;
  label: string;
  icon: string;
};

const SERVICES: ServiceLookup[] = HEALTH_SERVICES.map((service) => ({
  id: service,
  label: SERVICE_PRESENTATION[service].label,
  icon: SERVICE_PRESENTATION[service].icon
}));

type HealthMap = {
  [K in HealthService]: HealthStatus | null;
};

type SidebarHealthStatusProps = {
  collapsed?: boolean;
};

function createInitialState(): HealthMap {
  return SERVICES.reduce<HealthMap>((state, service) => {
    state[service.id] = null;
    return state;
  }, {} as HealthMap);
}

const STATUS_STYLES: Record<
  "ok" | "error" | "loading",
  { label: string; textClass: string; dotClass: string }
> = {
  ok: {
    label: "Online",
    textClass: "text-emerald-600 dark:text-emerald-400",
    dotClass: "bg-emerald-500"
  },
  error: {
    label: "Offline",
    textClass: "text-destructive",
    dotClass: "bg-destructive"
  },
  loading: {
    label: "Checking...",
    textClass: "text-muted-foreground",
    dotClass: "bg-muted-foreground"
  }
};

function summariseMeta(service: HealthService, status: HealthStatus | null): string | undefined {
  if (!status || status.status !== "ok") {
    return undefined;
  }

  const meta = status.meta ?? {};

  if (service === "ollama") {
    const countCandidate = meta["model_count"];
    if (typeof countCandidate === "number") {
      const label = countCandidate === 1 ? "model" : "models";
      return `${countCandidate} ${label} available`;
    }
  }

  if (service === "database") {
    return "Connection established";
  }

  if (service === "backend") {
    const response = meta["response"];
    if (response === "ok") {
      return "API responsive";
    }
    return "API reachable";
  }

  return undefined;
}

function sanitizeDetail(detail: unknown): string | undefined {
  if (typeof detail !== "string") {
    return undefined;
  }
  const trimmed = detail.trim();
  if (!trimmed.length) {
    return undefined;
  }
  const MAX_LENGTH = 180;
  if (trimmed.length > MAX_LENGTH) {
    return `${trimmed.slice(0, MAX_LENGTH - 1)}…`;
  }
  return trimmed;
}

function formatErrorDetail(service: HealthService, detail: unknown): string | undefined {
  const message = sanitizeDetail(detail);
  if (!message) {
    return undefined;
  }

  if (service === "ollama") {
    const hostMatch = message.match(/host='([^']+)'/i);
    const portMatch = message.match(/port=(\d+)/i);
    const host = hostMatch?.[1];
    const port = portMatch?.[1];
    const location = host ? `${host}${port ? `:${port}` : ""}` : "the configured host";
    const lower = message.toLowerCase();
    const connectionIndicators = [
      "failed to establish a new connection",
      "connection refused",
      "network is unreachable",
      "max retries exceeded",
      "connection aborted",
      "timed out",
    ];
    if (connectionIndicators.some(indicator => lower.includes(indicator))) {
      return `Unable to reach Ollama at ${location}. Confirm the service is running and accessible.`;
    }
  }

  return message;
}

export default function SidebarHealthStatus({ collapsed = false }: SidebarHealthStatusProps) {
  const [statuses, setStatuses] = useState<HealthMap>(createInitialState);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      const results = await Promise.allSettled(
        SERVICES.map((service) => fetchHealthStatus(service.id))
      );

      if (cancelled) {
        return;
      }

      const nextState = createInitialState();

      results.forEach((result, index) => {
        const serviceId = SERVICES[index]?.id;
        if (!serviceId) {
          return;
        }

        if (result.status === "fulfilled") {
          nextState[serviceId] = result.value;
          return;
        }

        const message =
          result.reason instanceof Error ? result.reason.message : "Unable to fetch status";

        nextState[serviceId] = {
          service: serviceId,
          status: "error",
          attempts: 0,
          elapsed_seconds: 0,
          detail: message
        };
      });

      setStatuses(nextState);
      setLoading(false);
    }

    load();
    const interval = setInterval(load, POLL_INTERVAL_MS);

    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  return (
    <div className={cn("rounded-lg border bg-background", collapsed ? "p-2" : "p-4")}
      data-state={collapsed ? "collapsed" : "expanded"}>
      <h2
        className={cn(
          "text-sm font-semibold text-muted-foreground",
          collapsed ? "sr-only" : undefined
        )}
      >
        System Status
      </h2>
      <ul
        className={cn(
          "mt-3 space-y-3 text-sm",
          collapsed ? "mt-0 flex flex-col items-center gap-3 space-y-0 text-xs" : undefined
        )}
      >
        {SERVICES.map((service) => {
          const status = statuses[service.id];
          const stateKey: "ok" | "error" | "loading" =
            status?.status ?? (loading ? "loading" : "error");
          const presentation = STATUS_STYLES[stateKey];

          const detail =
            stateKey === "ok"
              ? summariseMeta(service.id, status)
              : formatErrorDetail(service.id, status?.detail) ?? (loading ? undefined : "Status unavailable");

          return (
            <li key={service.id} className={cn("flex flex-col gap-1", collapsed && "items-center")}
              aria-label={`${service.label} status: ${presentation.label}`}>
              {collapsed ? (
                <div className="flex flex-col items-center gap-2">
                  <span className={cn("h-2 w-2 rounded-full", presentation.dotClass)} aria-hidden />
                  <span aria-hidden className="text-lg">{service.icon}</span>
                </div>
              ) : (
                <>
                  <div className="flex items-center justify-between">
                    <span className="flex items-center gap-2 font-medium">
                      <span aria-hidden className="text-lg leading-none">{service.icon}</span>
                      {service.label}
                    </span>
                    <span className={cn("flex items-center gap-2 text-xs font-medium", presentation.textClass)}>
                      <span className={cn("h-2 w-2 rounded-full", presentation.dotClass)} />
                      {presentation.label}
                    </span>
                  </div>
                  {detail && <p className="text-xs text-muted-foreground">{detail}</p>}
                </>
              )}
              {collapsed && detail && (
                <p className="sr-only">{detail}</p>
              )}
            </li>
          );
        })}
      </ul>
    </div>
  );
}
