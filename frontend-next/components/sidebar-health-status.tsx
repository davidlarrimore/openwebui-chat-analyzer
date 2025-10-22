"use client";

import { useEffect, useState } from "react";
import { fetchHealthStatus, HealthService, HealthStatus, HEALTH_SERVICES } from "@/lib/health";
import { cn } from "@/lib/utils";

const POLL_INTERVAL_MS = 10000;

const SERVICE_LABELS: Record<HealthService, string> = {
  backend: "Backend API",
  ollama: "Ollama",
  database: "Database"
};

type ServiceLookup = {
  id: HealthService;
  label: string;
};

const SERVICES: ServiceLookup[] = HEALTH_SERVICES.map((service) => ({
  id: service,
  label: SERVICE_LABELS[service]
}));

type HealthMap = {
  [K in HealthService]: HealthStatus | null;
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

export default function SidebarHealthStatus() {
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
    <div className="rounded-lg border bg-background p-4">
      <h2 className="text-sm font-semibold text-muted-foreground">System Status</h2>
      <ul className="mt-3 space-y-3 text-sm">
        {SERVICES.map((service) => {
          const status = statuses[service.id];
          const stateKey: "ok" | "error" | "loading" =
            status?.status ?? (loading ? "loading" : "error");
          const presentation = STATUS_STYLES[stateKey];

          const detail =
            stateKey === "ok"
              ? summariseMeta(service.id, status)
              : status?.detail ?? (loading ? undefined : "Status unavailable");

          return (
            <li key={service.id} className="flex flex-col gap-1">
              <div className="flex items-center justify-between">
                <span className="font-medium">{service.label}</span>
                <span className={cn("flex items-center gap-2 text-xs font-medium", presentation.textClass)}>
                  <span className={cn("h-2 w-2 rounded-full", presentation.dotClass)} />
                  {presentation.label}
                </span>
              </div>
              {detail && <p className="text-xs text-muted-foreground">{detail}</p>}
            </li>
          );
        })}
      </ul>
    </div>
  );
}
