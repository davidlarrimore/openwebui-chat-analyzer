import fs from "node:fs";
import path from "node:path";

import { apiGet } from "@/lib/api";
import type { DirectConnectSettings } from "@/lib/types";

type EnvCache = Record<string, string>;

let cachedEnv: EnvCache | null = null;

export type DirectConnectDefaultResult = {
  host: string;
  apiKey: string;
  hostSource: "database" | "environment" | "default";
  apiKeySource: "database" | "environment" | "empty";
  databaseHost: string;
  databaseApiKey: string;
};

function collectSearchDirectories(): string[] {
  const seeds = [
    process.cwd(),
    __dirname,
    process.env.PWD,
    process.env.INIT_CWD,
    process.env.ORIGINAL_CWD
  ]
    .filter((value): value is string => Boolean(value))
    .map((value) => path.resolve(value));

  const seen = new Set<string>();
  const ordered: string[] = [];

  const registerDir = (dir: string) => {
    if (!seen.has(dir)) {
      seen.add(dir);
      ordered.push(dir);
    }
  };

  for (const seed of seeds) {
    let current = seed;
    let depth = 0;
    while (true) {
      registerDir(current);
      const parent = path.dirname(current);
      if (parent === current || depth >= 10) {
        break;
      }
      current = parent;
      depth += 1;
    }
  }

  return ordered;
}

function loadEnvFile(): EnvCache {
  if (cachedEnv) {
    return cachedEnv;
  }

  const searchDirs = collectSearchDirectories();
  const envPaths: string[] = [];
  const pathSeen = new Set<string>();

  for (const dir of searchDirs) {
    for (const filename of [".env.local", ".env"]) {
      const candidate = path.join(dir, filename);
      if (!pathSeen.has(candidate)) {
        pathSeen.add(candidate);
        envPaths.push(candidate);
      }
    }
  }

  const result: EnvCache = {};

  for (const envPath of envPaths) {
    if (!fs.existsSync(envPath)) {
      continue;
    }
    const content = fs.readFileSync(envPath, "utf8");
    const lines = content.split(/\r?\n/);

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith("#")) {
        continue;
      }

      const equalsIndex = trimmed.indexOf("=");
      if (equalsIndex === -1) {
        continue;
      }

      const key = trimmed.slice(0, equalsIndex).trim();
      let value = trimmed.slice(equalsIndex + 1).trim();

      if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
        value = value.slice(1, -1);
      }

      if (!(key in result)) {
        result[key] = value;
      }
    }
  }

  cachedEnv = result;
  return result;
}

function resolveEnv(key: string): string | undefined {
  const directValue = process.env[key];
  if (directValue && directValue.trim()) {
    return directValue.trim();
  }

  const cache = loadEnvFile();
  const value = cache[key];
  return value?.trim() ? value.trim() : undefined;
}

const FALLBACK_DIRECT_HOST =
  resolveEnv("NEXT_PUBLIC_OWUI_DIRECT_HOST") ??
  resolveEnv("OWUI_DIRECT_HOST") ??
  "http://localhost:4000";

function loadEnvDefaults(): DirectConnectDefaultResult {
  const envHost =
    resolveEnv("NEXT_PUBLIC_OWUI_DIRECT_HOST") ?? resolveEnv("OWUI_DIRECT_HOST") ?? "";
  const envApiKey =
    resolveEnv("NEXT_PUBLIC_OWUI_DIRECT_API_KEY") ?? resolveEnv("OWUI_DIRECT_API_KEY") ?? "";

  const host = envHost || FALLBACK_DIRECT_HOST;
  const hostSource: DirectConnectDefaultResult["hostSource"] = envHost ? "environment" : "default";

  const apiKey = envApiKey || "";
  const apiKeySource: DirectConnectDefaultResult["apiKeySource"] = envApiKey ? "environment" : "empty";

  return {
    host,
    apiKey,
    hostSource,
    apiKeySource,
    databaseHost: "",
    databaseApiKey: ""
  };
}

export async function getDirectConnectDefaults(): Promise<DirectConnectDefaultResult> {
  const fallback = loadEnvDefaults();

  try {
    const payload = await apiGet<DirectConnectSettings>("api/v1/admin/settings/direct-connect");
    if (!payload || typeof payload !== "object") {
      return fallback;
    }

    const hostSource = payload.host_source ?? fallback.hostSource;
    const apiKeySource = payload.api_key_source ?? fallback.apiKeySource;

    const rawHost = typeof payload.host === "string" ? payload.host.trim() : "";
    const rawApiKey = typeof payload.api_key === "string" ? payload.api_key : "";
    const rawDatabaseHost =
      typeof payload.database_host === "string" ? payload.database_host.trim() : "";
    const rawDatabaseApiKey =
      typeof payload.database_api_key === "string" ? payload.database_api_key : "";

    const databaseHost = rawDatabaseHost || (hostSource === "database" ? rawHost : "");
    const databaseApiKey = rawDatabaseApiKey || (apiKeySource === "database" ? rawApiKey : "");

    const host =
      rawHost || fallback.host;
    const apiKey =
      rawApiKey || fallback.apiKey;

    return {
      host,
      apiKey,
      hostSource,
      apiKeySource,
      databaseHost,
      databaseApiKey
    };
  } catch (error) {
    return fallback;
  }
}
