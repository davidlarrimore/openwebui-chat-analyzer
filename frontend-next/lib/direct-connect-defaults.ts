import fs from "node:fs";
import path from "node:path";

type EnvCache = Record<string, string>;

let cachedEnv: EnvCache | null = null;

function loadEnvFile(): EnvCache {
  if (cachedEnv) {
    return cachedEnv;
  }

  const envPaths = [
    path.resolve(process.cwd(), ".env.local"),
    path.resolve(process.cwd(), ".env"),
    path.resolve(process.cwd(), "..", ".env.local"),
    path.resolve(process.cwd(), "..", ".env")
  ];

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

export function getDirectConnectDefaults() {
  const host =
    resolveEnv("NEXT_PUBLIC_OWUI_DIRECT_HOST") ??
    resolveEnv("OWUI_DIRECT_HOST") ??
    "http://localhost:4000";

  const apiKey =
    resolveEnv("NEXT_PUBLIC_OWUI_DIRECT_API_KEY") ??
    resolveEnv("OWUI_DIRECT_API_KEY") ??
    "";

  return { host, apiKey };
}
