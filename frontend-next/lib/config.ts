const FALLBACK_BACKEND_BASE_URL = "http://localhost:8502";

function normalizeBackendBaseUrl(raw?: string): string {
  const candidate = (raw ?? "").trim();
  if (!candidate) {
    return FALLBACK_BACKEND_BASE_URL;
  }

  try {
    const parsed = new URL(candidate.startsWith("http") ? candidate : `http://${candidate}`);
    if (parsed.hostname === "backend" && process.env.NODE_ENV !== "production") {
      // Default docker-compose hostname does not resolve during local dev; fall back to localhost.
      return FALLBACK_BACKEND_BASE_URL;
    }
    return parsed.origin;
  } catch {
    return FALLBACK_BACKEND_BASE_URL;
  }
}

export const BACKEND_BASE_URL = normalizeBackendBaseUrl(
  process.env.NEXT_PUBLIC_BACKEND_BASE_URL ?? process.env.BACKEND_BASE_URL ?? process.env.FRONTEND_NEXT_BACKEND_BASE_URL
);

export const APP_BASE_URL = process.env.APP_BASE_URL ?? process.env.NEXT_PUBLIC_APP_BASE_URL ?? "http://localhost:3000";

export function getServerConfig() {
  return {
    BACKEND_BASE_URL,
    APP_BASE_URL
  };
}
