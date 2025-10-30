const FALLBACK_TIMEZONE = "UTC";

function normaliseTimezone(value: string | null | undefined): string | null {
  if (!value) {
    return null;
  }

  const trimmed = value.trim();
  if (!trimmed) {
    return null;
  }

  try {
    // Throws if the provided time zone identifier is invalid.
    new Intl.DateTimeFormat("en-US", { timeZone: trimmed }).format(new Date("2000-01-01T00:00:00Z"));
    return trimmed;
  } catch {
    return null;
  }
}

function detectRuntimeTimezone(): string | null {
  if (typeof Intl === "undefined" || typeof Intl.DateTimeFormat !== "function") {
    return null;
  }

  try {
    const resolved = Intl.DateTimeFormat().resolvedOptions().timeZone ?? null;
    return normaliseTimezone(resolved);
  } catch {
    return null;
  }
}

let envTimezone: string | null = null;

if (typeof process !== "undefined" && process.env) {
  envTimezone =
    process.env.NEXT_PUBLIC_DISPLAY_TIMEZONE ?? process.env.DISPLAY_TIMEZONE ?? null;
}

const resolvedEnvTimezone = normaliseTimezone(envTimezone);
const runtimeTimezone = detectRuntimeTimezone();

export const DISPLAY_TIMEZONE = runtimeTimezone ?? resolvedEnvTimezone ?? FALLBACK_TIMEZONE;
