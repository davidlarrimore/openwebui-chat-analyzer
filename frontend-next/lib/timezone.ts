const FALLBACK_TIMEZONE = "UTC";

function validateTimezone(value: string | null | undefined): string {
  if (!value) {
    return FALLBACK_TIMEZONE;
  }

  const trimmed = value.trim();
  if (!trimmed) {
    return FALLBACK_TIMEZONE;
  }

  try {
    // Throws if the provided time zone identifier is invalid.
    new Intl.DateTimeFormat("en-US", { timeZone: trimmed }).format(new Date("2000-01-01T00:00:00Z"));
    return trimmed;
  } catch {
    return FALLBACK_TIMEZONE;
  }
}

let envTimezone: string | null = null;

if (typeof process !== "undefined" && process.env) {
  envTimezone =
    process.env.NEXT_PUBLIC_DISPLAY_TIMEZONE ?? process.env.DISPLAY_TIMEZONE ?? null;
}

export const DISPLAY_TIMEZONE = validateTimezone(envTimezone);
