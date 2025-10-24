type LogLevel = "debug" | "info" | "warn" | "error";

function resolveConsoleMethod(level: LogLevel): (...args: unknown[]) => void {
  switch (level) {
    case "debug":
      return typeof console.debug === "function" ? console.debug.bind(console) : console.log.bind(console);
    case "info":
      return typeof console.info === "function" ? console.info.bind(console) : console.log.bind(console);
    case "warn":
      return typeof console.warn === "function" ? console.warn.bind(console) : console.log.bind(console);
    case "error":
      return typeof console.error === "function" ? console.error.bind(console) : console.log.bind(console);
    default:
      return console.log.bind(console);
  }
}

function serialiseContext(context?: Record<string, unknown>): string | undefined {
  if (!context || Object.keys(context).length === 0) {
    return undefined;
  }
  try {
    return JSON.stringify(context);
  } catch {
    return "[unserializable-context]";
  }
}

export function logAuthEvent(level: LogLevel, message: string, context?: Record<string, unknown>): void {
  const emitter = resolveConsoleMethod(level);
  const serialised = serialiseContext(context);
  emitter(`[auth] ${message}${serialised ? ` ${serialised}` : ""}`);
}
