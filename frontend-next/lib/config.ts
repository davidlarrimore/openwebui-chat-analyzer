export const ENABLE_GITHUB_OAUTH =
  (process.env.NEXT_PUBLIC_GITHUB_OAUTH_ENABLED ?? process.env.GITHUB_OAUTH_ENABLED ?? "false").toLowerCase() === "true";

export const BACKEND_BASE_URL = process.env.BACKEND_BASE_URL ?? "http://localhost:8000";
export const NEXTAUTH_SECRET = process.env.NEXTAUTH_SECRET ?? "dev-secret";
export const NEXTAUTH_URL = process.env.NEXTAUTH_URL ?? "http://localhost:3000";

export function getServerConfig() {
  return {
    BACKEND_BASE_URL,
    NEXTAUTH_SECRET,
    NEXTAUTH_URL
  };
}
