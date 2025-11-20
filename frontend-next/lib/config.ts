export const BACKEND_BASE_URL = process.env.BACKEND_BASE_URL ?? "http://localhost:8502";
export const APP_BASE_URL = process.env.APP_BASE_URL ?? process.env.NEXT_PUBLIC_APP_BASE_URL ?? "http://localhost:3000";

export function getServerConfig() {
  return {
    BACKEND_BASE_URL,
    APP_BASE_URL
  };
}
