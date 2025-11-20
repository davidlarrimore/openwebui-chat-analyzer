import { config as loadEnv } from "dotenv";
import { resolve } from "path";

loadEnv({ path: resolve(process.cwd(), "../.env"), override: false, quiet: true });
loadEnv({ path: resolve(process.cwd(), ".env"), override: false, quiet: true });

const FALLBACK_BACKEND_BASE_URL = "http://localhost:8502";

function normalizeBackendBaseUrl(rawUrl) {
  const candidate = (rawUrl ?? "").trim();
  if (!candidate) {
    return FALLBACK_BACKEND_BASE_URL;
  }

  try {
    const parsed = new URL(candidate.startsWith("http") ? candidate : `http://${candidate}`);
    if (parsed.hostname === "backend" && process.env.NODE_ENV !== "production") {
      // The docker-compose default uses the internal service name; make local dev default to localhost.
      return FALLBACK_BACKEND_BASE_URL;
    }
    return parsed.origin;
  } catch (error) {
    console.warn(`Invalid BACKEND_BASE_URL '${candidate}'; falling back to ${FALLBACK_BACKEND_BASE_URL}.`);
    return FALLBACK_BACKEND_BASE_URL;
  }
}

const backendBaseUrl = normalizeBackendBaseUrl(
  process.env.BACKEND_BASE_URL ?? process.env.FRONTEND_NEXT_BACKEND_BASE_URL
);

const nextConfig = {
  eslint: {
    ignoreDuringBuilds: false
  },
  typescript: {
    ignoreBuildErrors: false
  },
  output: "standalone",
  async rewrites() {
    return [
      {
        source: "/api/backend/:path*",
        destination: `${backendBaseUrl.replace(/\/$/, "")}/api/backend/:path*`
      }
    ];
  }
};

export default nextConfig;
