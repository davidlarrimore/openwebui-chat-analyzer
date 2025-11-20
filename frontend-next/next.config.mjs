import { config as loadEnv } from "dotenv";
import { resolve } from "path";

loadEnv({ path: resolve(process.cwd(), "../.env"), override: false, quiet: true });
loadEnv({ path: resolve(process.cwd(), ".env"), override: false, quiet: true });

const backendBaseUrl = process.env.BACKEND_BASE_URL ?? "http://localhost:8502";

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
