import { config as loadEnv } from "dotenv";
import { resolve } from "path";

loadEnv({ path: resolve(process.cwd(), "../.env"), override: false, quiet: true });
loadEnv({ path: resolve(process.cwd(), ".env"), override: false, quiet: true });

const nextConfig = {
  eslint: {
    ignoreDuringBuilds: false
  },
  typescript: {
    ignoreBuildErrors: false
  },
  output: "standalone"
};

export default nextConfig;
