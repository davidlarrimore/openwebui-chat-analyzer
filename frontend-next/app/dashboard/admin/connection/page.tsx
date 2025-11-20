import { apiGet } from "@/lib/api";
import type { OpenWebUISettingsResponse } from "@/lib/api";
import ConnectionClient from "./connection-client";

// Force dynamic rendering to always get fresh data
export const revalidate = 0;
export const dynamic = "force-dynamic";
export const fetchCache = "force-no-store";

export default async function AdminConnectionPage() {
  let initialSettings: OpenWebUISettingsResponse | undefined;

  try {
    initialSettings = await apiGet<OpenWebUISettingsResponse>("api/v1/admin/settings/direct-connect");
  } catch (error) {
    // If fetching settings fails, let the client component handle fetching
    console.error("Failed to fetch initial settings:", error);
  }

  // TODO: Fetch dataset metadata
  // TODO: Fetch last sync status
  // Example:
  // const datasetMeta = await apiGet("api/v1/datasets/meta");

  return <ConnectionClient initialSettings={initialSettings} />;
}
