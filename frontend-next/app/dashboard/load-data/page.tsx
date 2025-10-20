import LoadDataClient from "./load-data-client";
import { apiGet } from "@/lib/api";
import type { DatasetMeta } from "@/lib/types";
import { getDirectConnectDefaults } from "@/lib/direct-connect-defaults";

export const revalidate = 0;

export default async function LoadDataPage() {
  let initialMeta: DatasetMeta | null = null;
  let initialError: string | null = null;

  try {
    initialMeta = await apiGet<DatasetMeta>("api/v1/datasets/meta");
  } catch (error) {
    initialError = error instanceof Error ? error.message : "Unable to reach backend API.";
  }

  const defaults = getDirectConnectDefaults();

  return <LoadDataClient defaults={defaults} initialError={initialError} initialMeta={initialMeta} />;
}
