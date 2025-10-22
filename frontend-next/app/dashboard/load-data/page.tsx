import { redirect } from "next/navigation";

import LoadDataClient from "./load-data-client";
import { apiGet } from "@/lib/api";
import { getServerAuthSession } from "@/lib/auth";
import type { DatasetMeta } from "@/lib/types";
import { getDirectConnectDefaults } from "@/lib/direct-connect-defaults";

export const revalidate = 0;
export const dynamic = "force-dynamic";
export const fetchCache = "force-no-store";

export default async function LoadDataPage() {
  const session = await getServerAuthSession();
  if (!session) {
    redirect(`/login?callbackUrl=${encodeURIComponent("/dashboard/load-data")}`);
  }

  let initialMeta: DatasetMeta | null = null;
  let initialError: string | null = null;

  try {
    initialMeta = await apiGet<DatasetMeta>("api/v1/datasets/meta");
  } catch (error) {
    initialError = error instanceof Error ? error.message : "Unable to reach backend API.";
  }

  const defaults = await getDirectConnectDefaults();

  return <LoadDataClient defaults={defaults} initialError={initialError} initialMeta={initialMeta} />;
}
