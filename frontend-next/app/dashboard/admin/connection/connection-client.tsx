"use client";

import { ConnectionInfoPanel } from "@/components/connection-info-panel";
import type { OpenWebUISettingsResponse } from "@/lib/api";

interface ConnectionClientProps {
  initialSettings?: OpenWebUISettingsResponse;
  // TODO: Add more props for initial data from server component
  // initialDatasetMeta?: DatasetMeta;
  // initialSyncStatus?: SyncStatus;
}

export default function ConnectionClient({ initialSettings }: ConnectionClientProps) {
  // TODO: Set up client-side state management
  // TODO: Implement data fetching/mutations
  // TODO: Handle real-time updates (WebSocket/SSE)

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-3xl font-bold tracking-tight">Admin: Connection Management</h2>
        <p className="text-muted-foreground mt-2">
          Manage your data source connection, sync settings, and monitor processing logs.
        </p>
      </div>

      <ConnectionInfoPanel initialSettings={initialSettings} />
    </div>
  );
}
