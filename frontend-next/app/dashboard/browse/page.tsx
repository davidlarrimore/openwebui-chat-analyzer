import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { apiGet } from "@/lib/api";
import { normaliseBrowseChats, normaliseBrowseMessages } from "@/lib/browse";
import type { SummarizerSettingsResponse } from "@/lib/api";
import BrowseClient from "./browse-client";

interface RawBrowseData {
  rawChats: unknown;
  rawMessages: unknown;
  rawUsers: unknown;
  rawModels: unknown;
  summarizerEnabled: boolean;
}

async function fetchBrowseData(): Promise<RawBrowseData | null> {
  try {
    const [rawChats, rawMessages, rawUsers, rawModels, summarizerSettings] = await Promise.all([
      apiGet<unknown>("api/v1/chats"),
      apiGet<unknown>("api/v1/messages"),
      apiGet<unknown>("api/v1/users"),
      apiGet<unknown>("api/v1/models"),
      apiGet<SummarizerSettingsResponse>("api/v1/admin/settings/summarizer")
    ]);
    return { rawChats, rawMessages, rawUsers, rawModels, summarizerEnabled: summarizerSettings.enabled };
  } catch {
    return null;
  }
}

export default async function BrowseChatsPage() {
  const data = await fetchBrowseData();

  if (!data) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Unable to load conversations</CardTitle>
          <CardDescription>The backend API could not be reached. Confirm the FastAPI service is running and reachable.</CardDescription>
        </CardHeader>
      </Card>
    );
  }

  const chats = normaliseBrowseChats(data.rawChats, data.rawUsers);
  const messages = normaliseBrowseMessages(data.rawMessages, data.rawModels);

  if (!chats.length) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>No conversations available yet</CardTitle>
          <CardDescription>Import chats or sync with Open WebUI to start browsing your datasets.</CardDescription>
        </CardHeader>
      </Card>
    );
  }

  if (!messages.length) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>No messages available yet</CardTitle>
          <CardDescription>Messages will appear here after a dataset is loaded from Open WebUI or an export file.</CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">
            Once messages are ingested you&apos;ll be able to expand conversations, review transcripts, and download raw threads for deeper analysis.
          </p>
        </CardContent>
      </Card>
    );
  }

  return <BrowseClient chats={chats} messages={messages} summarizerEnabled={data.summarizerEnabled} />;
}
