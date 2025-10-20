import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { apiGet } from "@/lib/api";
import { normaliseBrowseChats, normaliseBrowseMessages } from "@/lib/browse";
import SearchClient from "./search-client";

interface RawSearchData {
  rawChats: unknown;
  rawMessages: unknown;
  rawUsers: unknown;
  rawModels: unknown;
}

async function fetchSearchData(): Promise<RawSearchData | null> {
  try {
    const [rawChats, rawMessages, rawUsers, rawModels] = await Promise.all([
      apiGet<unknown>("api/v1/chats"),
      apiGet<unknown>("api/v1/messages"),
      apiGet<unknown>("api/v1/users"),
      apiGet<unknown>("api/v1/models")
    ]);
    return { rawChats, rawMessages, rawUsers, rawModels };
  } catch {
    return null;
  }
}

export default async function SearchChatsPage() {
  const data = await fetchSearchData();

  if (!data) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Unable to load conversations</CardTitle>
          <CardDescription>We could not reach the backend API. Confirm the FastAPI service is running.</CardDescription>
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
          <CardDescription>Import chats or use Direct Connect before searching transcripts.</CardDescription>
        </CardHeader>
      </Card>
    );
  }

  if (!messages.length) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>No messages available yet</CardTitle>
          <CardDescription>Messages will appear once an Open WebUI dataset or export is loaded.</CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">
            After loading messages you&apos;ll be able to search across every conversation, highlight matches, and export
            individual threads.
          </p>
        </CardContent>
      </Card>
    );
  }

  return <SearchClient chats={chats} messages={messages} />;
}
