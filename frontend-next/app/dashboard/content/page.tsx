import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { ContentAnalysisClient } from "./content-analysis-client";
import { apiGet } from "@/lib/api";
import { buildChatUserMap, normaliseChats, normaliseMessages, normaliseModels, normaliseUsers } from "@/lib/overview";
import type { ContentChat, ContentMessage } from "@/lib/content-analysis";

interface RawContentPayload {
  rawChats: unknown;
  rawMessages: unknown;
  rawUsers: unknown;
  rawModels: unknown;
}

async function fetchContentData(): Promise<RawContentPayload | null> {
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

export default async function ContentAnalysisPage() {
  const data = await fetchContentData();

  if (!data) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Unable to load content analysis</CardTitle>
          <CardDescription>The backend API could not be reached. Verify that the FastAPI service is running.</CardDescription>
        </CardHeader>
      </Card>
    );
  }

  const modelsMap = normaliseModels(data.rawModels);
  const usersMap = normaliseUsers(data.rawUsers);
  const chats = normaliseChats(data.rawChats);
  const messages = normaliseMessages(data.rawMessages, modelsMap);

  if (!messages.length) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>No messages available yet</CardTitle>
          <CardDescription>Upload chat exports or sync with Open WebUI to populate the content analysis dashboard.</CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">
            Once messages are ingested you&apos;ll be able to explore word clouds and message length distributions similar to the Streamlit experience.
          </p>
        </CardContent>
      </Card>
    );
  }

  const chatUserMap = buildChatUserMap(chats, usersMap);
  const userDisplayMap: Record<string, string> = Object.fromEntries(usersMap.entries());

  for (const chat of chats) {
    const userId = chat.userId?.trim();
    if (!userId) {
      continue;
    }
    if (!userDisplayMap[userId]) {
      const fallback = chatUserMap.get(chat.chatId);
      userDisplayMap[userId] = fallback ?? userId;
    }
  }

  const chatsPayload: ContentChat[] = chats.map((chat) => ({
    chatId: chat.chatId,
    userId: chat.userId ?? null
  }));

  const messagesPayload: ContentMessage[] = messages.map((message) => ({
    chatId: message.chatId,
    role: message.role,
    content: message.content ?? "",
    model: message.model ?? "",
    timestamp: message.timestamp ? message.timestamp.toISOString() : null
  }));

  return <ContentAnalysisClient chats={chatsPayload} messages={messagesPayload} userDisplayMap={userDisplayMap} />;
}

