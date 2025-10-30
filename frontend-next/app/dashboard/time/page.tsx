import { redirect } from "next/navigation";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { TimeAnalysisClient } from "./time-analysis-client";
import { ApiError, apiGet } from "@/lib/api";
import {
  buildChatUserMap,
  normaliseChats,
  normaliseMessages,
  normaliseModels,
  normaliseUsers
} from "@/lib/overview";
import type { TimeAnalysisChat, TimeAnalysisMessage } from "@/lib/time-analysis";

interface RawTimePayload {
  rawChats: unknown;
  rawMessages: unknown;
  rawUsers: unknown;
  rawModels: unknown;
}

async function fetchTimeData(): Promise<RawTimePayload | null> {
  try {
    const [rawChats, rawMessages, rawUsers, rawModels] = await Promise.all([
      apiGet<unknown>("api/v1/chats"),
      apiGet<unknown>("api/v1/messages"),
      apiGet<unknown>("api/v1/users"),
      apiGet<unknown>("api/v1/models")
    ]);
    return { rawChats, rawMessages, rawUsers, rawModels };
  } catch (error) {
    if (error instanceof ApiError && error.status === 401) {
      redirect(`/login?error=AuthRequired&callbackUrl=${encodeURIComponent("/dashboard/time")}`);
    }
    return null;
  }
}

export default async function TimeAnalysisPage() {
  const data = await fetchTimeData();

  if (!data) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Unable to load time analysis</CardTitle>
          <CardDescription>The backend API could not be reached. Verify that the FastAPI service is running.</CardDescription>
        </CardHeader>
      </Card>
    );
  }

  const usersMap = normaliseUsers(data.rawUsers);
  const modelsMap = normaliseModels(data.rawModels);
  const chats = normaliseChats(data.rawChats);
  const messages = normaliseMessages(data.rawMessages, modelsMap);

  if (!messages.length) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>No messages available yet</CardTitle>
          <CardDescription>Upload chat exports or sync with Open WebUI to populate the time analysis dashboard.</CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">
            Once messages are ingested you&apos;ll see the same conversation cadence charts that were available in the legacy
            application.
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

  const chatsPayload: TimeAnalysisChat[] = chats.map((chat) => ({
    chatId: chat.chatId,
    userId: chat.userId ?? null,
    tags: chat.tags ?? []
  }));

  const messagesPayload: TimeAnalysisMessage[] = messages.map((message) => ({
    chatId: message.chatId,
    role: message.role,
    model: message.model ?? "",
    content: message.content ?? "",
    timestamp: message.timestamp ? message.timestamp.toISOString() : null
  }));

  return <TimeAnalysisClient chats={chatsPayload} messages={messagesPayload} userDisplayMap={userDisplayMap} />;
}
