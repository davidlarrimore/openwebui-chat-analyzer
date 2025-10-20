import { notFound } from "next/navigation";
import { CustomVisxChart } from "@/components/charts/custom-visx-chart";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { apiGet } from "@/lib/api";
import type { Chat, Message } from "@/lib/types";

interface ThreadPageProps {
  params: {
    id: string;
  };
}

async function fetchThread(id: string) {
  try {
    const [chat, messages] = await Promise.all([
      apiGet<Chat>(`api/v1/chats/${id}`),
      apiGet<Message[]>(`api/v1/messages?chat_id=${id}`)
    ]);
    return { chat, messages };
  } catch {
    return null;
  }
}

export default async function ThreadDetailPage({ params }: ThreadPageProps) {
  const data = await fetchThread(params.id);

  if (!data) {
    notFound();
  }

  const { chat, messages } = data;
  const chartData = messages.map((message) => ({
    timestamp: message.created_at ?? new Date().toISOString(),
    value: message.content?.length ?? 0,
    author: message.sender ?? "unknown"
  }));

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>{chat.title || "Untitled conversation"}</CardTitle>
          <CardDescription>
            Updated {chat.updated_at ? new Date(chat.updated_at).toLocaleString() : "recently"}
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <h3 className="text-sm font-semibold text-muted-foreground">Participants</h3>
            <p className="text-sm">{chat.participants?.join(", ") || "Unknown participants"}</p>
          </div>
          <div>
            <h3 className="text-sm font-semibold text-muted-foreground">Tags</h3>
            <p className="text-sm">{chat.tags?.join(", ") || "No tags"}</p>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Message Length Over Time</CardTitle>
          <CardDescription>Quick visual check of engagement across the thread.</CardDescription>
        </CardHeader>
        <CardContent>
          <CustomVisxChart data={chartData} height={260} width={640} />
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Messages</CardTitle>
          <CardDescription>Chronological log of this conversation.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {messages.map((message) => (
            <article className="rounded-md border bg-card px-4 py-3" key={message.id}>
              <header className="mb-2 flex items-center justify-between text-xs uppercase text-muted-foreground">
                <span>{message.sender ?? "Unknown sender"}</span>
                <span>{message.created_at ? new Date(message.created_at).toLocaleString() : "unknown time"}</span>
              </header>
              <p className="whitespace-pre-wrap text-sm leading-relaxed">{message.content}</p>
            </article>
          ))}
          {!messages.length && <p className="text-sm text-muted-foreground">No messages yet.</p>}
        </CardContent>
      </Card>
    </div>
  );
}
