"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  ALL_MODELS_OPTION,
  ALL_USERS_OPTION,
  BrowseChat,
  BrowseMessage,
  buildModelOptions,
  buildThreadExportPayload,
  buildUserOptions,
  filterMessagesByUserAndModel,
  formatOutcome
} from "@/lib/browse";
import { DISPLAY_TIMEZONE } from "@/lib/timezone";

interface BrowseClientProps {
  chats: BrowseChat[];
  messages: BrowseMessage[];
  summarizerEnabled: boolean;
}

interface ThreadEntry {
  chat: BrowseChat;
  messages: BrowseMessage[];
  firstUserTimestamp: number;
}

const THREADS_PER_PAGE_OPTIONS = [5, 10, 20, 50] as const;

const DATE_TIME_FORMATTER = new Intl.DateTimeFormat("en-US", {
  dateStyle: "medium",
  timeStyle: "short",
  timeZone: DISPLAY_TIMEZONE
});

function formatTimestamp(value: string | null | undefined | number): string {
  if (typeof value === "number") {
    if (!Number.isFinite(value) || value <= 0) {
      return "Unknown";
    }
    return DATE_TIME_FORMATTER.format(new Date(value));
  }

  if (!value) {
    return "Unknown";
  }

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "Unknown";
  }
  return DATE_TIME_FORMATTER.format(date);
}

function timestampToNumber(value: string | null | undefined): number {
  if (!value) {
    return 0;
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return 0;
  }
  return date.getTime();
}

function getFileLabel(value: unknown): string | null {
  if (!value || typeof value !== "object") {
    if (typeof value === "string" && value.trim()) {
      return value.trim();
    }
    return null;
  }
  const record = value as Record<string, unknown>;
  const filename = typeof record.filename === "string" && record.filename.trim() ? record.filename.trim() : null;
  if (filename) {
    return filename;
  }
  const name = typeof record.name === "string" && record.name.trim() ? record.name.trim() : null;
  return name;
}

export default function BrowseClient({ chats, messages, summarizerEnabled }: BrowseClientProps) {
  const [selectedUser, setSelectedUser] = useState<string>(ALL_USERS_OPTION);
  const [selectedModel, setSelectedModel] = useState<string>(ALL_MODELS_OPTION);
  const [threadsPerPage, setThreadsPerPage] = useState<number>(10);
  const [page, setPage] = useState<number>(1);
  const [expandedChatId, setExpandedChatId] = useState<string | null>(null);

  const userDisplayMap = useMemo(() => {
    const map: Record<string, string> = {};
    for (const chat of chats) {
      const userId = chat.userId?.trim();
      if (!userId) {
        continue;
      }
      if (!map[userId]) {
        map[userId] = chat.userDisplay.trim() || userId;
      }
    }
    return map;
  }, [chats]);

  const userOptions = useMemo(() => buildUserOptions(chats, userDisplayMap), [chats, userDisplayMap]);
  const modelOptions = useMemo(() => buildModelOptions(messages), [messages]);

  const filtered = useMemo(
    () => filterMessagesByUserAndModel(chats, messages, selectedUser, selectedModel),
    [chats, messages, selectedUser, selectedModel]
  );

  const matchingChatIds = useMemo(() => new Set(filtered.matchingChatIds), [filtered.matchingChatIds]);

  const filteredChats = useMemo(
    () => chats.filter((chat) => matchingChatIds.has(chat.chatId)),
    [chats, matchingChatIds]
  );

  const threads: ThreadEntry[] = useMemo(() => {
    if (!filteredChats.length || !filtered.filteredMessages.length) {
      return [];
    }

    const grouped = new Map<string, BrowseMessage[]>();
    for (const message of filtered.filteredMessages) {
      const existing = grouped.get(message.chatId);
      if (existing) {
        existing.push(message);
      } else {
        grouped.set(message.chatId, [message]);
      }
    }

    const rolePriority = (role: string) => {
      if (role === "user") return 0;
      if (role === "assistant") return 1;
      return 2;
    };

    return filteredChats
      .map((chat) => {
        const threadMessages = [...(grouped.get(chat.chatId) ?? [])];
        if (!threadMessages.length) {
          return null;
        }

        threadMessages.sort((left, right) => {
          const delta = timestampToNumber(left.timestamp) - timestampToNumber(right.timestamp);
          if (delta !== 0) {
            return delta;
          }
          return rolePriority(left.role) - rolePriority(right.role);
        });

        const firstUser = threadMessages.find((message) => message.role === "user");
        const firstTimestamp = firstUser
          ? timestampToNumber(firstUser.timestamp)
          : timestampToNumber(threadMessages[0]?.timestamp);

        return {
          chat,
          messages: threadMessages,
          firstUserTimestamp: firstTimestamp
        } satisfies ThreadEntry;
      })
      .filter((entry): entry is ThreadEntry => Boolean(entry))
      .sort((a, b) => b.firstUserTimestamp - a.firstUserTimestamp);
  }, [filtered.filteredMessages, filteredChats]);

  const totalPages = threads.length ? Math.max(1, Math.ceil(threads.length / threadsPerPage)) : 1;

  useEffect(() => {
    setPage(1);
  }, [selectedUser, selectedModel, threadsPerPage, threads.length]);

  useEffect(() => {
    setPage((previous) => {
      if (previous < 1) {
        return 1;
      }
      if (previous > totalPages) {
        return totalPages;
      }
      return previous;
    });
  }, [totalPages]);

  const paginatedThreads = useMemo(() => {
    if (!threads.length) {
      return [];
    }
    const startIndex = (page - 1) * threadsPerPage;
    return threads.slice(startIndex, startIndex + threadsPerPage);
  }, [threads, page, threadsPerPage]);

  useEffect(() => {
    if (!expandedChatId) {
      return;
    }
    const stillVisible = paginatedThreads.some((entry) => entry.chat.chatId === expandedChatId);
    if (!stillVisible) {
      setExpandedChatId(null);
    }
  }, [paginatedThreads, expandedChatId]);

  const handleDownload = useCallback((chat: BrowseChat, threadMessages: BrowseMessage[]) => {
    const payload = buildThreadExportPayload(chat, threadMessages);
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `thread_browse_${chat.chatId}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    setTimeout(() => URL.revokeObjectURL(link.href), 500);
  }, []);

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Filter Conversations</CardTitle>
          <CardDescription>Limit the list by user or model before browsing transcripts.</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-2">
            <label className="flex flex-col gap-2 text-sm font-medium">
              <span>User</span>
              <select
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm shadow-sm focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2"
                value={selectedUser}
                onChange={(event) => {
                  setSelectedUser(event.target.value);
                  setPage(1);
                }}
              >
                {userOptions.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>
            <label className="flex flex-col gap-2 text-sm font-medium">
              <span>Model</span>
              <select
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm shadow-sm focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2"
                value={selectedModel}
                onChange={(event) => {
                  setSelectedModel(event.target.value);
                  setPage(1);
                }}
              >
                {modelOptions.map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <div className="mt-4 flex flex-wrap items-center gap-3 text-sm">
            <label className="flex items-center gap-2">
              <span>Threads per page</span>
              <select
                className="rounded-md border border-input bg-background px-2 py-1 text-sm focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2"
                value={threadsPerPage}
                onChange={(event) => {
                  const value = Number(event.target.value);
                  setThreadsPerPage(Number.isFinite(value) && value > 0 ? value : 10);
                }}
              >
                {THREADS_PER_PAGE_OPTIONS.map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
                ))}
              </select>
            </label>
            <span className="text-muted-foreground">
              Showing {filteredChats.length} thread{filteredChats.length === 1 ? "" : "s"} matching the current filters.
            </span>
          </div>
        </CardContent>
      </Card>

      {!paginatedThreads.length ? (
        <Card>
          <CardHeader>
            <CardTitle>No conversations match these filters</CardTitle>
            <CardDescription>Try widening the user or model selections to see more threads.</CardDescription>
          </CardHeader>
      </Card>
    ) : (
      <div className="space-y-4">
        {paginatedThreads.map(({ chat, messages: threadMessages, firstUserTimestamp }) => {
          const startedLabel = formatTimestamp(firstUserTimestamp);
          const attachmentNames = chat.files
            .map((file) => getFileLabel(file))
            .filter((name): name is string => Boolean(name));
          const headerMeta = [
            startedLabel ? `Started ${startedLabel}` : null,
            chat.userDisplay || null,
            chat.filesUploaded > 0 ? `📎 ${chat.filesUploaded} attachment${chat.filesUploaded === 1 ? "" : "s"}` : null
          ]
            .filter(Boolean)
            .join(" • ");
          const isOpen = expandedChatId === chat.chatId;

          return (
            <div className="overflow-hidden rounded-lg border border-muted bg-card" key={chat.chatId}>
              <button
                type="button"
                className="flex w-full items-center justify-between gap-4 px-4 py-3 text-left transition hover:bg-muted"
                aria-expanded={isOpen}
                onClick={() => setExpandedChatId(isOpen ? null : chat.chatId)}
              >
                <div>
                  <p className="text-sm font-semibold text-foreground">{chat.title || chat.chatId}</p>
                  {headerMeta && <p className="text-xs text-muted-foreground">{headerMeta}</p>}
                </div>
                <span className="text-lg font-medium text-muted-foreground">{isOpen ? "−" : "+"}</span>
              </button>
              {isOpen && (
                <div className="space-y-4 border-t border-muted bg-background px-4 py-4">
                  {summarizerEnabled && (
                    <div className="rounded-md border border-indigo-200 bg-indigo-50 px-4 py-3 text-sm text-slate-800">
                      <div className="mb-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs font-medium uppercase tracking-wide text-indigo-700">
                        <span>🧠 Summary</span>
                        {chat.userDisplay && <span>👤 {chat.userDisplay}</span>}
                        {chat.models.length > 0 && <span>🤖 {chat.models[0]}</span>}
                        {formatOutcome(chat.outcome) && <span>{formatOutcome(chat.outcome)}</span>}
                      </div>
                      <p className="whitespace-pre-wrap leading-relaxed">
                        {chat.summary?.trim() ? chat.summary.trim() : "No summary available."}
                      </p>
                    </div>
                  )}

                  <div className="space-y-3">
                    {threadMessages.map((message) => {
                      const isUser = message.role === "user";
                      const background = isUser ? "bg-sky-50 border-sky-200" : "bg-slate-100 border-slate-200";
                      const author = isUser
                        ? chat.userDisplay || "User"
                        : `🤖 Assistant${message.model ? ` (${message.model})` : ""}`;

                      return (
                        <div className={`rounded-md border px-3 py-2 text-sm ${background}`} key={message.messageId}>
                          <div className="mb-1 flex flex-wrap items-center justify-between gap-2 text-xs font-medium text-muted-foreground">
                            <span>{author}</span>
                            <span>{formatTimestamp(message.timestamp)}</span>
                          </div>
                          <p className="whitespace-pre-wrap leading-relaxed text-slate-900">
                            {message.content?.trim() ? message.content : "(no content)"}
                          </p>
                        </div>
                      );
                    })}
                  </div>

                  {attachmentNames.length > 0 && (
                    <div className="rounded-md border border-slate-200 bg-slate-50 px-3 py-2 text-sm">
                      <p className="font-medium text-slate-700">Attachments</p>
                      <ul className="mt-1 list-disc pl-5 text-slate-600">
                        {attachmentNames.map((name) => (
                          <li key={name}>{name}</li>
                        ))}
                      </ul>
                    </div>
                  )}

                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <div className="text-xs text-muted-foreground">
                      Updated {formatTimestamp(chat.updatedAt)} • Chat ID {chat.chatId}
                    </div>
                    <Button variant="outline" onClick={() => handleDownload(chat, threadMessages)}>
                      Download Thread (JSON)
                    </Button>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    )}

      {threads.length > 0 && (
        <div className="flex flex-wrap items-center justify-between gap-3 rounded-md border border-muted bg-background px-4 py-3 text-sm">
          <Button
            type="button"
            variant="outline"
            disabled={page <= 1}
            onClick={() => setPage((prev) => Math.max(1, prev - 1))}
          >
            Previous
          </Button>
          <span className="text-muted-foreground">
            Page {page} of {totalPages}
          </span>
          <Button
            type="button"
            variant="outline"
            disabled={page >= totalPages}
            onClick={() => setPage((prev) => Math.min(totalPages, prev + 1))}
          >
            Next
          </Button>
        </div>
      )}
    </div>
  );
}
