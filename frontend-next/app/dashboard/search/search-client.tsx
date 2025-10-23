"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { ALL_MODELS_OPTION, ALL_USERS_OPTION, BrowseChat, BrowseMessage, buildModelOptions, buildThreadExportPayload, buildUserOptions, filterMessagesByUserAndModel } from "@/lib/browse";
import { DISPLAY_TIMEZONE } from "@/lib/timezone";

interface SearchClientProps {
  chats: BrowseChat[];
  messages: BrowseMessage[];
}

interface SearchResultEntry {
  chat: BrowseChat;
  messages: BrowseMessage[];
  firstTimestamp: number;
}

type RoleFilterOption = "All" | "user" | "assistant";

const MAX_CONVERSATIONS_OPTIONS = [5, 10, 15, 20] as const;

const DATE_TIME_FORMATTER = new Intl.DateTimeFormat("en-US", {
  dateStyle: "medium",
  timeStyle: "short",
  timeZone: DISPLAY_TIMEZONE
});

function formatTimestamp(value: string | number | null | undefined): string {
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

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function highlightContent(content: string, query: string): (string | JSX.Element)[] {
  if (!query.trim()) {
    return [content];
  }

  const safeQuery = escapeRegExp(query.trim());
  const regex = new RegExp(`(${safeQuery})`, "gi");
  const segments = content.split(regex);
  const result: (string | JSX.Element)[] = [];

  for (let index = 0; index < segments.length; index += 1) {
    const segment = segments[index];
    if (!segment) {
      continue;
    }
    const isMatch = index % 2 === 1;
    if (isMatch) {
      result.push(
        <mark className="rounded-sm bg-amber-200 px-1 py-0.5 text-foreground" key={`match-${index}`}>
          {segment}
        </mark>
      );
    } else {
      result.push(segment);
    }
  }

  return result;
}

export default function SearchClient({ chats, messages }: SearchClientProps) {
  const [selectedUser, setSelectedUser] = useState<string>(ALL_USERS_OPTION);
  const [selectedModel, setSelectedModel] = useState<string>(ALL_MODELS_OPTION);
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [roleFilter, setRoleFilter] = useState<RoleFilterOption>("All");
  const [maxConversations, setMaxConversations] = useState<number>(10);
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

  const conversationLookup = useMemo(() => {
    const grouped = new Map<string, BrowseMessage[]>();
    for (const message of filtered.filteredMessages) {
      const existing = grouped.get(message.chatId);
      if (existing) {
        existing.push(message);
      } else {
        grouped.set(message.chatId, [message]);
      }
    }
    return grouped;
  }, [filtered.filteredMessages]);

  const searchResults = useMemo(() => {
    const trimmedQuery = searchQuery.trim();
    if (!trimmedQuery) {
      return {
        matches: [] as SearchResultEntry[],
        totalMatches: 0
      };
    }

    const normalizedQuery = trimmedQuery.toLowerCase();
    let matchingMessages = filtered.filteredMessages.filter((message) =>
      (message.content ?? "").toLowerCase().includes(normalizedQuery)
    );

    if (roleFilter !== "All") {
      matchingMessages = matchingMessages.filter((message) => message.role === roleFilter);
    }

    const matchedChatIds = new Set<string>();
    for (const message of matchingMessages) {
      matchedChatIds.add(message.chatId);
    }

    const rolePriority = (role: string) => {
      if (role === "user") return 0;
      if (role === "assistant") return 1;
      return 2;
    };

    const unsortedMatches: SearchResultEntry[] = [];
    for (const chatId of matchedChatIds) {
      const chat = chats.find((entry) => entry.chatId === chatId);
      const threadMessages = conversationLookup.get(chatId);
      if (!chat || !threadMessages || !threadMessages.length) {
        continue;
      }

      const sortedMessages = [...threadMessages].sort((left, right) => {
        const delta = timestampToNumber(left.timestamp) - timestampToNumber(right.timestamp);
        if (delta !== 0) {
          return delta;
        }
        return rolePriority(left.role) - rolePriority(right.role);
      });

      const firstTimestamp = sortedMessages.length ? timestampToNumber(sortedMessages[0]?.timestamp) : 0;
      unsortedMatches.push({
        chat,
        messages: sortedMessages,
        firstTimestamp
      });
    }

    unsortedMatches.sort((a, b) => a.firstTimestamp - b.firstTimestamp);

    return {
      matches: unsortedMatches,
      totalMatches: matchedChatIds.size
    };
  }, [chats, conversationLookup, filtered.filteredMessages, roleFilter, searchQuery]);

  const paginatedMatches = useMemo(() => {
    if (!searchResults.matches.length) {
      return [];
    }
    return searchResults.matches.slice(0, maxConversations);
  }, [searchResults.matches, maxConversations]);

  useEffect(() => {
    if (!paginatedMatches.length) {
      setExpandedChatId(null);
      return;
    }
    if (expandedChatId && !paginatedMatches.some((entry) => entry.chat.chatId === expandedChatId)) {
      setExpandedChatId(null);
    }
  }, [expandedChatId, paginatedMatches]);

  const handleDownload = useCallback((chat: BrowseChat, threadMessages: BrowseMessage[]) => {
    const payload = buildThreadExportPayload(chat, threadMessages);
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `thread_search_${chat.chatId}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    setTimeout(() => URL.revokeObjectURL(link.href), 500);
  }, []);

  const trimmedQuery = searchQuery.trim();
  const showResults = Boolean(trimmedQuery);
  const hasMatches = paginatedMatches.length > 0;

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Filter Conversations</CardTitle>
          <CardDescription>Select a user or model to scope the search before querying transcripts.</CardDescription>
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
                  setExpandedChatId(null);
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
                  setExpandedChatId(null);
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
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Search Conversations</CardTitle>
          <CardDescription>Enter a keyword or phrase and refine by role or the number of result threads.</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-3">
            <label className="flex flex-col gap-2 text-sm font-medium md:col-span-1">
              <span>Keyword</span>
              <Input
                placeholder="Search message text…"
                value={searchQuery}
                onChange={(event) => setSearchQuery(event.target.value)}
              />
            </label>
            <label className="flex flex-col gap-2 text-sm font-medium">
              <span>Role</span>
              <select
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm shadow-sm focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2"
                value={roleFilter}
                onChange={(event) => setRoleFilter(event.target.value as RoleFilterOption)}
              >
                <option value="All">All</option>
                <option value="user">User</option>
                <option value="assistant">Assistant</option>
              </select>
            </label>
            <label className="flex flex-col gap-2 text-sm font-medium">
              <span>Max conversations</span>
              <select
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm shadow-sm focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2"
                value={maxConversations}
                onChange={(event) => {
                  const value = Number(event.target.value);
                  setMaxConversations(Number.isFinite(value) && value > 0 ? value : 10);
                }}
              >
                {MAX_CONVERSATIONS_OPTIONS.map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
                ))}
              </select>
            </label>
          </div>
          {!trimmedQuery && (
            <p className="mt-4 text-sm text-muted-foreground">
              Start typing a keyword to find conversations containing matching messages. Filters will further narrow the
              results.
            </p>
          )}
        </CardContent>
      </Card>

      {showResults && (
        <Card>
          <CardContent>
            {hasMatches ? (
              <p className="text-sm text-emerald-600">
                ✅ Found {searchResults.totalMatches} conversation{searchResults.totalMatches === 1 ? "" : "s"} containing
                &nbsp;
                <span className="font-semibold">&ldquo;{trimmedQuery}&rdquo;</span>.
              </p>
            ) : (
              <p className="text-sm text-muted-foreground">
                No conversations matched &ldquo;{trimmedQuery}&rdquo;. Try a different keyword or adjust the filters.
              </p>
            )}
          </CardContent>
        </Card>
      )}

      {showResults && hasMatches && (
        <div className="space-y-4">
          {paginatedMatches.map(({ chat, messages: threadMessages }, index) => {
            const attachmentNames = chat.files
              .map((file) => getFileLabel(file))
              .filter((name): name is string => Boolean(name));
            const headerMeta = [
              formatTimestamp(threadMessages[0]?.timestamp),
              chat.userDisplay || null,
              chat.models.length > 0 ? chat.models[0] : null,
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
                    <p className="text-sm font-semibold text-foreground">
                      Thread #{index + 1}: {chat.title || chat.chatId}
                    </p>
                    {headerMeta && <p className="text-xs text-muted-foreground">{headerMeta}</p>}
                  </div>
                  <span className="text-lg font-medium text-muted-foreground">{isOpen ? "−" : "+"}</span>
                </button>
                {isOpen && (
                  <div className="space-y-4 border-t border-muted bg-background px-4 py-4">
                    <div className="rounded-md border border-indigo-200 bg-indigo-50 px-4 py-3 text-sm text-slate-800">
                      <div className="mb-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs font-medium uppercase tracking-wide text-indigo-700">
                        <span>🧠 Summary</span>
                        {chat.userDisplay && <span>👤 {chat.userDisplay}</span>}
                        {chat.models.length > 0 && <span>🤖 {chat.models[0]}</span>}
                      </div>
                      <p className="whitespace-pre-wrap leading-relaxed">
                        {chat.summary?.trim() ? chat.summary.trim() : "No summary available."}
                      </p>
                    </div>

                    <div className="space-y-3">
                      {threadMessages.map((message) => {
                        const isUser = message.role === "user";
                        const background = isUser ? "bg-sky-50 border-sky-200" : "bg-slate-100 border-slate-200";
                        const author = isUser
                          ? chat.userDisplay || "User"
                          : `🤖 Assistant${message.model ? ` (${message.model})` : ""}`;
                        const content = message.content?.trim() ?? "";

                        return (
                          <div className={`rounded-md border px-3 py-2 text-sm ${background}`} key={message.messageId}>
                            <div className="mb-1 flex flex-wrap items-center justify-between gap-2 text-xs font-medium text-muted-foreground">
                              <span>{author}</span>
                              <span>{formatTimestamp(message.timestamp)}</span>
                            </div>
                            <p className="whitespace-pre-wrap leading-relaxed text-slate-900">
                              {content ? highlightContent(content, trimmedQuery) : "(no content)"}
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

      {showResults && !hasMatches && (
        <Card>
          <CardHeader>
            <CardTitle>Need inspiration?</CardTitle>
            <CardDescription>
              Try searching for a common term like &ldquo;meeting&rdquo;, &ldquo;api&rdquo;, or &ldquo;summary&rdquo; to validate the search workflow.
            </CardDescription>
          </CardHeader>
        </Card>
      )}
    </div>
  );
}
