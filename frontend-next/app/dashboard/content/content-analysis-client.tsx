"use client";

import { useEffect, useMemo, useState } from "react";
import { AverageMessageLengthChart, MessageLengthHistogram } from "@/components/charts/content-charts";
import { WordCloud } from "@/components/charts/word-cloud";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import {
  ALL_MODELS_OPTION,
  ALL_USERS_OPTION,
  buildModelOptions,
  buildUserOptions,
  calculateAverageMessageLength,
  buildMessageLengthHistogram,
  extractWordFrequencies,
  filterMessagesByUserAndModel,
  hasTextContent,
  hasUserMessages,
  type ContentChat,
  type ContentMessage
} from "@/lib/content-analysis";

interface ContentAnalysisClientProps {
  chats: ContentChat[];
  messages: ContentMessage[];
  userDisplayMap: Record<string, string>;
}

function SummaryPill({ children }: { children: React.ReactNode }) {
  return <span className="rounded-full bg-primary/10 px-3 py-1 text-xs font-medium text-primary">{children}</span>;
}

function AlertMessage({ children }: { children: React.ReactNode }) {
  return (
    <div className="rounded-md border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-800">
      {children}
    </div>
  );
}

export function ContentAnalysisClient({ chats, messages, userDisplayMap }: ContentAnalysisClientProps) {
  const [selectedUser, setSelectedUser] = useState<string>(ALL_USERS_OPTION);
  const [selectedModel, setSelectedModel] = useState<string>(ALL_MODELS_OPTION);

  const userOptions = useMemo(() => buildUserOptions(chats, userDisplayMap), [chats, userDisplayMap]);
  const modelOptions = useMemo(() => buildModelOptions(messages), [messages]);

  useEffect(() => {
    if (!userOptions.find((option) => option.value === selectedUser)) {
      setSelectedUser(ALL_USERS_OPTION);
    }
  }, [selectedUser, userOptions]);

  useEffect(() => {
    if (!modelOptions.includes(selectedModel)) {
      setSelectedModel(ALL_MODELS_OPTION);
    }
  }, [modelOptions, selectedModel]);

  const { filteredMessages, matchingChatIds } = useMemo(
    () => filterMessagesByUserAndModel(chats, messages, selectedUser, selectedModel),
    [chats, messages, selectedModel, selectedUser]
  );

  const averageLengthData = useMemo(() => calculateAverageMessageLength(filteredMessages), [filteredMessages]);
  const histogramData = useMemo(() => buildMessageLengthHistogram(filteredMessages), [filteredMessages]);
  const wordFrequencies = useMemo(() => extractWordFrequencies(filteredMessages), [filteredMessages]);

  const userMessagesAvailable = useMemo(() => hasUserMessages(filteredMessages), [filteredMessages]);
  const textAvailable = useMemo(() => hasTextContent(filteredMessages), [filteredMessages]);

  let wordCloudMessage: string | null = null;
  if (!filteredMessages.length) {
    wordCloudMessage = "Word cloud unavailable — no messages available. Upload data or adjust filters.";
  } else if (!textAvailable) {
    wordCloudMessage = "Word cloud unavailable — messages are present but no textual content was found.";
  } else if (!userMessagesAvailable) {
    wordCloudMessage = "Word cloud unavailable — no messages labeled with role 'user' were found.";
  } else if (!wordFrequencies.length) {
    wordCloudMessage = "Word cloud unavailable — text processing did not surface any significant terms.";
  }

  return (
    <div className="space-y-6">
      <section className="space-y-2">
        <h1 className="text-2xl font-semibold tracking-tight text-foreground">Content Analysis</h1>
        <p className="text-sm text-muted-foreground">
          Explore user prompt trends and conversation lengths by narrowing in on specific users or models.
        </p>
        <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
          <SummaryPill>{filteredMessages.length.toLocaleString()} messages</SummaryPill>
          <SummaryPill>{matchingChatIds.length.toLocaleString()} chats</SummaryPill>
          <SummaryPill>
            User:{" "}
            {selectedUser === ALL_USERS_OPTION
              ? "All"
              : userDisplayMap[selectedUser] ?? selectedUser}
          </SummaryPill>
          <SummaryPill>Model: {selectedModel === ALL_MODELS_OPTION ? "All" : selectedModel}</SummaryPill>
        </div>
      </section>

      <Card>
        <CardHeader>
          <CardTitle>Filters</CardTitle>
          <CardDescription>Focus the metrics on a single user or assistant model.</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 sm:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor="content-user-filter">User</Label>
              <select
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm shadow-sm focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/40"
                id="content-user-filter"
                onChange={(event) => setSelectedUser(event.target.value)}
                value={selectedUser}
              >
                {userOptions.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </div>
            <div className="space-y-2">
              <Label htmlFor="content-model-filter">Model</Label>
              <select
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm shadow-sm focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/40"
                id="content-model-filter"
                onChange={(event) => setSelectedModel(event.target.value)}
                value={selectedModel}
              >
                {modelOptions.map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Word Cloud (User Messages)</CardTitle>
          <CardDescription>Most frequent user prompt terms after filtering out common stop words.</CardDescription>
        </CardHeader>
        <CardContent>
          {wordFrequencies.length ? <WordCloud words={wordFrequencies} /> : <AlertMessage>{wordCloudMessage}</AlertMessage>}
        </CardContent>
      </Card>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Average Message Length by Role</CardTitle>
            <CardDescription>Compares the average character count for user and assistant messages.</CardDescription>
          </CardHeader>
          <CardContent>
            <AverageMessageLengthChart data={averageLengthData} />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Message Length Distribution</CardTitle>
            <CardDescription>Shows how message lengths are distributed across the filtered set.</CardDescription>
          </CardHeader>
          <CardContent>
            <MessageLengthHistogram data={histogramData} />
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

