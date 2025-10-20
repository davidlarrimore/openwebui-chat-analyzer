"use client";

import type { ReactNode } from "react";
import { useEffect, useMemo, useState } from "react";
import { DailyMessageAreaChart, ConversationLengthHistogram, ActivityHeatmap } from "@/components/charts/time-charts";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import {
  ALL_MODELS_OPTION,
  ALL_USERS_OPTION,
  buildModelOptions,
  buildUserOptions,
  filterMessagesByUserAndModel
} from "@/lib/content-analysis";
import {
  buildActivityHeatmap,
  buildConversationLengthHistogram,
  buildDailyMessageSeries,
  summariseCounts,
  summariseDateRange,
  type ActivityHeatmapData,
  type ConversationLengthBin,
  type TimeAnalysisChat,
  type TimeAnalysisMessage,
  type TimeSeriesPoint
} from "@/lib/time-analysis";

interface TimeAnalysisClientProps {
  chats: TimeAnalysisChat[];
  messages: TimeAnalysisMessage[];
  userDisplayMap: Record<string, string>;
}

function SummaryPill({ children }: { children: ReactNode }) {
  return <span className="rounded-full bg-primary/10 px-3 py-1 text-xs font-medium text-primary">{children}</span>;
}

export function TimeAnalysisClient({ chats, messages, userDisplayMap }: TimeAnalysisClientProps) {
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

  const dailySeries: TimeSeriesPoint[] = useMemo(
    () => buildDailyMessageSeries(filteredMessages),
    [filteredMessages]
  );
  const conversationHistogram: ConversationLengthBin[] = useMemo(
    () => buildConversationLengthHistogram(filteredMessages),
    [filteredMessages]
  );
  const activityHeatmap: ActivityHeatmapData = useMemo(
    () => buildActivityHeatmap(filteredMessages),
    [filteredMessages]
  );

  const summaryCounts = useMemo(
    () => summariseCounts(filteredMessages, matchingChatIds),
    [filteredMessages, matchingChatIds]
  );

  const dateRange = useMemo(() => summariseDateRange(filteredMessages), [filteredMessages]);

  const selectedUserLabel =
    selectedUser === ALL_USERS_OPTION ? "All" : userDisplayMap[selectedUser] ?? selectedUser;
  const selectedModelLabel = selectedModel === ALL_MODELS_OPTION ? "All" : selectedModel;

  return (
    <div className="space-y-6">
      <section className="space-y-2">
        <h1 className="text-2xl font-semibold tracking-tight text-foreground">Time Analysis</h1>
        <p className="text-sm text-muted-foreground">
          Examine message cadence, conversation length distribution, and daily engagement heatmaps with the same filters
          available in the Streamlit dashboard.
        </p>
        <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
          <SummaryPill>{summaryCounts.messages.toLocaleString()} messages</SummaryPill>
          <SummaryPill>{summaryCounts.chats.toLocaleString()} chats</SummaryPill>
          <SummaryPill>User: {selectedUserLabel}</SummaryPill>
          <SummaryPill>Model: {selectedModelLabel}</SummaryPill>
          {dateRange.start && dateRange.end ? (
            <SummaryPill>
              Date Range: {dateRange.start} – {dateRange.end}
            </SummaryPill>
          ) : (
            <SummaryPill>Date Range: Unavailable</SummaryPill>
          )}
        </div>
      </section>

      <Card>
        <CardHeader>
          <CardTitle>Filters</CardTitle>
          <CardDescription>Limit the calculations to a single user or model.</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 sm:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor="time-user-filter">User</Label>
              <select
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm shadow-sm focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/40"
                id="time-user-filter"
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
              <Label htmlFor="time-model-filter">Model</Label>
              <select
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm shadow-sm focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/40"
                id="time-model-filter"
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

      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Messages Over Time</CardTitle>
            <CardDescription>Daily message volume with zero-fill for inactive days.</CardDescription>
          </CardHeader>
          <CardContent>
            <DailyMessageAreaChart data={dailySeries} />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Conversation Length Distribution</CardTitle>
            <CardDescription>Histogram of chat lengths after applying filters.</CardDescription>
          </CardHeader>
          <CardContent>
            <ConversationLengthHistogram data={conversationHistogram} />
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Activity Heatmap</CardTitle>
          <CardDescription>Hourly message counts by weekday in the selected timezone.</CardDescription>
        </CardHeader>
        <CardContent>
          <ActivityHeatmap data={activityHeatmap} />
        </CardContent>
      </Card>
    </div>
  );
}
