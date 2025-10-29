"use client";

import { useMemo } from "react";
import { ResponsiveContainer, CartesianGrid, Cell, XAxis, YAxis, Tooltip, ScatterChart, Scatter } from "recharts";
import type { ContentChat } from "@/lib/content-analysis";

interface TopicDataPoint {
  topic: string;
  week: string;
  count: number;
  intensity: number;
}

interface TopicHeatmapProps {
  chats: ContentChat[];
}

function parseTopicsFromChats(chats: ContentChat[]): Map<string, Map<string, number>> {
  // Map of topic -> week -> count
  const topicsByWeek = new Map<string, Map<string, number>>();

  for (const chat of chats) {
    if (!chat.genTopics) continue;

    // Split the comma-separated topics
    const topics = chat.genTopics
      .split(",")
      .map((t) => t.trim().toLowerCase())
      .filter((t) => t.length > 0);

    // For now, we'll use a simple "All Chats" category since we don't have timestamps
    // In a future enhancement, we could group by week/month if we add timestamps to ContentChat
    const weekLabel = "All Chats";

    for (const topic of topics) {
      if (!topicsByWeek.has(topic)) {
        topicsByWeek.set(topic, new Map());
      }
      const weekMap = topicsByWeek.get(topic)!;
      weekMap.set(weekLabel, (weekMap.get(weekLabel) || 0) + 1);
    }
  }

  return topicsByWeek;
}

function buildHeatmapData(topicsByWeek: Map<string, Map<string, number>>): TopicDataPoint[] {
  const data: TopicDataPoint[] = [];

  // Find the maximum count for normalization
  let maxCount = 0;
  for (const weekMap of topicsByWeek.values()) {
    for (const count of weekMap.values()) {
      if (count > maxCount) maxCount = count;
    }
  }

  // Build data points
  const topicNames = Array.from(topicsByWeek.keys()).sort((a, b) => {
    // Sort by total frequency (descending)
    const aTotal = Array.from(topicsByWeek.get(a)!.values()).reduce((sum, c) => sum + c, 0);
    const bTotal = Array.from(topicsByWeek.get(b)!.values()).reduce((sum, c) => sum + c, 0);
    return bTotal - aTotal;
  });

  // Limit to top 20 topics
  const topTopics = topicNames.slice(0, 20);

  topTopics.forEach((topic, topicIndex) => {
    const weekMap = topicsByWeek.get(topic)!;
    const weeks = Array.from(weekMap.keys()).sort();

    weeks.forEach((week, weekIndex) => {
      const count = weekMap.get(week) || 0;
      const intensity = maxCount > 0 ? count / maxCount : 0;

      data.push({
        topic,
        week,
        count,
        intensity,
      });
    });
  });

  return data;
}

function getColorForIntensity(intensity: number): string {
  // Use a blue color scale
  const baseColor = [59, 130, 246]; // blue-500
  const lightColor = [239, 246, 255]; // blue-50

  const r = Math.round(lightColor[0] + (baseColor[0] - lightColor[0]) * intensity);
  const g = Math.round(lightColor[1] + (baseColor[1] - lightColor[1]) * intensity);
  const b = Math.round(lightColor[2] + (baseColor[2] - lightColor[2]) * intensity);

  return `rgb(${r}, ${g}, ${b})`;
}

export function TopicHeatmap({ chats }: TopicHeatmapProps) {
  const heatmapData = useMemo(() => {
    const topicsByWeek = parseTopicsFromChats(chats);
    return buildHeatmapData(topicsByWeek);
  }, [chats]);

  const topics = useMemo(() => {
    return Array.from(new Set(heatmapData.map((d) => d.topic)));
  }, [heatmapData]);

  const weeks = useMemo(() => {
    return Array.from(new Set(heatmapData.map((d) => d.week)));
  }, [heatmapData]);

  if (heatmapData.length === 0) {
    return (
      <div className="flex h-64 items-center justify-center text-sm text-muted-foreground">
        No topic data available. Run the summarizer to generate topic insights.
      </div>
    );
  }

  // Transform data for scatter plot with topic on Y axis
  const scatterData = heatmapData.map((point, index) => ({
    x: weeks.indexOf(point.week),
    y: topics.indexOf(point.topic),
    count: point.count,
    intensity: point.intensity,
    topic: point.topic,
    week: point.week,
  }));

  return (
    <ResponsiveContainer width="100%" height={Math.max(300, topics.length * 30)}>
      <ScatterChart
        margin={{ top: 20, right: 20, bottom: 20, left: 150 }}
      >
        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
        <XAxis
          type="number"
          dataKey="x"
          name="Period"
          tickFormatter={(value) => weeks[value] || ""}
          domain={[0, weeks.length - 1]}
          ticks={Array.from({ length: weeks.length }, (_, i) => i)}
        />
        <YAxis
          type="number"
          dataKey="y"
          name="Topic"
          tickFormatter={(value) => {
            const topic = topics[value];
            return topic ? topic.slice(0, 20) + (topic.length > 20 ? "..." : "") : "";
          }}
          domain={[0, topics.length - 1]}
          ticks={Array.from({ length: topics.length }, (_, i) => i)}
          width={140}
        />
        <Tooltip
          content={({ payload }) => {
            if (!payload || payload.length === 0) return null;
            const data = payload[0].payload;
            return (
              <div className="rounded-lg border bg-background p-3 shadow-lg">
                <p className="font-semibold">{data.topic}</p>
                <p className="text-sm text-muted-foreground">{data.week}</p>
                <p className="text-sm">
                  Count: <span className="font-medium">{data.count}</span>
                </p>
              </div>
            );
          }}
        />
        <Scatter data={scatterData} shape="square">
          {scatterData.map((entry, index) => (
            <Cell key={`cell-${index}`} fill={getColorForIntensity(entry.intensity)} />
          ))}
        </Scatter>
      </ScatterChart>
    </ResponsiveContainer>
  );
}
