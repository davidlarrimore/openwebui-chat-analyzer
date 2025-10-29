"use client";

import { useMemo } from "react";
import { ResponsiveContainer, Treemap, Tooltip } from "recharts";
import type { ContentChat } from "@/lib/content-analysis";

interface TreemapDatum {
  name: string;
  value: number;
  fill: string;
}

interface TopicTreemapProps {
  chats: ContentChat[];
}

const COLOR_PALETTE = [
  "#2563eb",
  "#1d4ed8",
  "#1e3a8a",
  "#0ea5e9",
  "#0284c7",
  "#38bdf8",
  "#4338ca",
  "#7c3aed",
  "#a855f7",
  "#c084fc",
  "#db2777",
  "#e11d48",
  "#f43f5e",
  "#f97316",
  "#fb923c",
  "#f59e0b",
  "#22c55e",
  "#16a34a",
  "#14b8a6",
  "#0f766e",
  "#8b5cf6",
  "#3b82f6",
  "#ec4899",
  "#ef4444",
  "#10b981"
];

function formatTagLabel(value: string): string {
  const trimmed = value.trim();
  if (!trimmed.length) {
    return "Unknown";
  }

  const hasMixedCase = /[a-z]/.test(trimmed) && /[A-Z]/.test(trimmed);
  if (hasMixedCase) {
    return trimmed;
  }

  return trimmed
    .toLowerCase()
    .split(/\s+/)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function buildTreemapData(chats: ContentChat[]): TreemapDatum[] {
  const counts = new Map<string, { name: string; value: number }>();

  for (const chat of chats) {
    if (!Array.isArray(chat.tags) || chat.tags.length === 0) {
      continue;
    }

    for (const rawTag of chat.tags) {
      const tag = rawTag?.toString().trim();
      if (!tag) {
        continue;
      }
      const key = tag.toLowerCase();
      const existing = counts.get(key);

      if (existing) {
        existing.value += 1;
        continue;
      }

      counts.set(key, {
        name: formatTagLabel(tag),
        value: 1
      });
    }
  }

  const sorted = Array.from(counts.values()).sort((a, b) => b.value - a.value);

  return sorted.slice(0, 25).map((entry, index) => ({
    ...entry,
    fill: COLOR_PALETTE[index % COLOR_PALETTE.length]
  }));
}

export function TopicTreemap({ chats }: TopicTreemapProps) {
  const treemapData = useMemo(() => buildTreemapData(chats), [chats]);

  if (treemapData.length === 0) {
    return (
      <div className="flex h-64 items-center justify-center text-sm text-muted-foreground">
        No tag data available yet. Add tags to Open WebUI chats to populate this treemap.
      </div>
    );
  }

  return (
    <div className="h-[360px]">
      <ResponsiveContainer height="100%" width="100%">
        <Treemap
          data={treemapData}
          dataKey="value"
          stroke="#fff"
          animationDuration={400}
          aspectRatio={4 / 3}
        >
          <Tooltip
            content={({ payload }) => {
              if (!payload || payload.length === 0) {
                return null;
              }
              const datum = payload[0].payload as TreemapDatum;
              return (
                <div className="rounded-lg border bg-background p-3 shadow-lg">
                  <p className="font-semibold">{datum.name}</p>
                  <p className="text-sm">
                    {datum.value.toLocaleString()} chats with this tag
                  </p>
                </div>
              );
            }}
          />
        </Treemap>
      </ResponsiveContainer>
    </div>
  );
}
