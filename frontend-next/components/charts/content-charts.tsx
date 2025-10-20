"use client";

import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { AverageLengthDatum, HistogramBin } from "@/lib/content-analysis";

interface AverageMessageLengthChartProps {
  data: AverageLengthDatum[];
}

interface MessageLengthHistogramProps {
  data: HistogramBin[];
}

function formatAverageTooltip(value: number) {
  return [`${value.toFixed(1)} characters`, "Average Length"] as const;
}

function renderEmptyState(message: string) {
  return (
    <div className="flex h-72 items-center justify-center text-sm text-muted-foreground">
      {message}
    </div>
  );
}

export function AverageMessageLengthChart({ data }: AverageMessageLengthChartProps) {
  if (!data.length) {
    return renderEmptyState("Average message length chart unavailable — no messages with text content.");
  }

  return (
    <div className="h-72 w-full">
      <ResponsiveContainer height="100%" width="100%">
        <BarChart data={data} margin={{ top: 16, right: 24, bottom: 24, left: 12 }}>
          <CartesianGrid stroke="#e4e4e7" strokeDasharray="4 4" />
          <XAxis dataKey="role" stroke="#71717a" />
          <YAxis allowDecimals={false} stroke="#71717a" />
          <Tooltip formatter={(value: number) => formatAverageTooltip(value)} />
          <Bar dataKey="averageLength" fill="#6366f1" name="Average Characters" radius={[6, 6, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function MessageLengthHistogram({ data }: MessageLengthHistogramProps) {
  if (!data.length) {
    return renderEmptyState("Message length distribution unavailable — no textual messages.");
  }

  return (
    <div className="h-72 w-full">
      <ResponsiveContainer height="100%" width="100%">
        <BarChart data={data} margin={{ top: 16, right: 24, bottom: 48, left: 12 }}>
          <CartesianGrid stroke="#e4e4e7" strokeDasharray="4 4" />
          <XAxis
            angle={-35}
            dataKey="range"
            height={70}
            interval={0}
            stroke="#71717a"
            textAnchor="end"
          />
          <YAxis allowDecimals={false} stroke="#71717a" />
          <Tooltip formatter={(value: number) => [`${value} messages`, "Messages"]} />
          <Bar dataKey="count" fill="#0ea5e9" name="Messages" radius={[6, 6, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

