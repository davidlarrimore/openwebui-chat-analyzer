"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";
import type {
  ModelUsageShareDatum,
  ModelUsageTimelinePoint,
  ModelAverageTokensDatum,
  ModelTopicDistributionDatum
} from "@/lib/model-analysis";
import { DISPLAY_TIMEZONE } from "@/lib/timezone";

const MODEL_COLOR_PALETTE = [
  "#1d4ed8",
  "#2563eb",
  "#38bdf8",
  "#0ea5e9",
  "#14b8a6",
  "#22c55e",
  "#f97316",
  "#f59e0b",
  "#f43f5e",
  "#a855f7"
];

const OTHER_SERIES_COLOR = "#94a3b8";

function getSeriesColor(index: number, model: string): string {
  if (model === "Other") {
    return OTHER_SERIES_COLOR;
  }
  return MODEL_COLOR_PALETTE[index % MODEL_COLOR_PALETTE.length];
}

function formatDateLabel(dateIso: string): string {
  const date = new Date(`${dateIso}T12:00:00Z`);
  return new Intl.DateTimeFormat("en-US", {
    timeZone: DISPLAY_TIMEZONE,
    month: "short",
    day: "numeric"
  }).format(date);
}

interface ModelUsageShareBarChartProps {
  data: ModelUsageShareDatum[];
}

export function ModelUsageShareBarChart({ data }: ModelUsageShareBarChartProps) {
  if (!data.length) {
    return (
      <div className="flex h-80 items-center justify-center text-sm text-muted-foreground">
        We need assistant messages with model metadata to calculate usage share.
      </div>
    );
  }

  const chartData = data.map((item) => ({
    ...item,
    percentageLabel: `${item.percentage.toFixed(1)}%`
  }));

  return (
    <div className="w-full" style={{ height: 360 }}>
      <ResponsiveContainer height="100%" width="100%">
        <BarChart data={chartData} layout="vertical" margin={{ top: 16, right: 32, bottom: 16, left: 180 }}>
          <CartesianGrid strokeDasharray="4 4" stroke="#e4e4e7" />
          <XAxis
            type="number"
            domain={[0, 100]}
            tickFormatter={(value: number) => `${value.toFixed(0)}%`}
            stroke="#71717a"
          />
          <YAxis dataKey="model" type="category" stroke="#71717a" />
          <Tooltip
            formatter={(value: number, _name, props) => {
              const datum = props.payload as (ModelUsageShareDatum & { percentageLabel: string }) | undefined;
              const shareLabel = `${value.toFixed(1)}%`;
              if (!datum) {
                return [shareLabel, "Usage Share"];
              }
              return [`${shareLabel} • ${datum.count.toLocaleString()} messages`, "Usage Share"];
            }}
            labelFormatter={(value) => `Model: ${value}`}
          />
          <Bar dataKey="percentage" name="Usage Share" fill="#2563eb" radius={[4, 4, 4, 4]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

interface ModelUsageTimelineChartProps {
  data: ModelUsageTimelinePoint[];
  models: string[];
}

export function ModelUsageTimelineChart({ data, models }: ModelUsageTimelineChartProps) {
  if (!data.length || !models.length) {
    return (
      <div className="flex h-96 items-center justify-center text-sm text-muted-foreground">
        Need timestamped assistant messages to build the model usage timeline.
      </div>
    );
  }

  const chartData = data.map((row) => ({
    ...row,
    label: formatDateLabel(row.date)
  }));

  return (
    <div className="w-full" style={{ height: 400 }}>
      <ResponsiveContainer height="100%" width="100%">
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="4 4" stroke="#e4e4e7" />
          <XAxis dataKey="label" stroke="#71717a" />
          <YAxis stroke="#71717a" />
          <Tooltip
            formatter={(value: number, seriesKey: string) => [`${value.toLocaleString()} messages`, seriesKey]}
            labelFormatter={(label: string) => `Date: ${label}`}
          />
          <Legend />
          {models.map((model, index) => (
            <Line
              key={model}
              dataKey={model}
              name={model}
              stroke={getSeriesColor(index, model)}
              strokeWidth={2}
              type="monotone"
              dot={false}
              activeDot={{ r: 4 }}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

interface ModelAverageTokensChartProps {
  data: ModelAverageTokensDatum[];
}

export function ModelAverageTokensChart({ data }: ModelAverageTokensChartProps) {
  if (!data.length) {
    return (
      <div className="flex h-80 items-center justify-center text-sm text-muted-foreground">
        We need assistant token counts to calculate model response lengths.
      </div>
    );
  }

  const chartData = data.map((item) => ({
    ...item,
    averageTokens: Number(item.averageTokens.toFixed(1))
  }));

  return (
    <div className="w-full" style={{ height: 320 }}>
      <ResponsiveContainer height="100%" width="100%">
        <BarChart data={chartData} margin={{ top: 16, right: 24, bottom: 48, left: 24 }}>
          <CartesianGrid strokeDasharray="4 4" stroke="#e4e4e7" />
          <XAxis
            dataKey="model"
            stroke="#71717a"
            tick={{ fontSize: 11 }}
            interval={0}
            angle={-30}
            textAnchor="end"
          />
          <YAxis stroke="#71717a" />
          <Tooltip
            formatter={(value: number, _name, props) => {
              const datum = props.payload as ModelAverageTokensDatum | undefined;
              const tokensLabel = `${value.toLocaleString(undefined, { maximumFractionDigits: 1 })} tokens`;
              const responsesLabel = datum ? `${datum.messageCount.toLocaleString()} responses` : "Average Tokens";
              return [tokensLabel, responsesLabel];
            }}
            labelFormatter={(label) => `Model: ${label}`}
          />
          <Bar dataKey="averageTokens" fill="#a855f7" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

interface ModelTopicStackedBarChartProps {
  data: ModelTopicDistributionDatum[];
  models: string[];
}

export function ModelTopicStackedBarChart({ data, models }: ModelTopicStackedBarChartProps) {
  if (!data.length || !models.length) {
    return (
      <div className="flex h-80 items-center justify-center text-sm text-muted-foreground">
        Tag data not available for the selected models. Apply tags in Open WebUI to populate this chart.
      </div>
    );
  }

  const chartData = data.map((row) => ({
    ...row,
    tagLabel: row.tag
  }));

  return (
    <div className="w-full" style={{ height: 360 }}>
      <ResponsiveContainer height="100%" width="100%">
        <BarChart data={chartData} layout="vertical" margin={{ top: 16, right: 32, bottom: 16, left: 160 }}>
          <CartesianGrid strokeDasharray="4 4" stroke="#e4e4e7" />
          <XAxis type="number" stroke="#71717a" />
          <YAxis dataKey="tagLabel" type="category" stroke="#71717a" />
          <Tooltip
            formatter={(value: number, seriesKey: string) => [`${value.toLocaleString()} chats`, seriesKey]}
            labelFormatter={(label: string) => `Tag: ${label}`}
          />
          <Legend />
          {models.map((model, index) => (
            <Bar
              key={model}
              dataKey={model}
              stackId="topics"
              name={model}
              fill={getSeriesColor(index, model)}
              radius={index === models.length - 1 ? [0, 4, 4, 0] : undefined}
            />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
