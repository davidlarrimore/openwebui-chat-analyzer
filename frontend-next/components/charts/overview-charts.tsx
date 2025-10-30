"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  Line,
  LineChart
} from "recharts";
import type {
  AdoptionSeriesPoint,
  DailyActiveUsersPoint,
  ModelChatUsageDatum,
  ModelUsageDatum,
  PieDatum,
  TokenSeriesPoint,
  TopicUsageDatum
} from "@/lib/overview";
import { DISPLAY_TIMEZONE } from "@/lib/timezone";

const palette = ["#6366f1", "#38bdf8", "#f97316", "#22c55e", "#c084fc", "#f43f5e", "#14b8a6", "#facc15"];

function formatDateLabel(dateIso: string): string {
  const date = new Date(`${dateIso}T12:00:00Z`);
  return new Intl.DateTimeFormat("en-US", {
    timeZone: DISPLAY_TIMEZONE,
    month: "short",
    day: "numeric"
  }).format(date);
}

export interface TokenConsumptionChartProps {
  data: TokenSeriesPoint[];
}

export function TokenConsumptionChart({ data }: TokenConsumptionChartProps) {
  const chartData = data.map((point) => ({
    ...point,
    label: formatDateLabel(point.date)
  }));

  return (
    <div className="h-72 w-full">
      <ResponsiveContainer height="100%" width="100%">
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="4 4" stroke="#e4e4e7" />
          <XAxis dataKey="label" stroke="#71717a" />
          <YAxis stroke="#71717a" />
          <Tooltip
            formatter={(value: number) => [`${value.toLocaleString()} tokens`, "Tokens"]}
            labelFormatter={(label: string) => `Date: ${label}`}
          />
          <Line
            dataKey="tokens"
            name="Tokens"
            stroke="#6366f1"
            strokeWidth={2}
            type="monotone"
            dot={{ r: 2 }}
            activeDot={{ r: 4 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export interface ModelUsageBarChartProps {
  data: ModelUsageDatum[];
  height?: number;
}

export function ModelUsageBarChart({ data, height }: ModelUsageBarChartProps) {
  const chartHeight = height ?? 288;
  return (
    <div className="w-full" style={{ height: chartHeight }}>
      <ResponsiveContainer height="100%" width="100%">
        <BarChart data={data} layout="vertical" margin={{ top: 16, right: 24, bottom: 16, left: 80 }}>
          <CartesianGrid strokeDasharray="4 4" stroke="#e4e4e7" />
          <XAxis type="number" stroke="#71717a" />
          <YAxis dataKey="model" type="category" width={160} stroke="#71717a" />
          <Tooltip formatter={(value: number) => [`${value.toLocaleString()} messages`, "Messages"]} />
          <Bar dataKey="count" fill="#6366f1" name="Messages" radius={[4, 4, 4, 4]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export interface TopModelsChartProps {
  data: ModelChatUsageDatum[];
  height?: number;
}

export function TopModelsChart({ data, height }: TopModelsChartProps) {
  const chartHeight = height ?? 288;
  return (
    <div className="w-full" style={{ height: chartHeight }}>
      <ResponsiveContainer height="100%" width="100%">
        <BarChart data={data} layout="vertical" margin={{ top: 16, right: 24, bottom: 16, left: 80 }}>
          <CartesianGrid strokeDasharray="4 4" stroke="#e4e4e7" />
          <XAxis type="number" stroke="#71717a" />
          <YAxis dataKey="model" type="category" width={160} stroke="#71717a" />
          <Tooltip formatter={(value: number) => [`${value.toLocaleString()} chats`, "Chats"]} />
          <Bar dataKey="chatCount" fill="#6366f1" name="Chats" radius={[4, 4, 4, 4]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export interface ModelUsagePieChartProps {
  data: PieDatum[];
}

export function ModelUsagePieChart({ data }: ModelUsagePieChartProps) {
  return (
    <div className="h-72 w-full">
      <ResponsiveContainer height="100%" width="100%">
        <PieChart>
          <Tooltip />
          <Legend verticalAlign="bottom" height={32} />
          <Pie
            data={data}
            dataKey="value"
            nameKey="name"
            cx="50%"
            cy="50%"
            outerRadius="70%"
            labelLine={false}
            label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
          >
            {data.map((entry, index) => (
              <Cell fill={palette[index % palette.length]} key={entry.name} />
            ))}
          </Pie>
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
}

export interface TopTopicsChartProps {
  data: TopicUsageDatum[];
  height?: number;
}

export function TopTopicsChart({ data, height }: TopTopicsChartProps) {
  const chartHeight = height ?? 288;
  return (
    <div className="w-full" style={{ height: chartHeight }}>
      <ResponsiveContainer height="100%" width="100%">
        <BarChart data={data} layout="vertical" margin={{ top: 16, right: 24, bottom: 16, left: 120 }}>
          <CartesianGrid strokeDasharray="4 4" stroke="#e4e4e7" />
          <XAxis type="number" stroke="#71717a" />
          <YAxis dataKey="topic" type="category" width={200} stroke="#71717a" />
          <Tooltip formatter={(value: number) => [`${value.toLocaleString()} chats`, "Chats"]} />
          <Bar dataKey="chatCount" fill="#0ea5e9" name="Chats" radius={[4, 4, 4, 4]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export interface UserAdoptionChartProps {
  data: AdoptionSeriesPoint[];
}

export function UserAdoptionChart({ data }: UserAdoptionChartProps) {
  const chartData = data.map((point) => ({
    ...point,
    label: formatDateLabel(point.date)
  }));

  return (
    <div className="h-72 w-full">
      <ResponsiveContainer height="100%" width="100%">
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="4 4" stroke="#e4e4e7" />
          <XAxis dataKey="label" stroke="#71717a" />
          <YAxis stroke="#71717a" />
          <Tooltip
            formatter={(value: number) => [`${value.toLocaleString()} users`, "Users"]}
            labelFormatter={(label: string) => `Date: ${label}`}
          />
          <Line
            dataKey="value"
            name="Cumulative Users"
            stroke="#10b981"
            strokeWidth={2}
            type="monotone"
            dot={{ r: 2 }}
            activeDot={{ r: 4 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export interface DailyActiveUsersChartProps {
  data: DailyActiveUsersPoint[];
}

export function DailyActiveUsersChart({ data }: DailyActiveUsersChartProps) {
  const chartData = data.map((point) => ({
    ...point,
    label: formatDateLabel(point.date)
  }));

  return (
    <div className="h-72 w-full">
      <ResponsiveContainer height="100%" width="100%">
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="4 4" stroke="#e4e4e7" />
          <XAxis dataKey="label" stroke="#71717a" />
          <YAxis stroke="#71717a" />
          <Tooltip
            formatter={(value: number) => [`${value.toLocaleString()} users`, "Active Users"]}
            labelFormatter={(label: string) => `Date: ${label}`}
          />
          <Line
            dataKey="activeUsers"
            name="Daily Active Users"
            stroke="#f97316"
            strokeWidth={2}
            type="monotone"
            dot={{ r: 2 }}
            activeDot={{ r: 4 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
