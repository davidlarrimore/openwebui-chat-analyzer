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
import type { AdoptionSeriesPoint, ModelUsageDatum, PieDatum, TokenSeriesPoint } from "@/lib/overview";

const DISPLAY_TIMEZONE = "America/New_York";
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
}

export function ModelUsageBarChart({ data }: ModelUsageBarChartProps) {
  return (
    <div className="h-72 w-full">
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
