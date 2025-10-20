"use client";

import { Area, AreaChart, Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { ActivityHeatmapData, ConversationLengthBin, TimeSeriesPoint } from "@/lib/time-analysis";

interface DailyMessageAreaChartProps {
  data: TimeSeriesPoint[];
}

interface ConversationLengthHistogramProps {
  data: ConversationLengthBin[];
}

interface ActivityHeatmapProps {
  data: ActivityHeatmapData;
}

function renderEmptyState(message: string) {
  return (
    <div className="flex h-72 items-center justify-center text-sm text-muted-foreground">
      {message}
    </div>
  );
}

export function DailyMessageAreaChart({ data }: DailyMessageAreaChartProps) {
  if (!data.length) {
    return renderEmptyState("Messages over time unavailable — no timestamped messages to plot.");
  }

  const chartData = data.map((point) => ({
    ...point,
    label: new Date(`${point.date}T00:00:00Z`).toLocaleDateString(undefined, {
      month: "short",
      day: "numeric"
    })
  }));

  return (
    <div className="h-72 w-full">
      <ResponsiveContainer height="100%" width="100%">
        <AreaChart data={chartData} margin={{ top: 16, right: 24, bottom: 24, left: 12 }}>
          <defs>
            <linearGradient id="dailyMessagesFill" x1="0" x2="0" y1="0" y2="1">
              <stop offset="5%" stopColor="#2563eb" stopOpacity={0.35} />
              <stop offset="95%" stopColor="#2563eb" stopOpacity={0.05} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke="#e4e4e7" strokeDasharray="4 4" />
          <XAxis dataKey="label" stroke="#71717a" />
          <YAxis allowDecimals={false} stroke="#71717a" />
          <Tooltip
            formatter={(value: number) => [`${value.toLocaleString()} messages`, "Messages"]}
            labelFormatter={(label: string) => `Date: ${label}`}
          />
          <Area dataKey="value" fill="url(#dailyMessagesFill)" stroke="#2563eb" strokeWidth={2} type="monotone" />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

export function ConversationLengthHistogram({ data }: ConversationLengthHistogramProps) {
  if (!data.length) {
    return renderEmptyState("Conversation length distribution unavailable — no chats with messages after filtering.");
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
          <Tooltip formatter={(value: number) => [`${value.toLocaleString()} chats`, "Chats"]} />
          <Bar dataKey="count" fill="#0ea5e9" name="Chats" radius={[6, 6, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function ActivityHeatmap({ data }: ActivityHeatmapProps) {
  const hasActivity = data.rows.some((row) => row.counts.some((count) => count > 0));

  if (!hasActivity) {
    return renderEmptyState("Activity heatmap unavailable — no timestamped messages in the selected range.");
  }

  const hours = Array.from({ length: 24 }, (_, index) => index);

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full border border-border text-xs">
        <thead>
          <tr>
            <th className="sticky left-0 z-10 bg-background px-2 py-2 text-right font-semibold">Day</th>
            {hours.map((hour) => (
              <th className="min-w-8 px-2 py-2 text-center font-normal text-muted-foreground" key={hour}>
                {hour}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.rows.map((row) => (
            <tr key={row.weekday}>
              <th className="sticky left-0 z-10 bg-background px-2 py-2 text-right font-medium">{row.weekday}</th>
              {row.counts.map((value, hour) => {
                const intensity = data.maxCount > 0 ? value / data.maxCount : 0;
                const backgroundColor = intensity
                  ? `rgba(37, 99, 235, ${0.15 + intensity * 0.55})`
                  : "rgba(226, 232, 240, 0.6)";
                const textColor = intensity > 0.6 ? "#f8fafc" : "#1f2937";
                const title = `${row.weekday} ${hour.toString().padStart(2, "0")}:00 • ${value.toLocaleString()} messages`;
                return (
                  <td
                    className="h-8 min-w-8 border border-border text-center align-middle"
                    key={`${row.weekday}-${hour}`}
                    style={{ backgroundColor, color: textColor }}
                    title={title}
                  >
                    {value > 0 ? value : ""}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
