"use client";

import { ResponsiveContainer, LineChart, CartesianGrid, Line, Tooltip, XAxis, YAxis } from "recharts";
import type { TimeSeriesPoint } from "@/lib/types";

interface LineTimeseriesProps {
  data: TimeSeriesPoint[];
}

export function LineTimeseries({ data }: LineTimeseriesProps) {
  const chartData = data.map((point) => ({
    date: new Date(point.date).toLocaleDateString(),
    value: point.value
  }));

  return (
    <div className="h-72 w-full">
      <ResponsiveContainer height="100%" width="100%">
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="4 4" stroke="#d4d4d8" />
          <XAxis dataKey="date" stroke="#71717a" />
          <YAxis stroke="#71717a" />
          <Tooltip />
          <Line dataKey="value" stroke="#2563eb" strokeWidth={2} type="monotone" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
