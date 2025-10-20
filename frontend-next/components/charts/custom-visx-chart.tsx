"use client";

import { useMemo } from "react";
import { AxisBottom, AxisLeft } from "@visx/axis";
import { scaleLinear, scaleTime } from "@visx/scale";
import { LinePath } from "@visx/shape";
import { curveMonotoneX } from "@visx/curve";
import { cn } from "@/lib/utils";

interface CustomVisxChartProps {
  data: { timestamp: string; value: number; author?: string }[];
  width?: number;
  height?: number;
}

export function CustomVisxChart({ data, width = 640, height = 260 }: CustomVisxChartProps) {
  const margin = { top: 20, right: 20, bottom: 40, left: 50 };

  const parsedData = useMemo(
    () =>
      data.map((item) => ({
        x: new Date(item.timestamp),
        y: item.value
      })),
    [data]
  );

  const xScale = useMemo(
    () =>
      scaleTime<number>({
        range: [margin.left, width - margin.right],
        domain: [
          parsedData[0]?.x ?? new Date(),
          parsedData[parsedData.length - 1]?.x ?? new Date()
        ]
      }),
    [parsedData, width, margin.left, margin.right]
  );

  const yScale = useMemo(
    () =>
      scaleLinear<number>({
        range: [height - margin.bottom, margin.top],
        domain: [
          0,
          parsedData.reduce((max, item) => Math.max(max, item.y), 0) + 10
        ]
      }),
    [parsedData, height, margin.top, margin.bottom]
  );

  return (
    <div className="overflow-x-auto">
      <svg className={cn("rounded-md border bg-card", "text-muted-foreground")} height={height} width={width}>
        <AxisBottom
          scale={xScale}
          top={height - margin.bottom}
          tickFormat={(value) => new Date(value as Date).toLocaleTimeString()}
          stroke="#d4d4d8"
          tickStroke="#d4d4d8"
          tickLabelProps={() => ({ fill: "#71717a", fontSize: 10 })}
        />
        <AxisLeft
          scale={yScale}
          left={margin.left}
          stroke="#d4d4d8"
          tickStroke="#d4d4d8"
          tickLabelProps={() => ({ fill: "#71717a", fontSize: 10 })}
        />
        <LinePath
          curve={curveMonotoneX}
          data={parsedData}
          stroke="#2563eb"
          strokeWidth={2}
          x={(d) => xScale(d.x) ?? 0}
          y={(d) => yScale(d.y) ?? 0}
        />
      </svg>
    </div>
  );
}
