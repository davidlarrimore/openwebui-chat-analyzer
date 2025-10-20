import type { ContentChat, ContentMessage } from "./content-analysis";
import type { TimeSeriesPoint } from "./types";

export type { TimeSeriesPoint };

const MS_PER_DAY = 24 * 60 * 60 * 1000;
const DISPLAY_TIMEZONE = "America/New_York";
const ORDERED_WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];

const dateKeyFormatter = new Intl.DateTimeFormat("en-CA", {
  timeZone: DISPLAY_TIMEZONE,
  year: "numeric",
  month: "2-digit",
  day: "2-digit"
});

const weekdayFormatter = new Intl.DateTimeFormat("en-US", {
  timeZone: DISPLAY_TIMEZONE,
  weekday: "long"
});

const hourFormatter = new Intl.DateTimeFormat("en-US", {
  timeZone: DISPLAY_TIMEZONE,
  hour: "numeric",
  hour12: false
});

const numberFormatter = new Intl.NumberFormat();

export type TimeAnalysisChat = ContentChat;

export type TimeAnalysisMessage = ContentMessage;

export interface ConversationLengthBin {
  range: string;
  count: number;
  min: number;
  max: number;
}

export interface ActivityHeatmapRow {
  weekday: string;
  counts: number[];
}

export interface ActivityHeatmapData {
  rows: ActivityHeatmapRow[];
  maxCount: number;
}

export interface SummaryCounts {
  chats: number;
  messages: number;
}

export function buildDailyMessageSeries(messages: TimeAnalysisMessage[]): TimeSeriesPoint[] {
  const totals = new Map<string, number>();

  for (const message of messages) {
    const timestamp = parseTimestamp(message.timestamp);
    if (!timestamp) {
      continue;
    }
    const key = dateKeyFormatter.format(timestamp);
    totals.set(key, (totals.get(key) ?? 0) + 1);
  }

  if (!totals.size) {
    return [];
  }

  const keys = Array.from(totals.keys()).sort();
  const start = new Date(`${keys[0]}T00:00:00Z`);
  const end = new Date(`${keys[keys.length - 1]}T00:00:00Z`);

  const series: TimeSeriesPoint[] = [];
  for (let time = start.getTime(); time <= end.getTime(); time += MS_PER_DAY) {
    const cursor = new Date(time);
    const key = cursor.toISOString().slice(0, 10);
    series.push({
      date: key,
      value: totals.get(key) ?? 0
    });
  }

  return series;
}

export function buildConversationLengthHistogram(messages: TimeAnalysisMessage[], desiredBins = 20): ConversationLengthBin[] {
  const countsByChat = new Map<string, number>();

  for (const message of messages) {
    const chatId = message.chatId.trim();
    if (!chatId) {
      continue;
    }
    countsByChat.set(chatId, (countsByChat.get(chatId) ?? 0) + 1);
  }

  const counts = Array.from(countsByChat.values()).filter((value) => Number.isFinite(value) && value > 0);

  if (!counts.length) {
    return [];
  }

  const max = Math.max(...counts);
  const min = Math.min(...counts);
  const binCount = Math.max(1, Math.min(desiredBins, Math.ceil(Math.sqrt(counts.length))));
  const span = max - min + 1;
  const binSize = Math.max(1, Math.ceil(span / binCount));

  const bins: ConversationLengthBin[] = Array.from({ length: binCount }, (_, index) => {
    const lower = min + index * binSize;
    const upper = min + (index + 1) * binSize - 1;
    return {
      min: lower,
      max: upper,
      range: formatRange(lower, upper),
      count: 0
    };
  });

  for (const value of counts) {
    const offset = Math.floor((value - min) / binSize);
    const index = Math.min(bins.length - 1, Math.max(0, offset));
    bins[index].count += 1;
  }

  return bins;
}

export function buildActivityHeatmap(messages: TimeAnalysisMessage[]): ActivityHeatmapData {
  const matrix = new Map<string, Map<number, number>>();
  let maxCount = 0;

  for (const message of messages) {
    const timestamp = parseTimestamp(message.timestamp);
    if (!timestamp) {
      continue;
    }
    const weekday = weekdayFormatter.format(timestamp);
    const hourRaw = hourFormatter.format(timestamp);
    const hour = Number.parseInt(hourRaw, 10);
    if (!Number.isFinite(hour)) {
      continue;
    }

    const row = matrix.get(weekday) ?? new Map<number, number>();
    const next = (row.get(hour) ?? 0) + 1;
    row.set(hour, next);
    matrix.set(weekday, row);

    if (next > maxCount) {
      maxCount = next;
    }
  }

  const rows: ActivityHeatmapRow[] = ORDERED_WEEKDAYS.map((weekday) => {
    const row = matrix.get(weekday) ?? new Map<number, number>();
    const counts = Array.from({ length: 24 }, (_, hour) => row.get(hour) ?? 0);
    return { weekday, counts };
  });

  return { rows, maxCount };
}

export function summariseCounts(messages: TimeAnalysisMessage[], chatIds: string[]): SummaryCounts {
  const uniqueChats = new Set(chatIds);
  const totalMessages = messages.length;
  return {
    chats: uniqueChats.size,
    messages: totalMessages
  };
}

export function summariseDateRange(messages: TimeAnalysisMessage[]): { start: string | null; end: string | null } {
  const timestamps: Date[] = [];
  for (const message of messages) {
    const timestamp = parseTimestamp(message.timestamp);
    if (timestamp) {
      timestamps.push(timestamp);
    }
  }

  if (!timestamps.length) {
    return { start: null, end: null };
  }

  timestamps.sort((a, b) => a.getTime() - b.getTime());
  const first = dateKeyFormatter.format(timestamps[0]);
  const last = dateKeyFormatter.format(timestamps[timestamps.length - 1]);

  return {
    start: first,
    end: last
  };
}

function parseTimestamp(value: string | null | undefined): Date | null {
  if (!value) {
    return null;
  }
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? null : parsed;
}

function formatRange(lower: number, upper: number): string {
  if (lower === upper) {
    return numberFormatter.format(lower);
  }
  return `${numberFormatter.format(lower)}-${numberFormatter.format(upper)}`;
}
