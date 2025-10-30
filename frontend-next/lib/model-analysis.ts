import { DISPLAY_TIMEZONE } from "@/lib/timezone";
import type { OverviewChat, OverviewMessage, ModelUsageDatum } from "@/lib/overview";

const DATE_KEY_FORMATTER = new Intl.DateTimeFormat("en-CA", {
  timeZone: DISPLAY_TIMEZONE,
  year: "numeric",
  month: "2-digit",
  day: "2-digit"
});

interface DateParts {
  year: number;
  month: number;
  day: number;
}

function toDateKey(value: Date): string {
  const parts = DATE_KEY_FORMATTER.formatToParts(value);
  const year = parts.find((part) => part.type === "year")?.value ?? "0000";
  const month = parts.find((part) => part.type === "month")?.value ?? "01";
  const day = parts.find((part) => part.type === "day")?.value ?? "01";
  return `${year}-${month}-${day}`;
}

function parseDateKey(key: string): DateParts | null {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(key);
  if (!match) {
    return null;
  }
  return {
    year: Number(match[1]),
    month: Number(match[2]),
    day: Number(match[3])
  };
}

function getNextDateKey(current: string): string | null {
  const parsed = parseDateKey(current);
  if (!parsed) {
    return null;
  }
  const base = new Date(Date.UTC(parsed.year, parsed.month - 1, parsed.day, 12));
  const next = new Date(base.getTime() + 24 * 60 * 60 * 1000);
  return toDateKey(next);
}

interface BuildTimelineOptions {
  topModelLimit?: number;
  includeOther?: boolean;
}

interface BuildTopicDistributionOptions {
  topModelLimit?: number;
  topTagLimit?: number;
}

interface BuildAverageTokensOptions {
  topModelLimit?: number;
}

export interface ModelUsageShareDatum {
  model: string;
  count: number;
  percentage: number;
}

export interface ModelUsageTimelineResult {
  models: string[];
  data: ModelUsageTimelinePoint[];
}

export interface ModelUsageTimelinePoint {
  date: string;
  [model: string]: number | string;
}

export interface ModelAverageTokensDatum {
  model: string;
  averageTokens: number;
  messageCount: number;
}

export interface ModelTopicDistributionResult {
  models: string[];
  data: ModelTopicDistributionDatum[];
}

export interface ModelTopicDistributionDatum {
  tag: string;
  total: number;
  [model: string]: number | string;
}

function normaliseTag(tag: string): string {
  const trimmed = tag.trim();
  if (!trimmed.length) {
    return "Unlabelled";
  }
  if (/[A-Z]/.test(trimmed) && /[a-z]/.test(trimmed)) {
    return trimmed;
  }
  return trimmed
    .toLowerCase()
    .split(/\s+/)
    .map((part) => (part ? part.charAt(0).toUpperCase() + part.slice(1) : ""))
    .join(" ")
    .trim();
}

function selectTopModels(breakdown: ModelUsageDatum[], limit: number): string[] {
  const sorted = [...breakdown].sort((a, b) => b.count - a.count);
  return sorted.slice(0, limit).map((item) => item.model);
}

export function buildModelUsageShare(breakdown: ModelUsageDatum[], topLimit = 10): ModelUsageShareDatum[] {
  if (!breakdown.length) {
    return [];
  }

  const total = breakdown.reduce((acc, { count }) => acc + count, 0);
  if (total <= 0) {
    return [];
  }

  const sorted = [...breakdown].sort((a, b) => b.count - a.count);
  const head = sorted.slice(0, topLimit);
  const tail = sorted.slice(topLimit);

  const shareData: ModelUsageShareDatum[] = head.map(({ model, count }) => ({
    model,
    count,
    percentage: (count / total) * 100
  }));

  if (tail.length) {
    const otherCount = tail.reduce((acc, item) => acc + item.count, 0);
    if (otherCount > 0) {
      shareData.push({
        model: "Other",
        count: otherCount,
        percentage: (otherCount / total) * 100
      });
    }
  }

  return shareData;
}

export function buildModelUsageTimeline(
  messages: OverviewMessage[],
  breakdown: ModelUsageDatum[],
  options: BuildTimelineOptions = {}
): ModelUsageTimelineResult {
  const topLimit = options.topModelLimit ?? 5;
  const includeOther = options.includeOther ?? true;

  const selectedModels = selectTopModels(breakdown, topLimit);
  const countsByModel = new Map<string, Map<string, number>>();
  const totalsByModel = new Map<string, number>();

  let minKey: string | null = null;
  let maxKey: string | null = null;

  for (const message of messages) {
    if (message.role !== "assistant" || !message.timestamp) {
      continue;
    }
    const model = message.model?.trim();
    if (!model) {
      continue;
    }
    const dateKey = toDateKey(message.timestamp);
    const modelCounts = countsByModel.get(model) ?? new Map<string, number>();
    modelCounts.set(dateKey, (modelCounts.get(dateKey) ?? 0) + 1);
    countsByModel.set(model, modelCounts);
    totalsByModel.set(model, (totalsByModel.get(model) ?? 0) + 1);
    if (!minKey || dateKey < minKey) {
      minKey = dateKey;
    }
    if (!maxKey || dateKey > maxKey) {
      maxKey = dateKey;
    }
  }

  if (!minKey || !maxKey) {
    return { models: [], data: [] };
  }

  const retainedModels = selectedModels.filter((model) => totalsByModel.get(model) ?? 0 > 0);
  const hasOtherGroup = includeOther && totalsByModel.size > retainedModels.length;

  const data: ModelUsageTimelinePoint[] = [];
  for (let key: string | null = minKey; key; key = getNextDateKey(key)) {
    const row: ModelUsageTimelinePoint = { date: key };
    for (const model of retainedModels) {
      const modelCounts = countsByModel.get(model);
      row[model] = modelCounts?.get(key) ?? 0;
    }
    if (hasOtherGroup) {
      let otherTotal = 0;
      for (const [model, modelCounts] of countsByModel.entries()) {
        if (retainedModels.includes(model)) {
          continue;
        }
        otherTotal += modelCounts.get(key) ?? 0;
      }
      row.Other = otherTotal;
    }
    data.push(row);
    if (key === maxKey) {
      break;
    }
  }

  const models = hasOtherGroup ? [...retainedModels, "Other"] : retainedModels;
  return { models, data };
}

export function buildAverageTokensByModel(
  messages: OverviewMessage[],
  breakdown: ModelUsageDatum[],
  options: BuildAverageTokensOptions = {}
): ModelAverageTokensDatum[] {
  const topLimit = options.topModelLimit ?? 8;
  const selectedModels = selectTopModels(breakdown, topLimit);

  const aggregates = new Map<string, { totalTokens: number; count: number }>();

  for (const message of messages) {
    if (message.role !== "assistant") {
      continue;
    }
    const model = message.model?.trim();
    if (!model) {
      continue;
    }
    const entry = aggregates.get(model) ?? { totalTokens: 0, count: 0 };
    entry.totalTokens += message.tokenCount ?? 0;
    entry.count += 1;
    aggregates.set(model, entry);
  }

  const data: ModelAverageTokensDatum[] = [];
  for (const model of selectedModels) {
    const aggregate = aggregates.get(model);
    if (!aggregate || aggregate.count === 0) {
      continue;
    }
    data.push({
      model,
      averageTokens: aggregate.totalTokens / aggregate.count,
      messageCount: aggregate.count
    });
  }

  return data.sort((a, b) => b.averageTokens - a.averageTokens);
}

export function buildModelTopicDistribution(
  messages: OverviewMessage[],
  chats: OverviewChat[],
  breakdown: ModelUsageDatum[],
  options: BuildTopicDistributionOptions = {}
): ModelTopicDistributionResult {
  const topModelLimit = options.topModelLimit ?? 5;
  const topTagLimit = options.topTagLimit ?? 8;

  const selectedModels = selectTopModels(breakdown, topModelLimit);
  if (!selectedModels.length) {
    return { models: [], data: [] };
  }

  const chatTagMap = new Map<string, string[]>();
  for (const chat of chats) {
    if (!Array.isArray(chat.tags) || !chat.tags.length) {
      continue;
    }
    const formattedTags = chat.tags
      .map((tag) => (typeof tag === "string" ? tag : String(tag ?? "")))
      .map((tag) => tag.trim())
      .filter((tag) => tag.length > 0)
      .map((tag) => normaliseTag(tag));
    if (!formattedTags.length) {
      continue;
    }
    chatTagMap.set(chat.chatId, Array.from(new Set(formattedTags)));
  }

  const tagCounts = new Map<string, Map<string, number>>();

  for (const message of messages) {
    if (message.role !== "assistant") {
      continue;
    }
    const model = message.model?.trim();
    if (!model) {
      continue;
    }
    const tags = chatTagMap.get(message.chatId);
    if (!tags || !tags.length) {
      continue;
    }
    const modelTagCounts = tagCounts;
    for (const tag of tags) {
      const perTag = modelTagCounts.get(tag) ?? new Map<string, number>();
      perTag.set(model, (perTag.get(model) ?? 0) + 1);
      modelTagCounts.set(tag, perTag);
    }
  }

  if (!tagCounts.size) {
    return { models: [], data: [] };
  }

  const topTags = Array.from(tagCounts.entries())
    .map(([tag, perModel]) => ({
      tag,
      total: selectedModels.reduce((acc, model) => acc + (perModel.get(model) ?? 0), 0)
    }))
    .filter((entry) => entry.total > 0)
    .sort((a, b) => b.total - a.total)
    .slice(0, topTagLimit)
    .map((entry) => entry.tag);

  if (!topTags.length) {
    return { models: [], data: [] };
  }

  const data: ModelTopicDistributionDatum[] = topTags.map((tag) => {
    const row: ModelTopicDistributionDatum = { tag, total: 0 };
    const modelCounts = tagCounts.get(tag);
    for (const model of selectedModels) {
      const value = modelCounts?.get(model) ?? 0;
      row[model] = value;
      row.total += value;
    }
    return row;
  });

  return {
    models: selectedModels,
    data
  };
}
