import { encode } from "gpt-tokenizer/encoding/cl100k_base";
import { DISPLAY_TIMEZONE } from "@/lib/timezone";

const MS_PER_DAY = 24 * 60 * 60 * 1000;
const DATE_KEY_FORMATTER = new Intl.DateTimeFormat("en-CA", {
  timeZone: DISPLAY_TIMEZONE,
  year: "numeric",
  month: "2-digit",
  day: "2-digit"
});

function countTokens(text: string): number {
  if (typeof text !== "string") {
    return 0;
  }
  if (!text) {
    return 0;
  }
  try {
    return encode(text).length;
  } catch {
    const trimmed = text.trim();
    if (!trimmed) {
      return 0;
    }
    return Math.max(1, Math.ceil(trimmed.length / 4));
  }
}

function parseTimestamp(value: unknown): Date | null {
  if (value instanceof Date) {
    return Number.isNaN(value.getTime()) ? null : value;
  }

  if (typeof value === "number" && Number.isFinite(value)) {
    if (value < 1e12) {
      return new Date(value * 1000);
    }
    return new Date(value);
  }

  if (typeof value === "string") {
    const trimmed = value.trim();
    if (!trimmed) {
      return null;
    }
    const numeric = Number(trimmed);
    if (!Number.isNaN(numeric)) {
      return parseTimestamp(numeric);
    }
    const parsed = new Date(trimmed);
    return Number.isNaN(parsed.getTime()) ? null : parsed;
  }

  return null;
}

function toDateKey(date: Date): string {
  const parts = DATE_KEY_FORMATTER.formatToParts(date);
  const year = parts.find((part) => part.type === "year")?.value ?? "0000";
  const month = parts.find((part) => part.type === "month")?.value ?? "01";
  const day = parts.find((part) => part.type === "day")?.value ?? "01";
  return `${year}-${month}-${day}`;
}

function formatMonthDay(date: Date | null): string {
  if (!date) {
    return "N/A";
  }
  return new Intl.DateTimeFormat("en-US", {
    timeZone: DISPLAY_TIMEZONE,
    month: "2-digit",
    day: "2-digit"
  }).format(date);
}

function cloneDateOnly(date: Date): Date {
  return new Date(Date.UTC(date.getUTCFullYear(), date.getUTCMonth(), date.getUTCDate()));
}

function parseDateKey(key: string): { year: number; month: number; day: number } | null {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(key);
  if (!match) {
    return null;
  }
  const [, year, month, day] = match;
  return {
    year: Number(year),
    month: Number(month),
    day: Number(day)
  };
}

function getNextDateKey(key: string): string | null {
  const parts = parseDateKey(key);
  if (!parts) {
    return null;
  }
  const base = new Date(Date.UTC(parts.year, parts.month - 1, parts.day, 12));
  const next = new Date(base.getTime() + MS_PER_DAY);
  const nextKey = toDateKey(next);
  return nextKey === key ? null : nextKey;
}

export interface OverviewChat {
  chatId: string;
  userId?: string;
  filesUploaded: number;
  createdAt: Date | null;
  updatedAt: Date | null;
  tags: string[];
  meta?: Record<string, unknown>;
}

export interface OverviewMessage {
  chatId: string;
  role: string;
  content: string;
  timestamp: Date | null;
  model: string;
  tokenCount: number;
}

export interface EngagementMetrics {
  totalChats: number;
  totalMessages: number;
  uniqueUsers: number;
  avgDailyActiveUsers: number;
  avgMessagesPerChat: number;
  avgInputTokensPerChat: number;
  avgOutputTokensPerChat: number;
  totalTokens: number;
  filesUploaded: number;
  webSearchesUsed: number;
  knowledgeBaseUsed: number;
}

export interface DateSummary {
  dateMinLabel: string;
  dateMaxLabel: string;
  totalDays: number;
  dateMin: Date | null;
  dateMax: Date | null;
}

export interface TokenSeriesPoint {
  date: string;
  tokens: number;
}

export interface ModelUsageDatum {
  model: string;
  count: number;
}

export interface PieDatum {
  name: string;
  value: number;
}

export interface AdoptionSeriesPoint {
  date: string;
  value: number;
}

function toStringOrNull(value: unknown): string | null {
  if (typeof value === "string") {
    const trimmed = value.trim();
    return trimmed ? trimmed : null;
  }
  if (typeof value === "number" && Number.isFinite(value)) {
    return String(value);
  }
  return null;
}

function toStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  const result: string[] = [];
  for (const entry of value) {
    const normalized = toStringOrNull(entry);
    if (normalized) {
      result.push(normalized);
    }
  }
  return result;
}

export function normaliseChats(raw: unknown): OverviewChat[] {
  if (!Array.isArray(raw)) {
    return [];
  }

  return raw
    .map((entry) => {
      if (!entry || typeof entry !== "object") {
        return null;
      }

      const record = entry as Record<string, unknown>;
      const chatId =
        toStringOrNull(record.chat_id) ??
        toStringOrNull(record.id) ??
        toStringOrNull(record.chatId);

      if (!chatId) {
        return null;
      }

      const userId = toStringOrNull(record.user_id);
      const filesUploadedRaw = record.files_uploaded ?? 0;
      const filesUploaded =
        typeof filesUploadedRaw === "number"
          ? filesUploadedRaw
          : typeof filesUploadedRaw === "string" && filesUploadedRaw.trim()
          ? Number(filesUploadedRaw)
          : 0;

      const createdAt = parseTimestamp(record.created_at ?? record.timestamp);
      const updatedAt = parseTimestamp(record.updated_at ?? record.timestamp);
      const tags = toStringArray(record.tags);
      const meta = record.meta && typeof record.meta === "object" ? record.meta as Record<string, unknown> : undefined;

      return {
        chatId,
        userId: userId ?? undefined,
        filesUploaded: Number.isFinite(filesUploaded) ? Number(filesUploaded) : 0,
        createdAt: createdAt ? cloneDateOnly(createdAt) : null,
        updatedAt: updatedAt ? cloneDateOnly(updatedAt) : null,
        tags,
        meta
      } satisfies OverviewChat;
    })
    .filter(Boolean) as OverviewChat[];
}

export function normaliseUsers(raw: unknown): Map<string, string> {
  const result = new Map<string, string>();
  if (!Array.isArray(raw)) {
    return result;
  }
  for (const entry of raw) {
    if (!entry || typeof entry !== "object") {
      continue;
    }
    const record = entry as Record<string, unknown>;
    const userId = toStringOrNull(record.user_id);
    if (!userId) {
      continue;
    }
    const providedName = toStringOrNull(record.name);
    const pseudonym = toStringOrNull(record.pseudonym);
    const realName = toStringOrNull(record.real_name);
    const displayName = providedName ?? pseudonym ?? realName ?? userId;
    result.set(userId, displayName);
  }
  return result;
}

export function normaliseModels(raw: unknown): Map<string, string> {
  const result = new Map<string, string>();
  if (!Array.isArray(raw)) {
    return result;
  }
  for (const entry of raw) {
    if (!entry || typeof entry !== "object") {
      continue;
    }
    const record = entry as Record<string, unknown>;
    const modelId = toStringOrNull(record.model_id) ?? toStringOrNull(record.id);
    if (!modelId) {
      continue;
    }
    const nameValue = record.name;
    const name =
      typeof nameValue === "string" && nameValue.trim() ? nameValue.trim() : modelId;
    result.set(modelId, name);
  }
  return result;
}

export function normaliseMessages(raw: unknown, modelMap: Map<string, string>): OverviewMessage[] {
  if (!Array.isArray(raw)) {
    return [];
  }

  return raw
    .map((entry) => {
      if (!entry || typeof entry !== "object") {
        return null;
      }
      const record = entry as Record<string, unknown>;

      const chatId = toStringOrNull(record.chat_id);
      if (!chatId) {
        return null;
      }

      const roleValue = record.role;
      const role = typeof roleValue === "string" ? roleValue.trim().toLowerCase() : "";
      if (!role) {
        return null;
      }

      const contentValue = record.content;
      const content = typeof contentValue === "string" ? contentValue : String(contentValue ?? "");

      const timestamp = parseTimestamp(record.timestamp ?? record.created_at);

      const modelCandidates: string[] = [];
      const modelRaw = record.model ?? record.model_name ?? null;
      const modelString = toStringOrNull(modelRaw);
      if (modelString) {
        modelCandidates.push(modelString);
      }
      if (Array.isArray(record.models)) {
        for (const item of record.models) {
          const candidate = toStringOrNull(item);
          if (candidate) {
            modelCandidates.push(candidate);
          }
        }
      }

      const modelResolved = modelCandidates.find((candidate) => modelMap.get(candidate)) ?? modelCandidates[0] ?? "";
      const modelDisplay = modelResolved ? modelMap.get(modelResolved) ?? modelResolved : "";
      const tokenCount = countTokens(content);

      return {
        chatId,
        role,
        content,
        timestamp,
        model: modelDisplay,
        tokenCount
      } satisfies OverviewMessage;
    })
    .filter(Boolean) as OverviewMessage[];
}

export function buildChatUserMap(chats: OverviewChat[], users: Map<string, string>): Map<string, string> {
  const result = new Map<string, string>();
  for (const chat of chats) {
    const userId = chat.userId;
    const display =
      (userId ? users.get(userId) : null) ??
      (userId ? userId : null) ??
      "User";
    result.set(chat.chatId, display);
  }
  return result;
}

export function calculateEngagementMetrics(chats: OverviewChat[], messages: OverviewMessage[]): EngagementMetrics | null {
  if (!messages.length) {
    return null;
  }

  const totalChats = chats.length;
  const totalMessages = messages.length;

  const uniqueUsers = new Set(
    chats
      .map((chat) => (chat.userId ?? "").trim())
      .filter((value) => value.length > 0)
  ).size;

  const filesUploaded = chats.reduce((acc, chat) => acc + (Number.isFinite(chat.filesUploaded) ? chat.filesUploaded : 0), 0);

  // Count chats with web search and knowledge base usage
  let webSearchesUsed = 0;
  let knowledgeBaseUsed = 0;
  for (const chat of chats) {
    if (chat.meta && typeof chat.meta === "object") {
      const actions = chat.meta.actions;
      if (Array.isArray(actions)) {
        if (actions.includes("web_search")) {
          webSearchesUsed++;
        }
        if (actions.includes("knowledge_search")) {
          knowledgeBaseUsed++;
        }
      }
    }
  }

  const messagesPerChat = new Map<string, number>();
  for (const message of messages) {
    messagesPerChat.set(message.chatId, (messagesPerChat.get(message.chatId) ?? 0) + 1);
  }
  const avgMessagesPerChat =
    messagesPerChat.size > 0
      ? Array.from(messagesPerChat.values()).reduce((acc, count) => acc + count, 0) / messagesPerChat.size
      : 0;

  const userMessages = messages.filter((message) => message.role === "user");
  const assistantMessages = messages.filter((message) => message.role === "assistant");

  const sumTokensByChat = (collection: OverviewMessage[]): Map<string, number> => {
    const map = new Map<string, number>();
    for (const message of collection) {
      const tokens = message.tokenCount ?? 0;
      if (tokens <= 0) {
        continue;
      }
      map.set(message.chatId, (map.get(message.chatId) ?? 0) + tokens);
    }
    return map;
  };

  const inputTokensPerChat = sumTokensByChat(userMessages);
  const outputTokensPerChat = sumTokensByChat(assistantMessages);

  const totalInputTokens = Array.from(inputTokensPerChat.values()).reduce((acc, value) => acc + value, 0);
  const totalOutputTokens = Array.from(outputTokensPerChat.values()).reduce((acc, value) => acc + value, 0);

  const avgInputTokensPerChat =
    inputTokensPerChat.size > 0 ? totalInputTokens / inputTokensPerChat.size : 0;
  const avgOutputTokensPerChat =
    outputTokensPerChat.size > 0 ? totalOutputTokens / outputTokensPerChat.size : 0;

  // Calculate average daily active users over the past 30 days
  const chatUserMap = new Map<string, string>();
  for (const chat of chats) {
    if (chat.userId) {
      chatUserMap.set(chat.chatId, chat.userId);
    }
  }

  // Find the most recent activity date
  const allTimestamps = [
    ...messages.map((msg) => msg.timestamp).filter((ts): ts is Date => ts instanceof Date),
    ...chats.map((chat) => chat.createdAt).filter((ts): ts is Date => ts instanceof Date)
  ];

  let avgDailyActiveUsers = 0;
  if (allTimestamps.length > 0) {
    const sortedTimestamps = allTimestamps.sort((a, b) => b.getTime() - a.getTime());
    const mostRecentDate = cloneDateOnly(sortedTimestamps[0]);
    const thirtyDaysAgo = new Date(mostRecentDate.getTime() - 30 * MS_PER_DAY);

    // Track active users by day for the past 30 days
    const activeUsersByDate = new Map<string, Set<string>>();

    // Count users who sent messages in the past 30 days
    for (const message of messages) {
      if (message.role !== "user" || !message.timestamp) {
        continue;
      }
      if (message.timestamp.getTime() < thirtyDaysAgo.getTime()) {
        continue;
      }
      const userId = chatUserMap.get(message.chatId);
      if (!userId) {
        continue;
      }
      const dateKey = toDateKey(message.timestamp);
      if (!activeUsersByDate.has(dateKey)) {
        activeUsersByDate.set(dateKey, new Set());
      }
      activeUsersByDate.get(dateKey)!.add(userId);
    }

    // Count users who created chats in the past 30 days
    for (const chat of chats) {
      if (!chat.createdAt || !chat.userId) {
        continue;
      }
      if (chat.createdAt.getTime() < thirtyDaysAgo.getTime()) {
        continue;
      }
      const dateKey = toDateKey(chat.createdAt);
      if (!activeUsersByDate.has(dateKey)) {
        activeUsersByDate.set(dateKey, new Set());
      }
      activeUsersByDate.get(dateKey)!.add(chat.userId);
    }

    // Calculate average across all days with activity
    if (activeUsersByDate.size > 0) {
      const totalActiveUsers = Array.from(activeUsersByDate.values()).reduce(
        (sum, userSet) => sum + userSet.size,
        0
      );
      avgDailyActiveUsers = totalActiveUsers / activeUsersByDate.size;
    }
  }

  return {
    totalChats,
    totalMessages,
    uniqueUsers,
    avgDailyActiveUsers,
    avgMessagesPerChat,
    avgInputTokensPerChat,
    avgOutputTokensPerChat,
    totalTokens: totalInputTokens + totalOutputTokens,
    filesUploaded,
    webSearchesUsed,
    knowledgeBaseUsed
  };
}

export function computeDateSummary(messages: OverviewMessage[]): DateSummary {
  const timestamps = messages
    .map((message) => message.timestamp)
    .filter((value): value is Date => value instanceof Date);

  if (!timestamps.length) {
    return {
      dateMinLabel: "N/A",
      dateMaxLabel: "N/A",
      totalDays: 0,
      dateMin: null,
      dateMax: null
    };
  }

  const sorted = timestamps.sort((a, b) => a.getTime() - b.getTime());
  const dateMin = cloneDateOnly(sorted[0]);
  const dateMax = cloneDateOnly(sorted[sorted.length - 1]);
  const totalDays = Math.floor((dateMax.getTime() - dateMin.getTime()) / MS_PER_DAY) + 1;

  return {
    dateMinLabel: formatMonthDay(dateMin),
    dateMaxLabel: formatMonthDay(dateMax),
    totalDays,
    dateMin,
    dateMax
  };
}

export function buildTokenConsumptionSeries(messages: OverviewMessage[]): TokenSeriesPoint[] {
  const tokensByDate = new Map<string, number>();
  let minKey: string | null = null;
  let maxKey: string | null = null;

  for (const message of messages) {
    if (!message.timestamp) {
      continue;
    }
    const key = toDateKey(message.timestamp);
    const tokens = message.tokenCount ?? 0;
    if (tokens <= 0) {
      continue;
    }
    tokensByDate.set(key, (tokensByDate.get(key) ?? 0) + tokens);
    if (!minKey || key < minKey) {
      minKey = key;
    }
    if (!maxKey || key > maxKey) {
      maxKey = key;
    }
  }

  if (!tokensByDate.size || !minKey || !maxKey) {
    return [];
  }

  const points: TokenSeriesPoint[] = [];
  for (let key: string | null = minKey; key; ) {
    points.push({
      date: key,
      tokens: tokensByDate.get(key) ?? 0
    });
    if (key === maxKey) {
      break;
    }
    const nextKey = getNextDateKey(key);
    if (!nextKey) {
      break;
    }
    key = nextKey;
  }

  return points;
}

export function buildModelUsageBreakdown(messages: OverviewMessage[]): ModelUsageDatum[] {
  const counts = new Map<string, number>();
  for (const message of messages) {
    if (message.role !== "assistant") {
      continue;
    }
    const model = (message.model ?? "").trim();
    if (!model) {
      continue;
    }
    counts.set(model, (counts.get(model) ?? 0) + 1);
  }

  return Array.from(counts.entries())
    .map(([model, count]) => ({ model, count }))
    .sort((a, b) => a.count - b.count);
}

export function buildModelUsagePie(messages: OverviewMessage[]): PieDatum[] {
  const breakdown = buildModelUsageBreakdown(messages);
  return breakdown.map((item) => ({ name: item.model, value: item.count }));
}

export function buildUserAdoptionSeries(
  messages: OverviewMessage[],
  chatUserMap: Map<string, string>,
  dateMin?: Date | null,
  dateMax?: Date | null
): AdoptionSeriesPoint[] {
  const firstMessageByUser = new Map<string, Date>();

  for (const message of messages) {
    if (message.role !== "user" || !message.timestamp) {
      continue;
    }
    const userDisplay = chatUserMap.get(message.chatId) ?? "User";
    const existing = firstMessageByUser.get(userDisplay);
    const candidate = cloneDateOnly(message.timestamp);
    if (!existing || candidate.getTime() < existing.getTime()) {
      firstMessageByUser.set(userDisplay, candidate);
    }
  }

  if (!firstMessageByUser.size) {
    return [];
  }

  const dailyCounts = new Map<string, number>();
  for (const date of firstMessageByUser.values()) {
    const key = toDateKey(date);
    dailyCounts.set(key, (dailyCounts.get(key) ?? 0) + 1);
  }

  const sortedDates = Array.from(dailyCounts.keys()).sort();
  const firstDate = new Date(`${sortedDates[0]}T00:00:00Z`);
  const lastDate = new Date(`${sortedDates[sortedDates.length - 1]}T00:00:00Z`);

  const rangeStartCandidate = new Date(firstDate.getTime() - MS_PER_DAY);
  const rangeStart =
    dateMin && dateMin.getTime() < rangeStartCandidate.getTime() ? cloneDateOnly(dateMin) : rangeStartCandidate;
  const rangeEnd =
    dateMax && dateMax.getTime() > lastDate.getTime() ? cloneDateOnly(dateMax) : lastDate;

  const points: AdoptionSeriesPoint[] = [];
  let cumulative = 0;
  for (let time = rangeStart.getTime(); time <= rangeEnd.getTime(); time += MS_PER_DAY) {
    const date = new Date(time);
    const key = toDateKey(date);
    cumulative += dailyCounts.get(key) ?? 0;
    points.push({
      date: key,
      value: cumulative
    });
  }

  return points;
}

export interface DailyActiveUsersPoint {
  date: string;
  activeUsers: number;
}

export function buildDailyActiveUsersSeries(
  messages: OverviewMessage[],
  chats: OverviewChat[],
  chatUserMap: Map<string, string>,
  dateMin?: Date | null,
  dateMax?: Date | null
): DailyActiveUsersPoint[] {
  // Track which users were active on which days
  const activeUsersByDate = new Map<string, Set<string>>();

  // Track users who sent messages
  for (const message of messages) {
    if (message.role !== "user" || !message.timestamp) {
      continue;
    }
    const userDisplay = chatUserMap.get(message.chatId) ?? "User";
    const key = toDateKey(message.timestamp);

    if (!activeUsersByDate.has(key)) {
      activeUsersByDate.set(key, new Set());
    }
    activeUsersByDate.get(key)!.add(userDisplay);
  }

  // Track users who created chats
  for (const chat of chats) {
    if (!chat.createdAt || !chat.userId) {
      continue;
    }
    const userDisplay = chatUserMap.get(chat.chatId) ?? "User";
    const key = toDateKey(chat.createdAt);

    if (!activeUsersByDate.has(key)) {
      activeUsersByDate.set(key, new Set());
    }
    activeUsersByDate.get(key)!.add(userDisplay);
  }

  if (!activeUsersByDate.size) {
    return [];
  }

  // Determine date range
  const sortedDates = Array.from(activeUsersByDate.keys()).sort();
  const firstDate = new Date(`${sortedDates[0]}T00:00:00Z`);
  const lastDate = new Date(`${sortedDates[sortedDates.length - 1]}T00:00:00Z`);

  const rangeStartCandidate = new Date(firstDate.getTime() - MS_PER_DAY);
  const rangeStart =
    dateMin && dateMin.getTime() < rangeStartCandidate.getTime() ? cloneDateOnly(dateMin) : rangeStartCandidate;
  const rangeEnd =
    dateMax && dateMax.getTime() > lastDate.getTime() ? cloneDateOnly(dateMax) : lastDate;

  // Build the series
  const points: DailyActiveUsersPoint[] = [];
  for (let time = rangeStart.getTime(); time <= rangeEnd.getTime(); time += MS_PER_DAY) {
    const date = new Date(time);
    const key = toDateKey(date);
    const activeUsers = activeUsersByDate.get(key)?.size ?? 0;
    points.push({
      date: key,
      activeUsers
    });
  }

  return points;
}
