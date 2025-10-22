import { getRandomCharacterName } from "./character-names";

const MS_PER_DAY = 24 * 60 * 60 * 1000;
const DISPLAY_TIMEZONE = "America/New_York";

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
  const utcMidnight = new Date(Date.UTC(date.getUTCFullYear(), date.getUTCMonth(), date.getUTCDate()));
  return utcMidnight.toISOString().slice(0, 10);
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

export interface OverviewChat {
  chatId: string;
  userId?: string;
  filesUploaded: number;
  createdAt: Date | null;
  updatedAt: Date | null;
}

export interface OverviewMessage {
  chatId: string;
  role: string;
  content: string;
  timestamp: Date | null;
  model: string;
}

export interface EngagementMetrics {
  totalChats: number;
  totalMessages: number;
  uniqueUsers: number;
  avgMessagesPerChat: number;
  avgInputTokensPerChat: number;
  avgOutputTokensPerChat: number;
  totalTokens: number;
  filesUploaded: number;
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

      return {
        chatId,
        userId: userId ?? undefined,
        filesUploaded: Number.isFinite(filesUploaded) ? Number(filesUploaded) : 0,
        createdAt: createdAt ? cloneDateOnly(createdAt) : null,
        updatedAt: updatedAt ? cloneDateOnly(updatedAt) : null
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
    // Use random character name instead of actual user name
    const randomName = getRandomCharacterName(userId);
    result.set(userId, randomName);
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

      return {
        chatId,
        role,
        content,
        timestamp,
        model: modelDisplay
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
      const length = message.content?.length ?? 0;
      if (length <= 0) {
        continue;
      }
      map.set(message.chatId, (map.get(message.chatId) ?? 0) + length);
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

  return {
    totalChats,
    totalMessages,
    uniqueUsers,
    avgMessagesPerChat,
    avgInputTokensPerChat,
    avgOutputTokensPerChat,
    totalTokens: totalInputTokens + totalOutputTokens,
    filesUploaded
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

  for (const message of messages) {
    if (!message.timestamp) {
      continue;
    }
    const key = toDateKey(message.timestamp);
    const length = message.content?.length ?? 0;
    tokensByDate.set(key, (tokensByDate.get(key) ?? 0) + length);
  }

  if (!tokensByDate.size) {
    return [];
  }

  const dates = Array.from(tokensByDate.keys()).sort();
  const start = new Date(`${dates[0]}T00:00:00Z`);
  const end = new Date(`${dates[dates.length - 1]}T00:00:00Z`);

  const points: TokenSeriesPoint[] = [];
  for (let time = start.getTime(); time <= end.getTime(); time += MS_PER_DAY) {
    const date = new Date(time);
    const key = toDateKey(date);
    points.push({
      date: key,
      tokens: tokensByDate.get(key) ?? 0
    });
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
