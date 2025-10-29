import { ALL_MODELS_OPTION, ALL_USERS_OPTION, buildModelOptions, buildUserOptions, filterMessagesByUserAndModel } from "./content-analysis";
import { normaliseModels, normaliseUsers } from "./overview";

type UnknownRecord = Record<string, unknown>;

function toStringOrNull(value: unknown): string | null {
  if (typeof value === "string") {
    const trimmed = value.trim();
    return trimmed.length ? trimmed : null;
  }

  if (typeof value === "number") {
    if (!Number.isFinite(value)) {
      return null;
    }
    return String(value);
  }

  return null;
}

function toBoolean(value: unknown): boolean {
  if (typeof value === "boolean") {
    return value;
  }
  if (typeof value === "number") {
    return value !== 0;
  }
  if (typeof value === "string") {
    const candidate = value.trim().toLowerCase();
    if (candidate === "true" || candidate === "1" || candidate === "yes" || candidate === "y") {
      return true;
    }
    if (candidate === "false" || candidate === "0" || candidate === "no" || candidate === "n") {
      return false;
    }
  }
  return false;
}

function toStringArray(value: unknown): string[] {
  if (!value) {
    return [];
  }
  if (Array.isArray(value)) {
    return value
      .map((entry) => toStringOrNull(entry))
      .filter((entry): entry is string => entry !== null);
  }
  const candidate = toStringOrNull(value);
  return candidate ? [candidate] : [];
}

function pickSummary(record: UnknownRecord, meta: UnknownRecord): string {
  const metaSummary =
    meta && typeof meta === "object"
      ? [
          meta.owca_gen_chat_summary as unknown,
          (meta.summary_full ?? meta.summary) as unknown,
          meta.owca_summary_full as unknown,
          meta.owca_summary as unknown
        ]
      : [];

  const chatValue = record.chat;
  const chatObject =
    chatValue && typeof chatValue === "object" && !Array.isArray(chatValue) ? (chatValue as UnknownRecord) : null;
  const chatSummaryCandidates = chatObject
    ? [
        chatObject.summary_full as unknown,
        chatObject.summary as unknown,
        chatObject.summaryText as unknown,
        chatObject.summary_text as unknown
      ]
    : [];

  const candidates: unknown[] = [
    record.summary_full,
    record.summary,
    record.summaryText,
    record.summary_text,
    record.full_summary,
    ...metaSummary,
    record.gen_chat_summary,
    record.summary_512,
    record.summary_256,
    ...chatSummaryCandidates
  ];

  for (const candidate of candidates) {
    const value = toStringOrNull(candidate);
    if (value) {
      return value;
    }
  }
  return "";
}

function coerceTimestamp(value: unknown): string | null {
  if (value instanceof Date) {
    return Number.isNaN(value.getTime()) ? null : value.toISOString();
  }

  if (typeof value === "number" && Number.isFinite(value)) {
    const epochMs = value < 1e12 ? value * 1000 : value;
    const date = new Date(epochMs);
    return Number.isNaN(date.getTime()) ? null : date.toISOString();
  }

  if (typeof value === "string") {
    const trimmed = value.trim();
    if (!trimmed) {
      return null;
    }

    const numeric = Number(trimmed);
    if (!Number.isNaN(numeric)) {
      return coerceTimestamp(numeric);
    }

    const parsed = new Date(trimmed);
    return Number.isNaN(parsed.getTime()) ? null : parsed.toISOString();
  }

  return null;
}

export { ALL_MODELS_OPTION, ALL_USERS_OPTION, buildModelOptions, buildUserOptions, filterMessagesByUserAndModel };

/**
 * Map outcome score (1-5) to emoji and category name.
 */
export function formatOutcome(outcome: number | null | undefined): string | null {
  if (!outcome || typeof outcome !== "number") {
    return null;
  }

  const rounded = Math.round(outcome);
  switch (rounded) {
    case 1:
      return "❌ Not Successful";
    case 2:
      return "⚠️ Partially Successful";
    case 3:
      return "😐 Moderately Successful";
    case 4:
      return "👍 Mostly Successful";
    case 5:
      return "🌟 Fully Successful";
    default:
      return null;
  }
}

export interface BrowseChat {
  chatId: string;
  userId: string | null;
  title: string;
  summary: string;
  outcome: number | null;
  createdAt: string | null;
  updatedAt: string | null;
  userDisplay: string;
  filesUploaded: number;
  files: unknown[];
  archived: boolean;
  pinned: boolean;
  tags: string[];
  models: string[];
  meta: UnknownRecord;
  params: UnknownRecord;
  shareId: string | null;
  folderId: string | null;
}

export interface BrowseMessage {
  messageId: string;
  chatId: string;
  role: string;
  content: string;
  timestamp: string | null;
  model: string;
  modelId: string | null;
  models: string[];
  files: unknown[];
}

export interface ThreadExportPayload {
  chat_id: string;
  user_id: string | null;
  user_name: string;
  title: string;
  summary: string | null;
  created_at: string | null;
  updated_at: string | null;
  archived: boolean;
  pinned: boolean;
  tags: string[];
  files: unknown[];
  models: string[];
  meta: UnknownRecord;
  params: UnknownRecord;
  share_id: string | null;
  folder_id: string | null;
  messages: Array<{
    message_id: string;
    chat_id: string;
    role: string;
    content: string;
    timestamp: string | null;
    model: string | null;
    model_display: string;
    models: string[];
    files: unknown[];
  }>;
}

export function normaliseBrowseChats(rawChats: unknown, rawUsers: unknown): BrowseChat[] {
  const usersMap = normaliseUsers(rawUsers);

  if (!Array.isArray(rawChats)) {
    return [];
  }

  return rawChats
    .map((entry) => {
      if (!entry || typeof entry !== "object") {
        return null;
      }

      const record = entry as UnknownRecord;
      const chatId =
        toStringOrNull(record.chat_id) ??
        toStringOrNull(record.id) ??
        toStringOrNull(record.chatId);

      if (!chatId) {
        return null;
      }

      const userId = toStringOrNull(record.user_id);
      const title = toStringOrNull(record.title) ?? chatId;
      const createdAt = coerceTimestamp(record.created_at ?? record.timestamp);
      const updatedAt = coerceTimestamp(record.updated_at ?? record.timestamp);
      const filesUploaded = Number(record.files_uploaded ?? 0) || 0;

      const rawFiles = Array.isArray(record.files) ? record.files : [];
      const metaValue = record.meta;
      const meta =
        metaValue && typeof metaValue === "object" && !Array.isArray(metaValue)
          ? (metaValue as UnknownRecord)
          : {};
      const paramsValue = record.params;
      const params =
        paramsValue && typeof paramsValue === "object" && !Array.isArray(paramsValue)
          ? (paramsValue as UnknownRecord)
          : {};
      const summary = pickSummary(record, meta);

      // Extract outcome score (1-5)
      const outcomeValue = record.gen_chat_outcome ?? record.outcome;
      let outcome: number | null = null;
      if (typeof outcomeValue === "number" && outcomeValue >= 1 && outcomeValue <= 5) {
        outcome = Math.round(outcomeValue);
      }

      const tagsSet = new Set<string>();
      for (const candidate of toStringArray(meta.tags)) {
        tagsSet.add(candidate);
      }
      for (const candidate of toStringArray(record.tags)) {
        tagsSet.add(candidate);
      }
      const tags = Array.from(tagsSet);

      const models = Array.from(new Set(toStringArray(record.models)));

      const userDisplay =
        (userId ? usersMap.get(userId) : null) ??
        (userId ?? null) ??
        "User";

      return {
        chatId,
        userId: userId ?? null,
        title,
        summary,
        outcome,
        createdAt,
        updatedAt,
        userDisplay,
        filesUploaded,
        files: rawFiles,
        archived: toBoolean(record.archived),
        pinned: toBoolean(record.pinned),
        tags,
        models,
        meta,
        params,
        shareId: toStringOrNull(record.share_id),
        folderId: toStringOrNull(record.folder_id)
      } satisfies BrowseChat;
    })
    .filter((chat): chat is BrowseChat => Boolean(chat));
}

export function normaliseBrowseMessages(rawMessages: unknown, rawModels: unknown): BrowseMessage[] {
  const modelsMap = normaliseModels(rawModels);

  if (!Array.isArray(rawMessages)) {
    return [];
  }

  const messages: BrowseMessage[] = [];

  for (const entry of rawMessages) {
    if (!entry || typeof entry !== "object") {
      continue;
    }

    const record = entry as UnknownRecord;
    const chatId = toStringOrNull(record.chat_id);
    if (!chatId) {
      continue;
    }

    const messageId =
      toStringOrNull(record.message_id) ??
      toStringOrNull(record.id) ??
      toStringOrNull(record.uuid);

    if (!messageId) {
      continue;
    }

    const roleRaw = toStringOrNull(record.role) ?? "unknown";
    const role = roleRaw.toLowerCase();
    const content = toStringOrNull(record.content) ?? "";
    const timestamp = coerceTimestamp(record.timestamp ?? record.created_at);

    const modelIdCandidate = toStringOrNull(record.model) ?? toStringOrNull(record.model_id);
    const modelList = toStringArray(record.models);
    const modelId = modelIdCandidate ?? modelList[0] ?? null;
    const modelDisplay = modelId ? modelsMap.get(modelId) ?? modelId : "";

    const files = Array.isArray(record.files) ? record.files : [];

    messages.push({
      messageId,
      chatId,
      role,
      content,
      timestamp,
      model: modelDisplay,
      modelId,
      models: modelList,
      files
    });
  }

  return messages;
}

export function buildThreadExportPayload(chat: BrowseChat, messages: BrowseMessage[]): ThreadExportPayload {
  return {
    chat_id: chat.chatId,
    user_id: chat.userId,
    user_name: chat.userDisplay,
    title: chat.title,
    summary: chat.summary || null,
    created_at: chat.createdAt,
    updated_at: chat.updatedAt,
    archived: chat.archived,
    pinned: chat.pinned,
    tags: [...chat.tags],
    files: [...chat.files],
    models: [...chat.models],
    meta: { ...chat.meta },
    params: { ...chat.params },
    share_id: chat.shareId,
    folder_id: chat.folderId,
    messages: messages.map((message) => ({
      message_id: message.messageId,
      chat_id: message.chatId,
      role: message.role,
      content: message.content,
      timestamp: message.timestamp,
      model: message.modelId,
      model_display: message.model,
      models: [...message.models],
      files: [...message.files]
    }))
  };
}
