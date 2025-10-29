const DEFAULT_MAX_WORDS = 80;
const DEFAULT_MIN_WORD_LENGTH = 3;

const STOP_WORDS = new Set([
  "the",
  "and",
  "for",
  "with",
  "that",
  "this",
  "from",
  "have",
  "will",
  "your",
  "about",
  "into",
  "when",
  "what",
  "where",
  "which",
  "like",
  "just",
  "they",
  "them",
  "were",
  "been",
  "while",
  "also",
  "could",
  "should",
  "would",
  "there",
  "their",
  "then",
  "than",
  "some",
  "more",
  "most",
  "such",
  "only",
  "other",
  "over",
  "after",
  "before",
  "because",
  "through",
  "between",
  "until",
  "above",
  "below",
  "again",
  "further",
  "those",
  "these",
  "ours",
  "ourselves"
]);

export const ALL_USERS_OPTION = "__ALL_USERS__";
export const ALL_MODELS_OPTION = "All Models";

export interface ContentChat {
  chatId: string;
  userId?: string | null;
  genTopics?: string | null;
}

export interface ContentMessage {
  chatId: string;
  role: string;
  content: string;
  model: string;
  timestamp?: string | null;
}

export interface UserOption {
  value: string;
  label: string;
}

export interface FilteredMessagesResult<T extends ContentMessage = ContentMessage> {
  filteredMessages: T[];
  matchingChatIds: string[];
}

export interface AverageLengthDatum {
  role: string;
  averageLength: number;
}

export interface HistogramBin {
  range: string;
  count: number;
  min: number;
  max: number;
}

export interface WordFrequency {
  text: string;
  count: number;
}

const NUMBER_FORMATTER = new Intl.NumberFormat();

function capitalise(value: string): string {
  if (!value) {
    return "Unknown";
  }
  return value.charAt(0).toUpperCase() + value.slice(1);
}

export function buildUserOptions<T extends ContentChat>(chats: T[], userDisplayMap: Record<string, string>): UserOption[] {
  const seen = new Set<string>();
  for (const chat of chats) {
    const candidate = chat.userId?.trim();
    if (candidate) {
      seen.add(candidate);
    }
  }

  const sorted = Array.from(seen).sort((a, b) => {
    const left = (userDisplayMap[a] ?? a).toLowerCase();
    const right = (userDisplayMap[b] ?? b).toLowerCase();
    return left.localeCompare(right);
  });

  const options: UserOption[] = [{ value: ALL_USERS_OPTION, label: "All Users" }];
  for (const userId of sorted) {
    const label = userDisplayMap[userId]?.trim();
    options.push({
      value: userId,
      label: label && label.length ? label : userId
    });
  }
  return options;
}

export function buildModelOptions<T extends ContentMessage>(messages: T[]): string[] {
  const seen = new Set<string>();
  for (const message of messages) {
    const candidate = message.model?.trim();
    if (candidate) {
      seen.add(candidate);
    }
  }
  const sorted = Array.from(seen).sort((a, b) => a.toLowerCase().localeCompare(b.toLowerCase()));
  return [ALL_MODELS_OPTION, ...sorted];
}

export function filterMessagesByUserAndModel<C extends ContentChat, M extends ContentMessage>(
  chats: C[],
  messages: M[],
  selectedUser: string,
  selectedModel: string
): FilteredMessagesResult<M> {
  const userFilteredChatIds =
    selectedUser && selectedUser !== ALL_USERS_OPTION
      ? new Set(chats.filter((chat) => chat.userId === selectedUser).map((chat) => chat.chatId))
      : null;

  let working = userFilteredChatIds ? messages.filter((message) => userFilteredChatIds.has(message.chatId)) : [...messages];

  if (selectedModel && selectedModel !== ALL_MODELS_OPTION) {
    working = working.filter((message) => message.model === selectedModel);
  }

  const chatIds = new Set<string>();
  for (const message of working) {
    chatIds.add(message.chatId);
  }

  return {
    filteredMessages: working,
    matchingChatIds: Array.from(chatIds)
  };
}

export function calculateAverageMessageLength(messages: ContentMessage[]): AverageLengthDatum[] {
  const aggregates = new Map<string, { total: number; count: number }>();

  for (const message of messages) {
    const content = (message.content ?? "").trim();
    if (!content.length) {
      continue;
    }
    const role = (message.role ?? "").toLowerCase() || "unknown";
    const entry = aggregates.get(role) ?? { total: 0, count: 0 };
    entry.total += content.length;
    entry.count += 1;
    aggregates.set(role, entry);
  }

  const preferredOrder = new Map<string, number>([
    ["user", 0],
    ["assistant", 1]
  ]);

  return Array.from(aggregates.entries())
    .map(([role, { total, count }]) => ({
      role: capitalise(role),
      averageLength: count > 0 ? total / count : 0
    }))
    .filter((datum) => datum.averageLength > 0)
    .sort((a, b) => {
      const rankA = preferredOrder.get(a.role.toLowerCase()) ?? 2;
      const rankB = preferredOrder.get(b.role.toLowerCase()) ?? 2;
      if (rankA !== rankB) {
        return rankA - rankB;
      }
      return a.role.localeCompare(b.role);
    });
}

export function buildMessageLengthHistogram(messages: ContentMessage[], desiredBins = 20): HistogramBin[] {
  const lengths = messages
    .map((message) => (message.content ?? "").trim().length)
    .filter((value) => value > 0);

  if (!lengths.length) {
    return [];
  }

  const max = Math.max(...lengths);
  const min = Math.min(...lengths);
  const binCount = Math.max(1, Math.min(desiredBins, Math.ceil(Math.sqrt(lengths.length))));
  const span = max - min + 1;
  const binSize = Math.max(1, Math.ceil(span / binCount));

  const bins: HistogramBin[] = Array.from({ length: binCount }, (_, index) => {
    const lower = min + index * binSize;
    const upper = min + (index + 1) * binSize - 1;
    return {
      min: lower,
      max: upper,
      range: formatRange(lower, upper),
      count: 0
    };
  });

  for (const length of lengths) {
    const offset = Math.floor((length - min) / binSize);
    const index = Math.min(bins.length - 1, Math.max(0, offset));
    bins[index].count += 1;
  }

  return bins;
}

export function extractWordFrequencies(
  messages: ContentMessage[],
  {
    maxWords = DEFAULT_MAX_WORDS,
    minWordLength = DEFAULT_MIN_WORD_LENGTH
  }: { maxWords?: number; minWordLength?: number } = {}
): WordFrequency[] {
  const counts = new Map<string, number>();

  for (const message of messages) {
    if ((message.role ?? "").toLowerCase() !== "user") {
      continue;
    }
    const content = (message.content ?? "").toLowerCase();
    if (!content.trim()) {
      continue;
    }

    const matches = content.match(/[a-z0-9']+/g);
    if (!matches) {
      continue;
    }

    for (const raw of matches) {
      const token = raw.replace(/^'+|'+$/g, "");
      if (token.length < minWordLength) {
        continue;
      }
      if (STOP_WORDS.has(token)) {
        continue;
      }
      counts.set(token, (counts.get(token) ?? 0) + 1);
    }
  }

  const table = Array.from(counts.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, maxWords)
    .map(([text, count]) => ({ text, count }));

  return table;
}

export function hasUserMessages(messages: ContentMessage[]): boolean {
  return messages.some((message) => (message.role ?? "").toLowerCase() === "user");
}

export function hasTextContent(messages: ContentMessage[]): boolean {
  return messages.some((message) => Boolean((message.content ?? "").trim()));
}

function formatRange(lower: number, upper: number): string {
  if (lower === upper) {
    return NUMBER_FORMATTER.format(lower);
  }
  return `${NUMBER_FORMATTER.format(lower)}-${NUMBER_FORMATTER.format(upper)}`;
}
