import {
  buildTokenConsumptionSeries,
  buildTopModelsByChats,
  buildTopTopics,
  type OverviewChat,
  type OverviewMessage
} from "@/lib/overview";
import { DISPLAY_TIMEZONE } from "@/lib/timezone";

const DATE_KEY_FORMATTER = new Intl.DateTimeFormat("en-CA", {
  timeZone: DISPLAY_TIMEZONE,
  year: "numeric",
  month: "2-digit",
  day: "2-digit"
});

describe("overview top aggregations", () => {
  it("ranks models by unique chat count", () => {
    const messages: OverviewMessage[] = [
      { chatId: "chat-1", role: "assistant", content: "", timestamp: null, model: "GPT-4o", tokenCount: 0 },
      { chatId: "chat-1", role: "assistant", content: "", timestamp: null, model: "GPT-4o", tokenCount: 0 },
      { chatId: "chat-2", role: "assistant", content: "", timestamp: null, model: "GPT-4o", tokenCount: 0 },
      { chatId: "chat-3", role: "assistant", content: "", timestamp: null, model: "sonnet", tokenCount: 0 },
      { chatId: "chat-4", role: "user", content: "", timestamp: null, model: "GPT-4o", tokenCount: 0 },
      { chatId: "chat-5", role: "assistant", content: "", timestamp: null, model: "", tokenCount: 0 }
    ];

    const result = buildTopModelsByChats(messages, 1);

    expect(result).toEqual([{ model: "GPT-4o", chatCount: 2 }]);
  });

  it("counts chats per topic using unique tags", () => {
    const chats: OverviewChat[] = [
      {
        chatId: "chat-1",
        userId: "user-1",
        filesUploaded: 0,
        createdAt: null,
        updatedAt: null,
        tags: ["analysis", "analysis", "Design"]
      },
      {
        chatId: "chat-2",
        userId: "user-2",
        filesUploaded: 0,
        createdAt: null,
        updatedAt: null,
        tags: ["design", "research"]
      },
      {
        chatId: "chat-3",
        userId: "user-3",
        filesUploaded: 0,
        createdAt: null,
        updatedAt: null,
        tags: []
      }
    ];

    const result = buildTopTopics(chats);

    expect(result).toEqual([
      { topic: "Design", chatCount: 2 },
      { topic: "Analysis", chatCount: 1 },
      { topic: "Research", chatCount: 1 }
    ]);
  });
});

function toDateKey(date: Date): string {
  return DATE_KEY_FORMATTER.format(date);
}

function iterateDates(start: string, end: string): string[] {
  const result: string[] = [];
  const startDate = new Date(`${start}T00:00:00Z`);
  const endDate = new Date(`${end}T00:00:00Z`);
  for (let time = startDate.getTime(); time <= endDate.getTime(); time += 24 * 60 * 60 * 1000) {
    const key = new Date(time).toISOString().slice(0, 10);
    result.push(key);
  }
  return result;
}

describe("overview token consumption series", () => {
  it("groups messages by local day boundaries and fills missing dates", () => {
    const messages: OverviewMessage[] = [
      {
        chatId: "chat-1",
        role: "user",
        content: "aa",
        timestamp: new Date("2024-03-09T23:30:00-05:00"),
        model: "gpt-4",
        tokenCount: 1
      },
      {
        chatId: "chat-1",
        role: "assistant",
        content: "bbb",
        timestamp: new Date("2024-03-10T00:15:00-05:00"),
        model: "gpt-4",
        tokenCount: 1
      },
      {
        chatId: "chat-1",
        role: "assistant",
        content: "cccc",
        timestamp: new Date("2024-03-10T03:15:00-04:00"),
        model: "gpt-4",
        tokenCount: 1
      },
      {
        chatId: "chat-2",
        role: "assistant",
        content: "d",
        timestamp: new Date("2024-03-12T10:00:00-04:00"),
        model: "gpt-4",
        tokenCount: 1
      }
    ];

    const series = buildTokenConsumptionSeries(messages);

    const totals = new Map<string, number>();
    for (const message of messages) {
      if (!message.timestamp) {
        continue;
      }
      const key = toDateKey(message.timestamp);
      totals.set(key, (totals.get(key) ?? 0) + (message.tokenCount ?? 0));
    }

    const sortedKeys = Array.from(totals.keys()).sort();
    const expectedDates = iterateDates(sortedKeys[0], sortedKeys[sortedKeys.length - 1]);

    expect(series.map((point) => point.date)).toEqual(expectedDates);
    for (const point of series) {
      expect(point.tokens).toBe(totals.get(point.date) ?? 0);
    }
  });

  it("assigns tokens using the detected local timezone even near UTC midnight", () => {
    const messages: OverviewMessage[] = [
      {
        chatId: "chat-3",
        role: "user",
        content: "hello",
        timestamp: new Date("2024-02-01T02:30:00Z"),
        model: "gpt-4",
        tokenCount: 1
      },
      {
        chatId: "chat-3",
        role: "assistant",
        content: "test",
        timestamp: new Date("2024-02-01T13:00:00Z"),
        model: "gpt-4",
        tokenCount: 1
      }
    ];

    const series = buildTokenConsumptionSeries(messages);

    const totals = new Map<string, number>();
    for (const message of messages) {
      if (!message.timestamp) {
        continue;
      }
      const key = toDateKey(message.timestamp);
      totals.set(key, (totals.get(key) ?? 0) + (message.tokenCount ?? 0));
    }

    const expectedKeys = Array.from(totals.keys()).sort();
    const actualKeys = series.filter((point) => totals.has(point.date)).map((point) => point.date);

    expect(actualKeys).toEqual(expectedKeys);
    for (const key of expectedKeys) {
      const entry = series.find((point) => point.date === key);
      expect(entry?.tokens ?? 0).toBe(totals.get(key) ?? 0);
    }
  });
});
