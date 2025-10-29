import { buildTokenConsumptionSeries, type OverviewMessage } from "@/lib/overview";

describe("overview token consumption series", () => {
  it("groups messages by Eastern Time day boundaries and fills missing dates", () => {
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

    expect(series).toEqual([
      { date: "2024-03-09", tokens: 1 },
      { date: "2024-03-10", tokens: 2 },
      { date: "2024-03-11", tokens: 0 },
      { date: "2024-03-12", tokens: 1 }
    ]);
  });

  it("assigns tokens based on Eastern Time even for timestamps near UTC midnight", () => {
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

    expect(series).toEqual([
      { date: "2024-01-31", tokens: 1 },
      { date: "2024-02-01", tokens: 1 }
    ]);
  });
});
