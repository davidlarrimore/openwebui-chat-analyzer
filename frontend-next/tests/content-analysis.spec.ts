import {
  ALL_MODELS_OPTION,
  ALL_USERS_OPTION,
  buildModelOptions,
  buildUserOptions,
  calculateAverageMessageLength,
  buildMessageLengthHistogram,
  extractWordFrequencies,
  filterMessagesByUserAndModel,
  type ContentChat,
  type ContentMessage
} from "@/lib/content-analysis";

describe("content analysis helpers", () => {
  const chats: ContentChat[] = [
    { chatId: "c1", userId: "u1" },
    { chatId: "c2", userId: "u2" }
  ];

  const messages: ContentMessage[] = [
    { chatId: "c1", role: "user", content: "Hello world from the user", model: "gpt-4" },
    { chatId: "c1", role: "assistant", content: "Assistant reply", model: "gpt-4" },
    { chatId: "c2", role: "user", content: "Exploring analytics metrics", model: "gpt-3.5" },
    { chatId: "c2", role: "assistant", content: "Details about analysis", model: "gpt-3.5" }
  ];

  it("builds sorted user and model options", () => {
    const userDisplayMap = { u1: "Alice", u2: "Bob" };
    const userOptions = buildUserOptions(chats, userDisplayMap);
    expect(userOptions[0]).toEqual({ value: ALL_USERS_OPTION, label: "All Users" });
    expect(userOptions.slice(1)).toEqual([
      { value: "u1", label: "Alice" },
      { value: "u2", label: "Bob" }
    ]);

    const modelOptions = buildModelOptions(messages);
    expect(modelOptions).toEqual([ALL_MODELS_OPTION, "gpt-3.5", "gpt-4"]);
  });

  it("filters messages by user and model selections", () => {
    const all = filterMessagesByUserAndModel(chats, messages, ALL_USERS_OPTION, ALL_MODELS_OPTION);
    expect(all.filteredMessages).toHaveLength(4);
    expect(new Set(all.matchingChatIds)).toEqual(new Set(["c1", "c2"]));

    const onlyUser = filterMessagesByUserAndModel(chats, messages, "u1", ALL_MODELS_OPTION);
    expect(onlyUser.filteredMessages).toHaveLength(2);
    expect(onlyUser.filteredMessages.every((entry) => entry.chatId === "c1")).toBe(true);

    const noMatch = filterMessagesByUserAndModel(chats, messages, "u1", "gpt-3.5");
    expect(noMatch.filteredMessages).toHaveLength(0);
    expect(noMatch.matchingChatIds).toEqual([]);
  });

  it("computes average message length by role", () => {
    const result = calculateAverageMessageLength(messages);
    const roles = result.map((entry) => entry.role);

    expect(roles).toEqual(["User", "Assistant"]);

    const userEntry = result.find((entry) => entry.role === "User");
    expect(userEntry?.averageLength).toBeGreaterThan(0);

    const assistantEntry = result.find((entry) => entry.role === "Assistant");
    expect(assistantEntry?.averageLength).toBeGreaterThan(0);
  });

  it("builds a histogram of message lengths", () => {
    const histogram = buildMessageLengthHistogram(messages, 5);
    expect(histogram).not.toHaveLength(0);
    expect(histogram.every((bin) => typeof bin.count === "number")).toBe(true);
    expect(histogram.every((bin) => typeof bin.range === "string" && bin.range.length > 0)).toBe(true);
  });

  it("extracts word frequencies excluding stop words", () => {
    const frequencies = extractWordFrequencies(messages);
    const words = frequencies.map((entry) => entry.text);

    expect(words).toContain("hello");
    expect(words).toContain("world");
    expect(words).not.toContain("the");

    const helloEntry = frequencies.find((entry) => entry.text === "hello");
    expect(helloEntry?.count).toBeGreaterThanOrEqual(1);
  });
});

