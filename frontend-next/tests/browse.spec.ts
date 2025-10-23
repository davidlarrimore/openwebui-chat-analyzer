import {
  ALL_MODELS_OPTION,
  ALL_USERS_OPTION,
  BrowseMessage,
  buildModelOptions,
  buildThreadExportPayload,
  buildUserOptions,
  normaliseBrowseChats,
  normaliseBrowseMessages
} from "@/lib/browse";

describe("browse helpers", () => {
  const fullSummary =
    "Quick recap of the discussion with extended analysis, key decisions, and follow-up actions captured for reviewers.";
  const rawChats = [
    {
      chat_id: "chat-123",
      user_id: "user-1",
      title: "Sample Conversation",
      summary_full: fullSummary,
      gen_chat_summary: "Quick recap of the discussion.",
      summary_128: "Quick recap of the discussion.",
      created_at: "2024-05-01T12:00:00Z",
      updated_at: "2024-05-01T13:00:00Z",
      files_uploaded: 2,
      files: [{ filename: "notes.md" }, { name: "diagram.png" }],
      tags: ["analysis"],
      models: ["gpt-4-turbo"],
      archived: false,
      pinned: true
    }
  ];

  const rawUsers = [
    {
      user_id: "user-1",
      name: "Analyst Alice"
    }
  ];

  const rawModels = [
    {
      model_id: "gpt-4-turbo",
      name: "GPT-4 Turbo"
    }
  ];

  const rawMessages = [
    {
      message_id: "msg-1",
      chat_id: "chat-123",
      role: "user",
      content: "Can you summarise the quarterly performance?",
      timestamp: "2024-05-01T12:00:15Z",
      model: "gpt-4-turbo"
    },
    {
      message_id: "msg-2",
      chat_id: "chat-123",
      role: "assistant",
      content: "Certainly! The quarter exceeded expectations...",
      timestamp: "2024-05-01T12:01:10Z",
      model: "gpt-4-turbo"
    }
  ];

  it("normalises chats with user display names and metadata", () => {
    const chats = normaliseBrowseChats(rawChats, rawUsers);
    expect(chats).toHaveLength(1);

    const chat = chats[0];
    expect(chat.chatId).toBe("chat-123");
    expect(chat.userDisplay).toBeTruthy();
    expect(chat.summary).toBe(fullSummary);
    expect(chat.tags).toEqual(["analysis"]);
    expect(chat.filesUploaded).toBe(2);
    expect(chat.models).toEqual(["gpt-4-turbo"]);
    expect(chat.createdAt).toBe("2024-05-01T12:00:00.000Z");
    expect(chat.updatedAt).toBe("2024-05-01T13:00:00.000Z");
  });

  it("normalises messages with model display names", () => {
    const messages = normaliseBrowseMessages(rawMessages, rawModels);
    expect(messages).toHaveLength(2);

    const assistantMessage = messages.find((message) => message.role === "assistant") as BrowseMessage;
    expect(assistantMessage.model).toBe("GPT-4 Turbo");
    expect(assistantMessage.modelId).toBe("gpt-4-turbo");
    expect(assistantMessage.timestamp).toBe("2024-05-01T12:01:10.000Z");
  });

  it("builds export payloads suitable for download", () => {
    const chats = normaliseBrowseChats(rawChats, rawUsers);
    const messages = normaliseBrowseMessages(rawMessages, rawModels);

    const payload = buildThreadExportPayload(chats[0], messages);
    expect(payload.chat_id).toBe("chat-123");
    expect(payload.user_name).toBe(chats[0].userDisplay);
    expect(payload.messages).toHaveLength(2);
    expect(payload.messages[0]).toHaveProperty("message_id", "msg-1");
  });

  it("uses generated chat summaries when full text is unavailable", () => {
    const chats = normaliseBrowseChats(
      [
        {
          chat_id: "chat-456",
          user_id: "user-1",
          title: "Secondary Conversation",
          gen_chat_summary: "Concise summary only."
        }
      ],
      rawUsers
    );

    expect(chats).toHaveLength(1);
    expect(chats[0].summary).toBe("Concise summary only.");
  });

  it("accepts legacy summary_128 fields for backwards compatibility", () => {
    const chats = normaliseBrowseChats(
      [
        {
          chat_id: "chat-legacy",
          user_id: "user-1",
          title: "Legacy Conversation",
          summary_128: "Legacy truncated summary."
        }
      ],
      rawUsers
    );

    expect(chats).toHaveLength(1);
    expect(chats[0].summary).toBe("Legacy truncated summary.");
  });

  it("uses summaries persisted in chat metadata when provided", () => {
    const chats = normaliseBrowseChats(
      [
        {
          chat_id: "chat-789",
          user_id: "user-1",
          title: "Metadata Conversation",
          gen_chat_summary: "Legacy truncated summary.",
          meta: {
            summary_full: "Full summary stored in metadata for the conversation."
          }
        }
      ],
      rawUsers
    );

    expect(chats[0].summary).toBe("Full summary stored in metadata for the conversation.");
  });

  it("provides user and model filter options", () => {
    const chats = normaliseBrowseChats(rawChats, rawUsers);
    const messages = normaliseBrowseMessages(rawMessages, rawModels);
    const userDisplayMap = chats.reduce<Record<string, string>>((acc, chat) => {
      if (chat.userId) {
        acc[chat.userId] = chat.userDisplay;
      }
      return acc;
    }, {});

    const userOptions = buildUserOptions(chats, userDisplayMap);
    expect(userOptions[0]).toEqual({ value: ALL_USERS_OPTION, label: "All Users" });
    expect(userOptions[1]).toEqual({ value: "user-1", label: chats[0].userDisplay });

    const modelOptions = buildModelOptions(messages);
    expect(modelOptions).toEqual([ALL_MODELS_OPTION, "GPT-4 Turbo"]);
  });
});
