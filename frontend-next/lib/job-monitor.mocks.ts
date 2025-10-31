import { updateDataSyncStages, type JobState } from "./job-monitor";

const now = new Date();

export const mockJobStates: JobState[] = [
  {
    jobKey: "dataSync:demo-sync",
    jobId: "demo-sync",
    type: "dataSync",
    label: "Data Load",
    subtitle: "openwebui.demo",
    status: "running",
    startedAt: new Date(now.getTime() - 90_000).toISOString(),
    updatedAt: new Date(now.getTime() - 5_000).toISOString(),
    stages: updateDataSyncStages(undefined, "persist"),
    logs: [],
  },
  {
    jobKey: "summarizer:42",
    jobId: "42",
    type: "summarizer",
    label: "Summarizer",
    subtitle: "Chat summaries",
    status: "success",
    startedAt: new Date(now.getTime() - 12 * 60_000).toISOString(),
    completedAt: new Date(now.getTime() - 60_000).toISOString(),
    updatedAt: new Date(now.getTime() - 60_000).toISOString(),
    progress: {
      completed: 128,
      total: 128,
      percent: 100,
      message: "Summary job completed successfully.",
    },
    logs: [],
    lastMessage: "Summary job completed successfully.",
  },
];
