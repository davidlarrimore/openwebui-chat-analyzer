import type { ProcessLogEvent } from "./api";

export type JobType = "summarizer" | "dataSync" | "generic";

export type JobLifecycleStatus = "pending" | "running" | "success" | "error";

export type JobStageVisualState = "pending" | "current" | "complete" | "error";

export interface JobStageState {
  id: string;
  label: string;
  state: JobStageVisualState;
}

export interface JobProgressState {
  completed: number;
  total: number;
  percent: number;
  message?: string | null;
}

export interface JobState {
  jobKey: string;
  jobId: string;
  type: JobType;
  label: string;
  subtitle?: string | null;
  status: JobLifecycleStatus;
  startedAt?: string;
  completedAt?: string;
  updatedAt: string;
  progress?: JobProgressState | null;
  lastMessage?: string | null;
  stages?: JobStageState[];
  logs: ProcessLogEvent[];
}

export const DATA_SYNC_STAGES: JobStageState[] = [
  { id: "connect", label: "Connect", state: "pending" },
  { id: "fetch", label: "Extract", state: "pending" },
  { id: "persist", label: "Load", state: "pending" },
  { id: "normalize", label: "Normalize", state: "pending" },
];

const PHASE_TO_STAGE_INDEX: Record<string, number> = {
  connect: 0,
  fetch: 1,
  persist: 2,
  normalize: 3,
  done: 3,
  error: 2,
};

export function deriveJobTypeFromLog(log: ProcessLogEvent): JobType {
  if (log.phase === "summarize" || /summarizer/i.test(log.message)) {
    return "summarizer";
  }
  if (/sync/i.test(log.message) || ["connect", "fetch", "persist", "done", "error"].includes(log.phase)) {
    return "dataSync";
  }
  return "generic";
}

export function normaliseJobKey(jobType: JobType, jobId: string | number | null | undefined): string {
  const keyId = jobId !== null && jobId !== undefined ? String(jobId) : "unknown";
  return `${jobType}:${keyId}`;
}

export function updateDataSyncStages(
  previous: JobStageState[] | undefined,
  phase: ProcessLogEvent["phase"]
): JobStageState[] {
  const currentStages = previous?.length ? previous.map((stage) => ({ ...stage })) : DATA_SYNC_STAGES.map((stage) => ({ ...stage }));
  const stageIndex = PHASE_TO_STAGE_INDEX[phase] ?? -1;

  if (phase === "done") {
    return currentStages.map((stage) => ({ ...stage, state: "complete" }));
  }

  currentStages.forEach((stage, index) => {
    if (phase === "error" && index === Math.max(stageIndex, 0)) {
      stage.state = "error";
    } else if (index < stageIndex) {
      stage.state = "complete";
    } else if (index === stageIndex) {
      stage.state = "current";
    } else if (stage.state !== "complete") {
      stage.state = "pending";
    }
  });

  return currentStages;
}

export function inferJobSubtitle(jobType: JobType, log: ProcessLogEvent): string | null {
  if (jobType === "dataSync") {
    const match = /Starting sync from\s+([^\s]+)/i.exec(log.message);
    if (match && match[1]) {
      return match[1].trim();
    }
    const details = log.details as Record<string, unknown> | undefined;
    const users = details?.["users"];
    if (typeof users === "number") {
      return `${users.toLocaleString()} users`;
    }
  }
  if (jobType === "summarizer") {
    return "Chat summaries";
  }
  return null;
}
