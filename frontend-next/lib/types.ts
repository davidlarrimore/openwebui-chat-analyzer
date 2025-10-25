export interface User {
  id: string;
  username?: string;
  email?: string;
  name?: string;
  avatar_url?: string;
  role?: string;
}

export interface Chat {
  id: string;
  title?: string;
  created_at?: string;
  updated_at?: string;
  participants?: string[];
  tags?: string[];
  message_count?: number;
  summary?: string;
}

export interface Message {
  id: string;
  chat_id: string;
  sender?: string;
  content?: string;
  created_at?: string;
}

export interface TimeSeriesPoint {
  date: string;
  value: number;
}

export interface ExportRecord {
  id: string;
  name: string;
  format: "csv" | "json" | string;
  created_at?: string;
  download_path: string;
}

export interface AppMetadata {
  dataset_source?: string;
  dataset_pulled_at?: string;
  chat_uploaded_at?: string;
  users_uploaded_at?: string;
  models_uploaded_at?: string;
  first_chat_day?: string;
  last_chat_day?: string;
  chat_count?: number;
  user_count?: number;
  message_count?: number;
  model_count?: number;
  [key: string]: unknown;
}

export interface DatasetMeta {
  dataset_id: string;
  chat_count: number;
  message_count: number;
  user_count: number;
  model_count: number;
  source?: string | null;
  app_metadata?: AppMetadata | null;
}

export interface UploadStats {
  mode?: "full" | "incremental" | "noop" | string;
  source_matched?: boolean;
  submitted_hostname?: string;
  normalized_hostname?: string;
  new_chats?: number;
  new_messages?: number;
  new_users?: number;
  new_models?: number;
  models_changed?: boolean;
  summarizer_enqueued?: boolean;
  total_chats?: number;
  total_messages?: number;
  total_users?: number;
  total_models?: number;
  queued_chat_ids?: string[];
  [key: string]: unknown;
}

export interface UploadResult {
  detail?: string | null;
  dataset?: Record<string, unknown> | null;
  stats?: UploadStats | null;
}

export interface SummaryStatus {
  state: string | null;
  total: number;
  completed: number;
  message?: string | null;
  last_event?: Record<string, unknown> | null;
}

export interface SummaryEvent {
  event_id: string;
  type?: string | null;
  message?: string | null;
  timestamp?: string | null;
  job_id?: number | string | null;
  outcome?: string | null;
  chat_id?: string | null;
  chunk_index?: number | null;
  chunk_count?: number | null;
  completed?: number | null;
  total?: number | null;
  reason?: string | null;
  [key: string]: unknown;
}

export interface SummaryEventsResponse {
  events: SummaryEvent[];
  reset?: boolean;
}

export interface OllamaModelTag {
  name: string;
  modified_at?: string;
  digest?: string;
  size?: number;
  [key: string]: unknown;
}

export interface AuthStatus {
  has_users: boolean;
}

export interface AuthSessionUser {
  id: string;
  username: string;
  email: string;
  name: string;
}

export interface AuthResponse {
  access_token: string;
  token_type: "bearer";
  user: AuthSessionUser;
}

export interface DirectConnectSettings {
  host: string;
  api_key: string;
  database_host: string;
  database_api_key: string;
  host_source: "database" | "environment" | "default";
  api_key_source: "database" | "environment" | "empty";
}
