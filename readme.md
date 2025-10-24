# 💬 Open WebUI Chat Analyzer

Local analytics stack for exploring Open WebUI chat exports. The project pairs a FastAPI backend with a Next.js dashboard (`frontend-next/`) so you can explore conversations, summaries, and engagement trends without sending data outside your machine.

## Highlights
- Local FastAPI backend plus Next.js dashboard—your exports never leave your environment
- Load data by connecting to a live Open WebUI instance or by dropping exports into `data/`
- Automatic ingestion of the latest `all-chats-export-*.json` plus optional `users.csv` for friendly display names
- Overview metrics for chats, messages, user activity, model usage, file uploads, and approximate token volume
- **Instant metrics display**—dataset statistics update immediately after data loads while summaries process in the background
- Filtering across every visualisation by Open WebUI user and model
- Built-in summariser generates one-line chat headlines using local sentence embeddings plus a bundled Ollama service (with Open WebUI completions as a fallback)
- **Incremental summary persistence**—batches are written to SQLite as they complete so partial work is never lost
- Time analysis (daily trend, conversation length, hour-by-day heatmap) and content analysis (word clouds, message length)
- Sentiment breakdown with TextBlob, full-text search, paginated browsing, per-thread JSON downloads, and CSV exports
- Modern dashboard built with Next.js, Tailwind, shadcn/ui primitives, and Auth.js for credential and GitHub sign-in

## Contributor Guide
Review [AGENTS.md](AGENTS.md) for consolidated contributor workflows, coding standards, and release expectations.

## Configuration
1. Copy `.env.example` to `.env` before running anything.
2. Configure backend connectivity:
   - `OWUI_API_BASE_URL` – Primary URL the dashboard should target (`http://localhost:8502` locally, `http://backend:8502` for Docker Compose).
   - `OWUI_API_ALLOWED_ORIGINS` – Comma-separated list of origins permitted to call the FastAPI backend (defaults cover `http://localhost:3000` and `http://localhost:8503`).
   - `OWUI_DATA_DIR` – Directory where default exports live (relative to the project root).
3. (Optional) Prefill the Direct Connect form:
   - `OWUI_DIRECT_HOST` – Default Open WebUI base URL shown on the Load Data page.
   - `OWUI_DIRECT_API_KEY` – Optional API token that appears in the Direct Connect form (stored only in your local `.env`).
4. (Optional) Tune the summariser and GenAI helpers:
   - `SUMMARY_MAX_CHARS` / `SALIENT_K` – Control headline length and how many salient utterances feed the LLM.
   - `EMB_MODEL` – Sentence transformer used to pick salient lines (`sentence-transformers/all-MiniLM-L6-v2` by default).
   - `OLLAMA_*` variables – Configure the Ollama runtime, timeout, preload list, and default models for summaries, long-form responses, and embeddings.
   - `OWUI_COMPLETIONS_MODEL` – Chat completion model requested from your Open WebUI deployment (legacy fallback path).
5. Frontend defaults:
   - `FRONTEND_NEXT_PORT` – Published Docker port for the Next.js dashboard (defaults to `8503`).
   - `FRONTEND_NEXT_PUBLIC_URL` – External URL used for Auth.js redirects.
   - `FRONTEND_NEXT_BACKEND_BASE_URL` – Internal URL the Next.js proxy should use for FastAPI.
   - `NEXTAUTH_SECRET` / `NEXTAUTH_URL` – Secrets and base URLs for Auth.js session management.
   - Optional GitHub OAuth: `GITHUB_OAUTH_ENABLED`, `NEXT_PUBLIC_GITHUB_OAUTH_ENABLED`, `GITHUB_CLIENT_ID`, `GITHUB_CLIENT_SECRET`.
6. Restart the backend after changing environment variables. The first summariser run downloads embeddings locally and may take a moment.

## Input Data
Load data through either workflow:

- **Direct Connect** – Use **Load Data → Direct Connect** to point the analyzer at a live Open WebUI deployment. Provide the base URL (for example `http://localhost:3000`) and an API key; the backend will pull chats and users, persist them locally, display updated metrics instantly, and queue the summariser in the background.
- **File Uploads / Local Directory** – Export `all-chats-export-*.json` from Open WebUI (Admin Panel → **Settings → Data & Privacy → Export All Chats**) and optionally `users.csv` (Admin Panel → **Settings → Database → Export Users**) plus `models.json` captured from `/api/v1/models` for friendly model names. Drop the files in `data/` for automatic loading on startup or upload them on the Load Data page. Uploaded artifacts live under `uploads/`.

## Quick Start

### Option A – Docker (recommended)
```bash
git clone https://github.com/davidlarrimore/openwebui-chat-analyzer.git
cd openwebui-chat-analyzer
cp .env.example .env
make up
```
The FastAPI backend listens on `http://localhost:8502` and the Next.js dashboard is exposed on `http://localhost:8503` (or the value of `FRONTEND_NEXT_PORT`). Use `make logs` to tail combined logs and `make down` to stop everything.

### Option B – Local Python + Next.js environment
```bash
git clone https://github.com/davidlarrimore/openwebui-chat-analyzer.git
cd openwebui-chat-analyzer
python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
python -m textblob.download_corpora   # first run only
uvicorn backend.app:app --reload --port 8502  # terminal 1
```
In a second terminal:
```bash
cd openwebui-chat-analyzer/frontend-next
pnpm install
pnpm dev
```
The dashboard runs on [http://localhost:3000](http://localhost:3000) during development. Keep the backend process running so the proxy routes can reach `http://localhost:8502`.

### Option C – Guided setup scripts
Run `scripts/setup.sh` for an interactive wizard that prepares Docker or the local development environment. The script can also generate a `run.sh` helper that launches both the backend and Next.js dev server together.

## Backend API
The Next.js dashboard communicates with a FastAPI service that normalizes and serves chat exports. Key endpoints:

- `GET /api/v1/datasets/meta` – Current dataset identifier, row counts, source label, and last updated timestamp.
- `GET /api/v1/chats` / `GET /api/v1/messages` / `GET /api/v1/users` – Hydrated chat metadata, messages, and optional user directory.
- `POST /api/v1/openwebui/sync` – Pull chats and users directly from Open WebUI using a hostname plus API token.
- `POST /api/v1/uploads/chat-export` – Upload a new `all-chats-export*.json`; replaces the in-memory dataset and bumps the dataset id.
- `POST /api/v1/uploads/users` – Upload a companion `users.csv` for friendly display names.
- `POST /api/v1/datasets/reset` – Delete all stored chat, message, and user records plus metadata.
- `GET /api/v1/summaries/status` / `POST /api/v1/summaries/rebuild` – Monitor or requeue the background summariser job.

Run `uvicorn backend.app:app --reload` during development to keep the API available to the dashboard.

## Automated Chat Summaries
- Summaries are persisted in the `gen_chat_summary` field for each chat and surface throughout the Browse and Overview experiences.
- Every dataset update (direct sync or file upload) queues the summariser; progress is shown in the Load Data processing log with toast notifications.
- **Incremental persistence**: summaries are written to the database immediately after each batch is processed by Ollama, rather than waiting for all chats to complete. This means:
  - Dataset metrics update instantly after data loads, before summaries begin generating.
  - Summaries become available progressively as each batch completes.
  - If the summariser is interrupted, all previously completed batches are already saved.
  - The dashboard remains responsive during long summarisation jobs.
- The summariser picks salient utterances with `sentence-transformers/all-MiniLM-L6-v2`, then calls the bundled Ollama service (`OLLAMA_SUMMARY_MODEL`) with an automatic fallback to the Open WebUI completions endpoint at `OWUI_DIRECT_HOST`.
- Rebuild summaries anytime from **Load Data → Admin Tools → Rerun summaries** or through the API (`POST /api/v1/summaries/rebuild` + `/summaries/status`).

## Dashboard Tour
- **Load Data page**: View dataset stats that update instantly after data loads, stream processing logs in real-time, Direct Connect to Open WebUI, upload exports (with optional `users.csv`), and access admin tools to reset or rerun summaries. Metrics display immediately even while summaries generate in the background.
- **Overview metrics**: Totals and averages for chats, messages, per-role counts, file uploads, and approximate input/output token volumes (derived from character length).
- **Time analysis**: Daily trends, conversation length distributions, and hour-by-day heatmaps with filters for model and user.
- **Content analysis**: Word clouds for salient terms plus message length histograms broken down by role/model.
- **Sentiment**: TextBlob polarity triages conversations into positive, neutral, or negative bins with per-user breakdowns.
- **Browse and Search**: Dive into individual conversations, filter by user/model, run full-text search, and download enriched JSON/CSV exports.
- **Connection Info page**: Manage your OpenWebUI Direct Connect settings, monitor data freshness, view real-time sync logs, and configure automatic sync scheduling. See detailed documentation below.

## Connection Info Page

The Connection Info page (accessible at `/dashboard/admin/connection`) provides comprehensive management of your OpenWebUI Direct Connect integration with real-time monitoring and automated sync capabilities.

### Test Connection Flow

The **Test Connection** button verifies your OpenWebUI instance is reachable without importing data:

1. Click **Test Connection** to validate credentials and connectivity
2. The system tests both anonymous and authenticated endpoints
3. Success displays a toast with:
   - OpenWebUI version detected
   - Number of chats found
   - Connection attempts (if retries occurred)
4. Failure shows an error toast with detailed diagnostics
5. All tests complete without modifying your local dataset

**Security**: API keys are never logged or displayed. Connection logs only show "Using authenticated connection" without exposing credentials.

### Full vs Incremental Sync

Choose between two sync modes using the **Mode** toggle:

**Full Sync** (replaces all data):
- Wipes existing chats, messages, users, and models
- Fetches complete dataset from OpenWebUI
- Use when: switching data sources, recovering from corruption, or resetting the analyzer
- Triggered automatically when changing the hostname setting

**Incremental Sync** (appends new data):
- Compares source hostnames to detect same-source syncs
- Uses watermark tracking (max `updated_at` timestamp) to identify new chats
- Merges new chats, messages, users, and models into existing dataset
- Recommended mode when `has_data: true` and source matches
- Faster and preserves locally-generated summaries

**Last Sync Display**:
- Shows timestamp of most recent successful sync
- Displays "Never" if no sync has occurred
- Updates immediately after **Load Data** completes
- Persists across application restarts

### Quote-Free Settings Storage

The settings system automatically normalizes input to prevent serialization issues:

- **Quote Stripping**: Values like `"http://example.com"` are stored as `http://example.com`
- **Defensive Parsing**: Legacy quoted values are stripped on read
- **Migration**: Database migration (Migration 2) retroactively strips quotes from existing settings
- **API Response**: Settings endpoints always return clean, unquoted values
- **Source Tracking**: Each setting shows its origin (database, environment, or default)

**Note**: The legacy `dataset_source_display` setting has been removed. The dataset source is now derived directly from `OWUI_DIRECT_HOST`.

### Hostname Change: Wipe & Reload

Changing the hostname triggers a safe, transactional data refresh:

1. **Detection**: System compares normalized current vs new hostname
2. **Wipe Phase** (if hostname differs):
   - Transactional deletion in FK-safe order: messages → chats → users → models
   - SQLite PRAGMA `foreign_keys=ON` enforced
   - All-or-nothing: rollback on any failure
   - Emits structured log: `"Starting transactional wipe of OpenWebUI data"`
3. **Reload Phase**:
   - Triggers full sync from new hostname
   - Uses updated API key (or existing if unchanged)
   - Inherits all sync logging from standard pipeline
   - Emits log: `"Starting full reload from {new_hostname}"`
4. **Completion**:
   - In-memory state cleared and version bumped
   - Dataset metadata refreshed
   - Final log: `"Hostname change complete: wipe and reload finished"`

**Error Handling**: Exceptions during wipe/reload emit error logs and re-raise to caller for proper HTTP error response.

### Processing Log Viewer

The right panel displays real-time structured logs from all sync operations:

**Log Structure**:
- **Timestamp**: ISO 8601 with millisecond precision
- **Level**: debug, info, warning, error (color-coded)
- **Phase**: connect, fetch, persist, summarize, done, error (with icons)
- **Job ID**: Optional identifier for filtering multi-operation workflows
- **Message**: Human-readable description
- **Details**: Optional JSON metadata (e.g., `{"count": 150}`)

**Features**:
- **Auto-scroll**: Follows new entries (disable by scrolling up)
- **Auto-refresh**: Polls `/api/v1/logs` every 2 seconds
- **Filtering**: Query by `job_id` to isolate specific operations
- **Circular Buffer**: Retains last 200 events in memory
- **Secret Redaction**: API keys never appear in logs (only `"Using authenticated connection"`)

**Example Log Sequence** (hostname change):
```
[12:00:00.123] 🔌 INFO     Hostname changed to http://new.com, triggering wipe and reload
[12:00:00.456] 💾 INFO     Starting transactional wipe of OpenWebUI data
[12:00:01.789] 💾 INFO     Wipe complete
[12:00:01.890] 📥 INFO     Starting full reload from http://new.com
[12:00:02.123] 🔌 INFO     Starting sync from http://new.com
[12:00:03.456] 📥 INFO     Fetched 150 chats
[12:00:04.789] 💾 INFO     Performing full sync (different source or first sync)
[12:00:05.012] ✅ INFO     Full sync complete: 150 chats, 1250 messages
```

### Data Freshness & Staleness

Visual indicators help you monitor data currency:

**Staleness Pill** (next to Last Sync):
- **Green "Up to date"**: Last sync within threshold (default 6 hours)
- **Amber "Stale"**: Last sync exceeds threshold
- Hidden when no sync has occurred
- Threshold configurable via `SYNC_STALENESS_THRESHOLD_HOURS` setting

**Staleness Calculation**:
- Compares `last_sync_at` to current time
- Uses `timedelta(hours=staleness_threshold_hours)`
- Data with no timestamp but existing chats marked stale
- Fresh data marked `is_stale: false`

**Local Counts** (in sync status API):
- Chats, messages, users, models
- Returned in `/api/v1/sync/status` response
- Useful for comparing local vs upstream counts

### Automatic Sync Scheduler

Configure periodic incremental syncs via the **Scheduler** button:

**Scheduler Drawer**:
- **Enable/Disable Toggle**: Turn automatic syncs on/off
- **Interval Selector**: 5-1440 minutes (5 min to 24 hours)
- **Save Interval**: Persists configuration to database
- **State Display**: Shows enabled status, interval, last run, next run

**How It Works**:
- Scheduler state stored in database (`SYNC_SCHEDULER_ENABLED`, `SYNC_SCHEDULER_INTERVAL_MINUTES`)
- Interval validated: 5 ≤ interval ≤ 1440
- Next run calculated as `last_run_at + interval`
- Manual syncs unaffected (always available via **Load Data** button)

**API Endpoints**:
- `GET /api/v1/sync/scheduler`: Fetch current config
- `POST /api/v1/sync/scheduler`: Update config with `{enabled: bool, interval_minutes: int}`

**Note**: The scheduler infrastructure tracks state and configuration. Actual background execution (cron-like functionality) requires additional threading/scheduling implementation (e.g., `APScheduler`, `threading.Timer`).

### Edit Mode & Settings Persistence

Data source settings are protected by edit mode:

1. **Default State**: Hostname and API Key fields are **disabled**
2. **Edit Mode**: Click **Edit Data Source** to enable fields
3. **In Edit Mode**:
   - Fields become editable
   - **Save Settings** and **Cancel** buttons appear
   - Changes tracked against original values
4. **Save**:
   - Only changed values sent to backend (`PUT /api/v1/admin/settings/direct-connect`)
   - Settings persisted to database
   - Hostname change triggers wipe/reload (see above)
   - Edit mode exits on success
5. **Cancel**:
   - Reverts fields to original values
   - Exits edit mode without API call

**Security**:
- API keys displayed as password field (`type="password"`)
- Backend logs show redacted keys: `supe...2345` (first 4 + last 4 chars)
- Full secrets never logged or displayed

### WAL Mode & Concurrency

The database uses SQLite optimizations for better performance during large syncs:

- **WAL Mode**: Write-Ahead Logging (`PRAGMA journal_mode=WAL`)
- **Foreign Keys**: Enforced (`PRAGMA foreign_keys=ON`)
- **Synchronous**: Set to NORMAL (`PRAGMA synchronous=NORMAL`) for speed while maintaining safety

**Benefits**:
- Prevents long locks during reload operations
- Allows concurrent reads during writes
- Maintains transactional integrity

## Working With the Data
- CSV downloads contain the same columns the dashboard uses, making follow-on analysis in pandas, spreadsheets, or BI tools straightforward.
- Per-thread JSON downloads include metadata, ISO timestamps, attachments, and every message shown in the interface.
- Sentiment scores and token estimates are heuristic: tokens are inferred from character counts, and sentiment uses TextBlob’s polarity scale (−1 to 1).

## Sample Data
`sample_data/sample_data_extract.json` and `sample_data/sample_users.csv` let you explore the dashboard without waiting for a fresh export. Copy them to `data/` or upload them through the UI to see the charts populate immediately.

## Development Notes
- `docker-compose.yml` defines the backend, Next.js frontend, and optional Nginx reverse proxy. Use `make dev` (or `docker compose -f docker-compose.yml -f docker-compose.dev.yml up`) for hot reload.
- The `Makefile` centralises build and lifecycle commands — start with `make help`.
- Python dependencies live under `backend/requirements.txt` (aggregated through the root `requirements.txt`).
- The Next.js app lives under `frontend-next/`; use `pnpm dev`, `pnpm build`, and `pnpm test` for local workflows.
- The root Dockerfile includes a dedicated Next.js stage so you can build `frontend-next` alongside the Python services (`docker build --target frontend-next .`).

## Frontend Architecture
- `frontend-next/app/` – App Router routes, layouts, and API proxy handlers.
- `frontend-next/components/` – Shared UI primitives, charts, tables, and providers.
- `frontend-next/lib/` – Auth helpers, type definitions, and backend proxy utilities.
- `frontend-next/tests/` – Jest smoke tests that exercise the server-side proxy handlers.
- Authentication flows are powered by Auth.js (NextAuth) with credentials and optional GitHub OAuth providers.

## Privacy & Storage
All requests stay on your machine—the Next.js dashboard only talks to the bundled FastAPI service. Uploaded files remain under the repository (`data/` and `uploads/`) until you remove them.

## Troubleshooting
- If the Next.js dev server fails to start, confirm Node 20+ and pnpm 8 are installed.
- Summaries failing or timing out? Confirm your Open WebUI deployment at `OWUI_DIRECT_HOST` is reachable, the API key is valid, and that the sentence-transformers model has been downloaded (first run may take a minute).
- Seeing 5xx errors from `/api/v1/genai/*`? Make sure your local Ollama runtime is up (`ollama list`) and that the required models are available.
- Some environments need a font package for `wordcloud`; installing system fonts (for example `sudo apt-get install fonts-dejavu`) fixes blank visuals.
- Seeing “Unable to connect to the backend API”? Make sure `uvicorn backend.app:app --port 8502` (or the Docker `backend` service) is running and reachable.

## License
MIT — see `LICENSE` for the full text.
