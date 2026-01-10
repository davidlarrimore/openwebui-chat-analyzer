# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture Overview

This is a **local-first analytics platform** for Open WebUI conversations, comprising:
- **Backend**: FastAPI service (Python 3.10+) with SQLite persistence
- **Frontend**: Next.js 14 dashboard with App Router
- **Authentication**: Hybrid local/OIDC system (Microsoft Entra ID supported)
- **AI Pipeline**: Sentence transformers + LLM summarization (Ollama/OpenAI/OpenWebUI providers)

### Core Data Flow

1. **Ingestion** (`backend/services.py:DataService`):
   - Direct Connect: Syncs from Open WebUI via API (`/api/v1/openwebui/sync`)
   - File Upload: Processes `all-chats-export-*.json` files
   - Writes to SQLite via `backend/storage.py:DatabaseStorage`
   - Maintains pseudonym mapping for privacy (`backend/data/pseudonyms.json`)

2. **Summarization Pipeline** (`backend/summarizer.py`):
   - Extracts salient utterances using sentence-transformers (configurable via `SALIENT_K`)
   - Generates one-line summaries via provider registry (`backend/provider_registry.py`)
   - Supports three connection types: Ollama (local), OpenAI (API), OpenWebUI (fallback)
   - Saves incrementally to avoid data loss during long-running jobs

3. **Provider Architecture** (`backend/providers/`):
   - Abstract base: `backend/providers/base.py:LLMProvider`
   - Implementations: `ollama.py`, `openai.py`, `openwebui.py`
   - Registry manages model discovery and validation (`backend/provider_registry.py`)
   - All providers support: list models, validate, generate completions

4. **Authentication System** (`backend/auth/`):
   - Three modes: `DEFAULT` (local only), `OAUTH` (OIDC only), `HYBRID` (both)
   - Session management via HTTP-only cookies with auto-refresh
   - OIDC callback preserves original destination URL
   - Token hashing uses `AUTH_TOKEN_HASH_SECRET`

5. **Frontend API Client** (`frontend-next/lib/api.ts`):
   - Auto-routes all requests through `/api/backend/*` prefix
   - Server-side: forwards cookies to backend for session validation
   - Client-side: credentials-included fetch with auth redirect handling

### Key Subsystems

**DataService Singleton** (`backend/services.py`):
- Mounted on `app.state.data_service` for test monkeypatching
- In-memory caches backed by SQLite (`backend/storage.py`)
- Thread-safe with RLock for sync operations
- Circular buffer for processing logs (200 events)

**Database Layer** (`backend/db.py`, `backend/storage.py`):
- SQLAlchemy 2.0 ORM with SQLite (WAL mode for concurrency)
- Models: `ChatRecord`, `MessageRecord`, `OpenWebUIUser`, `ModelRecord`, `Setting`, `Account`, `AccessToken`
- Automatic schema migrations in `db.py:_run_migrations()`
- Connection string: `OWUI_DB_URL` or `OWUI_SQLITE_PATH`

**Sync Scheduler** (`backend/services.py`):
- Background thread for automatic incremental syncs
- Configurable interval (5 min - 24 hours) via `SyncSchedulerConfig`
- Staleness threshold: `SYNC_STALENESS_THRESHOLD_HOURS` (default: 6)

**Frontend Structure**:
- **Pages**: `frontend-next/app/` (App Router)
- **Components**: `frontend-next/components/` (shared UI with shadcn/ui)
- **API/Types**: `frontend-next/lib/` (centralized API client + TypeScript types)
- **Auth**: Automatic redirects, session checks on protected routes

## Common Commands

### Development Setup

```bash
# Backend (local)
python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m textblob.download_corpora  # First run only
cp .env.example .env
uvicorn backend.app:app --reload --port 8502

# Frontend (local)
cd frontend-next
pnpm install
pnpm dev  # Runs on http://localhost:3000
```

### Docker Development

```bash
make up          # Start all services (production build)
make dev         # Start with hot-reload (uses docker-compose.dev.yml)
make logs        # Tail all logs
make down        # Stop all services
make rebuild     # Rebuild and restart

# Service-specific commands
make logs-backend
make logs-frontend-next
make shell-backend       # Access backend container
make shell-frontend-next # Access frontend container
make health-backend      # Test backend health endpoint
```

### Testing

```bash
# Backend tests
pytest backend/tests/              # All tests
pytest backend/tests/ -v           # Verbose output
pytest backend/tests/test_summarizer_pipeline.py  # Single file
pytest backend/tests/ -k test_name # Run tests matching pattern

# Frontend tests
cd frontend-next
pnpm test                          # Run Jest suite
pnpm lint                          # ESLint check
```

### Building

```bash
# Backend (runs in Docker)
docker compose build backend

# Frontend
cd frontend-next
pnpm build    # Production build
pnpm start    # Serve production build
```

### Database Operations

```bash
# Reset database (WARNING: destructive)
rm data/openwebui_chat_analyzer.db

# Check database location
echo $OWUI_SQLITE_PATH  # or check .env

# Manual migration (if needed)
# Migrations run automatically on startup via backend/db.py:_run_migrations()
```

## Development Patterns

### Adding a New API Endpoint

1. Define Pydantic models in `backend/models.py`
2. Add route handler in `backend/routes.py` (use `router` with `/api/v1` prefix)
3. Implement business logic in `backend/services.py:DataService`
4. Add frontend API call in `frontend-next/lib/api.ts` (use `apiGet`/`apiPost`)
5. Add TypeScript types in `frontend-next/lib/types.ts`
6. Write backend test in `backend/tests/test_*.py`

### Adding a New LLM Provider

1. Create `backend/providers/your_provider.py` implementing `LLMProvider` base class
2. Register in `backend/provider_registry.py:ProviderRegistry.__init__()`
3. Add configuration constants to `backend/config.py`
4. Update `.env.example` with new environment variables
5. Add tests in `backend/tests/test_providers.py` (or new file)

### Working with Authentication

- Protected routes must call `backend/auth/deps.py:require_session()` dependency
- Frontend: use `apiGet`/`apiPost` from `lib/api.ts` (handles auth automatically)
- Session cookie name: `SESSION_COOKIE_NAME` (default: `analyzer_session`)
- OIDC callback: `/api/backend/auth/oidc/callback` preserves original URL

### Modifying the Summarization Pipeline

- Model selection: stored in `Setting` table with key `SUMMARIZER_PRIMARY_MODEL`
- Connection type: `SUMMARIZER_CONNECTION_TYPE` (ollama|openai|openwebui)
- Runtime config: `backend/summarizer.py` module-level variables (`_SUMMARY_MODEL`, etc.)
- Context building: `_build_salient_context()` uses sentence-transformers
- Incremental saves: `DataService._update_chat_summary()` commits after each chat

### Testing with Sample Data

```bash
# Load sample data (auto-loaded on startup)
cp sample_data/sample_data_extract.json data/
cp sample_data/sample_users.csv data/
make restart-backend
```

## Critical Configuration

**Environment Variables** (`.env`):
- `OWUI_DIRECT_HOST`: Default Open WebUI URL
- `OLLAMA_BASE_URL`: Ollama service URL (`http://host.docker.internal:11434` in Docker)
- `OPENAI_API_KEY`: Required for OpenAI provider
- `AUTH_MODE`: `DEFAULT` (local) | `HYBRID` (local+OIDC) | `OAUTH` (OIDC only)
- `SESSION_SECRET`: MUST be changed in production
- `OWUI_SQLITE_PATH`: Database location (default: `data/openwebui_chat_analyzer.db`)

**Provider Registry** (`backend/data/models_registry.json`):
- Maps model IDs to friendly names
- Updated via `/api/v1/uploads/models` endpoint

**Pseudonym Pool** (`backend/data/pseudonyms.json`):
- Static list of privacy-preserving aliases
- Assigned deterministically via hash of real user ID
- Configurable exposure via `OWUI_EXPOSE_REAL_NAMES`

## Important Notes

- **SQLite WAL Mode**: Enabled by default for better concurrency during syncs
- **Thread Safety**: `DataService` uses `RLock` for sync operations; summary jobs check for cancellation
- **API Versioning**: All data endpoints prefixed with `/api/v1`
- **CORS**: Configure `OWUI_API_ALLOWED_ORIGINS` to allow frontend origins
- **Hot Reload**: Backend uses `uvicorn --reload`; frontend uses Next.js dev server
- **Docker Networking**: Frontend uses `FRONTEND_NEXT_BACKEND_BASE_URL=http://backend:8502` internally
- **Port Mapping**: Backend → 8502, Frontend → 8503 (configurable via `FRONTEND_NEXT_PORT`)
