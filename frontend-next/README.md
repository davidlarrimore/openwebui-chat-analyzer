# Frontend Next App

This directory contains the Next.js 14 + App Router implementation of the chat analyzer UI.

## Tech Stack

- Next.js App Router (TypeScript)
- Tailwind CSS with shadcn/ui primitives
- FastAPI-backed HttpOnly sessions (local or Microsoft Entra ID)
- Recharts and Visx for data visualisation
- FastAPI backend consumed through server-side proxy route handlers

## Prerequisites

- Node.js 20 (ships with npm; `corepack enable` enables pnpm)
- pnpm 8 (`corepack prepare pnpm@8.15.4 --activate`)
- FastAPI service running at `BACKEND_BASE_URL` (default `http://localhost:8000`)

## Getting Started

1. Ensure the project root `.env` includes the frontend-next variables (`FRONTEND_NEXT_PORT`, `BACKEND_BASE_URL`, `APP_BASE_URL`, etc.).
2. Install dependencies and start dev server:
   ```bash
   cd frontend-next
   pnpm install
   pnpm dev
   ```
   The app runs on [http://localhost:3000](http://localhost:3000) by default.
4. Run the minimal smoke test:
   ```bash
   pnpm test
   ```

### Docker

To containerise the frontend:

```bash
docker compose up --build frontend-next
```

This uses the root `.env` for configuration and only starts the Next.js frontend; run the FastAPI backend separately.

## Auth Configuration

- Client-side components call the backend session authority at `/api/backend/auth/*` using same-origin requests with `credentials: "include"`.
- On first launch, if no backend users exist, you'll be redirected to `/register` to create the initial administrator account.
- To enable Microsoft Entra ID, populate the `OIDC_*` variables in `.env` and set `AUTH_MODE=HYBRID` or `AUTH_MODE=OAUTH`.

## Backend CORS

Ensure FastAPI allows the Next.js origin (update and restart the backend):

```python
# fastapi_cors_snippet.py
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Project Structure

```
frontend-next/
  app/                # App Router routes & layouts
  components/         # UI primitives, charts, tables, providers
  lib/                # API client, config helpers, shared types
  tests/              # Jest smoke tests exercising the middleware
  middleware.ts       # Protects /dashboard routes via backend session checks
```

## Testing

- `pnpm test` runs `tests/smoke.spec.ts`, a Jest-based integration check that:
  - Verifies `/api/session` returns 401 when unauthenticated
  - Ensures the proxy login route properly forwards mocked requests

Add additional Jest or Playwright suites beside this smoke test as the migration progresses.

## Launch Checklist

- [ ] Validate feature parity against the legacy dashboard
- [ ] Extend automated tests to cover ingestion, summarisation, and charting flows
- [ ] Obtain QA sign-off on auth paths (credentials + optional GitHub)
- [ ] Ensure analytics/telemetry are configured as required
- [ ] Update docs, links, and infra configs to point to the Next.js app
