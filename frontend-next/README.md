# Frontend Next App

This directory contains a Next.js 14 + App Router rewrite of the chat analyzer UI. It lives beside the existing Streamlit experience (`frontend/`) and **must not** modify or depend on that code. Both interfaces can run in parallel during migration.

## Tech Stack

- Next.js App Router (TypeScript)
- Tailwind CSS with shadcn/ui primitives
- Auth.js (NextAuth) with credentials and optional GitHub OAuth
- Recharts and Visx for data visualisation
- FastAPI backend consumed through server-side proxy route handlers

## Prerequisites

- Node.js 20 (ships with npm; `corepack enable` enables pnpm)
- pnpm 8 (`corepack prepare pnpm@8.15.4 --activate`)
- FastAPI service running at `BACKEND_BASE_URL` (default `http://localhost:8000`)

## Getting Started

1. Ensure the project root `.env` includes the frontend-next variables (`FRONTEND_NEXT_PORT`, `NEXTAUTH_SECRET`, `BACKEND_BASE_URL`, etc.).
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

- Credentials provider posts to `${BACKEND_BASE_URL}/api/v1/auth/login`.
- On first launch, if no backend users exist, you'll be redirected to `/register` to create the initial administrator account.
- Set `GITHUB_OAUTH_ENABLED=true` **and** `NEXT_PUBLIC_GITHUB_OAUTH_ENABLED=true` to enable the GitHub login button. Fill in `GITHUB_CLIENT_ID` and `GITHUB_CLIENT_SECRET`.

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
  lib/                # Auth, proxy, and type utilities
  tests/              # Jest smoke test hitting API handlers
  middleware.ts       # Protects /dashboard routes
```

Route handlers under `app/api/` proxy requests to the FastAPI backend, injecting the authenticated session token server-side for SSR safety.

## Testing

- `pnpm test` runs `tests/smoke.spec.ts`, a Jest-based integration check that:
  - Verifies `/api/session` returns 401 when unauthenticated
  - Ensures the proxy login route properly forwards mocked requests

Add additional Jest or Playwright suites beside this smoke test as the migration progresses.

## Decommission Checklist (to run later)

- [ ] Feature parity validated against `frontend/` dashboards
- [ ] Automated tests cover ingestion, summarisation, and charting flows
- [ ] QA sign-off on auth paths (credentials + optional GitHub)
- [ ] Analytics/telemetry duplicated as needed
- [ ] Update docs, links, and infra configs to point to Next.js app
- [ ] Archive or remove Streamlit UI (`frontend/`) once parity is locked

> Keep the Streamlit UI untouched until every item above is complete.
