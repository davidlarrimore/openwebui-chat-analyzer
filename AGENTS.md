# Repository Guidelines

## Project Structure & Module Organization
- `backend/` hosts the FastAPI service: `app.py` (entrypoint), `models.py` (Pydantic schemas), and `services.py` (ingest and parsing logic). Tests live in `backend/tests/`.
- `frontend-next/` contains the Next.js 14 dashboard. `app/` houses routes, `components/` holds shared UI elements, and `lib/` centralises API helpers plus type utilities. Jest tests live in `frontend-next/tests/`.
- Root assets include Docker tooling (`Dockerfile`, `docker-compose.yml`), automation (`Makefile`, `scripts/`), configuration (`config.yaml`, `.env`), and datasets (`data/`, `sample_data/`, `uploads/`).

## Build, Test, and Development Commands
- `pip install -r requirements.txt` installs Python dependencies; run on new environments.
- `python -m textblob.download_corpora` primes corpora used by sentiment features.
- `make up` launches the full stack in Docker; `make dev` starts the API plus the hot-reload Next.js profile.
- Local dev: `uvicorn backend.app:app --reload --port 8502` for the API, `pnpm dev` inside `frontend-next/` for the dashboard.
- Tests: `pytest backend/tests -q` for backend coverage and `pnpm test` inside `frontend-next/` for frontend smoke tests.

## Coding Style & Naming Conventions
- Target Python 3.10+, 4-space indentation, type hints, descriptive docstrings, and grouped imports (stdlib → third-party → local).
- Keep parsing/data manipulation in `backend/services.py` or `frontend-next/lib/*.ts`; pages focus on layout and wiring.
- Follow the Next.js project conventions: colocate route handlers under `app/`, share UI under `components/`, and keep types/helpers under `lib/`. Use `UPPER_SNAKE_CASE` for constants and `lowerCamelCase`/`lower_snake_case` per language norms.

## Testing Guidelines
- Pytest is the standard framework for the backend. Mirror module names for new tests and leverage fixtures in `backend/tests/conftest.py`.
- Cover happy-path and error handling for ingestion/summarizer flows. Use samples from `sample_data/` to create deterministic cases.
- Run `pytest backend/tests -q` and `pnpm test` (frontend) before submitting changes; ensure failures are addressed or documented.

## Commit & Pull Request Guidelines
- Follow existing history with Title Case commit messages (e.g., `Add Model Metadata Support`). Combine feature code, tests, and docs when feasible.
- PRs should call out user-facing impact, list verification steps (`pytest`, `pnpm test`, `make dev`, etc.), and attach UI screenshots or GIFs when altering the Next.js dashboards.
- Reference related issues or TODOs so reviewers can trace scope quickly. Keep large exports out of version control; rely on `sample_data/` for reproducible fixtures.

## Configuration & Environment Tips
- Copy `.env.example` to `.env` and adjust backend/frontend URLs before running locally.
- Tweak analyzer defaults in `config.yaml`; reload the backend after edits.
- Store reusable fixtures in `sample_data/` and keep `uploads/` clean of large artifacts before commits.
