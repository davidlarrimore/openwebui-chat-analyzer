# Repository Guidelines

## Project Structure & Module Organization
The FastAPI service lives under `backend/` (entrypoint `app.py`, typed data models in `models.py`, ingest logic in `services.py`). Streamlit code sits in `frontend/`, split into `core/` for API and dataframe helpers, `ui/` for shared widgets, and `pages/NN_Title.py` modules that define each dashboard page. Docker assets (`Dockerfile`, `docker-compose.yml`) and lifecycle helpers (`Makefile`, `scripts/`) are in the root. Default datasets load from `data/`, while throwaway uploads land in `uploads/`; sample fixtures live in `sample_data/` for quick testing.

## Build, Test, and Development Commands
Use `make up` to start the frontend and backend containers together, or `make dev` to run the backend plus the hot-reload Streamlit profile. For local Python work, run `uvicorn backend.app:app --reload --port 8502` and `streamlit run frontend/home.py` in separate terminals. Install dependencies with `pip install -r requirements.txt`, and fetch TextBlob data once via `python -m textblob.download_corpora`. Run automated checks with `pytest frontend/tests -q`; `make logs-backend` and `make logs-frontend` are handy when debugging container runs.

## Coding Style & Naming Conventions
Code is Python 3.10+ with 4-space indentation, type hints, and descriptive docstrings. Keep parsing and data munging inside `backend/services.py` or `frontend/core/*.py`, leaving Streamlit files primarily for layout. Name new Streamlit pages `NN_Title.py` to preserve sidebar order, and mirror existing module-level constants (UPPER_SNAKE_CASE) and functions (lower_snake_case). Follow import grouping (stdlib, third-party, local) and keep functions pure where practical so they stay easy to test.

## Testing Guidelines
Pytest drives the suite; add cases under `frontend/tests/` (mirroring module names) or `backend/tests/` if you introduce backend logic. Focus on dataframe utilities and service functions—use the fixtures in `frontend/tests/conftest.py` or craft lightweight sample payloads rather than real exports. When adding summarizer or ingestion features, assert both happy-path and error handling; aim to keep new logic covered before opening a PR.

## Commit & Pull Request Guidelines
Commit messages follow the short, Title Case summaries seen in `git log` (for example `Add model metadata support`). Push cohesive changesets that pair feature code with tests. Pull requests should explain the user-facing impact, list manual verification steps (e.g., `pytest frontend/tests -q`, `make dev` against `sample_data/`), and include screenshots or GIFs whenever the UI shifts. Link issues or TODO references in the description so maintainers can track scope quickly.

## Configuration & Environment Tips
Copy `.env.example` to `.env` and adjust Streamlit ↔ backend URLs before first run. Tune analyzer defaults in `config.yaml`—changes require a backend restart. Keep large exports out of version control; drop reusable fixtures inside `sample_data/` so collaborators can reproduce charts without touching production data.
