# Repository Guidelines

## Project Structure & Module Organization
- `backend/` houses the FastAPI service (`app.py` entrypoint, `models.py` for typed schemas, `services.py` for ingest logic); backend tests live in `backend/tests/`.
- `frontend/` contains Streamlit code: `core/` for API/dataframe helpers, `ui/` for shared widgets, and `pages/NN_Title.py` modules for dashboards. Frontend tests live in `frontend/tests/`.
- Root assets include Docker (`Dockerfile`, `docker-compose.yml`), automation (`Makefile`, `scripts/`), configuration (`config.yaml`, `.env`), and datasets (`data/`, `sample_data/`, `uploads/`).

## Build, Test, and Development Commands
- `pip install -r requirements.txt` installs Python dependencies; run once per environment.
- `python -m textblob.download_corpora` primes TextBlob data needed by sentiment features.
- `make up` spins up frontend and backend containers together; use when validating the integrated stack.
- `make dev` runs the backend API and hot-reload Streamlit profile for rapid iteration.
- Local terminals: `uvicorn backend.app:app --reload --port 8502` for the API and `streamlit run frontend/home.py` for the UI.

## Coding Style & Naming Conventions
- Target Python 3.10+, 4-space indentation, type hints, and descriptive docstrings.
- Keep parsing/data manipulation inside `backend/services.py` or `frontend/core/*.py`; reserve Streamlit pages for layout and wiring.
- Name new pages `NN_Title.py` to maintain sidebar ordering; use UPPER_SNAKE_CASE for constants and lower_snake_case for functions.
- Group imports by stdlib, third-party, then local modules.

## Testing Guidelines
- Pytest drives the suite; run `pytest frontend/tests -q` (or add `backend/tests` if relevant) before submitting.
- Mirror module names when adding tests; prefer fixtures from `frontend/tests/conftest.py` or lightweight sample payloads in `sample_data/`.
- Cover both happy-path and error handling for new ingestion or summarizer logic; unblock PRs only when coverage feels appropriate.

## Commit & Pull Request Guidelines
- Follow Title Case commit messages matching existing history (e.g., `Add Model Metadata Support`).
- Combine feature code with corresponding tests and docs in the same commit when feasible.
- PRs should explain user-facing impact, list manual verification (`pytest frontend/tests -q`, `make dev` with `sample_data/`), and attach screenshots/GIFs for UI updates.
- Reference related issues or TODOs so reviewers can trace scope quickly.

## Configuration & Environment Tips
- Copy `.env.example` to `.env` and adjust backend/frontend URLs before first run.
- Tweak analyzer defaults in `config.yaml`; restart the backend after changes.
- Keep large exports out of version control; store reusable fixtures in `sample_data/` to enable reproducible charts.
