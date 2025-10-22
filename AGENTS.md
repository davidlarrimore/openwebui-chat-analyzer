# Repository Guidelines

## Project Structure & Module Organization
- `backend/` hosts the FastAPI service: `app.py` (entrypoint), `models.py` (Pydantic schemas), and `services.py` (ingest and parsing logic). Tests live in `backend/tests/`.
- `frontend/` contains Streamlit code. Use `core/` for API/dataframe helpers, `ui/` for shared widgets, and `pages/NN_Title.py` for dashboards. Tests live in `frontend/tests/`.
- Root assets include Docker tooling (`Dockerfile`, `docker-compose.yml`), automation (`Makefile`, `scripts/`), configuration (`config.yaml`, `.env`), and datasets (`data/`, `sample_data/`, `uploads/`).

## Build, Test, and Development Commands
- `pip install -r requirements.txt` installs Python dependencies; run on new environments.
- `python -m textblob.download_corpora` primes corpora used by sentiment features.
- `make up` launches the full stack in Docker; `make dev` starts the API and Streamlit with reload.
- Local dev: `uvicorn backend.app:app --reload --port 8502` for the API, `streamlit run frontend/home.py` for the UI.
- Tests: `pytest frontend/tests -q` or append `backend/tests` when backend changes apply.

## Coding Style & Naming Conventions
- Target Python 3.10+, 4-space indentation, type hints, descriptive docstrings, and grouped imports (stdlib → third-party → local).
- Keep parsing/data manipulation in `backend/services.py` or `frontend/core/*.py`; pages focus on layout and wiring.
- Name new pages `NN_Title.py` to preserve sidebar order. Use `UPPER_SNAKE_CASE` for constants and `lower_snake_case` for functions.

## Testing Guidelines
- Pytest is the standard framework. Mirror module names for new tests and lean on fixtures in `frontend/tests/conftest.py`.
- Cover happy-path and error handling for ingestion/summarizer flows. Use samples from `sample_data/` to create deterministic cases.
- Run `pytest frontend/tests -q` (and backend where relevant) before submitting changes; ensure failures are addressed or documented.

## Commit & Pull Request Guidelines
- Follow existing history with Title Case commit messages (e.g., `Add Model Metadata Support`). Combine feature code, tests, and docs when feasible.
- PRs should call out user-facing impact, list verification steps (`pytest`, `make dev`, etc.), and attach UI screenshots or GIFs when altering Streamlit pages.
- Reference related issues or TODOs so reviewers can trace scope quickly. Keep large exports out of version control; rely on `sample_data/` for reproducible fixtures.

## Configuration & Environment Tips
- Copy `.env.example` to `.env` and adjust backend/frontend URLs before running locally.
- Tweak analyzer defaults in `config.yaml`; reload the backend after edits.
- Store reusable fixtures in `sample_data/` and keep `uploads/` clean of large artifacts before commits.
