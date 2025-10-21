# 💬 Open WebUI Chat Analyzer

Local analytics stack for exploring Open WebUI chat exports. The Streamlit UI remains the primary experience while a new Next.js frontend (`frontend-next/`) is under active development.

> The project is in a transitional phase: both frontends ship side by side while we migrate functionality from Streamlit to Next.js.

## Highlights

- Local FastAPI backend plus Streamlit UI—your exports never leave your machine
- Load data either by pulling directly from a running Open WebUI instance or by dropping exports into `data/`
- Backend auto-loads the latest `all-chats-export-*.json` from `data/` and supports an optional `users.csv` for friendly names
- Overview metrics for chats, messages, per-role activity, file uploads, and approximate token volume
- Filters every visualization by Open WebUI user and model
- Built-in summarizer generates one-line chat headlines using local sentence embeddings plus the bundled Ollama service (with Open WebUI completions as a fallback)
- Time analysis (daily trend, conversation length, hour-by-day heatmap) and content analysis (word cloud, message length)
- Sentiment breakdown with TextBlob plus full-text search, paginated browsing, and per-thread JSON downloads
- CSV exports for both chat metadata and individual messages
- Preview Next.js dashboard with richer charts, modern auth, and dashboard layouts (work in progress)

## Configuration

1. Copy `.env.example` to `.env`.
2. Configure backend connectivity:
   - `OWUI_API_BASE_URL` – Primary URL the Streamlit UI should hit (`http://localhost:8502` locally, `http://backend:8502` for Docker Compose).
   - `OWUI_API_FALLBACKS` – Optional comma-separated list of backup URLs the frontend will try before surfacing an error.
   - `OWUI_API_ALLOWED_ORIGINS` – Comma-separated list of origins permitted to call the FastAPI backend.
   - `OWUI_DATA_DIR` – Directory where default exports live (relative to the project root).
3. (Optional) Prefill the Direct Connect form:
   - `OWUI_DIRECT_HOST` – Default Open WebUI base URL shown on the Load Data page.
   - `OWUI_DIRECT_API_KEY` – Optional API token that appears in the Direct Connect form (stored only in your local `.env`).
4. (Optional) Tune the summarizer and GenAI helpers:
   - `SUMMARY_MAX_CHARS` / `SALIENT_K` – Control headline length and how many salient utterances feed the LLM.
   - `EMB_MODEL` – Sentence-transformer used to pick salient lines (`sentence-transformers/all-MiniLM-L6-v2` by default).
   - `OLLAMA_BASE_URL` / `OLLAMA_PORT` – Where the FastAPI backend reaches the Ollama runtime (`http://host.docker.internal:11434` when the backend runs in Docker, `http://localhost:11434` for bare-metal).
   - `OLLAMA_PRELOAD_MODELS` – Space-separated list of models pulled on container start (defaults to summary, long-form, and embedding models).
   - `OLLAMA_SUMMARY_MODEL` / `OLLAMA_LONGFORM_MODEL` / `OLLAMA_EMBED_MODEL` – Defaults for summaries (`llama3.1`), long-form generation (`phi3:mini`), and embeddings (`nomic-embed-text`).
   - `OLLAMA_DEFAULT_TEMPERATURE` / `OLLAMA_TIMEOUT_SECONDS` – Runtime defaults applied to requests.
   - `OWUI_COMPLETIONS_MODEL` – Chat completion model identifier requested from your Open WebUI deployment (legacy fallback path).
5. Restart the backend (and Streamlit) after changing environment variables. The first summarizer run will download embeddings locally.

## Input Data

Load data through either workflow:

- **Direct Connect** – Use the **Load Data → Direct Connect** panel to point the analyzer at a live Open WebUI deployment. Provide the base URL (for example `http://localhost:3000`) and an API key; the backend will pull chats and users via `/api/v1/openwebui/sync`, persist them locally, and kick off the summarizer.
- **File Uploads / Local Directory** – Export `all-chats-export-*.json` from Open WebUI (Admin Panel → **Settings → Data & Privacy → Export All Chats**) and optionally `users.csv` (Admin Panel → **Settings → Database → Export Users** with `user_id` plus a display column) plus `models.json` captured from the `/api/v1/models` endpoint for friendly model names. Drop the files in `data/` for automatic loading on startup or upload them on the Load Data page. Uploaded artifacts live under `uploads/`.

## Quick Start

### Option A – Docker (recommended)

```bash
git clone https://github.com/davidlarrimore/openwebui-chat-analyzer.git
cd openwebui-chat-analyzer
cp .env.example .env          # optional: set OWUI_API_BASE_URL to http://backend:8502 for docker compose
make up            # or: docker compose up -d
```

The Streamlit UI listens on `http://localhost:8501`, the FastAPI backend on `http://localhost:8502`, and the experimental Next.js frontend on `http://localhost:8503`. Use `make down` to stop, `make logs` to tail the container, and `make help` for the complete command catalog.
The compose stack now bind-mounts the `frontend/` directory, so Streamlit changes land immediately without rebuilding.

### Ollama GenAI Service

- The main `docker-compose.yml` includes an `ollama` service exposed on `http://localhost:11434`; it comes up automatically with `make up`.
- When running services outside Docker, point `OLLAMA_BASE_URL` to `http://localhost:11434`.
- Models listed in `OLLAMA_PRELOAD_MODELS` are pulled automatically on startup by `scripts/ollama-bootstrap.sh`; edit the env var to change the preload set.
- By default the `.env.example` preloads `llama3.1` (summaries), `phi3:mini` (general generation), and `nomic-embed-text` (embeddings); swap in heavier weights like `llama3.1:70b` when you have the hardware.
- Pull additional models manually with `docker compose run --rm ollama ollama pull <model>` and inspect the cache via `docker compose exec ollama ollama list`.
- Tune runtime behaviour with the `OLLAMA_*` environment variables (`OLLAMA_DEFAULT_TEMPERATURE`, `OLLAMA_TIMEOUT_SECONDS`, etc.) and restart the backend.
- For GPU acceleration, add the appropriate `--gpus` flag or device mapping to the `ollama` service before rebuilding the stack.

### Handy Make Commands

- `make help` – List every available helper target with a short description.
- `make up` / `make down` – Start or stop all services (backend, Streamlit frontend, and the Next.js preview).
- `make up-frontend` / `make up-backend` – Launch a single service.
- `make build` / `make rebuild` – Build images (all) or rebuild and restart.
- `make destroy` – Remove all services, volumes, and orphan containers.
- `make logs` / `make logs-frontend` / `make logs-backend` – Tail logs.
- `make restart` / `make restart-frontend` / `make restart-backend` – Restart running services.
- `make dev` – Start the backend plus the hot-reload frontend profile.

Run `make help` for the full list (build, deploy, debug, tooling helpers, etc.).

### Option B – Local Python environment

```bash
git clone https://github.com/davidlarrimore/openwebui-chat-analyzer.git
cd openwebui-chat-analyzer
python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
python -m textblob.download_corpora   # first run only
uvicorn backend.app:app --reload --port 8502  # terminal 1
streamlit run frontend/home.py
```

Run the backend and Streamlit UI in separate terminals (or background the FastAPI process) so the dashboard can reach `http://localhost:8502`.

### Option C – Guided setup scripts

Run `scripts/setup.sh` for an interactive wizard that can prepare either Docker or the virtual environment. After setup, use the `make` targets (`make up`, `make down`, `make logs`, etc.) for day-to-day lifecycle commands.

Once Streamlit is running, visit `http://localhost:8501`, open the **Load Data** page, and either Direct Connect to Open WebUI or upload your latest exports to populate the dashboard.

## Backend API

The Streamlit front-end now talks to a FastAPI service that normalizes and serves the chat exports. Key endpoints:

- `GET /api/v1/datasets/meta` – current dataset identifier, row counts, source label, and last updated timestamp.
- `GET /api/v1/chats` / `GET /api/v1/messages` / `GET /api/v1/users` – hydrated chat metadata, messages, and optional user directory.
- `POST /api/v1/openwebui/sync` – pull chats and users directly from Open WebUI using a hostname plus API token.
- `POST /api/v1/uploads/chat-export` – upload a new `all-chats-export*.json`; replaces the in-memory dataset and bumps the dataset id.
- `POST /api/v1/uploads/users` – upload a companion `users.csv` for friendly display names.
- `POST /api/v1/datasets/reset` – delete all stored chat, message, and user records plus metadata.
- `GET /api/v1/summaries/status` / `POST /api/v1/summaries/rebuild` – monitor or requeue the background summarizer job.

Run `uvicorn backend.app:app --reload` during development to keep the API available to the dashboard.

## Automated Chat Summaries

- Summaries are persisted in the `summary_128` field for each chat and surface throughout the Browse and Overview experiences.
- Every dataset update (direct sync or file upload) queues the summarizer; progress is shown in the Load Data processing log with toast notifications.
- The summarizer picks salient utterances with `sentence-transformers/all-MiniLM-L6-v2`, then calls the bundled Ollama service (`OLLAMA_SUMMARY_MODEL`) with an automatic fallback to the Open WebUI completions endpoint at `OWUI_DIRECT_HOST`.
- Rebuild summaries anytime from **Load Data → Admin Tools → Rerun summaries** or through the API (`POST /api/v1/summaries/rebuild` + `/summaries/status`).

## Dashboard Tour

- **Load Data page**: View dataset stats, stream processing logs, Direct Connect to Open WebUI, upload exports (with optional `users.csv`), and access admin tools to reset or rerun summaries.
- **Overview metrics**: Totals and averages for chats, messages, per-role counts, file uploads, and approximate input/output token volumes (derived from character length).
- **Model usage**: Horizontal bar chart plus quick stats for each model encountered across the filtered dataset.
- **Filters**: Slice all visuals by Open WebUI user and model; filter changes reset pagination so the browse experience stays predictable.
- **Tabs**:
  - `🧾 Overview`: Metrics recomputed for the active filters.
  - `📈 Time Analysis`: Daily activity timeline, conversation-length histogram, and hour-by-day heatmap.
  - `💭 Content Analysis`: Word cloud for user messages, average message length by role, and length distribution.
  - `😊 Sentiment`: TextBlob polarity grouped into positive/neutral/negative segments with a time series and supporting metrics.
  - `🔍 Search`: Full-text search with role filter, highlighted matches, attachment badges, and per-thread JSON downloads.
  - `🗂 Browse Data`: Paginated conversation browser with expanders, attachment indicators, and download buttons.
- **Export section**: Download enriched chat metadata and message tables as CSV for external analysis.

## Working With the Data

- CSV downloads contain the same columns the dashboard uses, making follow-on analysis in pandas, spreadsheets, or BI tools straightforward.
- Per-thread JSON downloads include metadata, ISO timestamps, attachments, and every message shown in the interface.
- Sentiment scores and token estimates are heuristic: tokens are inferred from character counts, and sentiment uses TextBlob’s polarity scale (−1 to 1).

## Sample Data

`sample_data/sample_data_extract.json` and `sample_data/sample_users.csv` let you explore the dashboard without waiting for a fresh export. Copy them to `data/` or upload them through the UI to see the charts populate immediately.

## Development Notes

- `docker-compose.yml` defines production, development (live reload), and optional Nginx proxy profiles. Use `docker compose --profile development up frontend-dev` or `make dev` for auto-reload.
- The `Makefile` centralizes build and lifecycle commands — start with `make help`.
- Python dependencies are split between `backend/requirements.txt` and `frontend/requirements.txt` (aggregate via root `requirements.txt`). The multi-stage Dockerfile builds dedicated images for each service and downloads the TextBlob corpora for the Streamlit frontend.
- The root Dockerfile now includes a dedicated Next.js stage so you can build `frontend-next` alongside the Python services (`docker build --target frontend-next .`). The preview app also has its own `frontend-next/Dockerfile` optimized for the standalone project.

## Next.js Preview Frontend

- The preview lives under `frontend-next/` and mirrors the backend API endpoints used by Streamlit.
- Run `docker compose up frontend-next` (or `pnpm dev` inside `frontend-next/`) to explore the new dashboard at `http://localhost:8503`.
- Environment variables such as `FRONTEND_NEXT_BACKEND_BASE_URL` and `NEXTAUTH_SECRET` control API routing and auth. See `.env.example` for defaults.
- Expect ongoing changes while we migrate Streamlit pages — gaps or regressions are tracked in `AGENTS.md` and upcoming milestones.
## Frontend Architecture

### Adding a New Page or Chart
1. Copy an existing module under `frontend/pages/` (for example `01_Overview.py`) or create a new `NN_Title.py` file so it appears in Streamlit's sidebar.
2. Call `ensure_data_ready()` from `frontend.ui.page_state` to load dataset metadata, render the shared controls (direct connect, uploads), and fetch cached dataframes.
3. Use `components.render_filters(...)` to reuse the global user/model filters; pass a unique `filter_prefix` so widget keys stay distinct.
4. Build figures with helpers in `frontend/ui/charts.py` or add new chart builders there if you need fresh visuals, then render them with `st.plotly_chart`.
5. Keep Streamlit calls inside the page or `frontend/ui/components.py`; push data munging into `frontend/core/processing.py` or `frontend/core/api.py` so it remains testable.

### Data Fetching & Processing
- HTTP access to the FastAPI backend lives in `frontend/core/api.py`, which exposes typed helpers like `get_dataset_meta()` and `build_processed_data()`.
- Dataset transformations, metrics, and filtering logic are implemented in `frontend/core/processing.py`.
- Streamlit caching for expensive calls is centralized in `frontend/ui/data_access.py`, while session helpers (reruns, widget state) live in `frontend/utils/state.py`.
- UI widgets and charts are separated: `frontend/ui/components.py` handles layout, and `frontend/ui/charts.py` returns ready-to-render Plotly figures or word clouds.

## Privacy & Storage

All requests stay on your machine—the Streamlit UI only talks to the bundled FastAPI service. Uploaded files remain under the repository (`data/` and `uploads/`) until you remove them.

## Troubleshooting

- If Streamlit crashes during sentiment analysis, install the TextBlob corpora with `python -m textblob.download_corpora`.
- Summaries failing or timing out? Confirm your Open WebUI deployment at `OWUI_DIRECT_HOST` is reachable, the API key is valid, and that the sentence-transformers model has been downloaded (first run may take a minute).
- Seeing 5xx errors from `/api/v1/genai/*`? Ensure the `ollama` container is healthy (`docker compose ps`, `docker compose logs ollama`) and that the required models are preloaded (`docker compose exec ollama ollama list`).
- Some environments need a font package for `wordcloud`; installing system fonts (for example `sudo apt-get install fonts-dejavu`) fixes blank visuals.
- Adjust `STREAMLIT_SERVER_PORT` or the Docker port mapping if 8501 is already in use.
- Seeing “Unable to connect to the backend API”? Make sure `uvicorn backend.app:app --port 8502` (or the Docker `backend` service) is running and reachable.

## License

MIT — see `LICENSE` for the full text.
